"""Bounded, resumable training on streaming JSONL data and scratch tokenizers."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import mmap
import os
import random
import struct
import tempfile
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .formatting import collate_examples, format_example, question_and_history, text_value
from .model import ArenaModel, ModelConfig


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_hash(path):
    path = Path(path)
    return file_sha256(path / "tokenizer.json" if path.is_dir() else path)


def retrieval_signature(index_path):
    """Hash external retrieval inputs once at startup, never for each sample."""
    if not index_path:
        return None
    directory = Path(index_path)
    reranker = directory / "reranker.json"
    return {"version": 1, "index_sha256": file_sha256(directory / "index.sqlite3"),
            "reranker_sha256": file_sha256(reranker) if reranker.is_file() else None}


def atomic_torch_save(payload, path):
    """Serialize once and atomically replace, tolerating short Windows locks.

    A failed replacement never removes the previous checkpoint. Keep a fully
    serialized temporary file for recovery; only incomplete writes are cleaned
    up, and cleanup errors must not hide the original failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    serialized = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        serialized = True
        retry_delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
        for attempt in range(len(retry_delays) + 1):
            try:
                os.replace(temporary, path)
                return
            except OSError as error:
                # WinError 32/33 are sharing/lock violations, not generic
                # permission errors. Retrying disk-full or access-denied errors
                # would merely hide an actionable failure.
                if getattr(error, "winerror", None) not in (32, 33) or attempt == len(retry_delays):
                    raise
                time.sleep(retry_delays[attempt])
    except BaseException as error:
        if serialized:
            error.add_note(f"Checkpoint replacement failed; the previous {path} was not removed. "
                           f"Complete temporary checkpoint retained at {temporary}. "
                           "Release the file lock and resume from the existing last.pt, or inspect the retained file before recovery.")
        else:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(f"Could not remove incomplete temporary file {temporary}: {cleanup_error}")
        raise


def rng_state():
    result = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path, model, optimizer=None, scaler=None, training_state=None,
                    tokenizer_path=None, training_config=None):
    if tokenizer_path is None:
        raise ValueError("A tokenizer path is required for a verifiable checkpoint")
    atomic_torch_save({
        "format_version": 1, "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "training_state": dict(training_state or {}),
        "training_config": dict(training_config or {}),
        "tokenizer_sha256": tokenizer_hash(tokenizer_path),
        "rng_state": rng_state(),
    }, path)


def load_model_checkpoint(checkpoint_path, device="cpu", tokenizer_path=None):
    """Load a locally produced checkpoint; never use this on untrusted pickle files."""
    from .tokenization import ArenaTokenizer
    if tokenizer_path is None:
        raise ValueError("Pass the exact tokenizer used for training")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("tokenizer_sha256") != tokenizer_hash(tokenizer_path):
        raise ValueError("Tokenizer SHA256 does not match the checkpoint")
    tokenizer = ArenaTokenizer.load(tokenizer_path)
    config = ModelConfig(**checkpoint["model_config"])
    for name in ("vocab_size", "pad_id", "bos_id", "eos_id"):
        if getattr(config, name) != getattr(tokenizer, name):
            raise ValueError(f"Tokenizer/model mismatch: {name}")
    model = ArenaModel(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint


class IndexedJsonlDataset(Dataset):
    """Disk-backed byte offsets: memory is O(batch), not O(corpus text).

    An adjacent binary index stores little-endian uint64 offsets and is memory
    mapped. Rebuilt after size/mtime/split changes; one row is parsed per request.
    """
    def __init__(self, path, tokenizer, split="train", stage="dialogue", seed=42,
                 max_source_length=512, max_target_length=96, retriever=None,
                 retrieval_dropout=0.2, cache_dir=None, retrieval_cache=None,
                 persona_dropout=0.0, persona_shuffle=False, language_window_policy="legacy"):
        self.path = Path(path).resolve()
        self.tokenizer, self.split, self.stage, self.seed = tokenizer, split, stage, seed
        self.max_source_length, self.max_target_length = max_source_length, max_target_length
        self.retriever, self.retrieval_dropout = retriever, retrieval_dropout
        self.retrieval_cache = retrieval_cache
        self.persona_dropout, self.persona_shuffle = persona_dropout, persona_shuffle
        self.language_window_policy = language_window_policy
        self._context_cache = OrderedDict()
        self._handle = self._index_handle = self._offsets = None
        if not 0 <= persona_dropout <= 1:
            raise ValueError("persona_dropout must be in [0,1]")
        if stage not in ("language", "dialogue"):
            raise ValueError("stage must be language or dialogue")
        if language_window_policy not in ("legacy", "full-span"):
            raise ValueError("language_window_policy must be legacy or full-span")
        if language_window_policy == "full-span" and (max_source_length < 3 or max_target_length < 2):
            raise ValueError("full-span requires source length >=3 and target length >=2")
        stat = self.path.stat()
        signature = {"path": str(self.path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                     "split": split, "stage": stage, "version": 3}
        cache = Path(cache_dir or self.path.parent / ".arena_offsets")
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        self.index_path, metadata_path = cache / f"{key}.idx", cache / f"{key}.json"
        if not self.index_path.exists() or not metadata_path.exists():
            count, malformed = 0, 0
            content_hash = hashlib.sha256()
            temporary = self.index_path.with_suffix(f".{os.getpid()}.tmp")
            with self.path.open("rb") as source, temporary.open("wb") as output:
                while True:
                    offset = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    content_hash.update(line)
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        malformed += 1
                        continue
                    row_split = row.get("split")
                    if split == "validation":
                        accepted = row_split in ("validation", "val", "dev")
                    else:
                        accepted = row_split == split
                    if stage == "language":
                        # Dialogue history can contain rejected earlier answers.
                        # It must never become a language-model training target.
                        eligible = row.get("kind") == "text" and bool(row.get("text"))
                    else:
                        eligible = row.get("kind") == "dialogue" and bool(row.get("response"))
                    if accepted and eligible:
                        if not row.get("group_id"):
                            raise ValueError(f"Missing group_id at byte {offset}: {self.path}")
                        output.write(struct.pack("<Q", offset))
                        count += 1
            if malformed:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"{malformed} malformed JSONL records in {self.path}; fix input first")
            os.replace(temporary, self.index_path)
            metadata_path.write_text(json.dumps(signature | {"count": count, "sha256": content_hash.hexdigest()}), encoding="utf-8")
        self.signature = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.count = self.index_path.stat().st_size // 8
        if not self.count:
            raise ValueError(f"No {stage} examples with split={split!r} in {self.path}")

    def __len__(self):
        return self.count

    def _open(self):
        if self._handle is None:
            self._handle = self.path.open("rb")
            self._index_handle = self.index_path.open("rb")
            self._offsets = mmap.mmap(self._index_handle.fileno(), 0, access=mmap.ACCESS_READ)

    def record(self, index):
        self._open()
        offset = struct.unpack_from("<Q", self._offsets, index * 8)[0]
        self._handle.seek(offset)
        return json.loads(self._handle.readline())

    def close(self):
        for name in ("_offsets", "_index_handle", "_handle"):
            value = getattr(self, name, None)
            if value is not None:
                value.close()
                setattr(self, name, None)

    def __del__(self):
        self.close()

    def _language_example(self, row, rng):
        if row.get("kind") != "text" or not row.get("text"):
            raise ValueError("Language warmup accepts only nonempty kind='text' records")
        text = text_value(row.get("text"))
        ids = self.tokenizer.encode(text)
        if len(ids) < 2:
            ids = ids + [self.tokenizer.eos_id]
        if self.language_window_policy == "full-span":
            return self._full_span_language_example(ids, rng)
        # Keep legacy window selection and RNG draw order unchanged for old runs.
        max_window = self.max_source_length + self.max_target_length - 4
        start = rng.randrange(max(1, len(ids) - max_window + 1))
        ids = ids[start:start + max_window]
        if rng.random() < 0.5 and len(ids) >= 4:
            # Causal continuation: the target is absent from the source.
            cut = min(self.max_source_length - 2, max(1, len(ids) // 2))
            source, target = ids[:cut], ids[cut:cut + self.max_target_length - 1]
        else:
            # Denoise spans, removing 30-40% rather than training identity only.
            target = ids[:self.max_target_length - 1]
            source, i = [], 0
            mask_id = self.tokenizer.token_id("<mask>")
            while i < len(target):
                if rng.random() < 0.3:
                    source.append(mask_id)
                    i += rng.randint(1, 3)
                else:
                    source.append(target[i])
                    i += 1
            if target and source == target:
                source[rng.randrange(len(source))] = mask_id
        source = [self.tokenizer.bos_id, self.tokenizer.token_id("<history>")] + source[:self.max_source_length - 2]
        forbidden = {self.tokenizer.pad_id, self.tokenizer.bos_id, self.tokenizer.eos_id,
                     self.tokenizer.token_id("<mask>"), self.tokenizer.token_id("<history>")}
        return {"input_ids": source, "attention_mask": [True] * len(source),
                "copy_mask": [x not in forbidden for x in source],
                "retrieval_mask": [False] * len(source),
                "labels": target + [self.tokenizer.eos_id], "retrieval_target": -1}

    def _full_span_language_example(self, ids, rng):
        """Select targets across the whole block, then construct their source.

        Continuation only sees preceding tokens. Denoising reconstructs a random
        target span, so the end of a short block is no longer unreachable.
        """
        target_budget = self.max_target_length - 1
        if rng.random() < 0.5:
            cut = rng.randint(1, len(ids) - 1)
            source = ids[max(0, cut - (self.max_source_length - 2)):cut]
            target = ids[cut:cut + target_budget]
        else:
            width = min(len(ids), target_budget)
            start = rng.randrange(len(ids) - width + 1)
            target = ids[start:start + width]
            source, i = [], 0
            mask_id = self.tokenizer.token_id("<mask>")
            while i < len(target):
                if rng.random() < 0.3:
                    source.append(mask_id)
                    i += rng.randint(1, 3)
                else:
                    source.append(target[i])
                    i += 1
            if target and source == target:
                source[rng.randrange(len(source))] = mask_id
        source = [self.tokenizer.bos_id, self.tokenizer.token_id("<history>")] + source[:self.max_source_length - 2]
        forbidden = {self.tokenizer.pad_id, self.tokenizer.bos_id, self.tokenizer.eos_id,
                     self.tokenizer.token_id("<mask>"), self.tokenizer.token_id("<history>")}
        return {"input_ids": source, "attention_mask": [True] * len(source),
                "copy_mask": [x not in forbidden for x in source],
                "retrieval_mask": [False] * len(source),
                "labels": target + [self.tokenizer.eos_id], "retrieval_target": -1}

    @staticmethod
    def _history_memory(row, persona):
        from .memory import ConversationMemory
        question, _ = question_and_history(row)
        memory_state = ConversationMemory(persona=persona)
        for turn in row.get("context", []):
            if not isinstance(turn, dict):
                continue
            if turn.get("role") == "assistant":
                memory_state.add_assistant(text_value(turn))
            else:
                memory_state.add_user(text_value(turn))
        return memory_state.retrieve(question, limit=6)

    def __getitem__(self, index):
        row = self.record(index)
        rng = random if self.split == "train" else random.Random(self.seed + index)
        if self.stage == "language":
            return self._language_example(row, rng)
        cached = self._context_cache.get(index)
        if cached is None:
            question, history = question_and_history(row)
            retrieved = row.get("retrieved", [])
            if self.retrieval_cache is not None:
                if not row.get("id"):
                    raise ValueError("Retrieval cache lookup requires a record id")
                retrieved = self.retrieval_cache.get(row["id"], row["group_id"])
                if retrieved is None:
                    raise ValueError(f"Retrieval cache entry missing for id={row['id']!r}, group_id={row['group_id']!r}; finish rebuilding the cache before training")
                if any(not isinstance(item, dict) or not item.get("group_id") or item["group_id"] == row["group_id"] for item in retrieved):
                    raise ValueError("Retrieval cache contains an untraceable or same-dialogue candidate")
            elif self.retriever is not None:
                retrieved = self.retriever.search(question, history=history, persona=row.get("persona"),
                                                 k=2, exclude_groups={row["group_id"]})
            elif retrieved:
                # Offline retrieval preprocessing must preserve group ids for audit.
                if isinstance(retrieved, dict):
                    retrieved = [retrieved]
                if any(not isinstance(x, dict) or not x.get("group_id") for x in retrieved):
                    raise ValueError("Precomputed retrieval needs candidate group_id; untraceable text is rejected")
                retrieved = [x for x in retrieved if x["group_id"] != row["group_id"]]
            memory = row.get("memory")
            if memory is None:
                memory = self._history_memory(row, row.get("persona"))
            cached = {"retrieved": retrieved, "memory": memory, "history_memory": None}
            self._context_cache[index] = cached
            if len(self._context_cache) > 8192:
                self._context_cache.popitem(last=False)
        else:
            self._context_cache.move_to_end(index)
        retrieved, memory = cached["retrieved"], cached["memory"]
        if self.split == "train" and row.get("persona"):
            if self.persona_dropout and rng.random() < self.persona_dropout:
                row = {**row, "persona": []}
                # Precomputed/derived memory can contain the same persona facts.
                # Rebuild from actual history when dropping persona; this keeps
                # facts spoken earlier but cannot reintroduce a hidden profile.
                if cached["history_memory"] is None:
                    cached["history_memory"] = self._history_memory(row, [])
                memory = cached["history_memory"]
            elif self.persona_shuffle:
                persona = row["persona"]
                if isinstance(persona, list):
                    persona = list(persona)
                    rng.shuffle(persona)
                elif isinstance(persona, dict):
                    persona = dict(persona)
                    facts = list(persona.get("facts", []))
                    rng.shuffle(facts)
                    persona["facts"] = facts
                else:
                    raise TypeError("persona_shuffle requires persona facts as a list or dict")
                row = {**row, "persona": persona}
        if self.split == "train" and rng.random() < self.retrieval_dropout:
            retrieved = []
        return format_example(row, self.tokenizer, self.max_source_length,
                              self.max_target_length, retrieved=retrieved, memory=memory)


class EpochBatchSampler(Sampler):
    def __init__(self, size, batch_size, seed, epoch=0, start_batch=0, shuffle=True):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.start_batch = epoch, start_batch
        self.shuffle = shuffle

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)) if self.shuffle else torch.arange(self.size)
        for start in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield order[start:start + self.batch_size].tolist()

    def __len__(self):
        return max(0, math.ceil(self.size / self.batch_size) - self.start_batch)


def normalize_source_weights(value):
    """Canonical source shares for configuration comparisons and sampling."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("source-weights must be a JSON object") from error
    if not isinstance(value, dict) or not value:
        raise ValueError("source-weights must be a nonempty object of positive finite numbers")
    for name, weight in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("source-weights names must be nonempty strings")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("source-weights must contain positive finite numbers")
    maximum = max(value.values())
    scaled = {name: value[name] / maximum for name in sorted(value)}
    denominator = math.fsum(scaled.values())
    result = {name: weight / denominator for name, weight in scaled.items()}
    if any(weight <= 0 for weight in result.values()):
        raise ValueError("source-weights relative magnitudes are too extreme")
    return result


def source_sampling_weights(dataset, source_weights):
    """Count eligible training rows, then assign equal weight within a source."""
    if dataset.split != "train" or dataset.stage != "dialogue":
        raise ValueError("source-weights are supported only for dialogue training")
    counts, names, row_sources = Counter(), {}, []
    for index in range(len(dataset)):
        name = dataset.record(index).get("source")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Weighted training row {index} lacks a source")
        if name not in names:
            names[name] = len(names)
        row_sources.append(names[name])
        counts[name] += 1
    missing, unknown = set(counts) - set(source_weights), set(source_weights) - set(counts)
    if missing or unknown:
        raise ValueError(f"source-weights must exactly cover training sources; missing={sorted(missing)}, unknown={sorted(unknown)}")
    per_source = torch.tensor([source_weights[name] / counts[name] for name in names], dtype=torch.float64)
    weights = per_source[torch.tensor(row_sources, dtype=torch.long)]
    return weights, dict(sorted(counts.items()))


class SourceWeightedBatchSampler(EpochBatchSampler):
    """Fixed-length replacement draws, reproducible from epoch and batch offset."""
    def __init__(self, weights, batch_size, seed, epoch=0, start_batch=0):
        super().__init__(len(weights), batch_size, seed, epoch, start_batch)
        self.weights = weights

    def __iter__(self):
        order = torch.multinomial(self.weights, self.size, replacement=True,
                                  generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for start in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield order[start:start + self.batch_size]


class ModelDiagnostics:
    """Detached, token-masked counters; materialize once per log/evaluation.

    Gate statistics exclude examples without retrieved input. Label coverage
    refers only to the optional auxiliary gate loss, not to NLL gradients.
    """
    def __init__(self):
        self.totals = None

    @torch.no_grad()
    def update(self, output, batch):
        labels = batch["labels"]
        valid = labels.ne(-100)
        retrieved = batch["retrieval_mask"].bool().any(1)
        supervised = batch["retrieval_targets"].ge(0) & retrieved
        p_generate = output["p_generate"].detach().float().squeeze(-1)
        values = torch.stack([
            (output["gate"].detach().float() * retrieved).sum(),
            retrieved.float().sum(),
            (p_generate * valid).sum(),
            valid.float().sum(),
            supervised.float().sum(),
            valid.new_tensor(labels.shape[0], dtype=torch.float),
        ])
        self.totals = values if self.totals is None else self.totals + values

    def summary(self):
        if self.totals is None:
            return {}
        gate_sum, retrieved, generate_sum, tokens, supervised, examples = self.totals.cpu().tolist()
        return {"mean_retrieval_gate": gate_sum / retrieved if retrieved else None,
                "mean_p_generate": generate_sum / tokens if tokens else None,
                "retrieval_supervision_fraction": supervised / retrieved if retrieved else None,
                "retrieved_example_fraction": retrieved / examples if examples else None,
                "supervised_retrieval_examples": int(supervised),
                "diagnostic_examples": int(examples), "diagnostic_target_tokens": int(tokens)}


def evaluate(model, loader, device, max_batches=20, amp_dtype=None, return_diagnostics=False):
    was_training = model.training
    model.eval()
    loss_sum, tokens = 0.0, 0
    diagnostics = ModelDiagnostics() if return_diagnostics else None
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= max_batches:
                    break
                batch = {key: value.to(device) for key, value in batch.items()}
                ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
                with ctx:
                    output = model(**batch)
                count = int(batch["labels"].ne(-100).sum())
                loss_sum += float(output["nll"]) * count
                tokens += count
                if diagnostics is not None:
                    diagnostics.update(output, batch)
    finally:
        model.train(was_training)
    result = loss_sum / max(1, tokens)
    if not math.isfinite(result):
        raise FloatingPointError("Validation loss is not finite")
    return {"nll": result, **diagnostics.summary()} if diagnostics is not None else result


def _parser(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config")
    known, _ = preliminary.parse_known_args(argv)
    defaults = {}
    if known.config:
        import yaml
        raw = yaml.safe_load(Path(known.config).read_text(encoding="utf-8")) or {}
        for key, value in raw.items():
            if key in ("model", "training") and isinstance(value, dict):
                defaults.update(value)
            else:
                defaults[key.replace("-", "_")] = value
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument("--data", "--train-data", dest="data")
    parser.add_argument("--val-data")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output-dir", default="runs/arena")
    parser.add_argument("--preset", choices=("small", "main", "large"), default="main")
    parser.add_argument("--stage", choices=("language", "dialogue"), default="dialogue")
    parser.add_argument("--steps", "--max-steps", dest="steps", type=int, default=2000, help="Total optimizer steps, including resumed steps")
    parser.add_argument("--stop-at-step", type=int, default=0,
                        help="Stop this segment at an absolute step without changing --steps or its LR horizon; 0 disables")
    parser.add_argument("--hours", "--max-hours", dest="hours", type=float, default=0, help="Cumulative run time budget including resumed elapsed time; 0 disables")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--validation-sampling", choices=("fixed-shuffle", "sequential"),
                        help="New runs use a fixed shuffled validation order; resume preserves the saved policy")
    parser.add_argument("--language-window-policy", choices=("legacy", "full-span"),
                        help="Language target selection; new runs default to legacy, resume inherits its saved policy")
    parser.add_argument("--source-weights",
                        help="Dialogue train source shares as a JSON object; omitted resume inherits saved shares")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint", "--init-from", dest="init_checkpoint", help="Start a new stage from weights; reset optimizer and counters")
    parser.add_argument("--retrieval-index")
    parser.add_argument("--retrieval-cache", help="Completed persistent retrieval cache; missing entries are errors, never online fallback")
    parser.add_argument("--retrieval-dropout", type=float, default=0.2)
    parser.add_argument("--persona-dropout", type=float,
                        help="Train-only probability of hiding persona and its derived memory; new runs default to 0, resume preserves saved value")
    parser.add_argument("--persona-shuffle", action=argparse.BooleanOptionalAction, default=None,
                        help="Shuffle training persona fact order; disabled for new runs unless requested, resume preserves saved value")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--encoder-layers", type=int)
    parser.add_argument("--decoder-layers", type=int)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    valid = {action.dest for action in parser._actions}
    unknown = set(defaults) - valid
    if unknown:
        parser.error(f"Unknown config keys: {sorted(unknown)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for name in ("data", "tokenizer"):
        if not getattr(args, name):
            parser.error(f"--{name} is required")
    for name in ("steps", "batch_size", "accumulation", "threads", "eval_every", "save_every", "log_every", "eval_batches"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be >=1")
    if args.hours < 0 or not 0 <= args.retrieval_dropout <= 1:
        parser.error("hours must be >=0; retrieval-dropout must be in [0,1]")
    if args.stop_at_step < 0 or args.stop_at_step > args.steps:
        parser.error("stop-at-step must be 0 or between 1 and steps")
    if args.persona_dropout is not None and not 0 <= args.persona_dropout <= 1:
        parser.error("persona-dropout must be in [0,1]")
    if args.resume and args.init_checkpoint:
        parser.error("Choose --resume OR --init-checkpoint")
    return args


def main(argv=None):
    from arena.console import configure_utf8
    configure_utf8()
    from .tokenization import ArenaTokenizer
    args = _parser(argv)
    destination = Path(args.output_dir)
    if not args.resume and any((destination / name).exists() for name in ("last.pt", "best.pt")):
        raise FileExistsError(f"Checkpoints already exist in {destination}. Use --resume or a new --output-dir; existing weights were not overwritten.")
    caller_rng = rng_state() if args.resume else None
    caller_threads = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation has no available CUDA GPU")
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    tokenizer = ArenaTokenizer.load(args.tokenizer)
    checkpoint = None
    if args.resume or args.init_checkpoint:
        model, tokenizer, checkpoint = load_model_checkpoint(args.resume or args.init_checkpoint, device, args.tokenizer)
        config = model.config
        if args.no_copy and config.copy_enabled:
            if args.resume:
                raise ValueError("Resume requires unchanged copy_enabled; use --init-from with --no-copy for a new run")
            # The pointer parameters remain present, so weights load unchanged.
            # Only a new run may change how they participate in the forward pass.
            config.copy_enabled = False
        # The actual saved architecture controls formatting on resume.
        args.max_source_length, args.max_target_length = config.max_source_length, config.max_target_length
    else:
        overrides = {key: getattr(args, key) for key in ("d_model", "n_heads", "encoder_layers", "decoder_layers") if getattr(args, key) is not None}
        config = ModelConfig.from_preset(args.preset, tokenizer.vocab_size, **overrides,
                     max_source_length=args.max_source_length, max_target_length=args.max_target_length,
                     pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
                     dropout=args.dropout, copy_enabled=not args.no_copy)
        model = ArenaModel(config).to(device)
    previous_sampling = checkpoint.get("training_config", {}).get("validation_sampling", "sequential") if args.resume else None
    if args.validation_sampling is None:
        args.validation_sampling = previous_sampling if args.resume else "fixed-shuffle"
    if args.resume and args.validation_sampling != previous_sampling:
        raise ValueError("Resume requires unchanged validation_sampling so best validation losses remain comparable; use --init-from for a new run")
    saved_window_policy = checkpoint.get("training_config", {}).get("language_window_policy", "legacy") if args.resume else "legacy"
    if args.language_window_policy is None:
        args.language_window_policy = saved_window_policy
    if args.resume and args.language_window_policy != saved_window_policy:
        raise ValueError("Resume requires unchanged language_window_policy; use --init-from for a new run")
    saved_source_weights = checkpoint.get("training_config", {}).get("source_weights") if args.resume else None
    if saved_source_weights is not None:
        # Validate, but retain the saved floating-point shares exactly. Repeated
        # normalization can round them differently across segment restarts.
        normalize_source_weights(saved_source_weights)
    if args.source_weights is None:
        args.source_weights = saved_source_weights
    else:
        args.source_weights = normalize_source_weights(args.source_weights)
    if args.resume:
        same = args.source_weights == saved_source_weights
        if not same and args.source_weights is not None and saved_source_weights is not None:
            same = (args.source_weights.keys() == saved_source_weights.keys()
                    and all(math.isclose(args.source_weights[key], saved_source_weights[key], rel_tol=1e-12, abs_tol=0)
                            for key in saved_source_weights))
        if not same:
            raise ValueError("Resume requires unchanged source_weights; use --init-from for a new run")
        args.source_weights = saved_source_weights
    if args.source_weights is not None and args.stage != "dialogue":
        raise ValueError("source-weights are supported only for dialogue training")
    for key, disabled in (("persona_dropout", 0.0), ("persona_shuffle", False)):
        saved = checkpoint.get("training_config", {}).get(key, disabled) if args.resume else disabled
        if getattr(args, key) is None:
            setattr(args, key, saved)
        if args.resume and getattr(args, key) != saved:
            raise ValueError(f"Resume requires unchanged {key}; use --init-from for a new run")
    if args.stage != "dialogue" and (args.persona_dropout or args.persona_shuffle):
        raise ValueError("Persona augmentation is supported only for the dialogue stage")
    persistent_cache = None
    current_cache_signature = None
    if args.retrieval_cache:
        from .retrieval_cache import RetrievalCache
        persistent_cache = RetrievalCache(args.retrieval_cache, index_dir=args.retrieval_index,
                                          readonly=True, require_complete=True)
        if not persistent_cache.complete:
            persistent_cache.close()
            raise ValueError("Retrieval cache is incomplete; finish rebuilding it before training")
        current_cache_signature = json.loads(json.dumps(persistent_cache.signature))
    if persistent_cache is not None and args.retrieval_index:
        # RetrievalCache already verified these files. Avoid rehashing the large
        # SQLite index for the second time during startup.
        current_retrieval_signature = {"version": 1,
            "index_sha256": current_cache_signature["index_sha256"],
            "reranker_sha256": current_cache_signature["reranker_sha256"]}
    else:
        current_retrieval_signature = retrieval_signature(args.retrieval_index)
    if args.resume:
        previous = checkpoint.get("training_config", {})
        if previous.get("retrieval_cache_signature") != current_cache_signature:
            if persistent_cache is not None:
                persistent_cache.close()
            raise ValueError("Resume retrieval cache signature changed; use --init-from for a new stage")
        if "retrieval_signature" not in previous:
            if previous.get("retrieval_index") or current_retrieval_signature is not None:
                if persistent_cache is not None:
                    persistent_cache.close()
                raise ValueError("Resume retrieval signature is missing in this checkpoint; use --init-from for a new stage")
        elif previous["retrieval_signature"] != current_retrieval_signature:
            if persistent_cache is not None:
                persistent_cache.close()
            raise ValueError("Resume retrieval index or reranker changed; use --init-from for a new stage")
    retriever = None
    if args.retrieval_index and persistent_cache is None:
        from .retrieval import HybridRetriever
        retriever = HybridRetriever.load(args.retrieval_index)
    data_kwargs = dict(tokenizer=tokenizer, stage=args.stage, seed=args.seed,
                       max_source_length=config.max_source_length, max_target_length=config.max_target_length,
                       retriever=retriever, retrieval_dropout=args.retrieval_dropout,
                       retrieval_cache=persistent_cache, persona_dropout=args.persona_dropout,
                       persona_shuffle=args.persona_shuffle, language_window_policy=args.language_window_policy)
    training = IndexedJsonlDataset(args.data, split="train", **data_kwargs)
    validation = IndexedJsonlDataset(args.val_data or args.data, split="validation", **data_kwargs)
    row_sampling_weights, source_counts = None, None
    if args.source_weights is not None:
        row_sampling_weights, source_counts = source_sampling_weights(training, args.source_weights)
    if persistent_cache is not None:
        expected_data = {"train": training.signature["sha256"], "val": validation.signature["sha256"]}
        if current_cache_signature.get("data_sha256") != expected_data:
            training.close()
            validation.close()
            persistent_cache.close()
            raise ValueError("Retrieval cache data signatures do not match train/validation files")
    collate = lambda examples: collate_examples(examples, tokenizer)
    validation_sampler = EpochBatchSampler(len(validation), args.batch_size, args.seed + 100000,
                                          shuffle=args.validation_sampling == "fixed-shuffle")
    validation_loader = DataLoader(validation, batch_sampler=validation_sampler,
                                   num_workers=0, collate_fn=collate,
                                   generator=torch.Generator().manual_seed(args.seed))
    params = list(model.named_parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for _, p in params if p.dim() >= 2], "weight_decay": args.weight_decay},
        {"params": [p for _, p in params if p.dim() < 2], "weight_decay": 0.0}], lr=args.lr, foreach=False)
    state = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "best_val_loss": float("inf"),
             "target_tokens": 0, "processed_tokens": 0, "elapsed_seconds": 0.0}
    if args.resume:
        if checkpoint.get("optimizer_state") is None:
            raise ValueError("Checkpoint has no optimizer state; use --init-checkpoint instead")
        previous = checkpoint.get("training_config", {})
        for key in ("stage", "batch_size", "accumulation", "seed"):
            if key in previous and previous[key] != getattr(args, key):
                raise ValueError(f"Resume requires unchanged {key}; use --init-checkpoint for a new run")
        for key, current in (("train_data_signature", training.signature), ("val_data_signature", validation.signature)):
            if key in previous and previous[key].get("sha256") != current["sha256"]:
                raise ValueError(f"Resume data changed: {key}; use --init-checkpoint for a new stage")
        state.update(checkpoint["training_state"])
        exhausted = []
        if state["step"] >= args.steps:
            exhausted.append("steps")
        if args.hours and state["elapsed_seconds"] >= args.hours * 3600:
            exhausted.append("hours")
        if args.stop_at_step and state["step"] >= args.stop_at_step:
            exhausted.append("stop_at_step")
        if exhausted:
            # All compatibility/input checks above still run. Without optimizer
            # updates there is no new model to validate or save: even tiny CUDA
            # rounding changes must not rewrite best.pt or duplicate reports.
            try:
                training.close()
                validation.close()
                if retriever is not None:
                    retriever.close()
                if persistent_cache is not None:
                    persistent_cache.close()
            finally:
                restore_rng(caller_rng)
                torch.set_num_threads(caller_threads)
            print(json.dumps({"event": "resume_complete", "step": state["step"],
                              "reason": exhausted, "optimizer_updates": 0,
                              "checkpoint": str(Path(args.resume).resolve())}), flush=True)
            return state
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state") or {})
        restore_rng(checkpoint["rng_state"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {"train_data_signature": training.signature, "val_data_signature": validation.signature,
                               "retrieval_signature": current_retrieval_signature,
                               "retrieval_cache_signature": current_cache_signature}
    (output_dir / "config.json").write_text(json.dumps({"model": asdict(config), "training": run_config}, ensure_ascii=False, indent=2), encoding="utf-8")
    start, prior_elapsed = time.monotonic(), state["elapsed_seconds"]
    history_path = output_dir / "metrics.jsonl"

    def log(values):
        line = json.dumps(values, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def save(name):
        state["elapsed_seconds"] = prior_elapsed + time.monotonic() - start
        save_checkpoint(output_dir / name, model, optimizer, scaler, state, args.tokenizer, run_config)

    def make_iterator():
        if row_sampling_weights is None:
            sampler = EpochBatchSampler(len(training), args.batch_size, args.seed, state["epoch"], state["batch_in_epoch"])
        else:
            sampler = SourceWeightedBatchSampler(row_sampling_weights, args.batch_size, args.seed,
                                                 state["epoch"], state["batch_in_epoch"])
        return iter(DataLoader(training, batch_sampler=sampler, num_workers=0, collate_fn=collate,
                               generator=torch.Generator().manual_seed(args.seed + state["epoch"])))

    iterator = make_iterator()
    model.train()
    log({"event": "start", "parameters": model.parameter_count(), "device": str(device),
         "amp": str(amp_dtype), "train_examples": len(training), "val_examples": len(validation),
         "validation_sampling": args.validation_sampling,
         "persona_dropout": args.persona_dropout, "persona_shuffle": args.persona_shuffle,
         "language_window_policy": args.language_window_policy,
         "source_weights": args.source_weights, "train_source_counts": source_counts,
         "schedule_steps": args.steps, "stop_at_step": args.stop_at_step,
         "step": state["step"], "stage": args.stage})
    if not args.resume:
        save("last.pt")
    window_start, window_tokens, window_all_tokens, last_val_step = time.monotonic(), 0, 0, -1
    window_diagnostics = ModelDiagnostics()
    try:
        segment_end = args.stop_at_step or args.steps
        while state["step"] < segment_end:
            # hours is the cumulative training-run budget, including resumed time.
            elapsed = prior_elapsed + time.monotonic() - start
            if args.hours and elapsed >= args.hours * 3600:
                break
            step = state["step"]
            if step < args.warmup_steps:
                lr = args.lr * (step + 1) / max(1, args.warmup_steps)
            else:
                progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
                lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, progress))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(args.accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    state["epoch"] += 1
                    state["batch_in_epoch"] = 0
                    iterator = make_iterator()
                    batch = next(iterator)
                state["batch_in_epoch"] += 1
                batch = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}
                ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
                with ctx:
                    result = model(**batch)
                    loss = result["loss"] / args.accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at optimizer step {step}; last.pt is intact")
                scaler.scale(loss).backward()
                window_diagnostics.update(result, batch)
                total_loss += float(loss.detach())
                count = int(batch["labels"].ne(-100).sum())
                processed_count = count + int(batch["attention_mask"].sum())
                state["target_tokens"] += count
                state["processed_tokens"] = state.get("processed_tokens", 0) + processed_count
                window_tokens += count
                window_all_tokens += processed_count
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            state["step"] += 1
            if state["step"] % args.log_every == 0 or state["step"] == 1:
                duration = time.monotonic() - window_start
                log({"event": "train", "step": state["step"], "loss": total_loss, "lr": lr,
                     "grad_norm": float(grad_norm), "target_tokens_per_second": window_tokens / max(duration, 1e-6),
                     "all_tokens_per_second": window_all_tokens / max(duration, 1e-6),
                     "processed_tokens": state["processed_tokens"],
                     "target_tokens": state["target_tokens"], "elapsed_seconds": prior_elapsed + time.monotonic() - start,
                     **window_diagnostics.summary()})
                window_start, window_tokens, window_all_tokens = time.monotonic(), 0, 0
                window_diagnostics = ModelDiagnostics()
            if state["step"] % args.eval_every == 0:
                val_metrics = evaluate(model, validation_loader, device, args.eval_batches, amp_dtype, return_diagnostics=True)
                val_loss = val_metrics["nll"]
                last_val_step = state["step"]
                log({"event": "validation", "step": state["step"], **val_metrics,
                     "perplexity": math.exp(min(50, val_loss))})
                if val_loss < state["best_val_loss"]:
                    state["best_val_loss"] = val_loss
                    save("best.pt")
            if state["step"] % args.save_every == 0:
                save("last.pt")
        segment_complete = bool(args.stop_at_step and state["step"] >= args.stop_at_step and state["step"] < args.steps)
        # Segment boundaries do not add validation checkpoints that an otherwise
        # identical uninterrupted run would not evaluate. Scheduled checks above
        # still run; last.pt always preserves optimizer and RNG for the next part.
        if not segment_complete and last_val_step != state["step"]:
            val_metrics = evaluate(model, validation_loader, device, args.eval_batches, amp_dtype, return_diagnostics=True)
            val_loss = val_metrics["nll"]
            log({"event": "validation", "step": state["step"], **val_metrics,
                 "perplexity": math.exp(min(50, val_loss))})
            if val_loss < state["best_val_loss"]:
                state["best_val_loss"] = val_loss
                save("best.pt")
        save("last.pt")
        log({"event": "segment_complete" if segment_complete else "complete", "step": state["step"],
             "schedule_steps": args.steps,
             "best_val_loss": state["best_val_loss"] if math.isfinite(state["best_val_loss"]) else None,
             "checkpoint": str((output_dir / "last.pt").resolve())})
    except KeyboardInterrupt:
        print("Interrupted. Resume from the last completed atomic last.pt checkpoint; incomplete gradients were not saved.", flush=True)
        # A launcher running language -> dialogue must not interpret Ctrl+C as a
        # successful stage and immediately start the next long GPU run.
        raise SystemExit(130)
    finally:
        training.close()
        validation.close()
        if retriever is not None:
            retriever.close()
        if persistent_cache is not None:
            persistent_cache.close()
    return state


if __name__ == "__main__":
    main()
