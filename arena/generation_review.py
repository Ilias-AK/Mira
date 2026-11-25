"""Reproducible raw-checkpoint review on cached validation examples, offline.

This bypasses runtime tools, response rejection and fallback replies. Surface
checks are diagnostics; they never certify conversational relevance or quality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import json
import math
from pathlib import Path
import random
import re
import time
import unicodedata

import torch

from .formatting import collate_examples, question_and_history, text_value
from .retrieval_cache import RetrievalCache
from .training import IndexedJsonlDataset, file_sha256, load_model_checkpoint, tokenizer_hash


def select_validation_indices(dataset, max_per_source=2000, seed=42):
    """Fixed per-source shuffle; changing one source does not reshuffle others."""
    if max_per_source < 1:
        raise ValueError("max_per_source must be positive")
    pools = defaultdict(list)
    for index in range(len(dataset)):
        pools[str(dataset.record(index).get("source", "unknown"))].append(index)
    selected, available = {}, {}
    for source in sorted(pools):
        indices = pools[source]
        available[source] = len(indices)
        random.Random(f"generation-review:{seed}:{source}").shuffle(indices)
        selected[source] = indices[:max_per_source]
    return selected, available


def generation_indices(selected, count=200, seed=42):
    if count < 0:
        raise ValueError("generation_examples must be nonnegative")
    indices = [index for source in sorted(selected) for index in selected[source]]
    random.Random(f"generation-review:{seed}:outputs").shuffle(indices)
    return indices[:count]


def _words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFC", str(text)).casefold())


def persona_facts(persona):
    if isinstance(persona, dict):
        persona = persona.get("facts", [])
    elif isinstance(persona, str):
        persona = [persona]
    return [str(fact) for fact in persona or [] if isinstance(fact, str) and _words(fact)]


def matching_persona_facts(text, persona):
    """Full fact as contiguous words: NFC, casefold, punctuation ignored, ё kept."""
    words = _words(text)
    matches = []
    for fact in persona_facts(persona):
        pattern = _words(fact)
        if any(words[start:start + len(pattern)] == pattern
               for start in range(len(words) - len(pattern) + 1)):
            matches.append(fact)
    return matches


def repeated_word_ngrams(text, n=3):
    words = _words(text)
    grams = Counter(tuple(words[index:index + n]) for index in range(len(words) - n + 1))
    return sum(max(0, count - 1) for count in grams.values())


def generation_summary(samples):
    with_persona = [row for row in samples if persona_facts(row["persona"])]
    denominator = len(with_persona)
    generated_matches = sum(bool(row["generated_persona_matches"]) for row in with_persona)
    reference_matches = sum(bool(row["reference_persona_matches"]) for row in with_persona)
    return {
        "examples": len(samples), "persona_bearing_examples": denominator,
        "examples_by_source": dict(Counter(row.get("source", "unknown") for row in samples)),
        "generated_full_fact_matches": generated_matches,
        "reference_full_fact_matches": reference_matches,
        "generated_full_fact_match_rate": generated_matches / denominator if denominator else None,
        "reference_full_fact_match_rate": reference_matches / denominator if denominator else None,
        "persona_matching_rule": "Any entire nonempty persona fact occurs as contiguous Unicode word tokens; NFC and casefold, punctuation ignored, ё preserved. Denominator is persona-bearing examples. This is matching, not proof of the causal source of copied text.",
        "unique_exact_outputs": len({row["generated"] for row in samples}),
        "unique_normalized_word_outputs": len({tuple(_words(row["generated"])) for row in samples}),
        "empty_outputs": sum(not row["generated"].strip() for row in samples),
        "outputs_with_repeated_word_trigrams": sum(row["repeated_word_trigrams"] > 0 for row in samples),
        "generation_timeouts": sum(row["generation_timeout"] for row in samples),
        "outputs_without_eos": sum(not row["ended_with_eos"] for row in samples),
        "note": "Lower repetition or persona matching is not a semantic quality pass. Inspect the saved question/history/reference/generated pairs.",
    }


def _autocast(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else contextlib.nullcontext()


def review_model(model, tokenizer, dataset, selected, generated_indices, *, device,
                 batch_size=4, amp_dtype=None, max_new_tokens=None,
                 generation_timeout_seconds=30.0, no_repeat_ngram_size=0):
    model.eval()
    losses, samples = {}, []
    with torch.inference_mode():
        for source, indices in selected.items():
            nll_sum, tokens, gate_sum, retrieved_count = 0.0, 0, 0.0, 0
            p_generate_sum, supervised_count = 0.0, 0
            for start in range(0, len(indices), batch_size):
                examples = [dataset[index] for index in indices[start:start + batch_size]]
                batch = {key: value.to(device) for key, value in collate_examples(examples, tokenizer).items()}
                with _autocast(device, amp_dtype):
                    output = model(**batch)
                valid = batch["labels"].ne(-100)
                count = int(valid.sum())
                value = float(output["nll"])
                if not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite validation NLL for {source}")
                nll_sum += value * count
                tokens += count
                retrieved = batch["retrieval_mask"].any(1)
                gate_sum += float(output["gate"][retrieved].float().sum())
                retrieved_count += int(retrieved.sum())
                supervised_count += int((batch["retrieval_targets"].ge(0) & retrieved).sum())
                p_generate_sum += float((output["p_generate"].float().squeeze(-1) * valid).sum())
            nll = nll_sum / tokens
            losses[source] = {"examples": len(indices), "target_tokens": tokens,
                "nll": nll, "perplexity": math.exp(nll) if nll < 700 else None,
                "retrieved_examples": retrieved_count,
                "mean_retrieval_gate_when_present": gate_sum / retrieved_count if retrieved_count else None,
                "auxiliary_gate_labeled_examples": supervised_count,
                "mean_p_generate_on_reference_tokens": p_generate_sum / tokens}
            print(f"Validation {source}: {len(indices)} examples, NLL {nll:.5f}", flush=True)
        for number, index in enumerate(generated_indices, 1):
            record, example = dataset.record(index), dataset[index]
            inputs = {key: torch.tensor([example[key]], device=device) for key in
                      ("input_ids", "attention_mask", "copy_mask", "retrieval_mask")}
            with _autocast(device, amp_dtype):
                result = model.generate(**inputs, max_new_tokens=max_new_tokens,
                    temperature=0.0, top_k=0, deadline_seconds=generation_timeout_seconds,
                    no_repeat_ngram_size=no_repeat_ngram_size, return_info=True)
            ids = result["tokens"][0].detach().cpu().tolist()
            text = tokenizer.decode(ids)
            question, history = question_and_history(record)
            persona, reference = record.get("persona", []), text_value(record.get("response", ""))
            samples.append({"id": record["id"], "group_id": record["group_id"],
                "source": str(record.get("source", "unknown")), "dataset_index": index,
                "question": question, "history": history, "persona": persona,
                "reference": reference, "generated": text, "generated_token_ids": ids,
                "input_tokens": len(example["input_ids"]),
                "generation_timeout": bool(result["deadline_reached"]),
                "ended_with_eos": bool(ids and ids[-1] == tokenizer.eos_id),
                "seconds": result["elapsed_seconds"],
                "retrieval_gate": float(result["gate"][0]),
                "generated_persona_matches": matching_persona_facts(text, persona),
                "reference_persona_matches": matching_persona_facts(reference, persona),
                "repeated_word_trigrams": repeated_word_ngrams(text)})
            if number % 20 == 0 or number == len(generated_indices):
                print(f"Raw generation: {number}/{len(generated_indices)}", flush=True)
    return {"nll_by_source": losses, "generation_summary": generation_summary(samples),
            "generated_samples": samples}


def main(argv=None):
    from .console import configure_utf8
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--train-data", default="data/processed/train.jsonl")
    parser.add_argument("--val-data", default="data/processed/val.jsonl")
    parser.add_argument("--index", default="artifacts/retrieval")
    parser.add_argument("--cache", "--retrieval-cache", dest="cache", default="artifacts/retrieval_cache.sqlite3")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-source", type=int, default=2000)
    parser.add_argument("--generation-examples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--generation-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Preserve existing review and choose another --output: {destination}")
    if args.max_per_source < 1 or args.generation_examples < 0 or args.batch_size < 1:
        parser.error("max-per-source/batch-size must be positive and generation-examples nonnegative")
    if args.generation_timeout_seconds <= 0 or not math.isfinite(args.generation_timeout_seconds):
        parser.error("generation-timeout-seconds must be finite and positive")
    if args.no_repeat_ngram_size < 0 or (args.max_new_tokens is not None and args.max_new_tokens < 1):
        parser.error("no-repeat-ngram-size must be nonnegative and max-new-tokens positive")
    torch.set_num_threads(2)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None
    started = time.monotonic()
    data_hashes = {"train": file_sha256(args.train_data), "val": file_sha256(args.val_data)}
    cache, dataset = None, None
    try:
        cache = RetrievalCache(args.cache, index_dir=args.index, readonly=True, require_complete=True)
        if cache.signature.get("data_sha256") != data_hashes:
            raise ValueError("Retrieval cache data signatures do not match train/validation files")
        model, tokenizer, checkpoint = load_model_checkpoint(args.checkpoint, device=device, tokenizer_path=args.tokenizer)
        max_new_tokens = min(args.max_new_tokens or model.config.max_target_length, model.config.max_target_length)
        metadata = {"model_config": checkpoint["model_config"],
                    "training_state": checkpoint.get("training_state", {}),
                    "training_stage": checkpoint.get("training_config", {}).get("stage")}
        del checkpoint
        dataset = IndexedJsonlDataset(args.val_data, tokenizer, split="validation", stage="dialogue",
            seed=args.seed, max_source_length=model.config.max_source_length,
            max_target_length=model.config.max_target_length, retrieval_dropout=0.0, retrieval_cache=cache)
        if dataset.signature["sha256"] != data_hashes["val"]:
            raise ValueError("Validation offset index checksum is stale")
        selected, available = select_validation_indices(dataset, args.max_per_source, args.seed)
        generated_indices = generation_indices(selected, args.generation_examples, args.seed)
        inputs = {"checkpoint_sha256": file_sha256(args.checkpoint),
                  "tokenizer_sha256": tokenizer_hash(args.tokenizer),
                  "data_sha256": data_hashes, "cache_sha256": file_sha256(args.cache),
                  "index_sha256": cache.signature["index_sha256"],
                  "reranker_sha256": cache.signature["reranker_sha256"],
                  "cache_signature": cache.signature}
        result = review_model(model, tokenizer, dataset, selected, generated_indices,
            device=device, batch_size=args.batch_size, amp_dtype=amp_dtype,
            max_new_tokens=max_new_tokens, generation_timeout_seconds=args.generation_timeout_seconds,
            no_repeat_ngram_size=args.no_repeat_ngram_size)
        report = {"format_version": 1, "review_finished": True,
            "inputs": inputs, "checkpoint": metadata,
            "paths": {key: str(Path(getattr(args, key)).resolve()) for key in
                      ("checkpoint", "tokenizer", "train_data", "val_data", "index", "cache")},
            "environment": {"torch": torch.__version__, "device": str(device),
                "amp": str(amp_dtype), "cpu_threads": 2,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            "selection": {"seed": args.seed, "max_per_source": args.max_per_source,
                "available_by_source": available, "generation_examples_requested": args.generation_examples,
                "rule": "Per-source fixed shuffle then cap; generation is a separate fixed shuffle of the selected pool. Validation only; no augmentation.",
                "selected_ids_by_source": {source: [dataset.record(index)["id"] for index in indices]
                                           for source, indices in selected.items()}},
            "generation_settings": {"temperature": 0.0, "top_k": 0,
                "max_new_tokens": max_new_tokens, "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "per_example_cooperative_deadline_seconds": args.generation_timeout_seconds,
                "runtime_tools_filters_fallback": False},
            **result, "elapsed_seconds": time.monotonic() - started,
            "note": "Fixed selections and raw greedy outputs enable paired checkpoint review. CUDA arithmetic may vary. Timeouts/length-stopped outputs remain in the report. No automatic semantic quality or competition readiness verdict."}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Saved raw checkpoint review: {destination}", flush=True)
        return report
    finally:
        if dataset is not None:
            dataset.close()
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    main()
