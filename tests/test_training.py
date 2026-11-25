import errno
import hashlib
import json
import random
import sqlite3
import sys
import types

import pytest
import torch

import arena.training as training_module
from arena.model import ArenaModel, ModelConfig
from arena.tokenization import train_tokenizer
from arena.training import (EpochBatchSampler, IndexedJsonlDataset, load_model_checkpoint,
                            restore_rng, save_checkpoint, retrieval_signature)


def windows_lock_error(code=32):
    error = PermissionError("The file is being used by another process")
    error.winerror = code
    return error


def test_atomic_save_retries_windows_locks_without_reserializing_or_overwriting_orphans(tmp_path, monkeypatch):
    destination = tmp_path / "last.pt"
    destination.write_bytes(b"previous checkpoint")
    orphan = tmp_path / f"last.pt.{training_module.os.getpid()}.tmp"
    orphan.write_bytes(b"previous orphan")
    real_save, real_replace = torch.save, training_module.os.replace
    replacements, serializations, sleeps = [], [], []

    def serialize(*args, **kwargs):
        serializations.append(True)
        return real_save(*args, **kwargs)

    def replace(source, target):
        replacements.append(source)
        if len(replacements) < 3:
            assert destination.read_bytes() == b"previous checkpoint"
            raise windows_lock_error(32 if len(replacements) == 1 else 33)
        return real_replace(source, target)

    monkeypatch.setattr(torch, "save", serialize)
    monkeypatch.setattr(training_module.os, "replace", replace)
    monkeypatch.setattr(training_module.time, "sleep", sleeps.append)
    training_module.atomic_torch_save({"step": 1200}, destination)
    assert torch.load(destination, weights_only=False) == {"step": 1200}
    assert len(serializations) == 1 and len(replacements) == 3
    assert len(set(replacements)) == 1 and replacements[0] != orphan
    assert sleeps == [.05, .1]
    assert orphan.read_bytes() == b"previous orphan"
    assert list(tmp_path.glob("*.tmp")) == [orphan]


def test_atomic_save_persistent_lock_keeps_previous_and_complete_recovery_file(tmp_path, monkeypatch):
    destination = tmp_path / "last.pt"
    destination.write_bytes(b"previous checkpoint")
    error, replacements, sleeps = windows_lock_error(), [], []

    def replace(source, target):
        replacements.append(source)
        raise error

    monkeypatch.setattr(training_module.os, "replace", replace)
    monkeypatch.setattr(training_module.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        training_module.atomic_torch_save({"step": 1200}, destination)
    assert caught.value is error
    assert len(replacements) == 7 and len(sleeps) == 6
    assert sum(sleeps) == pytest.approx(3.15)
    assert destination.read_bytes() == b"previous checkpoint"
    retained = list(tmp_path.glob("last.pt.*.tmp"))
    assert len(retained) == 1
    assert torch.load(retained[0], weights_only=False) == {"step": 1200}
    assert any(str(retained[0]) in note and "previous" in note for note in error.__notes__)


@pytest.mark.parametrize("error", [PermissionError(errno.EACCES, "access denied"), OSError(errno.ENOSPC, "disk full")])
def test_atomic_save_does_not_retry_other_replace_errors(tmp_path, monkeypatch, error):
    destination = tmp_path / "last.pt"
    destination.write_bytes(b"previous checkpoint")
    replacements, sleeps = [], []

    def replace(source, target):
        replacements.append(source)
        raise error

    monkeypatch.setattr(training_module.os, "replace", replace)
    monkeypatch.setattr(training_module.time, "sleep", sleeps.append)
    with pytest.raises(OSError) as caught:
        training_module.atomic_torch_save({"step": 1200}, destination)
    assert caught.value is error and len(replacements) == 1 and not sleeps
    assert destination.read_bytes() == b"previous checkpoint"


def test_atomic_save_cleanup_cannot_hide_serialization_error(tmp_path, monkeypatch):
    from pathlib import Path
    destination = tmp_path / "last.pt"
    destination.write_bytes(b"previous checkpoint")
    serialization_error = OSError(errno.ENOSPC, "serialization failed: disk full")
    real_unlink = Path.unlink

    def serialize(*args, **kwargs):
        raise serialization_error

    def unlink(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise windows_lock_error()
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", serialize)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError) as caught:
        training_module.atomic_torch_save({}, destination)
    assert caught.value is serialization_error
    assert destination.read_bytes() == b"previous checkpoint"
    assert any("incomplete temporary" in note for note in serialization_error.__notes__)


def test_atomic_save_cleans_incomplete_write_when_possible(tmp_path, monkeypatch):
    destination = tmp_path / "last.pt"
    destination.write_bytes(b"previous checkpoint")

    def serialize(payload, handle):
        handle.write(b"incomplete")
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(torch, "save", serialize)
    with pytest.raises(OSError):
        training_module.atomic_torch_save({}, destination)
    assert destination.read_bytes() == b"previous checkpoint"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.fixture
def tiny_corpus(tmp_path):
    torch.set_num_threads(2)
    rows = []
    for split in ("train", "val"):
        for i in range(4):
            rows.append({"id": f"{split}-{i}", "group_id": f"{split}-{i}", "kind": "dialogue", "split": split,
                         "context": [{"role": "user", "text": "Ты любишь чай?"}], "response": "Да, люблю чай.", "persona": []})
        for i in range(2):
            rows.append({"id": f"{split}-text-{i}", "group_id": f"{split}-article-{i}",
                         "kind": "text", "split": split,
                         "text": "Река течёт через город. На берегу расположены дома и деревья. " * 4})
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    tok_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer([path], tok_path, 320, show_progress=False)
    return path, tok_path, tokenizer


def test_indexed_data_excludes_validation_and_current_retrieval_group(tiny_corpus):
    path, _, tokenizer = tiny_corpus
    class Retriever:
        def __init__(self):
            self.exclusions = []
        def search(self, query, **kwargs):
            self.exclusions.append(kwargs["exclude_groups"])
            return [{"response": "Да, люблю.", "context": [], "group_id": "another-dialogue"}]
    retriever = Retriever()
    dataset = IndexedJsonlDataset(path, tokenizer, retriever=retriever, retrieval_dropout=0)
    assert len(dataset) == 4
    assert dataset.record(0)["split"] == "train"
    assert dataset[0]["retrieval_mask"].count(True) > 0
    dataset[0]
    assert retriever.exclusions == [{"train-0"}]  # bounded cache keyed by record, not question
    assert len(dataset.signature["sha256"]) == 64
    dataset.close()


def test_persistent_cache_miss_never_falls_back_and_empty_result_is_valid(tiny_corpus):
    path, _, tokenizer = tiny_corpus
    class Cache:
        value = None
        calls = []
        def get(self, record_id, group_id):
            self.calls.append((record_id, group_id))
            return self.value
    class ForbiddenOnlineRetriever:
        def search(self, *args, **kwargs):
            raise AssertionError("Online retrieval must not run when a cache is supplied")
    cache = Cache()
    dataset = IndexedJsonlDataset(path, tokenizer, retriever=ForbiddenOnlineRetriever(),
                                 retrieval_cache=cache, retrieval_dropout=1.0)
    with pytest.raises(ValueError, match="Retrieval cache entry missing"):
        dataset[0]
    cache.value = []
    assert not any(dataset[0]["retrieval_mask"])
    assert cache.calls == [("train-0", "train-0"), ("train-0", "train-0")]
    dataset.close()


def test_persistent_cache_cannot_inject_current_dialogue(tiny_corpus):
    path, _, tokenizer = tiny_corpus
    class Cache:
        def get(self, record_id, group_id):
            return [{"id": record_id, "group_id": group_id, "response": "Ответ из текущего диалога"}]
    dataset = IndexedJsonlDataset(path, tokenizer, retrieval_cache=Cache())
    with pytest.raises(ValueError, match="same-dialogue"):
        dataset[0]
    dataset.close()


@pytest.fixture
def persona_corpus(tiny_corpus):
    path, tokenizer_path, tokenizer = tiny_corpus
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["kind"] == "dialogue":
            row["persona"] = ["Меня зовут Персона.", "Я люблю чай.", "Я живу у реки."]
            row["context"] = [{"role": "user", "text": "Меня зовут История."},
                              {"role": "assistant", "text": "Рад знакомству."},
                              {"role": "user", "text": "Как меня зовут?"}]
            row["response"] = "История."
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return path, tokenizer_path, tokenizer


@pytest.mark.parametrize("explicit_memory", [False, True])
def test_persona_dropout_removes_profile_memory_and_preserves_history(persona_corpus, monkeypatch, explicit_memory):
    path, _, tokenizer = persona_corpus
    if explicit_memory:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["kind"] == "dialogue":
                row["memory"] = [{"text": "Меня зовут Персона.", "source_turn": -1}]
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    captured = []
    real_format = training_module.format_example
    def capture(row, *args, **kwargs):
        captured.append((row, kwargs["memory"]))
        return real_format(row, *args, **kwargs)
    monkeypatch.setattr(training_module, "format_example", capture)
    dataset = IndexedJsonlDataset(path, tokenizer, retrieval_dropout=0)
    try:
        before = dataset[0]
        assert "Персона" in str(captured[-1][1])
        dataset.persona_dropout = 1.0
        dropped = dataset[0]  # same cached item must not restore profile memory
        assert captured[-1][0]["persona"] == []
        assert "Персона" not in str(captured[-1][1])
        assert "История" in str(captured[-1][1])
        assert all(fact["source_turn"] >= 0 for fact in captured[-1][1])
        assert dropped["labels"] == before["labels"]
        dataset.persona_dropout = 0.0
        assert dataset[0] == before
        assert dataset.record(0)["persona"][0] == "Меня зовут Персона."
    finally:
        dataset.close()


def test_persona_shuffle_runs_after_cache_and_validation_never_augments(persona_corpus, monkeypatch):
    path, _, tokenizer = persona_corpus
    captured, shuffle_calls = [], []
    real_format = training_module.format_example
    def capture(row, *args, **kwargs):
        captured.append(row["persona"])
        return real_format(row, *args, **kwargs)
    def reverse(values):
        shuffle_calls.append(list(values))
        values.reverse()
    monkeypatch.setattr(training_module, "format_example", capture)
    monkeypatch.setattr(random, "shuffle", reverse)
    training = IndexedJsonlDataset(path, tokenizer, retrieval_dropout=0, persona_shuffle=True)
    validation = IndexedJsonlDataset(path, tokenizer, split="validation", persona_dropout=1, persona_shuffle=True)
    try:
        original = training.record(0)["persona"]
        training[0]
        training[0]
        assert captured == [list(reversed(original)), list(reversed(original))]
        assert shuffle_calls == [original, original]
        assert training.record(0)["persona"] == original
        rng_before = random.getstate()
        first = validation[0]
        assert validation[0] == first
        assert captured[-2:] == [original, original]
        assert random.getstate() == rng_before
        assert len(shuffle_calls) == 2
    finally:
        training.close()
        validation.close()


def test_model_diagnostics_masks_padding_and_counts_only_present_retrieval():
    diagnostic = training_module.ModelDiagnostics()
    output = {"gate": torch.tensor([.2, .9, .6], requires_grad=True),
              "p_generate": torch.tensor([[[.2], [.4]], [[.6], [1.0]], [[.8], [.2]]], requires_grad=True)}
    batch = {"labels": torch.tensor([[3, 3], [3, -100], [3, 3]]),
             "retrieval_mask": torch.tensor([[1, 0], [0, 0], [1, 1]]),
             "retrieval_targets": torch.tensor([1., 1., -1.])}
    diagnostic.update(output, batch)
    diagnostic.update(output, batch)
    summary = diagnostic.summary()
    assert summary["mean_retrieval_gate"] == pytest.approx(.4)
    assert summary["mean_p_generate"] == pytest.approx(2.2 / 5)
    assert summary["retrieval_supervision_fraction"] == .5
    assert summary["retrieved_example_fraction"] == pytest.approx(2 / 3)
    assert summary["diagnostic_examples"] == 6
    assert summary["diagnostic_target_tokens"] == 10
    assert summary["supervised_retrieval_examples"] == 2
    assert not diagnostic.totals.requires_grad
    batch["retrieval_mask"].zero_()
    empty = training_module.ModelDiagnostics()
    empty.update(output, batch)
    assert empty.summary()["mean_retrieval_gate"] is None
    assert empty.summary()["retrieval_supervision_fraction"] is None


def test_training_persists_cache_signature_and_rejects_changed_search_or_data(tiny_corpus, tmp_path, monkeypatch):
    path, tok_path, _ = tiny_corpus
    module = types.ModuleType("arena.retrieval_cache")
    class ContractCache:
        def __init__(self, cache_path, index_dir=None, readonly=True, require_complete=True):
            with open(cache_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.signature, self.complete = metadata["signature"], metadata["complete"]
            if require_complete and not self.complete:
                raise ValueError("Retrieval cache is incomplete")
        def get(self, record_id, group_id):
            return []
        def close(self):
            pass
    module.RetrievalCache = ContractCache
    monkeypatch.setitem(sys.modules, "arena.retrieval_cache", module)
    digest = training_module.file_sha256(path)
    signature = {"version": 1, "index_sha256": "0" * 64, "reranker_sha256": None,
                 "data_sha256": {"train": digest, "val": digest},
                 "search": {"k": 1, "candidate_pool": 256, "postings_per_term": 128, "history": True}}
    cache_path = tmp_path / "cache_header.json"
    cache_path.write_text(json.dumps({"complete": True, "signature": signature}), encoding="utf-8")
    output = tmp_path / "cached_run"
    common = ["--data", str(path), "--tokenizer", str(tok_path), "--retrieval-cache", str(cache_path),
              "--device", "cpu", "--threads", "2", "--steps", "1", "--batch-size", "2",
              "--accumulation", "1", "--eval-batches", "1", "--d-model", "16", "--n-heads", "2",
              "--encoder-layers", "1", "--decoder-layers", "1", "--max-source-length", "32", "--max-target-length", "16"]
    training_module.main(common + ["--output-dir", str(output)])
    checkpoint = torch.load(output / "last.pt", weights_only=False)
    assert checkpoint["training_config"]["retrieval_cache_signature"] == signature
    signature["search"]["k"] = 2
    cache_path.write_text(json.dumps({"complete": True, "signature": signature}), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume retrieval cache signature changed"):
        training_module.main(common + ["--output-dir", str(output), "--resume", str(output / "last.pt")])
    signature["data_sha256"]["train"] = "1" * 64
    cache_path.write_text(json.dumps({"complete": True, "signature": signature}), encoding="utf-8")
    with pytest.raises(ValueError, match="cache data signatures"):
        training_module.main(common + ["--output-dir", str(tmp_path / "wrong_data")])
    cache_path.write_text(json.dumps({"complete": False, "signature": signature}), encoding="utf-8")
    with pytest.raises(ValueError, match="cache is incomplete"):
        training_module.main(common + ["--output-dir", str(tmp_path / "partial_cache")])


def test_checkpoint_roundtrip_restores_optimizer_rng_and_checks_tokenizer(tiny_corpus, tmp_path):
    _, tok_path, tokenizer = tiny_corpus
    model = ArenaModel(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2,
        encoder_layers=1, decoder_layers=1, max_source_length=32, max_target_length=16))
    optimizer = torch.optim.AdamW(model.parameters())
    source, decoder = torch.tensor([[2, 5, 30]]), torch.tensor([[2, 31]])
    model(source, decoder, labels=torch.tensor([[31, 3]]))["loss"].backward()
    optimizer.step()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, training_state={"step": 1}, tokenizer_path=tok_path)
    expected_python, expected_torch = random.random(), torch.rand(3)
    restored, _, metadata = load_model_checkpoint(checkpoint_path, tokenizer_path=tok_path)
    assert metadata["training_state"]["step"] == 1
    assert metadata["optimizer_state"]["state"]
    for original, loaded in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, loaded)
    restore_rng(metadata["rng_state"])
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)
    changed = tmp_path / "changed.json"
    changed.write_bytes(tok_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        load_model_checkpoint(checkpoint_path, tokenizer_path=changed)


def test_denoising_and_continuation_have_hidden_target_tokens(tiny_corpus):
    path, _, tokenizer = tiny_corpus
    dataset = IndexedJsonlDataset(path, tokenizer, stage="language", split="validation",
                                 max_source_length=32, max_target_length=16)
    assert len(dataset) == 2
    assert all(dataset.record(i)["kind"] == "text" for i in range(len(dataset)))
    example = dataset[0]
    assert not any(example["retrieval_mask"])
    assert example["labels"][-1] == tokenizer.eos_id
    assert example["input_ids"][2:] != example["labels"][:-1]
    assert dataset[0] == example  # validation noise is fixed
    dataset.close()


class UniquePositionTokenizer:
    bos_id, eos_id, pad_id = 2, 3, 0

    def token_id(self, text):
        return {"<history>": 7, "<mask>": 14}[text]

    def encode(self, text):
        return list(range(100, 100 + int(text)))


def position_dataset(policy):
    dataset = IndexedJsonlDataset.__new__(IndexedJsonlDataset)
    dataset.tokenizer = UniquePositionTokenizer()
    dataset.max_source_length, dataset.max_target_length = 32, 16
    dataset.language_window_policy = policy
    return dataset


def test_legacy_language_examples_and_rng_match_pre_policy_snapshot():
    # Golden outputs were captured from the shipped checkpoint-save-fix code.
    # Guard both the examples and subsequent RNG draws used by exact resume.
    dataset, rng = position_dataset("legacy"), random.Random(71)
    examples = [dataset._language_example({"kind": "text", "text": str(n)}, rng)
                for n in (6, 20, 80, 300) * 20]
    assert hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest() == "ce3eadd3dd0d88adf687dd160a824aab1e57f688fade3b2e708e9ceed74cacfd"
    assert hashlib.sha256(repr(rng.getstate()).encode()).hexdigest() == "7f7858a1b824f22459aef8547701868ebb8d6512ca7993d53dca7312d4cde926"


@pytest.mark.parametrize("length", [8, 40, 250])
@pytest.mark.parametrize("continuation", [False, True])
def test_full_span_can_predict_final_position_without_future_context(length, continuation):
    class EndRng:
        first = True

        def random(self):
            if self.first:
                self.first = False
                return 0.0 if continuation else 1.0
            return 0.0  # Mask all denoising spans.

        def randint(self, low, high):
            return high

        def randrange(self, stop):
            return stop - 1

    example = position_dataset("full-span")._language_example(
        {"kind": "text", "text": str(length)}, EndRng())
    target, source = example["labels"][:-1], example["input_ids"][2:]
    assert target[-1] == 100 + length - 1
    assert len(source) <= 30 and 1 <= len(target) <= 15
    assert not set(source) & set(target)
    if continuation:
        assert source and max(source) < min(target)
    else:
        assert set(source) == {14}


def test_full_span_validation_is_fixed_and_rng_independent(tiny_corpus):
    path, _, tokenizer = tiny_corpus
    dataset = IndexedJsonlDataset(path, tokenizer, stage="language", split="validation",
                                 max_source_length=32, max_target_length=16,
                                 language_window_policy="full-span")
    before = random.getstate()
    assert dataset[0] == dataset[0]
    assert random.getstate() == before
    dataset.close()


def test_language_stage_rejects_dialogue_only_input(tiny_corpus, tmp_path):
    path, _, tokenizer = tiny_corpus
    dialogue_only = tmp_path / "dialogue_only.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    dialogue_only.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records
                                      if row["kind"] == "dialogue"), encoding="utf-8")
    with pytest.raises(ValueError, match="No language examples"):
        IndexedJsonlDataset(dialogue_only, tokenizer, stage="language")


def test_epoch_sampler_resumes_without_replaying_samples():
    whole = list(EpochBatchSampler(19, 3, seed=17, epoch=2))
    resumed = list(EpochBatchSampler(19, 3, seed=17, epoch=2, start_batch=3))
    assert resumed == whole[3:]
    assert sorted(sum(whole, [])) == list(range(19))


def test_fixed_validation_order_covers_later_records_and_preserves_training_rng():
    sampler = EpochBatchSampler(400, 4, seed=100042)
    rng_before = torch.get_rng_state().clone()
    first = list(sampler)
    assert first == list(sampler)
    assert torch.equal(torch.get_rng_state(), rng_before)
    selected = sum(first[:20], [])
    # Contiguous groups of ten represent articles/dialogues in source order.
    assert len({index // 10 for index in selected}) > 20
    assert max(selected) > 300
    assert len(set(selected)) == 80
    assert sum(list(EpochBatchSampler(12, 4, seed=1, shuffle=False)), []) == list(range(12))


def test_resume_preserves_legacy_validation_policy_and_rejects_silent_change(tiny_corpus, tmp_path):
    path, tokenizer_path, _ = tiny_corpus
    output = tmp_path / "legacy_validation"
    common = ["--data", str(path), "--tokenizer", str(tokenizer_path), "--output-dir", str(output),
              "--device", "cpu", "--threads", "2", "--batch-size", "2", "--accumulation", "1",
              "--eval-batches", "1", "--d-model", "16", "--n-heads", "2", "--encoder-layers", "1",
              "--decoder-layers", "1", "--max-source-length", "32", "--max-target-length", "16"]
    training_module.main(common + ["--steps", "1", "--validation-sampling", "sequential"])
    checkpoint_path = output / "last.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["training_config"].pop("validation_sampling")
    torch.save(checkpoint, checkpoint_path)
    training_module.main(common + ["--steps", "2", "--resume", str(checkpoint_path)])
    saved = torch.load(checkpoint_path, weights_only=False)
    assert saved["training_state"]["step"] == 2
    assert saved["training_config"]["validation_sampling"] == "sequential"
    with pytest.raises(ValueError, match="unchanged validation_sampling"):
        training_module.main(common + ["--steps", "3", "--resume", str(checkpoint_path),
                                       "--validation-sampling", "fixed-shuffle"])


def test_legacy_persona_defaults_resume_guard_and_new_stage(persona_corpus, tmp_path):
    path, tokenizer_path, _ = persona_corpus
    output = tmp_path / "legacy_persona"
    common = ["--data", str(path), "--tokenizer", str(tokenizer_path),
              "--device", "cpu", "--threads", "2", "--batch-size", "2", "--accumulation", "1",
              "--eval-batches", "1", "--d-model", "16", "--n-heads", "2", "--encoder-layers", "1",
              "--decoder-layers", "1", "--max-source-length", "32", "--max-target-length", "16"]
    training_module.main(common + ["--output-dir", str(output), "--steps", "1"])
    checkpoint_path = output / "last.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["training_config"].pop("persona_dropout")
    checkpoint["training_config"].pop("persona_shuffle")
    torch.save(checkpoint, checkpoint_path)
    training_module.main(common + ["--output-dir", str(output), "--steps", "2", "--resume", str(checkpoint_path)])
    saved = torch.load(checkpoint_path, weights_only=False)
    assert saved["training_config"]["persona_dropout"] == 0
    assert saved["training_config"]["persona_shuffle"] is False
    for change, key in [(["--persona-dropout", ".3"], "persona_dropout"),
                        (["--persona-shuffle"], "persona_shuffle")]:
        with pytest.raises(ValueError, match="unchanged " + key):
            training_module.main(common + ["--output-dir", str(output), "--steps", "3",
                                           "--resume", str(checkpoint_path)] + change)
    new_output = tmp_path / "new_persona_experiment"
    training_module.main(common + ["--output-dir", str(new_output), "--steps", "1",
        "--init-from", str(checkpoint_path), "--persona-dropout", ".3", "--persona-shuffle"])
    new = torch.load(new_output / "last.pt", weights_only=False)
    assert new["training_state"]["step"] == 1
    assert new["training_config"]["persona_dropout"] == .3
    assert new["training_config"]["persona_shuffle"] is True
    metrics = [json.loads(line) for line in (new_output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    for item in (item for item in metrics if item["event"] in ("train", "validation")):
        assert 0 <= item["mean_p_generate"] <= 1
        assert item["mean_retrieval_gate"] is None
        assert item["retrieval_supervision_fraction"] is None
        assert item["supervised_retrieval_examples"] == 0


def test_new_run_cannot_overwrite_existing_checkpoint(tiny_corpus, tmp_path):
    path, tok_path, _ = tiny_corpus
    output = tmp_path / "protected"
    output.mkdir()
    checkpoint = output / "last.pt"
    checkpoint.write_bytes(b"existing trained artifact")
    with pytest.raises(FileExistsError, match="--resume"):
        training_module.main(["--data", str(path), "--tokenizer", str(tok_path), "--output-dir", str(output)])
    assert checkpoint.read_bytes() == b"existing trained artifact"


@pytest.mark.parametrize("limit", ["steps", "hours"])
def test_completed_resume_is_read_only_and_preserves_caller_rng(tiny_corpus, tmp_path, monkeypatch, capsys, limit):
    path, tokenizer_path, _ = tiny_corpus
    output = tmp_path / "completed"
    common = ["--data", str(path), "--tokenizer", str(tokenizer_path), "--output-dir", str(output),
              "--device", "cpu", "--threads", "2", "--batch-size", "2", "--accumulation", "1",
              "--eval-batches", "1", "--d-model", "16", "--n-heads", "2", "--encoder-layers", "1",
              "--decoder-layers", "1", "--max-source-length", "32", "--max-target-length", "16"]
    original_state = training_module.main(common + ["--steps", "1"])
    checkpoint = output / "last.pt"
    before = {file.name: (file.read_bytes(), file.stat().st_mtime_ns) for file in output.iterdir()}
    capsys.readouterr()

    def forbidden(*args, **kwargs):
        raise AssertionError("A completed resume must not validate, restore optimizer or save")

    monkeypatch.setattr(training_module, "evaluate", forbidden)
    monkeypatch.setattr(training_module, "atomic_torch_save", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", forbidden)
    random.seed(954)
    torch.manual_seed(954)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    if limit == "steps":
        options = ["--steps", "1"]
    else:
        assert original_state["elapsed_seconds"] > 0
        options = ["--steps", "3", "--hours", str(original_state["elapsed_seconds"] / 7200)]
    returned = training_module.main(common + options + ["--resume", str(checkpoint)])
    assert returned == original_state
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    after = {file.name: (file.read_bytes(), file.stat().st_mtime_ns) for file in output.iterdir()}
    assert after == before
    event = json.loads(capsys.readouterr().out.strip())
    assert event["event"] == "resume_complete"
    assert event["reason"] == [limit] and event["optimizer_updates"] == 0
    # Completion must not skip policy/input checks and accept an incompatible run.
    with pytest.raises(ValueError, match="unchanged batch_size"):
        training_module.main(common + options + ["--resume", str(checkpoint), "--batch-size", "1"])


def test_resuming_with_extended_budget_continues_training(tiny_corpus, tmp_path):
    path, tokenizer_path, _ = tiny_corpus
    output = tmp_path / "extended"
    common = ["--data", str(path), "--tokenizer", str(tokenizer_path), "--output-dir", str(output),
              "--device", "cpu", "--threads", "2", "--batch-size", "2", "--accumulation", "1",
              "--eval-batches", "1", "--d-model", "16", "--n-heads", "2", "--encoder-layers", "1",
              "--decoder-layers", "1", "--max-source-length", "32", "--max-target-length", "16"]
    training_module.main(common + ["--steps", "1"])
    checkpoint = output / "last.pt"
    before = training_module.file_sha256(checkpoint)
    result = training_module.main(common + ["--steps", "2", "--hours", "1", "--resume", str(checkpoint)])
    assert result["step"] == 2
    assert training_module.file_sha256(checkpoint) != before
    assert torch.load(checkpoint, weights_only=False)["training_state"]["step"] == 2


def test_language_policy_inherits_legacy_and_rejects_resume_change(tiny_corpus, tmp_path):
    path, tokenizer_path, _ = tiny_corpus
    output = tmp_path / "language_policy"
    common = ["--data", str(path), "--tokenizer", str(tokenizer_path), "--stage", "language",
              "--device", "cpu", "--threads", "2", "--batch-size", "2", "--accumulation", "1",
              "--eval-batches", "1", "--d-model", "16", "--n-heads", "2", "--encoder-layers", "1",
              "--decoder-layers", "1", "--max-source-length", "32", "--max-target-length", "16"]
    training_module.main(common + ["--output-dir", str(output), "--steps", "1"])
    checkpoint = output / "last.pt"
    payload = torch.load(checkpoint, weights_only=False)
    payload["training_config"].pop("language_window_policy")
    torch.save(payload, checkpoint)
    training_module.main(common + ["--output-dir", str(output), "--steps", "2", "--resume", str(checkpoint)])
    assert torch.load(checkpoint, weights_only=False)["training_config"]["language_window_policy"] == "legacy"
    with pytest.raises(ValueError, match="unchanged language_window_policy"):
        training_module.main(common + ["--output-dir", str(output), "--steps", "3", "--resume", str(checkpoint),
                                       "--language-window-policy", "full-span"])
    new = tmp_path / "new_language_policy"
    training_module.main(common + ["--output-dir", str(new), "--steps", "1", "--init-from", str(checkpoint),
                                   "--language-window-policy", "full-span"])
    training_module.main(common + ["--output-dir", str(new), "--steps", "2", "--resume", str(new / "last.pt")])
    assert torch.load(new / "last.pt", weights_only=False)["training_config"]["language_window_policy"] == "full-span"


@pytest.mark.parametrize("stage,policy,weighted", [("language", "legacy", False), ("language", "full-span", False),
                                                 ("dialogue", "legacy", False), ("dialogue", "legacy", True)])
def test_fixed_horizon_segments_match_continuous_weights_optimizer_rng_and_lr(tiny_corpus, tmp_path, capsys, stage, policy, weighted):
    path, tokenizer_path, _ = tiny_corpus
    if weighted:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for i, row in enumerate(rows):
            if row["kind"] == "dialogue":
                row["source"] = "minority" if i % 4 == 0 else "majority"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    common = ["--data", str(path), "--tokenizer", str(tokenizer_path), "--stage", stage,
              "--language-window-policy", policy, "--device", "cpu", "--threads", "2", "--steps", "6",
              "--batch-size", "2", "--accumulation", "2", "--warmup-steps", "1", "--log-every", "1",
              "--eval-every", "3", "--save-every", "99", "--eval-batches", "1", "--d-model", "16",
              "--n-heads", "2", "--encoder-layers", "1", "--decoder-layers", "1",
              "--max-source-length", "32", "--max-target-length", "16"]
    if weighted:
        common += ["--source-weights", '{"majority": 0.35, "minority": 0.65}']
    full, segmented = tmp_path / "continuous", tmp_path / "segmented"
    training_module.main(common + ["--output-dir", str(full)])
    first = training_module.main(common + ["--output-dir", str(segmented), "--stop-at-step", "2"])
    assert first["step"] == 2
    assert not (segmented / "best.pt").exists()  # No unscheduled validation at segment boundary.
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in segmented.iterdir()}
    capsys.readouterr()
    repeated = training_module.main(common + ["--output-dir", str(segmented), "--stop-at-step", "2",
                                              "--resume", str(segmented / "last.pt")])
    assert repeated == first
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in segmented.iterdir()} == before
    assert json.loads(capsys.readouterr().out.strip())["reason"] == ["stop_at_step"]
    for stop in (4, 0):
        training_module.main(common + ["--output-dir", str(segmented), "--stop-at-step", str(stop),
                                       "--resume", str(segmented / "last.pt")])
    a, b = [torch.load(p / "last.pt", weights_only=False) for p in (full, segmented)]

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                equal(x, y)
        else:
            assert left == right

    for key in ("model_state", "optimizer_state", "scaler_state", "rng_state"):
        equal(a[key], b[key])
    for key in ("step", "epoch", "batch_in_epoch", "best_val_loss", "target_tokens", "processed_tokens"):
        equal(a["training_state"][key], b["training_state"][key])
    logs = [[json.loads(line) for line in (p / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            for p in (full, segmented)]
    trajectories = [[(r["step"], r["lr"], r["loss"]) for r in rows if r["event"] == "train"] for rows in logs]
    assert trajectories[0] == trajectories[1]
    assert [r["step"] for r in logs[1] if r["event"] == "segment_complete"] == [2, 4]
    assert logs[1][-1]["event"] == "complete"


@pytest.mark.parametrize("value", ['null', '{}', '{"x":0}', '{"x":-1}', '{"x":true}', '{"x":"1"}',
                                  '{"x":NaN}', '{"x":Infinity}', '{"":1}', '[]', 'bad json'])
def test_source_weights_reject_invalid_maps(value):
    with pytest.raises(ValueError, match="source-weights"):
        training_module.normalize_source_weights(value)


def test_source_weights_balance_sources_and_resume_identical_draws():
    class Rows:
        split, stage = "train", "dialogue"

        def __len__(self):
            return 10000

        def record(self, index):
            return {"source": "large" if index < 9900 else "small"}

    shares = training_module.normalize_source_weights('{"large":1,"small":1}')
    weights, counts = training_module.source_sampling_weights(Rows(), shares)
    assert counts == {"large": 9900, "small": 100}
    assert weights[:9900].sum().item() == pytest.approx(.5)
    assert weights[9900:].sum().item() == pytest.approx(.5)
    before = torch.get_rng_state().clone()
    full = list(training_module.SourceWeightedBatchSampler(weights, 7, seed=71, epoch=3))
    assert full == list(training_module.SourceWeightedBatchSampler(weights, 7, seed=71, epoch=3))
    assert full[13:] == list(training_module.SourceWeightedBatchSampler(weights, 7, seed=71, epoch=3, start_batch=13))
    assert torch.equal(torch.get_rng_state(), before)
    draws = sum(full, [])
    assert len(draws) == 10000
    assert .47 < sum(index >= 9900 for index in draws) / len(draws) < .53
    for wrong in ({"large": 1}, {"large": .4, "small": .4, "unknown": .2}):
        with pytest.raises(ValueError, match="exactly cover"):
            training_module.source_sampling_weights(Rows(), wrong)
    Rows.split = "validation"
    with pytest.raises(ValueError, match="only for dialogue training"):
        training_module.source_sampling_weights(Rows(), shares)


def test_source_weights_inherit_on_resume_and_reject_changes(tiny_corpus, tmp_path):
    path, tokenizer_path, _ = tiny_corpus
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for i, row in enumerate(rows):
        if row["kind"] == "dialogue":
            row["source"] = "small" if i % 4 == 0 else "large"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    output = tmp_path / "weighted"
    common = ["--data", str(path), "--tokenizer", str(tokenizer_path), "--output-dir", str(output),
              "--device", "cpu", "--threads", "2", "--batch-size", "2", "--accumulation", "1",
              "--eval-batches", "1", "--d-model", "16", "--n-heads", "2", "--encoder-layers", "1",
              "--decoder-layers", "1", "--max-source-length", "32", "--max-target-length", "16"]
    training_module.main(common + ["--steps", "1", "--source-weights", '{"large":35,"small":65}'])
    checkpoint = output / "last.pt"
    saved_weights = torch.load(checkpoint, weights_only=False)["training_config"]["source_weights"]
    training_module.main(common + ["--steps", "2", "--resume", str(checkpoint)])
    assert torch.load(checkpoint, weights_only=False)["training_config"]["source_weights"] == saved_weights
    training_module.main(common + ["--steps", "3", "--resume", str(checkpoint),
                                   "--source-weights", '{"small":0.65,"large":0.35}'])
    with pytest.raises(ValueError, match="unchanged source_weights"):
        training_module.main(common + ["--steps", "4", "--resume", str(checkpoint),
                                       "--source-weights", '{"small":0.5,"large":0.5}'])
    with pytest.raises(ValueError, match="only for dialogue training"):
        training_module.main(common + ["--output-dir", str(tmp_path / "invalid_language"), "--stage", "language",
                                       "--steps", "1", "--source-weights", '{"large":1,"small":1}'])
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    start = next(row for row in metrics if row["event"] == "start")
    assert start["source_weights"] == saved_weights
    assert start["train_source_counts"] == {"large": 3, "small": 1}


@pytest.mark.parametrize("stop", [-1, 7])
def test_stop_at_step_rejects_out_of_horizon_values(stop):
    with pytest.raises(SystemExit):
        training_module._parser(["--data", "unused", "--tokenizer", "unused", "--steps", "6", "--stop-at-step", str(stop)])


def test_no_copy_is_honored_for_new_stage_but_cannot_change_resume(tiny_corpus, tmp_path):
    path, tokenizer_path, _ = tiny_corpus
    common = ["--data", str(path), "--tokenizer", str(tokenizer_path),
              "--device", "cpu", "--threads", "2", "--batch-size", "2", "--accumulation", "1",
              "--eval-batches", "1", "--d-model", "16", "--n-heads", "2", "--encoder-layers", "1",
              "--decoder-layers", "1", "--max-source-length", "32", "--max-target-length", "16"]
    original = tmp_path / "copy_enabled"
    training_module.main(common + ["--output-dir", str(original), "--steps", "1"])
    checkpoint = original / "last.pt"
    original_hash = training_module.file_sha256(checkpoint)
    assert torch.load(checkpoint, weights_only=False)["model_config"]["copy_enabled"] is True
    with pytest.raises(ValueError, match="unchanged copy_enabled"):
        training_module.main(common + ["--output-dir", str(original), "--steps", "2",
                                       "--resume", str(checkpoint), "--no-copy"])
    assert training_module.file_sha256(checkpoint) == original_hash
    unchanged = tmp_path / "default_init"
    training_module.main(common + ["--output-dir", str(unchanged), "--steps", "1", "--init-from", str(checkpoint)])
    assert torch.load(unchanged / "last.pt", weights_only=False)["model_config"]["copy_enabled"] is True
    disabled = tmp_path / "copy_disabled"
    training_module.main(common + ["--output-dir", str(disabled), "--steps", "1",
                                   "--init-from", str(checkpoint), "--no-copy"])
    disabled_checkpoint = disabled / "last.pt"
    assert torch.load(disabled_checkpoint, weights_only=False)["model_config"]["copy_enabled"] is False
    # A disabled checkpoint resumes both with the explicit flag and without it.
    for step, flags in [(2, ["--no-copy"]), (3, [])]:
        training_module.main(common + ["--output-dir", str(disabled), "--steps", str(step),
                                       "--resume", str(disabled_checkpoint)] + flags)
        assert torch.load(disabled_checkpoint, weights_only=False)["model_config"]["copy_enabled"] is False
    metrics = [json.loads(line) for line in (disabled / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(item["mean_p_generate"] == 1.0 for item in metrics if item["event"] in ("train", "validation"))
    assert training_module.file_sha256(checkpoint) == original_hash


@pytest.mark.parametrize("changed_part", ["index", "reranker", "removed"])
def test_resume_rejects_changed_retrieval_but_new_stage_accepts_it(tiny_corpus, tmp_path, changed_part):
    from arena.retrieval import HybridRetriever, FEATURE_NAMES
    path, tok_path, tokenizer = tiny_corpus
    index_dir = tmp_path / "retrieval"
    HybridRetriever.build(path, index_dir).close()
    model = ArenaModel(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2,
        encoder_layers=1, decoder_layers=1, max_source_length=32, max_target_length=16))
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "original.pt"
    save_checkpoint(checkpoint, model, optimizer, tokenizer_path=tok_path,
                    training_config={"retrieval_index": str(index_dir), "retrieval_signature": retrieval_signature(index_dir)})
    if changed_part == "index":
        with sqlite3.connect(index_dir / "index.sqlite3") as connection:
            connection.execute("UPDATE docs SET response='Я люблю чай.' WHERE docid=1")
    elif changed_part == "reranker":
        (index_dir / "reranker.json").write_text(json.dumps({"version": 1, "feature_names": list(FEATURE_NAMES),
                                  "weights": [0.0] * len(FEATURE_NAMES), "bias": 0.25}), encoding="utf-8")
    common = ["--data", str(path), "--tokenizer", str(tok_path), "--output-dir", str(tmp_path / "new_stage"),
              "--device", "cpu", "--threads", "2", "--steps", "1", "--batch-size", "2",
              "--accumulation", "1", "--eval-batches", "1"]
    if changed_part != "removed":
        common += ["--retrieval-index", str(index_dir)]
    with pytest.raises(ValueError, match="Resume retrieval index or reranker changed"):
        training_module.main(common + ["--resume", str(checkpoint)])
    result = training_module.main(common + ["--init-from", str(checkpoint)])
    assert result["step"] == 1
    saved = torch.load(tmp_path / "new_stage" / "last.pt", weights_only=False)
    expected = None if changed_part == "removed" else retrieval_signature(index_dir)
    assert saved["training_config"]["retrieval_signature"] == expected


@pytest.mark.parametrize("stage,persona_enabled", [("language", False), ("dialogue", False), ("dialogue", True)])
def test_training_resume_matches_uninterrupted_run(persona_corpus, tmp_path, monkeypatch, stage, persona_enabled):
    path, tok_path, _ = persona_corpus
    common = ["--data", str(path), "--tokenizer", str(tok_path), "--steps", "6", "--stage", stage,
        "--batch-size", "2", "--accumulation", "2", "--warmup-steps", "1",
        "--d-model", "16", "--n-heads", "2", "--encoder-layers", "1", "--decoder-layers", "1",
        "--max-source-length", "32", "--max-target-length", "16", "--threads", "2",
        "--eval-every", "3", "--save-every", "3", "--eval-batches", "1", "--log-every", "3",
        "--device", "cpu", "--seed", "71"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    augmented = ["--persona-dropout", ".5", "--persona-shuffle"] if persona_enabled else []
    training_module.main(common + ["--output-dir", str(full)] + augmented)
    real_save = training_module.save_checkpoint
    interrupted = False
    def interrupt_after_save(checkpoint_path, model, optimizer, scaler, state, *args):
        nonlocal interrupted
        real_save(checkpoint_path, model, optimizer, scaler, state, *args)
        if str(checkpoint_path).endswith("last.pt") and state["step"] == 3 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
    monkeypatch.setattr(training_module, "save_checkpoint", interrupt_after_save)
    with pytest.raises(SystemExit) as stopped:
        training_module.main(common + ["--output-dir", str(resumed)] + augmented)
    assert stopped.value.code == 130
    monkeypatch.setattr(training_module, "save_checkpoint", real_save)
    state = training_module.main(common + ["--output-dir", str(resumed), "--resume", str(resumed / "last.pt")])
    assert state["step"] == 6
    a = torch.load(full / "last.pt", weights_only=False)
    b = torch.load(resumed / "last.pt", weights_only=False)
    assert all(torch.equal(a["model_state"][key], b["model_state"][key]) for key in a["model_state"])
    assert a["training_state"]["target_tokens"] == b["training_state"]["target_tokens"]
    assert b["training_config"]["persona_shuffle"] == persona_enabled
    assert b["training_config"]["persona_dropout"] == (.5 if persona_enabled else 0)
    if persona_enabled:
        with pytest.raises(ValueError, match="unchanged persona_shuffle"):
            training_module.main(common + ["--output-dir", str(resumed), "--resume", str(resumed / "last.pt"),
                                           "--no-persona-shuffle"])
