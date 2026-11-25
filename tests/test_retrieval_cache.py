import json
import sqlite3

import pytest

from arena.retrieval import FEATURE_NAMES, HybridRetriever
from arena.retrieval_cache import RetrievalCache, build_cache


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


@pytest.fixture
def cache_corpus(tmp_path):
    def dialogue(record_id, group_id, split, question="Ты пьёшь чай?"):
        return {"id": record_id, "group_id": group_id, "kind": "dialogue", "split": split,
                "source": "test", "context": [{"role": "user", "text": question}] if question else [],
                "response": "Да, люблю чай.", "persona": []}

    train_rows = [dialogue("t0", "shared", "train"), dialogue("t1", "shared", "train"),
                  dialogue("t2", "other", "train"), dialogue("t3", "books", "train", "Любишь читать книги?"),
                  {"id": "article", "group_id": "article", "kind": "text", "split": "train",
                   "text": "Река течёт через город. На берегу расположены дома и деревья."},
                  dialogue("heldout", "heldout", "test")]
    val_rows = [dialogue("v0", "validation", "val"), dialogue("v-empty", "empty", "dev", "")]
    train_path, val_path, index_dir = tmp_path / "train.jsonl", tmp_path / "val.jsonl", tmp_path / "index"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    HybridRetriever.build(train_path, index_dir).close()
    return train_path, val_path, index_dir, train_rows, val_rows


def test_partial_cache_resumes_excludes_whole_dialogue_and_distinguishes_missing(cache_corpus, tmp_path):
    train_path, val_path, index_dir, train_rows, val_rows = cache_corpus
    path = tmp_path / "cache.sqlite3"
    partial = build_cache(train_path, val_path, index_dir, path, workers=1, limit=2)
    assert partial["records"] == partial["computed_now"] == 2
    assert not partial["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        RetrievalCache(path)
    with RetrievalCache(path, index_dir, require_complete=False) as cache:
        first = cache.get("t0", "shared")
        assert first and first[0]["group_id"] == "other"
        assert cache.get("t2", "other") is None

    completed = build_cache(train_path, val_path, index_dir, path, workers=1)
    assert completed["complete"] and completed["records"] == 6
    assert completed["computed_now"] == 4
    assert completed["signature"] == partial["signature"]
    with RetrievalCache(path, index_dir) as cache:
        assert cache.get("t0", "shared") == first
        assert cache.get("v-empty", "empty") == []
        assert cache.get("v-empty", "wrong-group") is None
        assert cache.get("article", "article") is None
        assert cache.get("heldout", "heldout") is None
        for row in train_rows[:4] + val_rows:
            candidates = cache.get(row["id"], row["group_id"])
            assert candidates is not None
            assert all(item["group_id"] != row["group_id"] for item in candidates)
            assert all(item["id"] != "heldout" for item in candidates)
    unchanged = build_cache(train_path, val_path, index_dir, path, workers=1)
    assert unchanged["resumed"] and unchanged["computed_now"] == 0


@pytest.mark.parametrize("changed", ["train", "val", "index", "reranker"])
def test_cache_rejects_changed_frozen_inputs(cache_corpus, tmp_path, changed):
    train_path, val_path, index_dir, _, _ = cache_corpus
    path = tmp_path / "cache.sqlite3"
    build_cache(train_path, val_path, index_dir, path, workers=1, limit=1)
    if changed in ("train", "val"):
        source = train_path if changed == "train" else val_path
        with source.open("a", encoding="utf-8") as handle:
            handle.write("\n")
    elif changed == "index":
        with sqlite3.connect(index_dir / "index.sqlite3") as connection:
            connection.execute("UPDATE docs SET response='Люблю зелёный чай.' WHERE docid=1")
    else:
        (index_dir / "reranker.json").write_text(json.dumps({
            "version": 1, "enabled": False, "feature_names": list(FEATURE_NAMES),
            "weights": [0.0] * len(FEATURE_NAMES), "bias": 0.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="signature changed"):
        build_cache(train_path, val_path, index_dir, path, workers=1)
    if changed in ("index", "reranker"):
        with pytest.raises(ValueError, match="signature mismatch"):
            RetrievalCache(path, index_dir, require_complete=False)


def test_complete_cache_detects_missing_database_row(cache_corpus, tmp_path):
    train_path, val_path, index_dir, _, _ = cache_corpus
    path = tmp_path / "cache.sqlite3"
    build_cache(train_path, val_path, index_dir, path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM entries WHERE record_id='t0'")
    with pytest.raises(ValueError, match="record count mismatch"):
        RetrievalCache(path)


def test_two_spawn_workers_match_serial_retrieval(cache_corpus, tmp_path):
    train_path, val_path, index_dir, train_rows, val_rows = cache_corpus
    serial_path, parallel_path = tmp_path / "serial.sqlite3", tmp_path / "parallel.sqlite3"
    serial = build_cache(train_path, val_path, index_dir, serial_path, workers=1)
    parallel = build_cache(train_path, val_path, index_dir, parallel_path, workers=2)
    assert parallel["complete"] and parallel["records"] == serial["records"] == 6
    assert parallel["signature"] == serial["signature"]
    with RetrievalCache(serial_path) as one, RetrievalCache(parallel_path) as two:
        for row in train_rows + val_rows:
            assert one.get(row["id"], row["group_id"]) == two.get(row["id"], row["group_id"])


def test_real_cache_cpu_training_resume_matches_continuous(cache_corpus, tmp_path, monkeypatch):
    import torch
    import arena.training as training
    from arena.tokenization import train_tokenizer

    train_path, val_path, index_dir, _, _ = cache_corpus
    cache_path = tmp_path / "cache.sqlite3"
    header = build_cache(train_path, val_path, index_dir, cache_path)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer([train_path], tokenizer_path, 320, show_progress=False)
    with RetrievalCache(cache_path, index_dir) as cache:
        dataset = training.IndexedJsonlDataset(train_path, tokenizer, retrieval_cache=cache,
            retrieval_dropout=0, max_source_length=64, max_target_length=24)
        assert any(dataset[0]["retrieval_mask"])
        dataset.close()

    def forbid_online_load(*args, **kwargs):
        raise AssertionError("Cached training must not open an online retriever")
    monkeypatch.setattr(HybridRetriever, "load", forbid_online_load)
    common = ["--data", str(train_path), "--val-data", str(val_path), "--tokenizer", str(tokenizer_path),
              "--retrieval-index", str(index_dir), "--retrieval-cache", str(cache_path),
              "--retrieval-dropout", "0.2", "--device", "cpu", "--threads", "2", "--batch-size", "2",
              "--accumulation", "1", "--eval-batches", "1", "--warmup-steps", "10", "--d-model", "16",
              "--n-heads", "2", "--encoder-layers", "1", "--decoder-layers", "1",
              "--max-source-length", "64", "--max-target-length", "24"]
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    training.main(common + ["--steps", "2", "--output-dir", str(continuous)])
    training.main(common + ["--steps", "1", "--output-dir", str(resumed)])
    first = torch.load(resumed / "last.pt", weights_only=False)
    assert first["training_state"]["step"] == 1
    assert first["training_config"]["retrieval_cache_signature"] == header["signature"]
    training.main(common + ["--steps", "2", "--output-dir", str(resumed), "--resume", str(resumed / "last.pt")])
    expected = torch.load(continuous / "last.pt", weights_only=False)
    actual = torch.load(resumed / "last.pt", weights_only=False)
    assert actual["training_state"]["step"] == 2
    assert actual["training_state"]["processed_tokens"] == expected["training_state"]["processed_tokens"] > 0
    assert actual["training_config"]["retrieval_cache_signature"] == header["signature"]
    assert actual["optimizer_state"]["state"]
    for key, value in expected["model_state"].items():
        assert torch.equal(actual["model_state"][key], value), key
