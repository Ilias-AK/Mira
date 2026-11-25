import json
import sqlite3
import zlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from arena.bundle import build_bundle, verify_bundle
from arena.retrieval import FEATURE_NAMES, HybridRetriever
from arena.retrieval_cache import build_cache, RetrievalCache
from arena.cache_validation import validate_cache


def portable_project(tmp_path):
    root = tmp_path / "project"
    (root / "arena").mkdir(parents=True)
    for source in (Path(__file__).resolve().parents[1] / "arena").glob("*.py"):
        shutil.copy2(source, root / "arena" / source.name)
    (root / "README.md").write_text("Portable test project", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "persona.json").write_text('{"facts":[]}', encoding="utf-8")
    raw = root / "data" / "raw" / "example"
    raw.mkdir(parents=True)
    (raw / "README.md").write_text("Original source terms", encoding="utf-8")
    (raw / "corpus.zip").write_bytes(b"raw archive deliberately excluded")
    processed = root / "data" / "processed"
    processed.mkdir()
    row = {"id": "example:1", "group_id": "example", "split": "train", "source": "example",
           "kind": "dialogue", "context": [{"role": "user", "text": "Привет"}],
           "response": "Привет, как проходит день?", "persona": []}
    (processed / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    for split in ("val", "test"):
        (processed / f"{split}.jsonl").write_text("", encoding="utf-8")
    # This original-machine path must not be needed for inference after transfer.
    (processed / "manifest.json").write_text(json.dumps({"inputs": [{"path": "Z:/unavailable/raw.zip"}]}), encoding="utf-8")
    index = HybridRetriever.build([row], root / "artifacts" / "retrieval")
    index.close()
    (root / "artifacts" / "retrieval" / "reranker.json").write_text(json.dumps({
        "feature_names": list(FEATURE_NAMES), "weights": [0] * len(FEATURE_NAMES), "bias": 0,
    }), encoding="utf-8")
    (root / "artifacts" / "retrieval" / "inspection_base.json").write_text("{}", encoding="utf-8")
    (root / "artifacts" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "private.txt").write_text("not for export", encoding="utf-8")
    (root / ".env").write_text("PRIVATE=not_for_export", encoding="utf-8")
    (root / "runs" / "integration").mkdir(parents=True)
    (root / "runs" / "integration" / "best.pt").write_bytes(b"test checkpoint")
    return root


def test_verified_bundle_runs_retrieval_after_transfer_without_raw_archives(tmp_path):
    root = portable_project(tmp_path)
    archive_path = tmp_path / "kit.zip"
    result = build_bundle(root, archive_path, with_data=True)
    assert result["bytes"] > 0
    assert verify_bundle(archive_path)["includes_data"] is True
    extracted = tmp_path / "friend"
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        assert "data/raw/example/README.md" in members
        assert "data/raw/example/corpus.zip" not in members
        assert "artifacts/retrieval/inspection_base.json" not in members
        assert ".env" not in members
        assert not any(name.startswith((".venv/", "runs/")) for name in members)
        archive.extractall(extracted)
    command = [sys.executable, "-X", "utf8", "-m", "arena.cli", "--question", "Привет"]
    completed = subprocess.run(command, cwd=extracted, capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "Привет, как проходит день?"


def test_bundle_verification_detects_changed_member(tmp_path):
    root = portable_project(tmp_path)
    original = tmp_path / "original.zip"
    build_bundle(root, original, with_data=True)
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(damaged, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"modified" if name == "README.md" else source.read(name))
    with pytest.raises(ValueError, match="Corrupt ZIP member"):
        verify_bundle(damaged)


def test_bundle_with_data_rejects_missing_processed_file(tmp_path):
    root = portable_project(tmp_path)
    (root / "data" / "processed" / "test.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="Prepared data missing"):
        build_bundle(root, tmp_path / "incomplete.zip", with_data=True)


def test_bundle_with_data_requires_ready_artifacts(tmp_path):
    root = portable_project(tmp_path)
    (root / "artifacts" / "retrieval" / "reranker.json").unlink()
    with pytest.raises(FileNotFoundError, match="prepared artifact"):
        build_bundle(root, tmp_path / "missing-reranker.zip", with_data=True)


def test_bundle_rejects_unfinished_index(tmp_path):
    root = portable_project(tmp_path)
    (root / "artifacts" / "retrieval" / "index.building.sqlite3").write_bytes(b"incomplete")
    with pytest.raises(RuntimeError, match="still being built"):
        build_bundle(root, tmp_path / "unfinished.zip", with_data=True)


def _prepared_cache(root):
    path = root / "artifacts/retrieval_cache.sqlite3"
    build_cache(root / "data/processed/train.jsonl", root / "data/processed/val.jsonl",
                root / "artifacts/retrieval", path)
    return path


def test_bundle_with_cache_transfers_complete_cache_and_reuses_without_search(tmp_path, monkeypatch):
    import arena.retrieval_cache as module
    root = portable_project(tmp_path)
    _prepared_cache(root)
    archive_path = tmp_path / "cached.zip"
    build_bundle(root, archive_path, with_data=True, with_cache=True)
    assert verify_bundle(archive_path)["includes_retrieval_cache"]
    extracted = tmp_path / "friend"
    with zipfile.ZipFile(archive_path) as archive:
        assert "artifacts/retrieval_cache.sqlite3" in archive.namelist()
        assert not any(name.endswith(("-wal", "-shm", ".lock")) for name in archive.namelist())
        archive.extractall(extracted)
    def forbid_search(*args):
        raise AssertionError("A complete transferred cache must not start search workers")
    monkeypatch.setattr(module, "_worker_init", forbid_search)
    result = build_cache(extracted / "data/processed/train.jsonl", extracted / "data/processed/val.jsonl",
                        extracted / "artifacts/retrieval", extracted / "artifacts/retrieval_cache.sqlite3")
    assert result["complete"] and result["computed_now"] == 0
    with RetrievalCache(extracted / "artifacts/retrieval_cache.sqlite3") as cache:
        assert cache.get("example:1", "example") == []


def test_bundle_rejects_incomplete_cache(tmp_path):
    root = portable_project(tmp_path)
    path = _prepared_cache(root)
    with sqlite3.connect(path) as connection:
        header = json.loads(connection.execute("SELECT value FROM metadata WHERE key='header'").fetchone()[0])
        header["complete"] = False
        connection.execute("UPDATE metadata SET value=? WHERE key='header'", (json.dumps(header),))
    with pytest.raises(ValueError, match="incomplete"):
        build_bundle(root, tmp_path / "partial-cache.zip", with_data=True, with_cache=True)


def test_bundle_rejects_cache_for_different_data(tmp_path):
    root = portable_project(tmp_path)
    _prepared_cache(root)
    with (root / "data/processed/train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="data signatures"):
        build_bundle(root, tmp_path / "stale-cache.zip", with_data=True, with_cache=True)


def test_cache_validation_checks_candidate_content_against_train_index(tmp_path):
    root = portable_project(tmp_path)
    path = _prepared_cache(root)
    payload = zlib.compress(json.dumps([{"id":"missing", "group_id":"other",
                                        "context":[], "response":"Wrong response"}]).encode())
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE entries SET payload=?", (payload,))
    with pytest.raises(ValueError, match="does not match"):
        validate_cache(root / "data/processed/train.jsonl", root / "data/processed/val.jsonl",
                       root / "artifacts/retrieval", path)


def test_bundle_cache_requires_data(tmp_path):
    with pytest.raises(ValueError, match="requires --with-data"):
        build_bundle(portable_project(tmp_path), tmp_path / "cache-only.zip", with_cache=True)
