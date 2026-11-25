"""Verify every cached training example before transferring a completed cache."""
import argparse
import json
from pathlib import Path
import sqlite3

from .retrieval_cache import RetrievalCache, _records, _sha


def validate_cache(train_path, val_path, index_dir, cache_path):
    with RetrievalCache(cache_path, index_dir=index_dir) as cache:
        expected = {"train": _sha(train_path), "val": _sha(val_path)}
        if cache.signature.get("data_sha256") != expected:
            raise ValueError("Retrieval cache data signatures do not match train/validation files")
        integrity = cache.connection.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise ValueError(f"Retrieval cache integrity check failed: {integrity[:3]}")
        count, matches, empty = 0, 0, 0
        seen = set()
        database = (Path(index_dir) / "index.sqlite3").resolve()
        index = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        try:
            for row in _records(train_path, val_path):
                key = (str(row["id"]), str(row["group_id"]))
                if key in seen:
                    raise ValueError(f"Duplicate input record: {key}")
                seen.add(key)
                items = cache.get(*key)
                if items is None:
                    raise ValueError(f"Retrieval cache entry missing: {key}")
                if not isinstance(items, list) or len(items) > cache.signature["search"]["k"]:
                    raise ValueError(f"Invalid retrieval cache candidates: {key}")
                for item in items:
                    if not isinstance(item, dict) or not item.get("id") or not item.get("group_id") or item["group_id"] == key[1]:
                        raise ValueError(f"Untraceable or same-dialogue candidate: {key}")
                    original = index.execute("SELECT group_id,context,response FROM docs WHERE id=?", (item["id"],)).fetchone()
                    if original is None or original[0] != item["group_id"] or json.loads(original[1]) != item.get("context") or original[2] != item.get("response"):
                        raise ValueError(f"Candidate does not match the frozen train index: {key}")
                    matches += 1
                empty += int(not items)
                count += 1
        finally:
            index.close()
        stored = cache.connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if count != stored:
            raise ValueError("Retrieval cache contains unexpected records")
        signature = cache.signature
    wal = Path(str(cache_path) + "-wal")
    if wal.is_file() and wal.stat().st_size:
        raise ValueError("Retrieval cache WAL has pending pages; finish the build before packaging")
    return {"valid": True, "complete": True, "records": count, "candidates": matches,
            "empty_results": empty, "signature": signature, "sha256": _sha(cache_path)}


def main():
    from .console import configure_utf8
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--val", default="data/processed/val.jsonl")
    parser.add_argument("--index", default="artifacts/retrieval")
    parser.add_argument("--cache", default="artifacts/retrieval_cache.sqlite3")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate_cache(args.train, args.val, args.index, args.cache)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
