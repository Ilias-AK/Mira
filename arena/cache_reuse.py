"""Seed a new retrieval cache with verifiably unchanged frozen queries.

This never treats a missing entry as an empty search result. New or changed
queries remain missing for the normal resumable cache builder to compute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from .formatting import question_and_history
from .retrieval_cache import (SEARCH, RetrievalCache, _builder_lock,
                              _index_signature, _records, _sha, build_cache)


def query_fingerprint(row):
    question, history = question_and_history(row)
    value = {"question": question, "history": history,
             "persona": row.get("persona"), "exclude_group": str(row["group_id"])}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def seed_cache(source_cache, source_train, source_val, train, val, index, output, *, allow_partial_source=False):
    """Copy exact query matches under an identical index/search configuration."""
    output = Path(output).resolve()
    source_cache = Path(source_cache).resolve()
    if output == source_cache:
        raise ValueError("Use a new output; the original cache must be preserved")
    source_hashes = {"train": _sha(source_train), "val": _sha(source_val)}
    data_hashes = {"train": _sha(train), "val": _sha(val)}
    index_hashes = _index_signature(index)
    signature = {"version": 1, **index_hashes, "data_sha256": data_hashes,
                 "search": SEARCH.copy()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with RetrievalCache(source_cache, index_dir=index, require_complete=not allow_partial_source) as original:
        if original.signature.get("data_sha256") != source_hashes:
            raise ValueError("Source cache does not match its source data")
        if original.signature.get("search") != SEARCH:
            raise ValueError("Source cache search settings differ")
        originals = {}
        for row in _records(source_train, source_val):
            key = (str(row["id"]), str(row["group_id"]))
            if key in originals:
                raise ValueError("Duplicate source query identity")
            originals[key] = query_fingerprint(row)
        with _builder_lock(output):
            connection = sqlite3.connect(output)
            copied = existing_count = changed = new = missing_source = 0
            try:
                connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute("CREATE TABLE IF NOT EXISTS entries(record_id TEXT, group_id TEXT, payload BLOB NOT NULL, PRIMARY KEY(record_id,group_id)) WITHOUT ROWID")
                found = connection.execute("SELECT value FROM metadata WHERE key='header'").fetchone()
                header = json.loads(found[0]) if found else {"signature": signature, "complete": False, "records": 0}
                if header["signature"] != signature:
                    raise ValueError("Output cache signature changed; choose another path")
                if header["complete"]:
                    with RetrievalCache(output):
                        return {"already_complete": True, "copied": 0, "records": header["records"]}
                connection.execute("INSERT OR REPLACE INTO metadata VALUES('header',?)", (json.dumps(header),))
                seen = set()
                for row in _records(train, val):
                    key = (str(row["id"]), str(row["group_id"]))
                    if key in seen:
                        raise ValueError("Duplicate destination query identity")
                    seen.add(key)
                    if connection.execute("SELECT 1 FROM entries WHERE record_id=? AND group_id=?", key).fetchone():
                        existing_count += 1
                        continue
                    fingerprint = originals.get(key)
                    if fingerprint is None:
                        new += 1
                        continue
                    if fingerprint != query_fingerprint(row):
                        changed += 1
                        continue
                    found = original.connection.execute("SELECT payload FROM entries WHERE record_id=? AND group_id=?", key).fetchone()
                    if found is None:
                        if allow_partial_source and not original.complete:
                            missing_source += 1
                            continue
                        raise ValueError(f"Source cache missing a supposedly completed record: {key}")
                    connection.execute("INSERT INTO entries VALUES(?,?,?)", (*key, found[0]))
                    copied += 1
                if {"train": _sha(source_train), "val": _sha(source_val)} != source_hashes:
                    raise ValueError("Source data changed during seeding")
                if {"train": _sha(train), "val": _sha(val)} != data_hashes or _index_signature(index) != index_hashes:
                    raise ValueError("Destination data or index changed during seeding")
                total = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
                if total > len(seen):
                    raise ValueError("Output cache has unexpected entries")
                # Normal build_cache performs the final exhaustive check and
                # declares completeness, even if every row was reused here.
                header["records"] = total
                connection.execute("UPDATE metadata SET value=? WHERE key='header'", (json.dumps(header),))
                connection.commit()
                return {"copied": copied, "already_present": existing_count,
                        "changed_queries": changed, "new_queries": new,
                        "source_incomplete": not original.complete, "missing_source_entries": missing_source,
                        "records": total, "required": len(seen), "complete": False}
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()


def main(argv=None):
    from .console import configure_utf8
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", default="artifacts/retrieval_cache.sqlite3")
    parser.add_argument("--source-train", default="data/processed/train.jsonl")
    parser.add_argument("--source-val", default="data/processed/val.jsonl")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", default="data/processed/val.jsonl")
    parser.add_argument("--index", default="artifacts/retrieval")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--allow-partial-source", action="store_true",
                        help="Reuse committed matches from an interrupted cache; missing rows still require real search")
    args = parser.parse_args(argv)
    result = seed_cache(args.source_cache, args.source_train, args.source_val,
                        args.train, args.val, args.index, args.output,
                        allow_partial_source=args.allow_partial_source)
    print(json.dumps({"seeding": result}), flush=True)
    if not args.seed_only:
        print(json.dumps(build_cache(args.train, args.val, args.index, args.output,
                                     args.workers, args.limit), indent=2), flush=True)


if __name__ == "__main__":
    main()
