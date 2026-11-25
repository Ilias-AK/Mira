"""Resumable, bounded precomputation of retrieval for dialogue training."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time
import zlib

SEARCH = {"k": 1, "candidate_pool": 256, "postings_per_term": 128,
          "history": True, "use_reranker": None}
_retriever = None


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_signature(index_dir):
    directory = Path(index_dir)
    if (directory / "index.building.sqlite3").exists():
        raise ValueError("Wait for the retrieval index build to finish")
    return {"index_sha256": _sha(directory / "index.sqlite3"),
            "reranker_sha256": _sha(directory / "reranker.json")
            if (directory / "reranker.json").is_file() else None}


class RetrievalCache:
    def __init__(self, path, index_dir=None, readonly=True, require_complete=True):
        self.connection = None
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.connection = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"), uri=True)
        try:
            header = json.loads(self.connection.execute("SELECT value FROM metadata WHERE key='header'").fetchone()[0])
            self.signature, self.complete = header["signature"], header["complete"]
            if self.signature.get("version") != 1:
                raise ValueError("Unsupported retrieval cache version")
            if require_complete and not self.complete:
                raise ValueError("Retrieval cache is incomplete; resume the cache build first")
            if index_dir:
                actual = _index_signature(index_dir)
                if any(self.signature.get(key) != value for key, value in actual.items()):
                    raise ValueError("Retrieval cache index or reranker signature mismatch")
            if self.complete:
                count = self.connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
                if count != header["records"]:
                    raise ValueError("Retrieval cache record count mismatch")
        except BaseException:
            self.close()
            raise

    def get(self, record_id, group_id):
        found = self.connection.execute("SELECT payload FROM entries WHERE record_id=? AND group_id=?",
                                        (str(record_id), str(group_id))).fetchone()
        return json.loads(zlib.decompress(found[0])) if found is not None else None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextmanager
def _builder_lock(path):
    # OS lock releases on a crash. The tiny lock file may remain safely.
    with Path(str(path) + ".lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _records(train_path, val_path):
    for path, accepted in ((train_path, {"train"}), (val_path, {"val", "validation", "dev"})):
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") not in accepted or row.get("kind") != "dialogue" or not row.get("response"):
                    continue
                if not row.get("id") or not row.get("group_id"):
                    raise ValueError("Cache records require id and group_id")
                yield row


def _worker_init(index_dir):
    global _retriever
    from .retrieval import HybridRetriever
    _retriever = HybridRetriever.load(index_dir)
    # Read the frozen index through shared mapped pages. This changes only I/O,
    # not SQL ordering, filtering, scores, or the persistent cache signature.
    _retriever.connection.execute("PRAGMA mmap_size=1073741824")
    _retriever.connection.execute("PRAGMA cache_size=-65536")


def _compute(row):
    from .formatting import question_and_history
    question, history = question_and_history(row)
    candidates = _retriever.search(question, history=history, persona=row.get("persona"),
        exclude_groups={row["group_id"]}, k=SEARCH["k"], candidate_pool=SEARCH["candidate_pool"],
        postings_per_term=SEARCH["postings_per_term"], use_reranker=SEARCH["use_reranker"]) if question else []
    payload = []
    for item in candidates:
        if not item.get("group_id") or item["group_id"] == row["group_id"]:
            raise ValueError("Search returned an untraceable or same-dialogue candidate")
        payload.append({key: item[key] for key in ("id", "group_id", "context", "response", "score") if key in item})
    return str(row["id"]), str(row["group_id"]), zlib.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"), 1)


def build_cache(train_path, val_path, index_dir, cache_path, workers=1, limit=0):
    if not 1 <= workers <= 8 or limit < 0:
        raise ValueError("Use 1..8 workers and a nonnegative limit (0 means all records)")
    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signature = {"version": 1, **_index_signature(index_dir),
                 "data_sha256": {"train": _sha(train_path), "val": _sha(val_path)}, "search": SEARCH.copy()}
    with _builder_lock(cache_path):
        return _build_locked(train_path, val_path, index_dir, cache_path, workers, limit, signature)


def _build_locked(train_path, val_path, index_dir, path, workers, limit, signature):
    global _retriever
    connection = sqlite3.connect(path)
    started, completed, scanned, last_report = time.monotonic(), 0, 0, time.monotonic()
    header = {"signature": signature, "complete": False, "records": 0}
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS entries(record_id TEXT, group_id TEXT, payload BLOB NOT NULL, PRIMARY KEY(record_id,group_id)) WITHOUT ROWID")
        existing = connection.execute("SELECT value FROM metadata WHERE key='header'").fetchone()
        if existing:
            header = json.loads(existing[0])
            if header["signature"] != signature:
                raise ValueError("Cache signature changed; choose a new output path")
            if header["complete"]:
                with RetrievalCache(path):
                    return header | {"resumed": True, "computed_now": 0}
        connection.execute("INSERT OR REPLACE INTO metadata VALUES('header',?)", (json.dumps(header),))
        connection.commit()
        exhausted = False

        def missing():
            nonlocal scanned, exhausted
            submitted = 0
            for row in _records(train_path, val_path):
                scanned += 1
                found = connection.execute("SELECT 1 FROM entries WHERE record_id=? AND group_id=?", (str(row["id"]), str(row["group_id"]))).fetchone()
                if found:
                    continue
                if limit and submitted >= limit:
                    return
                submitted += 1
                yield row
            exhausted = True

        def store(result):
            nonlocal completed, last_report
            connection.execute("INSERT INTO entries VALUES(?,?,?)", result)
            completed += 1
            if completed % 100 == 0:
                connection.commit()
            now = time.monotonic()
            if now - last_report >= 30:
                connection.commit()
                print(json.dumps({"computed_now": completed, "scanned": scanned,
                    "elapsed_seconds": round(now - started, 1), "records_per_second": round(completed / (now - started), 2)}), flush=True)
                last_report = now

        if workers == 1:
            _worker_init(str(index_dir))
            try:
                for row in missing():
                    store(_compute(row))
            finally:
                _retriever.close()
                _retriever = None
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init, initargs=(str(index_dir),)) as executor:
                records, pending, ended = iter(missing()), set(), False
                while pending or not ended:
                    while not ended and len(pending) < workers * 4:
                        try:
                            pending.add(executor.submit(_compute, next(records)))
                        except StopIteration:
                            ended = True
                    if pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            store(future.result())
        # Detect edits made while workers were reading the frozen inputs.
        if _index_signature(index_dir) != {key: signature[key] for key in ("index_sha256", "reranker_sha256")}:
            raise ValueError("Index changed during cache build; choose a new output path")
        if {"train": _sha(train_path), "val": _sha(val_path)} != signature["data_sha256"]:
            raise ValueError("Data changed during cache build; choose a new output path")
        count = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if exhausted and count != scanned:
            raise ValueError("Duplicate record IDs or inconsistent cache count")
        header = {"signature": signature, "complete": exhausted, "records": count}
        connection.execute("UPDATE metadata SET value=? WHERE key='header'", (json.dumps(header),))
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return header | {"computed_now": completed, "elapsed_seconds": round(time.monotonic() - started, 3)}
    finally:
        # Valid completed rows survive an interruption; header remains incomplete.
        connection.commit()
        connection.close()


def main(argv=None):
    from .console import configure_utf8
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--train", default="data/processed/train.jsonl")
    build.add_argument("--val", default="data/processed/val.jsonl")
    build.add_argument("--index", default="artifacts/retrieval")
    build.add_argument("--output", default="artifacts/retrieval_cache.sqlite3")
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--limit", type=int, default=0, help="New records to compute; 0 completes the full cache")
    args = parser.parse_args(argv)
    print(json.dumps(build_cache(args.train, args.val, args.index, args.output, args.workers, args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
