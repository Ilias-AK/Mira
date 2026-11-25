"""Offline sparse retrieval and a small supervised reranker, without model weights.

Only records explicitly assigned to ``split=train`` enter an index.  The query
keys are dialogue contexts, never the response being predicted.  The SQLite
postings index streams construction and bounds work at query time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from .quality import response_issues

INDEX_VERSION = 1
FEATURE_NAMES = ("word_score", "char_score", "query_coverage", "length_fit", "negation_match")
WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)


def _words(text: str) -> list[str]:
    return WORD_RE.findall(str(text).casefold().replace("ё", "е"))


def _context_text(context: Any) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        return "\n".join(str(item.get("text", item.get("content", ""))) if isinstance(item, Mapping) else str(item) for item in context)
    return str(context or "")


def _retrieval_key(context: Any) -> str:
    """Match the turn being answered, while retaining full context as payload.

    Searching all prior turns at equal weight retrieves a later response to a
    different question whenever an earlier turn happened to match the query.
    """
    if isinstance(context, list):
        return _context_text(context[-1:])
    return _context_text(context)


def _keys(text: str, max_chars: int = 400) -> tuple[Counter, Counter]:
    # A bounded context window prevents long dialogues from dominating the index.
    tokens = _words(text[-max_chars:])[-64:]
    words = Counter(tokens)
    grams = Counter()
    for token in tokens:
        padded = "^" + token + "$"
        grams.update(padded[i:i + 3] for i in range(max(0, len(padded) - 2)))
    # Preserve the most frequent subwords; tie-breaks are stable across runs.
    grams = Counter(dict(sorted(grams.items(), key=lambda pair: (-pair[1], pair[0]))[:192]))
    return words, grams


def _iter_records(records_or_path: Any):
    if isinstance(records_or_path, (str, Path)):
        path = Path(records_or_path)
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON at {path}:{line_no}: {error}") from error
    else:
        yield from records_or_path


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


class HybridRetriever:
    """Hybrid BM25 word/character retriever with an optional fitted reranker."""

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.db_path = self.index_dir / "index.sqlite3"
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Retrieval index not found: {self.db_path}")
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=1000")
        self._deadline = None
        # Python 3.11 on Windows may implement monotonic() with the coarse
        # GetTickCount64 clock. perf_counter() is also monotonic and provides
        # the resolution needed for small query budgets.
        self._clock = time.perf_counter
        self.metadata = json.loads(self.connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        if self.metadata.get("version") != INDEX_VERSION:
            raise ValueError("Unsupported retrieval index version")
        self.reranker = None
        reranker_path = self.index_dir / "reranker.json"
        if reranker_path.is_file():
            self.reranker = json.loads(reranker_path.read_text(encoding="utf-8"))
            if self.reranker.get("feature_names") != list(FEATURE_NAMES):
                raise ValueError("Reranker feature version mismatch")
            coefficients = self.reranker.get("weights", []) + [self.reranker.get("bias", 0)]
            if len(coefficients) != len(FEATURE_NAMES) + 1 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in coefficients):
                raise ValueError("Invalid reranker coefficients")

    @classmethod
    def build(cls, records_or_path: Any, index_dir: str | Path, *, max_context_chars: int = 400, batch_size: int = 1000):
        if max_context_chars < 32 or max_context_chars > 4000:
            raise ValueError("max_context_chars must be in [32, 4000]")
        directory = Path(index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "index.building.sqlite3"
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(temporary)
        counts = Counter()
        source_counts = Counter()
        length_words = length_chars = 0
        start = time.monotonic()
        try:
            connection.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE docs(docid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                    group_id TEXT NOT NULL, context TEXT NOT NULL, response TEXT NOT NULL,
                    source TEXT NOT NULL, len_word INTEGER NOT NULL, len_char INTEGER NOT NULL);
                CREATE TABLE postings(kind TEXT NOT NULL, term TEXT NOT NULL,
                    docid INTEGER NOT NULL, tf INTEGER NOT NULL);
                CREATE TABLE terms(kind TEXT NOT NULL, term TEXT NOT NULL, df INTEGER NOT NULL,
                    PRIMARY KEY(kind,term));
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            for record in _iter_records(records_or_path):
                counts["seen"] += 1
                if not isinstance(record, Mapping):
                    raise ValueError("Each retrieval record must be a JSON object")
                if record.get("split") != "train":
                    counts["skipped_non_train"] += 1
                    continue
                group_id = record.get("group_id", record.get("dialogue_id"))
                if group_id is None or str(group_id).strip() == "":
                    counts["skipped_no_group"] += 1
                    continue
                context = record.get("context", record.get("question", ""))
                text = _retrieval_key(context)
                response = record.get("response", record.get("answer", ""))
                if not isinstance(response, str) or not response.strip() or not text.strip():
                    counts["skipped_empty"] += 1
                    continue
                issues = response_issues(response)
                if issues:
                    counts["skipped_response_quality"] += 1
                    for issue in issues:
                        counts["quality_" + issue] += 1
                    continue
                record_id = str(record.get("id") or hashlib.sha256((str(group_id) + "\0" + text + "\0" + response).encode()).hexdigest()[:24])
                words, chars = _keys(text, max_context_chars)
                if not words and not chars:
                    counts["skipped_empty"] += 1
                    continue
                nw, nc = sum(words.values()), sum(chars.values())
                source = record.get("source", "")
                cursor = connection.execute("INSERT OR IGNORE INTO docs(id,group_id,context,response,source,len_word,len_char) VALUES(?,?,?,?,?,?,?)", (record_id, str(group_id), json.dumps(context, ensure_ascii=False), response.strip(), json.dumps(source, ensure_ascii=False), nw, nc))
                if cursor.rowcount == 0:
                    counts["skipped_duplicate_id"] += 1
                    continue
                docid = cursor.lastrowid
                connection.executemany("INSERT INTO postings VALUES(?,?,?,?)", ((kind, term, docid, tf) for kind, terms in (("w", words), ("c", chars)) for term, tf in terms.items()))
                counts["indexed"] += 1
                source_counts[str(source)[:160]] += 1
                length_words += nw
                length_chars += nc
                if counts["indexed"] % max(1, batch_size) == 0:
                    connection.commit()
            if not counts["indexed"]:
                raise ValueError("No eligible records: require nonempty context/response, group_id and split='train'")
            connection.executescript("""
                CREATE INDEX postings_lookup ON postings(kind,term,tf DESC,docid);
                CREATE INDEX docs_group ON docs(group_id);
                INSERT INTO terms SELECT kind,term,COUNT(*) FROM postings GROUP BY kind,term;
            """)
            manifest = {"version": INDEX_VERSION, "counts": dict(counts), "sources": dict(source_counts), "max_context_chars": max_context_chars, "query_key": "last_context_turn", "avg_word_length": length_words / counts["indexed"], "avg_char_length": length_chars / counts["indexed"], "build_seconds": round(time.monotonic() - start, 3), "indexed_split": "train"}
            connection.execute("INSERT INTO metadata VALUES('manifest',?)", (json.dumps(manifest, ensure_ascii=False),))
            connection.commit()
        except BaseException:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        connection.close()
        temporary.replace(directory / "index.sqlite3")
        # A fitted model belongs to this exact index; rebuilding invalidates it.
        (directory / "reranker.json").unlink(missing_ok=True)
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return cls(directory)

    @classmethod
    def load(cls, index_dir: str | Path):
        return cls(index_dir)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def search(self, query: str, history=None, persona=None, k: int = 10, exclude_groups=None, *, exclude_group_id=None, limit=None, candidate_pool: int = 1500, postings_per_term: int = 800, use_reranker: bool | None = None, timeout_seconds: float | None = None) -> list[dict]:
        """Search with an optional wall-clock deadline including SQLite work.

        The default respects ``reranker.json``'s ``enabled`` flag. Passing True
        explicitly opts into an experimental comparison; False forces the base.
        """
        if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds < 0):
            raise ValueError("timeout_seconds must be finite and nonnegative")
        deadline = self._clock() + timeout_seconds if timeout_seconds is not None else None
        if use_reranker is None:
            use_reranker = bool(self.reranker and self.reranker.get("enabled", False))
        self._deadline = deadline
        try:
            if deadline is not None:
                milliseconds = min(1000, max(0, int(timeout_seconds * 1000)))
                self.connection.execute(f"PRAGMA busy_timeout={milliseconds}")
                self.connection.set_progress_handler(lambda: int(self._clock() >= deadline), 1000)
            self._check_deadline()
            return self._search(query, history=history, persona=persona, k=k, exclude_groups=exclude_groups,
                                exclude_group_id=exclude_group_id, limit=limit, candidate_pool=candidate_pool,
                                postings_per_term=postings_per_term, use_reranker=use_reranker)
        except sqlite3.OperationalError as error:
            if deadline is not None and self._clock() >= deadline:
                raise TimeoutError("Retrieval deadline exceeded") from error
            raise
        finally:
            self.connection.set_progress_handler(None, 0)
            self._deadline = None
            self.connection.execute("PRAGMA busy_timeout=1000")

    def _check_deadline(self):
        if self._deadline is not None and self._clock() >= self._deadline:
            raise TimeoutError("Retrieval deadline exceeded")

    def _search(self, query: str, history=None, persona=None, k: int = 10, exclude_groups=None, *, exclude_group_id=None, limit=None, candidate_pool: int = 1500, postings_per_term: int = 800, use_reranker: bool = True) -> list[dict]:
        """Return bounded sparse matches, filtering groups before score selection.

        ``limit``/``exclude_group_id`` are aliases for training adapters. History
        is caller-supplied past turns only; the index does not mutate or extend it.
        Persona is accepted for a common runtime interface, but cannot silently
        override a respondent's words or supply a semantic similarity model.
        """
        if limit is not None:
            k = limit
        if k <= 0 or not str(query).strip():
            return []
        k = min(int(k), 100)
        candidate_pool = max(k, min(int(candidate_pool), 10000))
        postings_per_term = max(k, min(int(postings_per_term), 5000))
        if isinstance(exclude_groups, str):
            exclude_groups = [exclude_groups]
        exclusions = {str(x) for x in (exclude_groups or [])}
        if exclude_group_id is not None:
            exclusions.add(str(exclude_group_id))
        if len(exclusions) > 100:
            raise ValueError("At most 100 excluded groups are supported per query")
        query_words, query_chars = _keys(str(query), self.metadata["max_context_chars"])
        weighted_terms = {"w": {term: 1.0 for term in query_words}, "c": {term: 1.0 for term in query_chars}}
        if history:
            history_text = _context_text(history[-2:] if isinstance(history, list) else history)
            hw, hc = _keys(history_text, 160)
            for kind, terms in (("w", hw), ("c", hc)):
                for term in terms:
                    weighted_terms[kind].setdefault(term, 0.2)
        n_docs = self.metadata["counts"]["indexed"]
        scores: dict[int, list[float]] = {}
        normalizers = [0.0, 0.0]
        exclusion_sql = " AND d.group_id NOT IN (" + ",".join("?" for _ in exclusions) + ")" if exclusions else ""
        for kind, index, term_cap in (("w", 0, 24), ("c", 1, 64)):
            known = []
            for term, weight in weighted_terms[kind].items():
                self._check_deadline()
                row = self.connection.execute("SELECT df FROM terms WHERE kind=? AND term=?", (kind, term)).fetchone()
                if row:
                    df = row["df"]
                    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    known.append((weight, idf, term))
            known.sort(key=lambda item: (-item[0], -item[1], item[2]))
            average = self.metadata["avg_word_length" if kind == "w" else "avg_char_length"]
            column = "len_word" if kind == "w" else "len_char"
            for weight, idf, term in known[:term_cap]:
                self._check_deadline()
                normalizers[index] += weight * idf * 2.2
                rows = self.connection.execute(f"SELECT p.docid,p.tf,d.{column} AS length FROM postings p JOIN docs d ON d.docid=p.docid WHERE p.kind=? AND p.term=?" + exclusion_sql + " ORDER BY p.tf DESC,p.docid LIMIT ?", (kind, term, *sorted(exclusions), postings_per_term))
                for row in rows:
                    tf = row["tf"]
                    value = weight * idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * row["length"] / max(average, 1.0)))
                    pair = scores.setdefault(row["docid"], [0.0, 0.0])
                    pair[index] += value
                if len(scores) > candidate_pool * 4:
                    scores = dict(sorted(scores.items(), key=lambda item: (-(item[1][0] + .35 * item[1][1]), item[0]))[:candidate_pool * 2])
        ranked = sorted(scores.items(), key=lambda item: (-self._base_score(item[1], normalizers), item[0]))[:candidate_pool]
        self._check_deadline()
        # Only a small head needs response materialization and learned reranking.
        head_size = min(candidate_pool, max(64, k * 8))
        results = []
        for docid, raw in ranked[:head_size]:
            self._check_deadline()
            row = self.connection.execute("SELECT * FROM docs WHERE docid=?", (docid,)).fetchone()
            context = json.loads(row["context"])
            normalized = [raw[i] / max(normalizers[i], 1e-9) for i in range(2)]
            features = self._features(query, context, row["response"], normalized)
            base = self._base_score(raw, normalizers)
            score = base
            if use_reranker and self.reranker is not None:
                score = _sigmoid(self.reranker["bias"] + sum(a * b for a, b in zip(self.reranker["weights"], features)))
            results.append({"id": row["id"], "group_id": row["group_id"], "context": context, "response": row["response"], "source": json.loads(row["source"]), "score": round(score, 8), "retrieval_score": round(base, 8), "features": dict(zip(FEATURE_NAMES, features))})
        results.sort(key=lambda result: (-result["score"], -result["retrieval_score"], result["id"]))
        self._check_deadline()
        return results[:k]

    @staticmethod
    def _base_score(raw, normalizers):
        return .72 * raw[0] / max(normalizers[0], 1e-9) + .28 * raw[1] / max(normalizers[1], 1e-9)

    @staticmethod
    def _features(query, context, response, normalized):
        query_set = set(_words(query))
        context_set = set(_words(_retrieval_key(context)))
        coverage = len(query_set & context_set) / max(1, len(query_set))
        # These are weak selection features, not a claim of semantic inference.
        negations = {"не", "нет", "никогда", "без"}
        negation_match = float(bool(query_set & negations) == bool(context_set & negations))
        length_fit = max(0.0, 1.0 - abs(len(response) - 100) / 300.0)
        return [float(normalized[0]), float(normalized[1]), coverage, length_fit, negation_match]

    def fit_reranker(self, labels_or_path=None, *, auto_negatives: bool = False, max_examples: int = 4000, epochs: int = 80, learning_rate: float = .15, seed: int = 17, enable: bool = False) -> dict:
        """Fit logistic regression from labels or explicit same-pair supervision.

        Label rows: {query, candidate_id, label:0|1, split:'train'}.
        ``auto_negatives`` uses an indexed training pair as positive and a
        different-group retrieved response as negative. Such negatives are noisy;
        this mode is recorded as ``pair_negatives`` and is never human evaluation.
        """
        import random
        randomizer = random.Random(seed)
        examples = []
        mode = "labels"
        if labels_or_path is not None:
            for label in _iter_records(labels_or_path):
                if label.get("split") != "train":
                    continue
                if label.get("label") not in (0, 1):
                    raise ValueError("Reranker labels must be 0 or 1")
                query = str(label.get("query", ""))
                candidates = self.search(query, k=100, use_reranker=False)
                match = next((candidate for candidate in candidates if candidate["id"] == str(label.get("candidate_id"))), None)
                if match:
                    examples.append(([match["features"][name] for name in FEATURE_NAMES], int(label["label"])))
                if len(examples) >= max_examples:
                    break
        elif auto_negatives:
            mode = "pair_negatives"
            # Stable pseudo-random IDs avoid an unbounded ORDER BY RANDOM().
            count = self.metadata["counts"]["indexed"]
            selected = randomizer.sample(range(1, count + 1), min(count, max_examples // 2))
            for docid in selected:
                row = self.connection.execute("SELECT id,group_id,context FROM docs WHERE docid=?", (docid,)).fetchone()
                if row is None:
                    continue
                query = _retrieval_key(json.loads(row["context"]))
                candidates = self.search(query, k=32, candidate_pool=256, postings_per_term=128, use_reranker=False)
                positive = next((item for item in candidates if item["id"] == row["id"]), None)
                negative = next((item for item in candidates if item["group_id"] != row["group_id"]), None)
                if positive and negative:
                    for match, label in ((positive, 1), (negative, 0)):
                        examples.append(([match["features"][name] for name in FEATURE_NAMES], label))
        else:
            raise ValueError("Provide labels_or_path or explicitly enable auto_negatives")
        if len(examples) < 4 or {label for _, label in examples} != {0, 1}:
            raise ValueError("At least four usable examples including both labels are required")
        weights = [0.0] * len(FEATURE_NAMES)
        bias = 0.0
        for _ in range(max(1, epochs)):
            randomizer.shuffle(examples)
            gradient = [0.0] * len(weights)
            db = 0.0
            for features, label in examples:
                error = _sigmoid(bias + sum(a * b for a, b in zip(weights, features))) - label
                db += error
                for i, value in enumerate(features):
                    gradient[i] += error * value
            size = len(examples)
            bias -= learning_rate * db / size
            weights = [weight - learning_rate * (gradient[i] / size + .001 * weight) for i, weight in enumerate(weights)]
        loss = -sum(label * math.log(max(1e-9, _sigmoid(bias + sum(a * b for a, b in zip(weights, features))))) + (1 - label) * math.log(max(1e-9, 1 - _sigmoid(bias + sum(a * b for a, b in zip(weights, features))))) for features, label in examples) / len(examples)
        self.reranker = {"version": 1, "enabled": bool(enable), "feature_names": list(FEATURE_NAMES), "weights": weights, "bias": bias, "training": {"mode": mode, "examples": len(examples), "epochs": epochs, "seed": seed, "split": "train", "training_loss": loss, "pair_search_candidate_pool": 256 if auto_negatives and labels_or_path is None else None, "pair_search_postings_per_term": 128 if auto_negatives and labels_or_path is None else None}}
        destination = self.index_dir / "reranker.json"
        temporary = destination.with_suffix(".building.json")
        temporary.write_text(json.dumps(self.reranker, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
        return self.reranker["training"]


RetrievalIndex = HybridRetriever


def main(argv=None):
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--data", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--max-context-chars", type=int, default=400)
    search = commands.add_parser("search")
    search.add_argument("--index", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--k", type=int, default=5)
    search.add_argument("--exclude-group", action="append", default=[])
    fit = commands.add_parser("fit-reranker")
    fit.add_argument("--index", required=True)
    fit.add_argument("--labels")
    fit.add_argument("--auto-negatives", action="store_true")
    fit.add_argument("--max-examples", type=int, default=4000)
    fit.add_argument("--epochs", type=int, default=80)
    fit.add_argument("--enable", action="store_true", help="Explicitly use the fitted reranker by default; validate it before enabling")
    args = parser.parse_args(argv)
    if args.command == "build":
        with HybridRetriever.build(args.data, args.output, max_context_chars=args.max_context_chars) as retriever:
            print(json.dumps(retriever.metadata, ensure_ascii=False, indent=2))
    elif args.command == "search":
        with HybridRetriever.load(args.index) as retriever:
            print(json.dumps(retriever.search(args.query, k=args.k, exclude_groups=args.exclude_group), ensure_ascii=False, indent=2))
    else:
        with HybridRetriever.load(args.index) as retriever:
            print(json.dumps(retriever.fit_reranker(args.labels, auto_negatives=args.auto_negatives, max_examples=args.max_examples, epochs=args.epochs, enable=args.enable), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
