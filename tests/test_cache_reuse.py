import json

import pytest

from arena.cache_reuse import seed_cache
from arena.retrieval import HybridRetriever
from arena.retrieval_cache import RetrievalCache, build_cache, _sha


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def row(name, question="Любишь читать книги?", split="train"):
    return {"id": name, "group_id": name, "kind": "dialogue", "split": split,
            "source": "fixture", "context": [{"role": "user", "text": question}],
            "persona": [], "response": "Люблю читать рассказы перед сном."}


def test_reuse_matches_fresh_build_and_recomputes_changed_queries(tmp_path):
    original, val, new = [tmp_path / name for name in ("old.jsonl", "val.jsonl", "new.jsonl")]
    index, cache, derived, fresh = [tmp_path / name for name in ("index", "old.db", "derived.db", "fresh.db")]
    write(original, [row("a"), row("b"), row("c")])
    write(val, [row("v", split="val")])
    HybridRetriever.build(original, index).close()
    build_cache(original, val, index, cache)
    before = _sha(cache)
    write(new, [row("a"), row("b", "Ты любишь путешествия?"), row("new")])
    seeded = seed_cache(cache, original, val, new, val, index, derived)
    assert seeded["copied"] == 2 and seeded["changed_queries"] == 1 and seeded["new_queries"] == 1
    with RetrievalCache(derived, require_complete=False) as reader:
        assert reader.get("a", "a") is not None
        assert reader.get("b", "b") is None
        assert reader.get("new", "new") is None
        assert reader.get("c", "c") is None
    completed = build_cache(new, val, index, derived)
    assert completed["computed_now"] == 2 and completed["complete"]
    build_cache(new, val, index, fresh)
    with RetrievalCache(derived) as one, RetrievalCache(fresh) as two:
        assert one.signature == two.signature
        for name in ("a", "b", "new", "v"):
            assert one.get(name, name) == two.get(name, name)
    assert _sha(cache) == before
    assert seed_cache(cache, original, val, new, val, index, derived)["already_complete"]


def test_reuse_refuses_bad_identity_and_original_destination(tmp_path):
    train, val, new = [tmp_path / name for name in ("train.jsonl", "val.jsonl", "new.jsonl")]
    index, cache = tmp_path / "index", tmp_path / "cache.db"
    write(train, [row("a"), row("b")])
    write(val, [row("v", split="val")])
    HybridRetriever.build(train, index).close()
    build_cache(train, val, index, cache)
    write(new, [row("a"), row("a")])
    with pytest.raises(ValueError, match="original cache"):
        seed_cache(cache, train, val, new, val, index, cache)
    with pytest.raises(ValueError, match="Duplicate destination"):
        seed_cache(cache, train, val, new, val, index, tmp_path / "other.db")
    with train.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Source cache does not match"):
        seed_cache(cache, train, val, new, val, index, tmp_path / "changed.db")


def test_changed_persona_is_not_reused(tmp_path):
    train, val, new = [tmp_path / name for name in ("train.jsonl", "val.jsonl", "new.jsonl")]
    index, cache, output = tmp_path / "index", tmp_path / "cache.db", tmp_path / "new.db"
    write(train, [row("a"), row("b")])
    write(val, [row("v", split="val")])
    HybridRetriever.build(train, index).close()
    build_cache(train, val, index, cache)
    changed = row("a") | {"persona": ["Я люблю готовить."]}
    write(new, [changed])
    report = seed_cache(cache, train, val, new, val, index, output)
    assert report["changed_queries"] == 1 and report["copied"] == 1


def test_partial_source_never_marks_missing_searches_empty_or_complete(tmp_path):
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    index, cache, output = tmp_path / "index", tmp_path / "partial.db", tmp_path / "new.db"
    write(train, [row("a"), row("b"), row("c")])
    write(val, [row("v", split="val")])
    HybridRetriever.build(train, index).close()
    build_cache(train, val, index, cache, limit=1)
    with pytest.raises(ValueError, match="incomplete"):
        seed_cache(cache, train, val, train, val, index, output)
    report = seed_cache(cache, train, val, train, val, index, output, allow_partial_source=True)
    assert report["copied"] == 1 and report["missing_source_entries"] == 3
    assert report["source_incomplete"] and not report["complete"]
    with RetrievalCache(output, require_complete=False) as reader:
        assert reader.get("b", "b") is None
    assert build_cache(train, val, index, output)["computed_now"] == 3
