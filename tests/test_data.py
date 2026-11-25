import collections
import csv
import gzip
import hashlib
import io
import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arena.data import (
    TOLOKA_URL, chunk_text, download_file, iter_oasst, iter_toloka,
    iter_wikipedia, parse_toloka_dialogue, prepare, record_fingerprint, split_group, validate,
)


def write_toloka(path, rows):
    content = io.StringIO(newline="")
    writer = csv.DictWriter(content, fieldnames=["persona_1_profile", "persona_2_profile", "dialogue"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("TlkPersonaChatRus/dialogues.tsv", content.getvalue())


def test_toloka_merges_speaker_messages_and_assigns_target_persona(tmp_path):
    markup = ('<span class=participant_2>Пользователь 2: Привет &amp; здравствуй!</span>'
              '<span class=participant_2>Пользователь 2: Как<br/>дела?</span>'
              '<span class=participant_1>Участник 1: Хорошо.</span>'
              '<span class=participant_2>Участник 2: Рад слышать.</span>')
    assert parse_toloka_dialogue(markup) == [("2", "Привет & здравствуй! Как дела?"), ("1", "Хорошо."), ("2", "Рад слышать.")]
    path = tmp_path / "toloka.zip"
    write_toloka(path, [{"persona_1_profile": "Люблю чай.<br/>Живу в городе.", "persona_2_profile": "Люблю кофе.", "dialogue": markup}])
    examples = list(iter_toloka(path))
    assert len(examples) == 2
    assert examples[0]["persona"] == ["Люблю чай.", "Живу в городе."]
    assert examples[1]["persona"] == ["Люблю кофе."]
    assert examples[0]["context"][-1]["role"] == "user"
    assert examples[0]["group_id"] == examples[1]["group_id"]
    assert "Участник" not in json.dumps(examples, ensure_ascii=False)
    assert "Пользователь" not in json.dumps(examples, ensure_ascii=False)
    assert "provenance" not in examples[0] and "license" not in examples[0]


def test_oasst_reconstructs_parents_not_file_neighbors_and_excludes_bad_ancestors(tmp_path):
    def node(node_id, parent, text, role="assistant", **extra):
        return {"message_id": node_id, "parent_id": parent, "message_tree_id": "tree1", "role": role,
                "text": text, "lang": "ru", "synthetic": False, "review_result": True, **extra}
    nodes = [
        node("a2", "q2", "Помню слово."),
        node("q2", "a1", "Помнишь слово?", "prompter"),
        node("bad_answer", "synthetic_parent", "Плохая ветвь"),
        node("missing", "unknown", "Нет родителя"),
        node("a1", "q1", "Запомнил."),
        node("q1", None, "Запомни жмуркоплюх.", "prompter"),
        node("synthetic_parent", None, "Синтетика", "prompter", synthetic=True),
        node("model_answer", "q1", "Скрытая модель", model_name="external"),
    ]
    path = tmp_path / "oasst.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for item in nodes:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    examples = list(iter_oasst(path, tmp_path / "nodes.sqlite"))
    assert {item["id"] for item in examples} == {"oasst:a1", "oasst:a2"}
    final = next(item for item in examples if item["id"] == "oasst:a2")
    assert [t["text"] for t in final["context"]] == ["Запомни жмуркоплюх.", "Запомнил.", "Помнишь слово?"]
    assert [t["role"] for t in final["context"]] == ["user", "assistant", "user"]


def test_prepare_groups_before_windows_and_removes_global_duplicate_pairs(tmp_path):
    raw = tmp_path / "raw"
    dialogue = ('<span class=participant_1>Участник 1: Привет!</span>'
                '<span class=participant_2>Участник 2: Здравствуй!</span>'
                '<span class=participant_1>Участник 1: Как день?</span>'
                '<span class=participant_2>Участник 2: Удачно.</span>')
    row = {"persona_1_profile": "", "persona_2_profile": "", "dialogue": dialogue}
    write_toloka(raw / "toloka" / "TlkPersonaChatRus.zip", [row, row])
    output = tmp_path / "processed"
    result = prepare(raw, output, sources=["toloka"], include_toloka=True)
    records = [json.loads(line) for split in ("train", "val", "test") for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert len({r["split"] for r in records}) == 1
    assert len({record_fingerprint(r) for r in records}) == 3
    assert result["counts"]["duplicate_records_removed"] == 3
    assert all(set(r) == {"id", "group_id", "source", "kind", "context", "response", "persona", "split"} for r in records)
    assert (output / "manifest.json").exists()
    assert validate(output)["valid"] is True


def test_wikipedia_streams_article_chunks_with_stable_group(tmp_path):
    path = tmp_path / "sample.parquet"
    text = "Это содержательная русская статья о природе и городе. " * 80
    pq.write_table(pa.Table.from_pylist([{"id": "123", "text": text}, {"id": "124", "text": "English data " * 100}]), path)
    examples = list(iter_wikipedia([path], max_chars=0, chunk_chars=300))
    assert len(examples) > 5
    assert all(100 <= len(e["text"]) <= 300 for e in examples)
    assert len({e["group_id"] for e in examples}) == 1
    assert all(e["kind"] == "text" for e in examples)
    assert sum(len(e["text"]) for e in iter_wikipedia([path], max_chars=650, chunk_chars=300)) <= 650


def test_prepare_filters_dialogue_targets_without_filtering_encyclopedia(tmp_path, monkeypatch):
    from arena import quality

    marker = "Отфильтрованный ответ."
    monkeypatch.setattr(quality, "response_issues", lambda text: ["test_quality"] if marker in text else [])
    raw = tmp_path / "raw"
    markup = ('<span class=participant_1>Пользователь 1: Начнём разговор?</span>'
              f'<span class=participant_2>Пользователь 2: {marker}</span>'
              '<span class=participant_1>Пользователь 1: Давай обсудим другое.</span>')
    write_toloka(raw / "toloka" / "TlkPersonaChatRus.zip", [{
        "persona_1_profile": "", "persona_2_profile": "", "dialogue": markup,
    }])
    wiki = raw / "wikipedia"
    wiki.mkdir()
    pq.write_table(pa.Table.from_pylist([{"id": "article", "text": (marker + " Энциклопедическое упоминание. ") * 10}]), wiki / "one.parquet")
    output = tmp_path / "processed"
    manifest = prepare(raw, output, sources=["toloka", "wikipedia"], include_toloka=True, max_wiki_chars=0)
    records = [json.loads(line) for split in ("train", "val", "test")
               for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["counts"]["dialogue_targets_filtered"] == 1
    assert manifest["counts"]["response_filter_test_quality"] == 1
    assert len(manifest["config"]["response_filter_sha256"]) == 64
    assert all(marker not in r["response"] for r in records if r["kind"] == "dialogue")
    assert any(marker in r["text"] for r in records if r["kind"] == "text")
    assert validate(output)["valid"] is True


def test_download_resumes_and_verifies_hash(tmp_path, monkeypatch):
    content = b"correct complete file"
    target = tmp_path / "dataset.bin"
    target.with_name("dataset.bin.part").write_bytes(content[:8])
    target.with_name("dataset.bin.part.json").write_text(json.dumps({"url": "https://example.test/data"}))
    requested = []

    class Response:
        status_code = 206
        headers = {"Content-Range": f"bytes 8-{len(content)-1}/{len(content)}"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield content[8:]

    def get(url, **kwargs):
        requested.append(kwargs["headers"])
        return Response()

    monkeypatch.setattr("arena.data.requests.get", get)
    report = download_file("https://example.test/data", target, hashlib.sha256(content).hexdigest(), len(content))
    assert requested[0]["Range"] == "bytes=8-"
    assert target.read_bytes() == content
    assert report["bytes"] == len(content)
    assert not target.with_name("dataset.bin.part").exists()


def test_group_split_is_deterministic_and_validates_fractions():
    assert split_group("conversation1") == split_group("conversation1")
    assert {split_group(str(i)) for i in range(1000)} == {"train", "val", "test"}
    with pytest.raises(ValueError):
        split_group("x", val_fraction=.7, test_fraction=.7)
