import hashlib
import json
from pathlib import Path

import pytest

import arena.dialogue_curation as curation


def row(identity, source="telegram_pairs", question="Какие книги ты любишь читать?", answer="Я люблю читать фантастику и приключенческие рассказы.", split="train", group=None):
    return {"id": identity, "group_id": group or identity, "source": source, "split": split,
            "kind": "dialogue", "context": [{"role": "user", "text": question}],
            "response": answer, "persona": []}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize("question,answer,expected", [
    ("Какие книги ты любишь читать?", "Я люблю читать фантастику и приключенческие рассказы.", None),
    ("Какую музыку ты чаще всего слушаешь?", "Мне нравится джаз, особенно живые концерты.", None),
    ("Какой суп ты любишь готовить?", "Я часто готовлю овощной суп с картошкой.", None),
    ("Какие книги ты любишь читать?", "Я слушаю джаз каждое утро.", "no_topic_or_content_link"),
    ("Какой фильм стоит посмотреть вечером?", "Хороший фильм про путешествия, но сам", "unfinished_answer"),
    ("Какой фильм стоит посмотреть вечером?", "Тебе вообще зачем смотреть кино?", "short_counterquestion"),
    ("Какой фильм стоит посмотреть вечером?", "Только идиоты смотрят такие фильмы.", "rude"),
    ("Что почитать про лечение болезней?", "Я читаю книги про лечение болезней.", "unreviewed_domain"),
])
def test_telegram_filter_keeps_connected_everyday_pairs_and_rejects_specific_noise(question, answer, expected):
    assert curation.telegram_problem(question, answer) == expected


def test_heldout_protection_covers_identity_pair_question_and_context(tmp_path):
    held = row("hold", split="val", group="shared_group", question="Что тебе особенно нравится в путешествиях?",
               answer="Мне нравится знакомиться с новыми городами и их историей.")
    path = tmp_path / "val.jsonl"
    write_rows(path, [held])
    guard = curation.HeldoutProtection([path])
    assert guard.problem(row("other", group="shared_group")) == "heldout_identity"
    assert guard.problem(held | {"id": "different", "group_id": "different"}) == "heldout_pair"
    assert guard.problem(row("new", question=held["context"][-1]["text"])) == "heldout_question"
    duplicate_context = row("new")
    duplicate_context["context"].insert(0, {"role": "assistant", "text": held["response"]})
    assert guard.problem(duplicate_context) == "heldout_utterance"
    # Generic repeated questions in the already-frozen group split are preserved.
    assert guard.problem(row("new", question=held["context"][-1]["text"]), strict_text=False) is None


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    frozen, expansion = tmp_path / "frozen", tmp_path / "expansion"
    old = [row("toloka:1", source="toloka", question="Как прошёл твой день?", answer="Сегодня я наконец закончил важный проект.")]
    old[0]["persona"] = ["Я работаю инженером."]
    write_rows(frozen / "train.jsonl", old)
    for split in ("val", "test"):
        write_rows(frozen / f"{split}.jsonl", [row(f"old:{split}", source="toloka", split=split,
                     question=f"Какой рассказ {split} тебе запомнился больше всего?", answer=f"В этом рассказе {split} очень интересный сюжет и герои.")])
        write_rows(expansion / f"oasst2/{split}.jsonl", [])
    write_rows(expansion / "oasst2/train.jsonl", [])
    write_rows(expansion / "telegram/train.jsonl", [
        row("tg:1"),
        row("tg:duplicate"),
        row("tg:2", question="Какую музыку ты слушаешь по вечерам?", answer="Мне нравится джаз и живая музыка."),
        row("tg:3", question="Какие книги ты читаешь в выходные?", answer="Я люблю читать фантастические книги и детективы.")])
    token_path = tmp_path / "tokenizer.json"
    token_path.write_text("test tokenizer", encoding="utf-8")

    class Tokenizer:
        def encode(self, text):
            return list(range(96 if "Слишком длинный" in text else 95 if "Граничный ответ" in text else len(text.split())))

    monkeypatch.setattr(curation.ArenaTokenizer, "load", lambda path: Tokenizer())
    return frozen, expansion, token_path, old


def test_prepare_preserves_text_split_and_inputs_and_deduplicates(source_tree, tmp_path):
    frozen, expansion, tokenizer, old = source_tree
    tracked = list(frozen.glob("*.jsonl")) + list(expansion.rglob("*.jsonl"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    destination = tmp_path / "output"
    result = curation.prepare(frozen, expansion, destination, tokenizer_path=tokenizer)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    assert before == after
    records = list(curation.read_jsonl(destination / "train.jsonl"))
    assert next(r for r in records if r["id"] == "toloka:1") == old[0]
    assert len(records) == 4
    assert result["counts"]["telegram_pairs"]["rejected_duplicate_pair_or_id"] == 1
    assert (destination / "val.jsonl").read_bytes() == (frozen / "val.jsonl").read_bytes()
    assert (destination / "oasst2_val.jsonl").read_bytes() == b""
    with pytest.raises(FileExistsError):
        curation.prepare(frozen, expansion, destination, tokenizer_path=tokenizer)


def test_target_token_limit_never_truncates_answer(source_tree, tmp_path):
    frozen, expansion, tokenizer, _ = source_tree
    long = row("oasst:long", source="oasst", answer="Слишком длинный ответ с полным исходным текстом.")
    boundary = row("oasst:boundary", source="oasst", answer="Граничный ответ содержит ровно допустимое количество токенов.")
    write_rows(frozen / "train.jsonl", [long, boundary])
    result = curation.prepare(frozen, expansion, tmp_path / "output", tokenizer_path=tokenizer)
    retained = list(curation.read_jsonl(tmp_path / "output/pools/oasst.jsonl"))
    assert retained == [boundary]
    assert result["counts"]["oasst"]["rejected_target_tokens_over95"] == 1
    assert result["counts"]["oasst"]["otherwise_eligible_le127"] == 2
    assert result["counts"]["oasst"]["target_tokens"] == 95


def test_mixture_caps_are_deterministic_and_do_not_repeat_examples(source_tree, tmp_path):
    frozen, expansion, tokenizer, _ = source_tree
    for name in ("first", "second"):
        curation.prepare(frozen, expansion, tmp_path / name, tokenizer_path=tokenizer,
                         seed=1234, telegram_limit=2)
    assert (tmp_path / "first/train.jsonl").read_bytes() == (tmp_path / "second/train.jsonl").read_bytes()
    rows = list(curation.read_jsonl(tmp_path / "first/train.jsonl"))
    assert len(rows) == 3 and len({r["id"] for r in rows}) == 3
    assert len(list(curation.read_jsonl(tmp_path / "first/pools/telegram_pairs.jsonl"))) == 3


def test_similar_word_prefixes_do_not_invent_everyday_topics():
    assert not curation.topics("кашель")
    assert not curation.topics("компрессор работает")
    assert not curation.topics("другому человеку")
    assert "cooking" in curation.topics("кашу готовлю")
    assert curation.telegram_problem("Как вылечить мокрый кашель после простуды?", "Попробуй выпить чай и отвар травы.") == "specialist_unreviewed"
    assert curation.telegram_problem("школьники дешевые просто", "Дешевые школьники звучит очень интересно.") == "not_explicit_question_or_request"


def test_refinement_verifies_parent_pools_and_preserves_selected_rows(source_tree, tmp_path):
    frozen, expansion, tokenizer, _ = source_tree
    first, second = tmp_path / "first", tmp_path / "second"
    curation.prepare(frozen, expansion, first, tokenizer_path=tokenizer)
    result = curation.prepare(frozen, expansion, second, tokenizer_path=tokenizer, from_pools=first)
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    assert result["parent_curation"]["sha256"] == curation.sha256_file(first / "manifest.json")
    with (first / "pools/telegram_pairs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="Parent curation pool changed"):
        curation.prepare(frozen, expansion, tmp_path / "third", tokenizer_path=tokenizer, from_pools=first)


def test_foreign_identity_filter_keeps_third_person_discussion_and_does_not_rewrite(source_tree, tmp_path):
    frozen, expansion, tokenizer, _ = source_tree
    bad = row("oasst:foreign", source="oasst", answer="Я — это языковая AI-модель, разработанная LAION.")
    good = row("oasst:discussion", source="oasst", answer="Языковые модели обучают предсказывать текст на основе контекста.")
    write_rows(frozen / "train.jsonl", [bad, good])
    destination = tmp_path / "output"
    result = curation.prepare(frozen, expansion, destination, tokenizer_path=tokenizer)
    assert list(curation.read_jsonl(destination / "pools/oasst.jsonl")) == [good]
    assert result["identity_excluded_records"] == 1
    assert json.loads((destination / "identity_exclusions.json").read_text(encoding="utf-8")) == [bad]
    assert result["counts"]["oasst"]["retained"] == 1
    assert result["counts"]["oasst"]["target_tokens"] == len(good["response"].split())
    before = (destination / "manifest.json").read_bytes()
    curation.finalize_identity_filter(destination)
    assert (destination / "manifest.json").read_bytes() == before


def test_precision_profile_separates_experience_and_concrete_help_without_generic_qa():
    assert curation.precision_class("Какие книги ты читаешь в выходные?", "Я люблю читать фантастику и исторические романы.") == "personal_experience"
    assert curation.precision_class("Что приготовить из картошки на ужин?", "Попробуй запечь картошку с луком и сыром в духовке.") == "concrete_help"
    assert curation.precision_class("Как научиться играть на гитаре?", "Начни с простых аккордов и повторяй их каждый день.") == "concrete_help"
    assert curation.precision_class("Бывает ли у собак рак?", "К сожалению, у собак он тоже бывает.") is None
    assert curation.precision_class("Как вам сама картина?", "Мне очень нравится эта картина и ее цвета.") is None
    assert curation.precision_class("Какой язык проще учить?", "Все зависит только от твоего желания.") is None


def test_manual_exclusions_preserve_pre_review_audit_and_remaining_source_text(source_tree, tmp_path):
    frozen, expansion, tokenizer, _ = source_tree
    output = tmp_path / "output"
    curation.prepare(frozen, expansion, output, tokenizer_path=tokenizer)
    before = list(curation.read_jsonl(output / "train.jsonl"))
    before_audit = (output / "precision_audit.json").read_bytes()
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"exclusions":[{"id":"tg:1","reason":"Fixture manual quality decision"}]}), encoding="utf-8")
    result = curation.apply_manual_exclusions(output, decisions)
    assert list(curation.read_jsonl(output / "train.jsonl")) == [r for r in before if r["id"] != "tg:1"]
    assert (output / "precision_audit_before_manual.json").read_bytes() == before_audit
    assert result["manual_exclusions_count"] == 1
    assert result["precision_class_counts"]["personal_experience"] == 2
    manifest_bytes = (output / "manifest.json").read_bytes()
    curation.apply_manual_exclusions(output, decisions)
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    assert (output / "val.jsonl").read_bytes() == (frozen / "val.jsonl").read_bytes()


def test_unknown_manual_ids_fail_before_mutating_any_output(source_tree, tmp_path):
    frozen, expansion, tokenizer, _ = source_tree
    output = tmp_path / "output"
    curation.prepare(frozen, expansion, output, tokenizer_path=tokenizer)
    before = {p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"exclusions":[{"id":"missing","reason":"Not an actual training ID"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing training IDs"):
        curation.apply_manual_exclusions(output, decisions)
    assert {p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()} == before
