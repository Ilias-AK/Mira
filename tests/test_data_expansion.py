import collections
import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arena.data_expansion import (FrozenProtection, candidate_problem,
                                 iter_new_oasst, iter_telegram, old_tree_ids)


def write_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def node(identity, parent, role, text, **kwargs):
    return dict(message_id=identity, parent_id=parent, role=role, text=text,
                lang="ru", synthetic=False, model_name=None, deleted=False,
                review_result=True, message_tree_id="root", **kwargs)


def test_oasst_strict_parent_and_old_tree_exclusion(tmp_path):
    path = tmp_path / "source.gz"
    rows = [node("root", None, "prompter", "Как провести выходные дома?"),
            node("answer", "root", "assistant", "Можно приготовить ужин и посмотреть кино.")]
    write_gzip(path, rows)
    accepted = list(iter_new_oasst(path, set(), collections.Counter()))
    assert len(accepted) == 1 and accepted[0]["group_id"] == "oasst2:root"
    assert old_tree_ids(path) == {"root"}
    assert not list(iter_new_oasst(path, {"root"}, collections.Counter()))
    for key, value in [("synthetic", True), ("synthetic", None), ("review_result", None),
                       ("deleted", True), ("model_name", "model")]:
        changed = [dict(row) for row in rows]
        changed[0][key] = value
        write_gzip(path, changed)
        assert not list(iter_new_oasst(path, set(), collections.Counter()))


def test_oasst_missing_parent_cycle_and_language_are_rejected(tmp_path):
    path = tmp_path / "source.gz"
    rows = [node("root", None, "prompter", "Что приготовить на ужин?"),
            node("answer", "root", "assistant", "Попробуй запечь картофель с овощами.")]
    for changed in ([rows[1]], [dict(rows[0], lang="en"), rows[1]],
                    [dict(rows[0], parent_id="answer"), rows[1]]):
        write_gzip(path, changed)
        assert not list(iter_new_oasst(path, set(), collections.Counter()))


def test_telegram_uses_only_positive_pairs_and_keeps_training_only(tmp_path):
    path = tmp_path / "0000.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"question": "Как провести выходные дома?", "answer": "Можно приготовить ужин и посмотреть кино.", "relevance": 1},
        {"question": "Как провести выходные дома?", "answer": "Я учусь на третьем курсе.", "relevance": 0},
        {"question": "Чем займемся сегодня вечером?", "answer": "Мне нравится читать интересные книги.", "relevance": 1},
        {"question": "Как связаться с тобой?", "answer": "Напиши мне на test@example.com", "relevance": 1},
    ]), path)
    counts = collections.Counter()
    rows = list(iter_telegram([path], counts))
    assert len(rows) == 2
    assert rows[0]["id"] == "telegram:0000:1"
    assert all(row["split"] == "train" and len(row["context"]) == 1 for row in rows)
    assert counts["rejected_negative_pair"] == 1
    assert counts["rejected_quality_contact_email"] == 1


def test_frozen_protection_includes_all_splits_and_heldout_context(tmp_path):
    for split, question in [("train", "Как прошел твой рабочий день?"),
                            ("val", "Чем займешься после работы?"),
                            ("test", "Какая книга тебе понравилась?")]:
        row = {"kind": "dialogue", "context": [{"role": "user", "text": question}],
               "response": "Мне понравилась книга о путешествиях."}
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    protection = FrozenProtection(tmp_path)
    for question in ["КАК ПРОШЕЛ ТВОЙ РАБОЧИЙ ДЕНЬ?", "Чем займешься после работы?", "Какая книга тебе понравилась?"]:
        row = {"kind": "dialogue", "context": [{"role": "user", "text": question}], "response": "Другой новый ответ."}
        assert protection.reason(row) == "frozen_question"
    row["context"][0]["text"] = "Новый вопрос для разговора"
    row["response"] = "Мне понравилась книга о путешествиях."
    assert protection.reason(row) == "heldout_utterance"
    row["response"] = "В отпуске собираюсь увидеть горы."
    assert protection.reason(row) is None


@pytest.mark.parametrize("answer", ["Да", "Hello my best friend", "слово слово слово слово слово слово", "Посмотри https://example.com", "Вот мой test@example.com", "```python print(1)```"])
def test_strict_chat_filters_noisy_fragments(answer):
    assert candidate_problem("Как провести выходные дома?", answer, strict_chat=True)


def test_concrete_short_conversation_is_accepted():
    assert candidate_problem("Что приготовить на ужин?", "Попробуй запечь картофель с овощами.", strict_chat=True) is None
    assert candidate_problem("Что приготовить на ужин?", "Что приготовить на ужин?", strict_chat=True) == "echo"
    assert candidate_problem("Как отправить бумажное письмо?", "Письмо можно отправить в почтовом отделении.", strict_chat=True) is None


def test_new_oasst_heldout_is_protected_from_later_telegram(tmp_path):
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.jsonl").write_text("", encoding="utf-8")
    protection = FrozenProtection(tmp_path)
    heldout = {"kind": "dialogue", "split": "val", "context": [{"role": "user", "text": "Куда отправиться на каникулы?"}],
               "response": "Я бы отправился в поход с друзьями."}
    protection.add_heldout(heldout)
    training = dict(heldout, split="train", response="Можно поехать в соседний город.")
    assert protection.reason(training) == "frozen_question"
    training["context"] = [{"role": "user", "text": "Что ты любишь делать летом?"}]
    training["response"] = heldout["response"]
    assert protection.reason(training) == "heldout_utterance"
    with pytest.raises(ValueError, match="validation/test"):
        protection.add_heldout(training)
