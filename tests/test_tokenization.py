import json
import unicodedata

import pytest

from arena.tokenization import ArenaTokenizer, SPECIAL_TOKENS, iter_training_texts, train_tokenizer


def test_tokenizer_fits_train_only_and_roundtrips_unseen_words(tmp_path):
    records = [
        {"split": "train", "kind": "dialogue", "context": [{"role": "user", "text": "Как дела?"}],
         "response": "Хорошо. Я пью чай.", "persona": ["Я люблю прогулки."]},
        {"split": "train", "kind": "text", "text": "Русский текст о погоде. " * 20},
        {"split": "val", "kind": "text", "text": "THIS_IS_HELD_OUT_AND_MUST_NOT_BE_FITTED"},
        {"split": "test", "kind": "dialogue", "context": [], "response": "TEST_ONLY"},
    ]
    corpus = tmp_path / "records.jsonl"
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    seen = list(iter_training_texts([corpus]))
    assert len(seen) == 4
    assert not any("HELD_OUT" in s or "TEST_ONLY" in s for s in seen)
    output = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer([corpus], output, vocab_size=400, show_progress=False)
    restored = ArenaTokenizer.load(output)
    for token_id, token in enumerate(SPECIAL_TOKENS):
        assert restored.token_id(token) == token_id
    for text in ("жмуркоплюх", "СовсемНовоеСлово-123 🦊", "Не кофе, а чай!", "и\u0306"):
        encoded = restored.encode(text)
        assert restored.token_id("<unk>") not in encoded
        assert restored.bos_id not in encoded and restored.eos_id not in encoded
        assert restored.decode(encoded) == unicodedata.normalize("NFC", text)
    assert tokenizer.vocab_size == restored.vocab_size
    assert json.loads(output.with_suffix(".meta.json").read_text())["split"] == "train"


def test_empty_or_test_only_training_is_rejected(tmp_path):
    source = tmp_path / "test.jsonl"
    source.write_text(json.dumps({"split": "test", "kind": "text", "text": "Секрет"}), encoding="utf-8")
    target = tmp_path / "tokenizer.json"
    with pytest.raises(ValueError, match="train-split"):
        train_tokenizer([source], target, vocab_size=300, show_progress=False)
    assert not target.exists()


def test_training_text_budget_is_respected(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"split": "train", "kind": "text", "text": "абвгдежз"}), encoding="utf-8")
    assert list(iter_training_texts([source], max_chars=3)) == ["абв"]
