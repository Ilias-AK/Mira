import json

import pytest
import torch

from arena.language_review import review
from arena.model import ArenaModel, ModelConfig
from arena.tokenization import train_tokenizer
from arena.training import file_sha256, save_checkpoint


def test_language_review_is_repeatable_and_keeps_checkpoint(tmp_path):
    torch.set_num_threads(1)
    data, tokenizer_path, checkpoint = [tmp_path / name for name in ("data.jsonl", "tokenizer.json", "language.pt")]
    rows = [{"id": str(i), "group_id": str(i), "split": "train" if i == 0 else "val",
             "kind": "text", "source": "test", "text": "Река проходит через город, на берегу растут деревья. " * (i + 2)} for i in range(4)]
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tokenizer = train_tokenizer([data], tokenizer_path, 320, show_progress=False)
    model = ArenaModel(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2,
                                   encoder_layers=1, decoder_layers=1, max_source_length=48, max_target_length=20))
    save_checkpoint(checkpoint, model, tokenizer_path=tokenizer_path, training_config={"stage": "language"})
    before = file_sha256(checkpoint)
    one = review(checkpoint, data, tokenizer_path, max_batches=1, batch_size=2)
    two = review(checkpoint, data, tokenizer_path, max_batches=1, batch_size=2)
    assert one == two and one["review_finished"] and one["metrics"]["nll"] > 0
    assert one["metrics"]["diagnostic_examples"] == 2
    assert file_sha256(checkpoint) == before
    save_checkpoint(checkpoint, model, tokenizer_path=tokenizer_path, training_config={"stage": "dialogue"})
    with pytest.raises(ValueError, match="language-stage"):
        review(checkpoint, data, tokenizer_path, max_batches=1)
