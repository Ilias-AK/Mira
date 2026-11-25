import json

import pytest
import torch

from arena.formatting import collate_examples, format_example
from arena.model import ArenaModel, ModelConfig
from arena.tokenization import train_tokenizer


@pytest.fixture(autouse=True)
def cap_threads():
    torch.set_num_threads(2)
    torch.manual_seed(7)


def tiny_model(vocab=64, dropout=0.0, source=64):
    return ArenaModel(ModelConfig(vocab_size=vocab, d_model=32, n_heads=4,
        encoder_layers=1, decoder_layers=1, dropout=dropout,
        max_source_length=source, max_target_length=16))


def test_probability_normalization_copy_mask_and_backward():
    model = tiny_model()
    source = torch.tensor([[2, 5, 20, 20, 21], [2, 5, 25, 0, 0]])
    target = torch.tensor([[2, 20, 21], [2, 25, 3]])
    allowed = torch.tensor([[False, False, True, True, True], [False] * 5])
    retrieved = torch.tensor([[False, False, True, True, True], [False] * 5])
    result = model(source, target, copy_mask=allowed, retrieval_mask=retrieved,
                   labels=torch.tensor([[20, 21, 3], [25, 3, -100]]),
                   retrieval_targets=torch.tensor([1.0, -1.0]))
    assert torch.allclose(result["log_probs"].exp().sum(-1), torch.ones(2, 3), atol=1e-6)
    assert torch.equal(result["p_generate"][1], torch.ones_like(result["p_generate"][1]))
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.retrieval_gate[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("copy_enabled", [True, False])
def test_retrieval_gate_receives_nll_gradients_without_auxiliary_labels(copy_enabled):
    model = tiny_model()
    model.config.copy_enabled = copy_enabled
    source = torch.tensor([[2, 5, 20, 8, 21, 22]])
    retrieved = torch.tensor([[False, False, False, True, True, True]])
    result = model(source, torch.tensor([[2, 21]]), retrieval_mask=retrieved,
                   labels=torch.tensor([[21, 3]]), retrieval_targets=torch.tensor([-1.0]))
    assert "gate_loss" not in result
    assert torch.equal(result["loss"], result["nll"])
    # Backpropagate only the token NLL, explicitly excluding auxiliary BCE.
    result["nll"].backward()
    gradient = model.retrieval_gate[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_cpu_bfloat16_autocast_forward_backward_with_retrieval_and_copy():
    """Exercise mixed dtypes on CPU; this is explicitly not a CUDA check."""
    if not torch.amp.autocast_mode.is_autocast_available("cpu"):
        pytest.skip("CPU autocast is unavailable in this PyTorch build")
    model = tiny_model()
    source = torch.tensor([[2, 5, 20, 8, 21, 22], [2, 5, 25, 0, 0, 0]])
    valid = source.ne(model.config.pad_id)
    copy = torch.tensor([[False, False, True, False, True, True], [False] * 6])
    retrieval = torch.tensor([[False, False, False, True, True, True], [False] * 6])
    decoder = torch.tensor([[2, 21, 22], [2, 25, 3]])
    labels = torch.tensor([[21, 22, 3], [25, 3, -100]])
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(source, decoder, attention_mask=valid, copy_mask=copy,
                           retrieval_mask=retrieval, labels=labels,
                           retrieval_targets=torch.tensor([1.0, -1.0]))
        output["loss"].backward()
    except RuntimeError as error:
        message = str(error).lower()
        if "bfloat16" in message and ("not implemented" in message or "not supported" in message):
            pytest.skip(f"CPU kernel explicitly lacks bfloat16 support: {error}")
        raise
    assert torch.isfinite(output["loss"])
    assert output["log_probs"].dtype == torch.float32
    assert torch.allclose(output["log_probs"].exp().sum(-1), torch.ones(2, 3), atol=1e-5)
    assert torch.equal(output["p_generate"][1], torch.ones_like(output["p_generate"][1]))
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.retrieval_gate[-1].weight.grad.abs().sum() > 0


def test_decoder_is_causal_and_primary_encoder_cannot_read_retrieval():
    model = tiny_model().eval()
    source = torch.tensor([[2, 5, 10, 11, 8, 22, 23]])
    retrieval = torch.tensor([[False, False, False, False, True, True, True]])
    a = model(source, torch.tensor([[2, 30, 31]]), retrieval_mask=retrieval)["log_probs"]
    b = model(source, torch.tensor([[2, 30, 41]]), retrieval_mask=retrieval)["log_probs"]
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6)
    assert not torch.allclose(a[:, 2], b[:, 2])
    changed = source.clone()
    changed[:, 5:] = torch.tensor([51, 52])
    original_memory = model.encode(source, retrieval_mask=retrieval)[0]
    changed_memory = model.encode(changed, retrieval_mask=retrieval)[0]
    assert torch.allclose(original_memory[:, :4], changed_memory[:, :4], atol=1e-6)


def test_pointer_can_reproduce_unseen_word_byte_tokens(tmp_path):
    corpus = tmp_path / "data.jsonl"
    corpus.write_text(json.dumps({"split": "train", "kind": "text", "text": "Обычный русский текст." * 20}), encoding="utf-8")
    tokenizer = train_tokenizer([corpus], tmp_path / "tokenizer.json", 300, show_progress=False)
    word = "жмуркоплюх🦊"
    unknown_word = tokenizer.encode(word)
    assert tokenizer.token_id("<unk>") not in unknown_word
    model = tiny_model(tokenizer.vocab_size, source=128).eval()
    with torch.no_grad():
        model.generate_gate.weight.zero_()
        model.generate_gate.bias.fill_(-30)
    source = torch.tensor([[tokenizer.bos_id, tokenizer.token_id("<question>")] + unknown_word])
    reconstructed = []
    for position in range(2, source.shape[1]):
        copy = torch.zeros_like(source, dtype=torch.bool)
        copy[0, position] = True
        result = model(source, torch.tensor([[tokenizer.bos_id]]), copy_mask=copy)
        reconstructed.append(int(result["log_probs"][0, 0].argmax()))
    assert tokenizer.decode(reconstructed) == word


def test_model_can_overfit_a_tiny_conditional_mapping():
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)
    source = torch.tensor([[2, 5, 20, 21], [2, 5, 25, 26]])
    decoder = torch.tensor([[2, 21], [2, 26]])
    labels = torch.tensor([[21, 3], [26, 3]])
    before = float(model(source, decoder, labels=labels)["loss"].detach())
    for _ in range(70):
        optimizer.zero_grad()
        loss = model(source, decoder, labels=labels)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    after = float(model(source, decoder, labels=labels)["loss"].detach())
    assert after < 0.2 and after < before / 10
    generated = model.generate(source, max_new_tokens=5)
    assert generated.tolist() == labels.tolist()


def test_length_and_deadline_are_bounded():
    model = tiny_model()
    source = torch.tensor([[2, 5, 20]])
    assert model.generate(source, max_new_tokens=3).shape[1] <= 3
    stopped = model.generate(source, deadline_seconds=0, return_info=True, no_repeat_ngram_size=3)
    assert stopped["deadline_reached"] and stopped["tokens"].shape[1] == 0


def scripted_generator(monkeypatch, preferences, *, fallback=3):
    """Preferred next tokens are scripted; actual encoder/batch plumbing stays."""
    model = tiny_model().eval()

    def decode(sequence, memory, valid, retrieved, gate, input_ids, copy_mask=None):
        batch, length = sequence.shape
        scores = torch.full((batch, length, model.config.vocab_size), -float("inf"))
        for row, script in enumerate(preferences):
            if fallback is not None:
                scores[row, -1, fallback] = -1.0
            scores[row, -1, script[min(length - 1, len(script) - 1)]] = 0.0
        return scores, torch.ones(batch, length, 1)

    monkeypatch.setattr(model, "decode", decode)
    return model


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_repeated_trigram_is_blocked_and_exhaustion_ends_with_eos(monkeypatch, temperature):
    model = scripted_generator(monkeypatch, [[10, 11, 12, 10, 11, 12]], fallback=None)
    source = torch.tensor([[2, 5, 10, 11, 12]])
    options = dict(max_new_tokens=6, temperature=temperature, top_k=1)
    assert model.generate(source, **options).tolist() == [[10, 11, 12, 10, 11, 12]]
    assert model.generate(source, no_repeat_ngram_size=3, **options).tolist() == [[10, 11, 12, 10, 11, 3]]


def test_ngram_filter_is_per_batch_allows_new_completion_and_ignores_source(monkeypatch):
    scripts = [[10, 11, 12, 10, 11, 12], [10, 11, 13, 10, 11, 12]]
    model = scripted_generator(monkeypatch, scripts)
    # Both inputs contain a target trigram; source copying is still permitted.
    source = torch.tensor([[2, 10, 11, 12], [2, 10, 11, 12]])
    output = model.generate(source, max_new_tokens=6, no_repeat_ngram_size=3)
    assert output.tolist() == [[10, 11, 12, 10, 11, 3], [10, 11, 13, 10, 11, 12]]


def test_unigram_filter_and_finished_batch_rows(monkeypatch):
    model = scripted_generator(monkeypatch, [[10, 10, 10], [11, 12, 13]], fallback=None)
    source = torch.tensor([[2, 5, 20], [2, 5, 21]])
    assert model.generate(source, max_new_tokens=3, no_repeat_ngram_size=1).tolist() == [
        [10, 3, 3], [11, 12, 13]]


@pytest.mark.parametrize("ngram_size", [-1, 1.5, True])
def test_invalid_ngram_constraint_rejected(ngram_size):
    with pytest.raises(ValueError, match="nonnegative integer"):
        tiny_model().generate(torch.tensor([[2, 5, 20]]), no_repeat_ngram_size=ngram_size)


def test_formatting_keeps_question_memory_and_retrieval_with_long_history(tmp_path):
    corpus = tmp_path / "data.jsonl"
    corpus.write_text(json.dumps({"split": "train", "kind": "text", "text": "текст вопрос ответ память " * 50}), encoding="utf-8")
    tokenizer = train_tokenizer([corpus], tmp_path / "tokenizer.json", 400, show_progress=False)
    record = {"context": [{"role": "user", "text": "длинная история " * 1000},
                         {"role": "user", "text": "Что такое жмуркоплюх?"}],
              "response": "Это наше слово", "persona": ["Люблю чай."]}
    example = format_example(record, tokenizer, memory=[{"text": "жмуркоплюх = громкий вздох"}],
                             retrieved=[{"context": "Что это?", "response": "Это наше слово."}])
    decoded = tokenizer.decode(example["input_ids"])
    assert "Что такое жмуркоплюх?" in decoded
    assert "жмуркоплюх = громкий вздох" in decoded
    assert "Это наше слово." in decoded
    assert any(example["retrieval_mask"])
    assert len(example["input_ids"]) <= 512
    assert example["labels"][-1] == tokenizer.eos_id
    batch = collate_examples([example, format_example({"context": [], "response": ""}, tokenizer)], tokenizer)
    assert batch["decoder_input_ids"][:, 0].eq(tokenizer.bos_id).all()
    assert batch["labels"][1, 0] == tokenizer.eos_id
    assert batch["labels"][1, 1:].eq(-100).all()
