import json
import pytest
import torch

from arena.generation_review import (generation_indices, generation_summary, main,
    matching_persona_facts, repeated_word_ngrams, select_validation_indices)


def test_selection_is_stratified_fixed_and_includes_entire_small_source():
    class Dataset:
        def __len__(self):
            return 17
        def record(self, index):
            return {"source": "large" if index < 15 else "small"}
    selected, available = select_validation_indices(Dataset(), max_per_source=5, seed=8)
    assert available == {"large": 15, "small": 2}
    assert len(selected["large"]) == 5 and set(selected["small"]) == {15, 16}
    assert max(selected["large"]) >= 5
    assert selected == select_validation_indices(Dataset(), max_per_source=5, seed=8)[0]
    outputs = generation_indices(selected, count=200, seed=8)
    assert len(outputs) == len(set(outputs)) == 7
    assert outputs == generation_indices(selected, count=200, seed=8)
    assert generation_indices(selected, count=0) == []


def test_persona_matching_uses_complete_word_sequences_and_clear_denominator():
    persona = ["Я учусь в университете.", "Люблю чай."]
    assert matching_persona_facts("Привет, я УЧУСЬ в университете, а ты?", persona) == persona[:1]
    assert not matching_persona_facts("Я учусь.", persona)
    assert not matching_persona_facts("Люблю чайники.", persona)
    assert matching_persona_facts("Я люблю чай!", {"facts": persona}) == persona[1:]
    samples = [dict(persona=persona, generated_persona_matches=persona[:1],
        reference_persona_matches=[], generated="Я учусь в университете.", repeated_word_trigrams=0,
        generation_timeout=False, ended_with_eos=True),
        dict(persona=[], generated_persona_matches=[], reference_persona_matches=[],
        generated="", repeated_word_trigrams=0, generation_timeout=True, ended_with_eos=False)]
    result = generation_summary(samples)
    assert result["persona_bearing_examples"] == 1
    assert result["generated_full_fact_match_rate"] == 1.0
    assert result["reference_full_fact_match_rate"] == 0.0
    assert result["generation_timeouts"] == result["empty_outputs"] == 1
    assert generation_summary([])["generated_full_fact_match_rate"] is None
    assert repeated_word_ngrams("я люблю чай, я люблю чай") == 1


@pytest.fixture
def review_fixture(tmp_path):
    from arena.model import ArenaModel, ModelConfig
    from arena.retrieval import HybridRetriever
    from arena.retrieval_cache import build_cache
    from arena.tokenization import train_tokenizer
    from arena.training import save_checkpoint
    torch.set_num_threads(2)
    torch.manual_seed(7)
    rows = []
    for split in ("train", "val"):
        for index, source in enumerate(("a", "b")):
            rows.append({"id": f"{split}-{index}", "group_id": f"{split}-{index}",
                "source": source, "kind": "dialogue", "split": split,
                "context": [{"role": "user", "text": "Как проведёшь свободный вечер?"}],
                "persona": ["Я люблю читать книги."] if index == 0 else [],
                "response": "Буду читать книгу дома."})
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows if row["split"] == split), encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer([tmp_path / "train.jsonl"], tokenizer_path, 320, show_progress=False)
    model = ArenaModel(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2,
        encoder_layers=1, decoder_layers=1, dropout=0, max_source_length=64, max_target_length=12))
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, tokenizer_path=tokenizer_path, training_config={"stage": "dialogue"})
    index = tmp_path / "index"
    HybridRetriever.build(tmp_path / "train.jsonl", index).close()
    cache = tmp_path / "cache.sqlite3"
    build_cache(tmp_path / "train.jsonl", tmp_path / "val.jsonl", index, cache)
    return ["--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer_path),
        "--train-data", str(tmp_path / "train.jsonl"), "--val-data", str(tmp_path / "val.jsonl"),
        "--index", str(index), "--cache", str(cache), "--device", "cpu",
        "--max-per-source", "1", "--generation-examples", "2", "--max-new-tokens", "4"]


def test_actual_tiny_checkpoint_review_uses_cache_and_preserves_output(review_fixture, tmp_path, monkeypatch):
    from arena.retrieval import HybridRetriever
    def forbidden_search(*args, **kwargs):
        raise AssertionError("Review must not query live retrieval")
    monkeypatch.setattr(HybridRetriever, "search", forbidden_search)
    path = tmp_path / "review.json"
    report = main(review_fixture + ["--output", str(path)])
    assert report["review_finished"] is True
    assert set(report["nll_by_source"]) == {"a", "b"}
    assert all(row["examples"] == 1 for row in report["nll_by_source"].values())
    assert {row["id"] for row in report["generated_samples"]} == {"val-0", "val-1"}
    assert report["generation_summary"]["examples"] == 2
    assert report["generation_summary"]["examples_by_source"] == {"a": 1, "b": 1}
    assert report["generation_settings"]["runtime_tools_filters_fallback"] is False
    assert len(report["inputs"]["cache_sha256"]) == 64
    assert json.loads(path.read_text(encoding="utf-8")) == report
    previous = path.read_bytes()
    with pytest.raises(FileExistsError, match="Preserve existing review"):
        main(review_fixture + ["--output", str(path)])
    assert path.read_bytes() == previous


def test_changed_validation_hash_rejected_before_review(review_fixture, tmp_path):
    with (tmp_path / "val.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    destination = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="cache data signatures"):
        main(review_fixture + ["--output", str(destination)])
    assert not destination.exists()


def test_timed_out_generations_are_saved_without_fallback(review_fixture, tmp_path, monkeypatch):
    from arena.model import ArenaModel
    def timed_out(self, input_ids, **kwargs):
        assert kwargs["return_info"]
        return {"tokens": input_ids.new_empty((1, 0)), "gate": torch.tensor([0.1]),
                "deadline_reached": True, "elapsed_seconds": 30.1}
    monkeypatch.setattr(ArenaModel, "generate", timed_out)
    report = main(review_fixture + ["--output", str(tmp_path / "timeouts.json")])
    assert report["generation_summary"]["generation_timeouts"] == 2
    assert report["generation_summary"]["empty_outputs"] == 2
    assert all(row["generated"] == "" and row["generation_timeout"] for row in report["generated_samples"])
