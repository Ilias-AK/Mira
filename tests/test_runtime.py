import pytest
import multiprocessing as mp
import time

from arena.runtime import bounded_text, output_problem
from arena.evaluation import create_blind_packet, tally_votes


def test_length_limit_preserves_complete_sentence():
    text = "Короткий ответ. " + "слово " * 100
    result = bounded_text(text, 60)
    assert result and len(result) <= 60
    assert not result.endswith(" ")


def test_repetition_detection():
    assert output_problem("это одна фраза " * 4) == "repetition"
    assert output_problem("Сегодня я выберу воду, потому что чая нет.") is None
    assert output_problem("?" * 32) == "no_content"
    assert output_problem("муршик" * 8) == "repetition"
    assert output_problem("😂") is None


def test_repeated_response_across_turns_is_narrow_and_question_aware():
    from arena.runtime import repeated_response
    history = [{"role": "user", "text": "Как проведёшь вечер?"},
               {"role": "assistant", "text": "Привет, я учусь в университете, а ты?"}]
    assert repeated_response("Я учусь в университете, а ты?", "А если дождь?", history) == 1
    assert repeated_response("Да, я учусь в университете, а ты?", "Что будешь делать?", history) == 1
    assert repeated_response("Я учусь в университете, а ты?", "Как проведёшь вечер!", history) is None
    assert repeated_response("Я учусь в университете, а ты?", "Повтори свой ответ", history) is None
    assert repeated_response("Да", "Ты уверен?", history) is None
    assert repeated_response("Я учусь в университете, а вечером собираюсь читать книгу.", "А если дождь?", history) is None


def test_conversation_checks_do_not_call_tools_dialogue_quality():
    from arena.evaluation import conversation_checks
    report = {"rounds": [{"id": "r", "turns": [
        {"question": "Как проведёшь вечер?", "answer": "Привет, я учусь в университете, а ты?", "route": "generator"},
        {"question": "А если дождь?", "answer": "Я учусь в университете, а ты?", "route": "generator"},
        {"question": "2+2", "answer": "4", "route": "tool"},
    ]}]}
    checks = conversation_checks(report)
    assert checks["free_dialogue_turns"] == 2
    assert checks["route_counts"] == {"generator": 2, "tool": 1}
    assert checks["repeated_turns"] == [{"round_id": "r", "turn": 2, "previous_turn": 1}]
    # A new round must not inherit the previous round's history.
    report["rounds"].append({"id": "new", "turns": [report["rounds"][0]["turns"][1]]})
    assert conversation_checks(report)["near_repeated_answers"] == 1


def test_runtime_blocks_repeats_from_both_generator_and_retrieval(tmp_path):
    from arena.retrieval import HybridRetriever
    from arena.runtime import ArenaRuntime

    class Generator:
        def generate(self, **kwargs):
            assert kwargs["no_repeat_ngram_size"] == 3
            return {"status": "ok", "text": "Я учусь в университете, а ты?"}
        def close(self):
            pass

    HybridRetriever.build([{"id": "1", "group_id": "1", "split": "train",
        "context": "дождь", "response": "Я учусь в университете, а ты?"}], tmp_path / "index").close()
    runtime = ArenaRuntime(tmp_path / "index", generator=Generator(), no_repeat_ngram_size=3)
    assert runtime.reject_repeated_responses is False
    runtime.reject_repeated_responses = True
    try:
        runtime.retriever.search = lambda *a, **k: [
            {"id": "bad", "response": "Я учусь в университете, а ты?"},
            {"id": "good", "response": "Останусь дома с книжкой."}]
        first = runtime.answer("Как проведёшь вечер?")
        assert first.route == "generator"
        second = runtime.answer("А если на улице дождь?")
        assert second.route == "retrieval"
        assert second.text == "Останусь дома с книжкой."
        assert second.diagnostics["generation_rejected"]["problem"] == "repeated_response"
        assert second.diagnostics["rejected"][0]["problem"] == "repeated_response"
        runtime.reset()
        assert runtime.answer("А если на улице дождь?").route == "generator"
        runtime.reject_repeated_responses = False
        assert runtime.answer("Что будешь делать?").route == "generator"
    finally:
        runtime.close()


def test_length_limit_also_counts_utf16_emoji():
    result = bounded_text("🙂" * 200, 280)
    assert len(result.encode("utf-16-le")) // 2 <= 280
    assert "\ufffd" not in result


def test_stage_output_rejects_foreign_sentences_and_code_only():
    assert output_problem("London is the capital of Great Britain") == "non_russian_text"
    assert output_problem("Hello! Do you speak Russian?") == "non_russian_text"
    assert output_problem('```go\npackage main\n```') == "code_fence"
    assert output_problem('Вот пример: ```print(1)```') == "code_fence"
    assert output_problem('~~~python\nprint(1)\n~~~') == "code_fence"
    for text in ("OK", "John Smith", "Linkin Park", "Мне нравится Linkin Park.", "12345678901", "🙂"):
        assert output_problem(text) is None


def test_blind_packets_keep_position_and_strip_diagnostics():
    def participant(name):
        return {"rounds": [{"id": "r1", "turns": [
            {"question": q, "answer": name + str(i), "route": "internal"}
            for i, q in enumerate(["Привет", "А потом?"])]}]}
    public, private = create_blind_packet(participant("bot"), participant("a"), participant("b"))
    position = private["rounds"]["r1"]["bot_booth"]
    for i, turn in enumerate(public["rounds"][0]["turns"]):
        assert turn["answers"][position - 1]["text"] == "bot" + str(i)
        assert set(turn["answers"][0]) == {"booth", "text"}
    votes = [{"round_id": "r1", "viewer_id": "v1", "booth": position}]
    assert tally_votes(private, votes)["rounds"][0]["identified_bot_fraction"] == 1
    with pytest.raises(ValueError, match="only once"):
        tally_votes(private, votes + votes)


def test_blind_packet_rejects_different_questions():
    first = {"rounds": [{"id": "r", "turns": [{"question": "a", "answer": "x"}]}]}
    other = {"rounds": [{"id": "r", "turns": [{"question": "b", "answer": "x"}]}]}
    with pytest.raises(ValueError, match="identical questions"):
        create_blind_packet(first, first, other)


def _stall_worker(pipe):
    pipe.recv()
    time.sleep(60)


def test_generator_timeout_terminates_worker():
    from arena.runtime import IsolatedGenerator
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=_stall_worker, args=(child,), daemon=True)
    process.start()
    child.close()
    generator = object.__new__(IsolatedGenerator)
    generator.connection, generator.process, generator.error = parent, process, None
    start = time.monotonic()
    result = generator.generate(seconds=.1, record={})
    assert result["status"] == "timeout"
    assert time.monotonic() - start < 3
    assert generator.process is None


def test_runtime_tools_memory_and_fallback(tmp_path):
    from arena.retrieval import HybridRetriever
    from arena.runtime import ArenaRuntime
    rows = [{"id": "1", "group_id": "1", "split": "train", "source": "unit",
             "kind": "dialogue", "context": [{"role": "user", "text": "Привет"}],
             "response": "Привет, как проходит день?"}]
    index = HybridRetriever.build(rows, tmp_path / "index")
    index.close()
    runtime = ArenaRuntime(tmp_path / "index")
    try:
        assert "391" in runtime.answer("Сколько будет 17 * 23?").text
        runtime.answer("Назовём муршиком желание остаться дома.")
        recall = runtime.answer("Что значит муршик?")
        assert recall.route == "tool"
        assert "дома" in recall.text
        assert len(runtime.answer("Привет").text) <= 280
        runtime.reset()
        assert runtime.memory.history == []
    finally:
        runtime.close()


def test_tool_result_cannot_bypass_response_filter(tmp_path, monkeypatch):
    from arena.retrieval import HybridRetriever
    from arena.runtime import ArenaRuntime
    import arena.constraints
    row = {"id": "1", "group_id": "1", "split": "train", "source": "unit",
           "context": [{"role": "user", "text": "Привет"}], "response": "Привет!"}
    HybridRetriever.build([row], tmp_path / "index").close()
    monkeypatch.setattr(arena.constraints, "execute_local_tool", lambda question: {"text": "test@example.com"})
    runtime = ArenaRuntime(tmp_path / "index")
    try:
        answer = runtime.answer("Привет")
        assert answer.route != "tool"
        assert "@" not in answer.text
        assert answer.diagnostics["tool_rejected"]
    finally:
        runtime.close()


def test_foreign_retrieval_and_generation_are_rejected_but_literal_tool_survives(tmp_path):
    from arena.retrieval import HybridRetriever
    from arena.runtime import ArenaRuntime

    class ForeignGenerator:
        def generate(self, **kwargs):
            return {"status": "ok", "text": "Hello! Do you speak Russian?"}

        def close(self):
            pass

    rows = [
        {"id": "0", "group_id": "0", "split": "train", "context": "Привет", "response": "London is the capital of Great Britain"},
        {"id": "1", "group_id": "1", "split": "train", "context": "Привет", "response": "Привет, как проходит день?"},
    ]
    HybridRetriever.build(rows, tmp_path / "index").close()
    runtime = ArenaRuntime(tmp_path / "index", generator=ForeignGenerator())
    try:
        answer = runtime.answer("Привет")
        assert answer.route == "retrieval"
        assert answer.text == "Привет, как проходит день?"
        assert any(item["problem"] == "non_russian_text" for item in answer.diagnostics["rejected"])
        assert answer.diagnostics["generation_rejected"]["problem"] == "non_russian_text"
        literal = runtime.answer('Напиши слово «this is English text» наоборот.')
        assert literal.route == "tool"
        assert literal.text == "txet hsilgnE si siht"
    finally:
        runtime.close()
