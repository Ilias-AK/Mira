from arena.diagnostics import compare_runtime


def test_ablation_options_and_generator_lifetime(tmp_path, monkeypatch):
    import arena.diagnostics as diagnostics
    instances = []
    generator = object()

    class Runtime:
        def __init__(self, index, **options):
            self.options = options
            self.generator = options["generator"]
            self.closed = False
            instances.append(self)
        def close(self):
            assert self.generator is None
            self.closed = True

    monkeypatch.setattr(diagnostics, "ArenaRuntime", Runtime)
    monkeypatch.setattr(diagnostics, "run_rounds", lambda runtime, rounds: {"rounds": rounds})
    result = compare_runtime(generator, tmp_path, {"facts": ["Учусь в университете."]}, {"rounds": []})
    assert len(result) == 6
    options = {key: instance.options for key, instance in zip(result, instances)}
    assert not options["baseline"]["reject_repeated_responses"]
    assert options["without_persona"]["persona"] == []
    assert options["without_retrieval"]["use_retrieval"] is False
    assert options["without_persona_or_retrieval"]["persona"] == []
    assert options["without_persona_or_retrieval"]["use_retrieval"] is False
    assert options["retrieval_only"]["generator"] is None
    assert options["with_repeat_guard"]["reject_repeated_responses"] is True
    assert all(instance.closed for instance in instances)
