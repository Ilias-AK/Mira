import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
import torch

from arena import next_training as runner


def option(arguments, name):
    return arguments[arguments.index(name) + 1]


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    config = runner.read_json(Path(__file__).resolve().parents[1] / "configs/training_v2.json")
    files = tmp_path / "inputs"
    files.mkdir()

    def data(name):
        path = files / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        return str(path)

    config["common"].update(tokenizer=data("tokenizer.json"), batch_size=2, accumulation=1)
    config["baseline_checkpoint"] = str(files / "baseline.pt")
    config["language"].update(init_checkpoint=str(files / "language.pt"), data=data("language_train.jsonl"),
                              val_data=data("language_val.jsonl"), steps=2, segment_steps=1,
                              eval_every=1, save_every=1, eval_batches=1, warmup_steps=1)
    config["dialogue"].update(steps=2, segment_steps=1, control_steps=1, eval_every=1,
                              save_every=1, eval_batches=1, warmup_steps=1,
                              retrieval_index=str(files / "index"))
    data("index/index.sqlite3")
    data("index/reranker.json")
    for branch in ("control", "candidate"):
        config["dialogue"][branch].update(data=data(branch + "_train.jsonl"),
                                          val_data=data("old_val.jsonl"), retrieval_cache=data(branch + ".sqlite3"))
    config["review"].update(train_data=config["dialogue"]["control"]["data"],
                            val_data=config["dialogue"]["control"]["val_data"],
                            cache=config["dialogue"]["control"]["retrieval_cache"],
                            index=config["dialogue"]["retrieval_index"],
                            rounds=data("rounds.json"), persona=data("persona.json"),
                            max_per_source=2, generation_examples=2, batch_size=2)
    for name, stage in ((config["baseline_checkpoint"], "dialogue"), (config["language"]["init_checkpoint"], "language")):
        torch.save({"model_config": {"d_model": 16},
                    "tokenizer_sha256": runner.sha256(config["common"]["tokenizer"]),
                    "training_state": {"step": 5}, "training_config": {"stage": stage}}, name)
    # Orchestration uses placeholder cache files. Cache validation itself has a
    # separate test below, and actual cache/training integration is CPU-smoked.
    monkeypatch.setattr(runner, "validate_cache_inputs", lambda config, hashes: None)
    return config, tmp_path / "new_run"


class FakeChildren:
    """Exercise orchestration with real checkpoint files, not language quality."""
    def __init__(self):
        self.calls = []
        self.fail_after_candidate = None
        self.mismatched_review = False

    def __call__(self, arguments, log_path):
        self.calls.append(list(arguments))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(arguments) + "\n")
        module = arguments[0]
        if module == "arena.doctor":
            runner.write_json(option(arguments, "--output"), {"cuda_available": False})
        elif module == "arena.training":
            output = Path(option(arguments, "--output-dir"))
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = option(arguments, "--resume") if "--resume" in arguments else option(arguments, "--init-from")
            saved = torch.load(checkpoint, weights_only=False)
            step = int(option(arguments, "--stop-at-step"))
            saved["training_state"] = {"step": step}
            saved["training_config"] = {"stage": option(arguments, "--stage"), "steps": int(option(arguments, "--steps"))}
            torch.save(saved, output / "last.pt")
            torch.save(saved, output / "best.pt")
            runner.write_json(output / "config.json", saved["training_config"])
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "train", "step": step}) + "\n")
            if output.name == "candidate" and step == self.fail_after_candidate:
                raise RuntimeError("simulated interruption after a completed checkpoint")
        elif module == "arena.generation_review":
            checkpoint = option(arguments, "--checkpoint")
            selected = "different" if self.mismatched_review and "candidate" in checkpoint else "heldout-id"
            runner.write_json(option(arguments, "--output"), {
                "review_finished": True,
                "inputs": {"checkpoint_sha256": runner.sha256(checkpoint), "data_sha256": "frozen"},
                "selection": {"selected_ids": [selected]},
                "generation_settings": {"temperature": 0},
                "generated_samples": [{"id": selected}],
            })
        elif module == "arena.evaluation":
            runner.write_json(option(arguments, "--output"), {"rounds": [], "metrics": {}})
        elif module == "arena.language_review":
            runner.write_json(option(arguments, "--output"), {
                "review_finished": True,
                "checkpoint_sha256": runner.sha256(option(arguments, "--checkpoint")),
                "tokenizer_sha256": runner.sha256(option(arguments, "--tokenizer")),
                "data_sha256": runner.sha256(option(arguments, "--data")),
                "settings": {"language_window_policy": "full-span", "validation_sampling": "fixed-shuffle",
                             "seed": int(option(arguments, "--seed")), "batch_size": int(option(arguments, "--batch-size")),
                             "max_batches": int(option(arguments, "--max-batches"))},
            })
        else:
            raise AssertionError(module)


def frozen_outputs(run):
    return {str(p.relative_to(run)): (runner.sha256(p), p.stat().st_mtime_ns)
            for p in run.rglob("*") if p.is_file() and p.suffix in (".pt", ".json", ".jsonl")}


def test_runner_preserves_shared_lr_horizon_and_uses_source_weights_only_for_candidate(recipe):
    config, run = recipe
    control = runner.training_arguments(config, "control", run, 1, "same.pt", "cpu")
    candidate = runner.training_arguments(config, "candidate", run, 1, "same.pt", "cpu")
    assert option(control, "--steps") == option(candidate, "--steps") == "2"
    assert option(control, "--stop-at-step") == option(candidate, "--stop-at-step") == "1"
    for flag in ("--lr", "--warmup-steps", "--seed", "--batch-size", "--accumulation", "--init-from"):
        assert option(control, flag) == option(candidate, flag)
    assert "--source-weights" not in control
    assert json.loads(option(candidate, "--source-weights")) == config["dialogue"]["candidate"]["source_weights"]
    language = runner.training_arguments(config, "language", run, 1, "old_language.pt", "cpu")
    assert option(language, "--language-window-policy") == "full-span"
    assert "--retrieval-cache" not in language


def test_complete_runner_resume_does_not_train_rewrite_weights_or_review_again(recipe, monkeypatch):
    config, run = recipe
    original = {p: runner.sha256(p) for p in (config["baseline_checkpoint"], config["language"]["init_checkpoint"])}
    children = FakeChildren()
    monkeypatch.setattr(runner, "invoke", children)
    archive = runner.run(config, run, device="cpu", engineering=True)
    training = [args for args in children.calls if args[0] == "arena.training"]
    assert len(training) == 5  # language 1/2, control1, candidate1/2
    dialogue_init = [option(args, "--init-from") for args in training
                     if option(args, "--stage") == "dialogue" and "--init-from" in args]
    assert len(dialogue_init) == 2 and dialogue_init[0] == dialogue_init[1]
    before = frozen_outputs(run)
    children.calls.clear()
    runner.run(config, run, resume=True, device="cpu", engineering=True)
    assert not children.calls
    assert frozen_outputs(run) == before
    assert {p: runner.sha256(p) for p in original} == original
    status = runner.read_json(run / "status.json")
    assert status["state"] == "complete" and status["promoted"] is False
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert not any(name.endswith(".pt") or "snapshots" in name for name in names)
        manifest = json.loads(zipped.read("RESULTS_MANIFEST.json"))
        assert manifest["weights_included"] is False
        for name, facts in manifest["files"].items():
            assert hashlib.sha256(zipped.read(name)).hexdigest() == facts["sha256"]


def test_interrupted_runner_resumes_without_replaying_completed_training(recipe, monkeypatch):
    config, run = recipe
    children = FakeChildren()
    children.fail_after_candidate = 1
    monkeypatch.setattr(runner, "invoke", children)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run(config, run, device="cpu", engineering=True)
    assert runner.read_json(run / "status.json")["state"] == "failed"
    assert (run / "training_v2_results.zip").exists()
    children.calls.clear()
    children.fail_after_candidate = None
    runner.run(config, run, resume=True, device="cpu", engineering=True)
    training = [args for args in children.calls if args[0] == "arena.training"]
    assert len(training) == 1
    assert option(training[0], "--stop-at-step") == "2"
    assert option(training[0], "--resume") == str(run / "candidate" / "last.pt")
    assert (run / "candidate" / "snapshots" / "step_000001.pt").exists()


def test_resume_rejects_changed_data_or_hyperparameters(recipe):
    config, run = recipe
    runner.freeze_settings(run, config, resume=False, device="cpu", engineering=True)
    with pytest.raises(FileExistsError):
        runner.freeze_settings(run, config, resume=False, device="cpu", engineering=True)
    changed = copy.deepcopy(config)
    changed["dialogue"]["lr"] *= 2
    with pytest.raises(ValueError, match="hashes changed"):
        runner.freeze_settings(run, changed, resume=True, device="cpu", engineering=True)
    Path(config["dialogue"]["candidate"]["data"]).write_text("modified data", encoding="utf-8")
    with pytest.raises(ValueError, match="hashes changed"):
        runner.freeze_settings(run, config, resume=True, device="cpu", engineering=True)


def test_runner_rejects_noncomparable_review(recipe, monkeypatch):
    config, run = recipe
    children = FakeChildren()
    children.mismatched_review = True
    monkeypatch.setattr(runner, "invoke", children)
    with pytest.raises(ValueError, match="not comparable"):
        runner.run(config, run, device="cpu", engineering=True)
    assert runner.read_json(run / "status.json")["state"] == "failed"
    assert not (run / "candidate" / "snapshots" / "step_000002.pt").exists()


def test_collection_whitelist_excludes_weights_and_unrelated_data(tmp_path):
    for name in ("settings.json", "candidate/config.json", "candidate/metrics.jsonl", "candidate/best.pt",
                 "reviews/one/raw.json", "reviews/one/secret.pt", "logs/run.log", "data/raw.jsonl", "private.txt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    with zipfile.ZipFile(runner.collect_results(tmp_path)) as zipped:
        assert set(zipped.namelist()) == {"settings.json", "candidate/config.json", "candidate/metrics.jsonl",
                                        "reviews/one/raw.json", "logs/run.log", "RESULTS_MANIFEST.json"}


def test_configuration_rejects_unsafe_or_unreviewed_segment_boundaries(recipe):
    config, _ = recipe
    config["language"].update(steps=20, segment_steps=7, eval_every=5)
    with pytest.raises(ValueError, match="multiple"):
        runner.validate_config(config)
    assert runner.segment_ends(10, 4) == [4, 8, 10]


def test_cpu_requires_explicit_engineering_label(recipe):
    config, run = recipe
    with pytest.raises(ValueError, match="engineering"):
        runner.run(config, run, device="cpu")
    assert not run.exists()


def test_preflight_checks_checkpoint_stage_and_hash_without_creating_run(recipe):
    config, run = recipe
    hashes, result = runner.preflight(config, device="cpu", engineering=True)
    assert result["ready"] and result["optimizer_updates"] == 0
    assert result["verified_input_files"] == len(hashes)
    assert not run.exists()
    checkpoint = Path(config["language"]["init_checkpoint"])
    saved = torch.load(checkpoint, weights_only=False)
    saved["training_config"]["stage"] = "dialogue"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="original language"):
        runner.preflight(config, device="cpu", engineering=True)


def test_cache_validation_requires_complete_matching_data(tmp_path, monkeypatch):
    from arena import retrieval_cache
    original_validator = runner.validate_cache_inputs
    train, val = tmp_path / "train", tmp_path / "val"
    hashes = {str(train.resolve()): {"sha256": "train"}, str(val.resolve()): {"sha256": "val"}}
    branch = {"data": str(train), "val_data": str(val), "retrieval_cache": "cache"}
    config = {"dialogue": {"control": branch, "candidate": branch, "retrieval_index": "index"},
              "review": {"train_data": str(train), "val_data": str(val), "cache": "cache", "index": "index"}}
    closed, signatures = [], [{"train": "train", "val": "val"}]

    class Cache:
        complete = True
        def __init__(self, path, **kwargs):
            assert kwargs == {"index_dir": "index", "readonly": True, "require_complete": True}
            self.signature = {"data_sha256": signatures[0]}
        def close(self):
            closed.append(True)

    monkeypatch.setattr(retrieval_cache, "RetrievalCache", Cache)
    original_validator(config, hashes)
    assert len(closed) == 1  # identical control/candidate/review cache checked once
    signatures[0] = {"train": "different", "val": "val"}
    with pytest.raises(ValueError, match="does not match"):
        original_validator(config, hashes)
    assert len(closed) == 2


def test_partial_review_is_preserved_and_atomic_replacement_retries(recipe, monkeypatch):
    config, run = recipe
    path = run / "reviews" / "some-checkpoint" / "raw.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"incomplete":', encoding="utf-8")
    calls = []

    def fail(arguments, log_path):
        destination = Path(option(arguments, "--output"))
        assert destination != path and destination.suffix == ".json"
        destination.write_text('{"partial":', encoding="utf-8")
        calls.append(destination)
        raise RuntimeError("simulated review interruption")

    monkeypatch.setattr(runner, "invoke", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.generate_report(["review", "--output", str(path)], path, run / "logs" / "review.log")
    assert not path.exists()
    assert len(list(path.parent.glob("raw.json.incomplete-*"))) == 1
    assert len(list(path.parent.glob("*.pending.json"))) == 1

    def success(arguments, log_path):
        destination = Path(option(arguments, "--output"))
        assert destination != calls[0]
        runner.write_json(destination, {"complete": True})

    monkeypatch.setattr(runner, "invoke", success)
    assert runner.generate_report(["review", "--output", str(path)], path, run / "logs" / "review.log") == {"complete": True}
    assert runner.read_json(path) == {"complete": True}


def test_concurrent_runner_is_rejected_and_lock_releases_after_failure(tmp_path):
    run = tmp_path / "one_run"
    with pytest.raises(RuntimeError, match="intentional failure"):
        with runner.run_lock(run):
            with pytest.raises(RuntimeError, match="Another training runner"):
                with runner.run_lock(run):
                    pytest.fail("Concurrent run should be blocked")
            raise RuntimeError("intentional failure")
    with runner.run_lock(run):
        pass


@pytest.mark.parametrize("writer", ["json", "snapshot", "zip", "review"])
def test_atomic_writers_retry_transient_windows_locks_once_prepared(tmp_path, monkeypatch, writer):
    real_replace = runner.os.replace
    attempts, sleeps = [], []
    destination = tmp_path / {"json": "status.json", "snapshot": "step.pt", "zip": "training_v2_results.zip", "review": "raw.json"}[writer]
    old = b"previous output" if writer in ("json", "zip") else None
    if old is not None:
        destination.write_bytes(old)

    def replace(source, target):
        if Path(target) == destination:
            attempts.append(Path(source))
            if len(attempts) < 3:
                assert destination.read_bytes() == old if old is not None else not destination.exists()
                error = PermissionError("temporary Windows lock")
                error.winerror = 32 if len(attempts) == 1 else 33
                raise error
        return real_replace(source, target)

    monkeypatch.setattr(runner.os, "replace", replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    if writer == "json":
        runner.write_json(destination, {"step": 4000})
        assert runner.read_json(destination) == {"step": 4000}
    elif writer == "snapshot":
        source = tmp_path / "last.pt"
        source.write_bytes(b"complete checkpoint bytes")
        runner.snapshot_checkpoint(source, destination)
        assert destination.read_bytes() == source.read_bytes()
    elif writer == "zip":
        (tmp_path / "settings.json").write_text('{"verified":true}', encoding="utf-8")
        runner.collect_results(tmp_path)
        with zipfile.ZipFile(destination) as zipped:
            assert zipped.testzip() is None
            assert json.loads(zipped.read("settings.json")) == {"verified": True}
    else:
        def child(arguments, log_path):
            Path(option(arguments, "--output")).write_text('{"finished":true}', encoding="utf-8")
        monkeypatch.setattr(runner, "invoke", child)
        runner.generate_report(["review", "--output", str(destination)], destination, tmp_path / "run.log")
        assert runner.read_json(destination) == {"finished": True}
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert sleeps == [.05, .1]
    assert not attempts[0].exists()


def test_persistent_lock_preserves_previous_and_prepared_json(tmp_path, monkeypatch):
    destination = tmp_path / "status.json"
    destination.write_bytes(b"previous status")
    error = PermissionError("persistent Windows lock")
    error.winerror = 32
    attempts, sleeps = [], []

    def replace(source, target):
        attempts.append(Path(source))
        raise error

    monkeypatch.setattr(runner.os, "replace", replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        runner.write_json(destination, {"step": 4000})
    assert caught.value is error
    assert len(attempts) == 7 and sum(sleeps) == pytest.approx(3.15)
    assert len(set(attempts)) == 1 and runner.read_json(attempts[0]) == {"step": 4000}
    assert destination.read_bytes() == b"previous status"
    assert any(str(attempts[0]) in note for note in error.__notes__)


def test_non_lock_error_is_not_retried_and_incomplete_cleanup_preserves_original_error(tmp_path, monkeypatch):
    destination = tmp_path / "status.json"
    destination.write_bytes(b"previous status")
    replacement_error = OSError("disk full")
    attempts, sleeps = [], []

    def replace(source, target):
        attempts.append(Path(source))
        raise replacement_error

    monkeypatch.setattr(runner.os, "replace", replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(OSError) as caught:
        runner.write_json(destination, {"step": 4000})
    assert caught.value is replacement_error
    assert len(attempts) == 1 and not sleeps
    assert runner.read_json(attempts[0]) == {"step": 4000}

    serialization_error = ValueError("cannot serialize this value")
    def dump(*args, **kwargs):
        raise serialization_error
    real_unlink = Path.unlink
    def unlink(path, *args, **kwargs):
        if path.parent == tmp_path and path.suffix == ".tmp":
            raise PermissionError("cleanup was blocked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(runner.json, "dump", dump)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(ValueError) as caught:
        runner.write_json(destination, {"bad": True})
    assert caught.value is serialization_error
    assert any("incomplete output" in note for note in serialization_error.__notes__)
    assert destination.read_bytes() == b"previous status"


def test_resume_rejects_changed_critical_code_without_rewriting_settings(recipe, monkeypatch, tmp_path):
    config, run = recipe
    code = tmp_path / "training_implementation.py"
    code.write_text("original behavior", encoding="utf-8")
    monkeypatch.setattr(runner, "critical_code_hashes", lambda: {"arena/training.py": {"sha256": runner.sha256(code), "bytes": code.stat().st_size}})
    frozen = runner.freeze_settings(run, config, resume=False, device="cpu", engineering=True)
    assert frozen["critical_code"]["arena/training.py"]["sha256"] == runner.sha256(code)
    settings = run / "settings.json"
    original = (settings.read_bytes(), settings.stat().st_mtime_ns)
    code.write_text("changed training behavior", encoding="utf-8")
    with pytest.raises(ValueError, match="critical code"):
        runner.freeze_settings(run, config, resume=True, device="cpu", engineering=True)
    assert (settings.read_bytes(), settings.stat().st_mtime_ns) == original
