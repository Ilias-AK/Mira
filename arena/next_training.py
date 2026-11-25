"""Resumable language expansion and a paired dialogue-data experiment.

Every segment keeps a fixed total LR horizon. Reviews use the original frozen
validation/cache for every checkpoint. This runner never promotes a candidate
or writes to the original runs/main checkpoints.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile


CRITICAL_CODE_FILES = (
    "training.py", "model.py", "formatting.py", "tokenization.py", "memory.py",
    "generation_review.py", "language_review.py", "runtime.py", "evaluation.py",
    "next_training.py", "retrieval.py", "retrieval_cache.py", "quality.py", "constraints.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def replace_ready_file(source, destination):
    """Retry only transient Windows locks; preserve prepared bytes on failure."""
    source, destination = Path(source), Path(destination)
    delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
    try:
        for attempt in range(len(delays) + 1):
            try:
                os.replace(source, destination)
                return
            except OSError as error:
                if getattr(error, "winerror", None) not in (32, 33) or attempt == len(delays):
                    raise
                time.sleep(delays[attempt])
    except BaseException as error:
        error.add_note(f"Replacement failed; the previous destination {destination} was not removed. "
                       f"Prepared file retained at {source} for inspection or recovery.")
        raise


@contextlib.contextmanager
def atomic_output(path, *, external_writer=False, preserve_incomplete=False):
    """Prepare once, fsync, then publish; cleanup never masks an earlier error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if external_writer:
        # generation_review refuses an existing output; reserve a unique name
        # without creating an empty file that would make the child refuse it.
        temporary = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.pending.json")
        if temporary.exists():
            raise FileExistsError(f"Temporary output already exists: {temporary}")
    else:
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        temporary = Path(name)
    prepared = False
    try:
        yield temporary
        prepared = True
        # The writer has closed its handle. Flush complete copies and ZIP files
        # too, not just the JSON path, before replacing the old artifact.
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        replace_ready_file(temporary, path)
    except BaseException as error:
        if prepared:
            if not any(str(temporary) in note for note in getattr(error, "__notes__", ())):
                error.add_note(f"Prepared output retained at {temporary}; the previous {path} was not removed.")
        elif preserve_incomplete:
            error.add_note(f"Any incomplete external report remains at {temporary}; it was not published as {path}.")
        else:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(f"Could not remove incomplete output {temporary}: {cleanup_error}")
        raise


def write_json(path, value):
    with atomic_output(path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")


def critical_code_hashes():
    directory = Path(__file__).resolve().parent
    return {f"arena/{name}": {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size}
            for name in CRITICAL_CODE_FILES}


def checkpoint_info(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "sha256": sha256(path),
        "model_config": checkpoint["model_config"],
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "training_state": checkpoint.get("training_state", {}),
        "training_config": checkpoint.get("training_config", {}),
    }


def segment_ends(limit, size):
    return list(range(size, limit, size)) + [limit]


def validate_config(config):
    if config.get("version") != 1:
        raise ValueError("Unsupported next-training configuration version")
    for section, names in (("common", ("batch_size", "accumulation", "threads")),
                           ("language", ("steps", "segment_steps", "eval_every", "save_every", "eval_batches")),
                           ("dialogue", ("steps", "segment_steps", "control_steps", "eval_every", "save_every", "eval_batches")),
                           ("review", ("max_per_source", "generation_examples", "batch_size"))):
        for key in names:
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if config["dialogue"]["control_steps"] > config["dialogue"]["steps"]:
        raise ValueError("control_steps cannot exceed the shared dialogue LR horizon")
    if config["language"].get("language_window_policy", "full-span") != "full-span":
        raise ValueError("This recipe requires full-span language windows")
    for name in ("language", "dialogue"):
        stage = config[name]
        if stage["lr"] <= 0 or stage["warmup_steps"] < 0:
            raise ValueError(f"Invalid learning-rate settings for {name}")
        if stage["segment_steps"] % stage["eval_every"]:
            raise ValueError(f"{name} segment_steps must be a multiple of eval_every")
        if stage["steps"] % stage["eval_every"]:
            raise ValueError(f"{name} steps must be a multiple of eval_every")
    if config["dialogue"]["control_steps"] % config["dialogue"]["eval_every"]:
        raise ValueError("control_steps must end at a scheduled validation point")


def input_files(config):
    common, language, dialogue, review = (config[k] for k in ("common", "language", "dialogue", "review"))
    paths = [config["baseline_checkpoint"], language["init_checkpoint"], common["tokenizer"],
             language["data"], language["val_data"], review["train_data"], review["val_data"],
             review["cache"], review["rounds"], review["persona"]]
    for branch in ("control", "candidate"):
        paths.extend(dialogue[branch][key] for key in ("data", "val_data", "retrieval_cache"))
    for directory in dict.fromkeys((dialogue["retrieval_index"], review["index"])):
        paths.append(str(Path(directory) / "index.sqlite3"))
        reranker = Path(directory) / "reranker.json"
        if reranker.exists():
            paths.append(str(reranker))
    return sorted(set(str(Path(path).resolve()) for path in paths))


def inspect_inputs(config):
    hashes = {}
    for name in input_files(config):
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Required prepared input is missing: {path}")
        hashes[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return hashes


def validate_cache_inputs(config, hashes):
    from .retrieval_cache import RetrievalCache
    dialogue, review = config["dialogue"], config["review"]
    definitions = [(dialogue[branch]["retrieval_cache"], dialogue["retrieval_index"],
                    dialogue[branch]["data"], dialogue[branch]["val_data"])
                   for branch in ("control", "candidate")]
    definitions.append((review["cache"], review["index"], review["train_data"], review["val_data"]))
    for cache_path, index, train, val in dict.fromkeys(definitions):
        cache = RetrievalCache(cache_path, index_dir=index, readonly=True, require_complete=True)
        try:
            expected = {"train": hashes[str(Path(train).resolve())]["sha256"],
                        "val": hashes[str(Path(val).resolve())]["sha256"]}
            if not cache.complete or cache.signature.get("data_sha256") != expected:
                raise ValueError(f"Retrieval cache does not match prepared train/validation: {cache_path}")
        finally:
            cache.close()


def preflight(config, *, device="cuda", engineering=False):
    """Validate all stages before spending GPU time; create no run or weights."""
    import torch
    validate_config(config)
    if device != "cuda" and not engineering:
        raise ValueError("CPU runs require --engineering; they are code checks, not production training")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the Lenovo CUDA environment")
    hashes = inspect_inputs(config)
    code = critical_code_hashes()
    tokenizer_digest = hashes[str(Path(config["common"]["tokenizer"]).resolve())]["sha256"]
    initial = [checkpoint_info(config["baseline_checkpoint"]), checkpoint_info(config["language"]["init_checkpoint"])]
    if any(info["tokenizer_sha256"] != tokenizer_digest for info in initial):
        raise ValueError("Initial checkpoints do not match the frozen tokenizer")
    if initial[0]["model_config"] != initial[1]["model_config"]:
        raise ValueError("Baseline and initial language checkpoint architectures differ")
    for info, expected in zip(initial, ("dialogue", "language")):
        if info["training_config"].get("stage") != expected:
            raise ValueError(f"Expected an original {expected} checkpoint")
    validate_cache_inputs(config, hashes)
    metadata = {"ready": True, "device": device, "engineering": bool(engineering),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "verified_input_files": len(hashes), "tokenizer_sha256": tokenizer_digest,
                "verified_code_files": len(code),
                "baseline_step": initial[0]["training_state"].get("step"),
                "language_initial_step": initial[1]["training_state"].get("step"),
                "optimizer_updates": 0, "note": "Input readiness only; not a model-quality verdict."}
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return hashes, metadata


def freeze_settings(run_dir, config, *, resume, device, engineering, hashes=None):
    run_dir = Path(run_dir)
    destination = run_dir / "settings.json"
    exists = run_dir.exists()
    if exists and not resume:
        raise FileExistsError(f"Run exists: {run_dir}. Use --resume or a new --run-name.")
    if exists and not destination.is_file():
        raise ValueError("Existing run lacks settings.json; preserve it and use a new run name")
    hashes = inspect_inputs(config) if hashes is None else hashes
    settings = {"version": 1, "config": config, "device": device,
                "engineering": bool(engineering), "inputs": hashes, "critical_code": critical_code_hashes()}
    if destination.is_file():
        if read_json(destination) != settings:
            raise ValueError("Run settings, critical code, or prepared input hashes changed; use a new run name")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(destination, settings)
    return settings


def invoke(arguments, log_path):
    """Stream child progress to the console and retain the same text for audit."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-X", "utf8", "-m", *map(str, arguments)]
    environment = dict(os.environ, PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment) as child:
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
                raise
        if code:
            raise RuntimeError(f"Child step failed with exit code {code}; see {log_path}")


@contextlib.contextmanager
def keep_awake():
    """Temporarily inhibit automatic Windows sleep; restore this thread's state."""
    previous = setter = None
    if os.name == "nt":
        import ctypes
        setter = ctypes.windll.kernel32.SetThreadExecutionState
        setter.argtypes = [ctypes.c_uint]
        setter.restype = ctypes.c_uint
        previous = setter(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
        if not previous:
            print("Windows did not grant a temporary sleep-inhibition request.", flush=True)
    try:
        yield
    finally:
        if previous and setter is not None:
            setter(previous)


@contextlib.contextmanager
def run_lock(run_dir):
    """An OS lock releases automatically on process exit, including a crash."""
    run_dir = Path(run_dir).resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.parent / f".{run_dir.name}.runner.lock"
    handle = lock_path.open("a+b")
    acquired = False
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as error:
            raise RuntimeError(f"Another training runner holds {run_dir}; do not launch the same run twice") from error
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def generate_report(arguments, path, log_path):
    """Publish completed JSON atomically; retain interrupted temporary reports."""
    path = Path(path)
    if path.exists():
        if path.is_symlink():
            raise ValueError(f"Refusing a report symlink: {path}")
        try:
            return read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if (path.parent / "manifest.json").exists():
                raise ValueError(f"A previously finalized report is damaged: {path}")
            preserved = path.with_name(path.name + f".incomplete-{time.time_ns()}")
            if preserved.resolve().parent != path.resolve().parent:
                raise ValueError("Incomplete report preservation escaped its directory")
            replace_ready_file(path, preserved)
            print(f"Preserved incomplete report: {preserved}", flush=True)
    with atomic_output(path, external_writer=True, preserve_incomplete=True) as temporary:
        args = list(arguments)
        args[args.index("--output") + 1] = str(temporary)
        invoke(args, log_path)
        report = read_json(temporary)
    return report


def review_initial_language(config, run_dir, device):
    language, common = config["language"], config["common"]
    destination = Path(run_dir) / "language_initial_review.json"
    report = generate_report(["arena.language_review", "--checkpoint", language["init_checkpoint"],
        "--data", language["val_data"], "--tokenizer", common["tokenizer"], "--device", device,
        "--max-batches", str(language["eval_batches"]), "--batch-size", str(common["batch_size"]),
        "--seed", str(common["seed"]), "--output", str(destination)],
        destination, Path(run_dir) / "logs" / "language_initial_review.log")
    if not report.get("review_finished") or report["checkpoint_sha256"] != sha256(language["init_checkpoint"]):
        raise ValueError("Initial language review does not match its checkpoint")
    if report["data_sha256"] != sha256(language["val_data"]) or report["tokenizer_sha256"] != sha256(common["tokenizer"]):
        raise ValueError("Initial language review input hashes differ")
    for key, value in {"batch_size": common["batch_size"], "max_batches": language["eval_batches"],
                       "seed": common["seed"], "language_window_policy": "full-span",
                       "validation_sampling": "fixed-shuffle"}.items():
        if report["settings"].get(key) != value:
            raise ValueError(f"Initial language review setting differs: {key}")
    return report


def training_arguments(config, branch, run_dir, stop_at, init_checkpoint, device):
    common = config["common"]
    stage = config["language"] if branch == "language" else config["dialogue"]
    data = stage if branch == "language" else stage[branch]
    output = Path(run_dir) / branch
    args = ["arena.training", "--data", data["data"], "--val-data", data["val_data"],
            "--tokenizer", common["tokenizer"], "--output-dir", str(output),
            "--stage", "language" if branch == "language" else "dialogue", "--device", device,
            "--steps", str(stage["steps"]), "--stop-at-step", str(stop_at), "--hours", "0",
            "--batch-size", str(common["batch_size"]), "--accumulation", str(common["accumulation"]),
            "--threads", str(common["threads"]), "--seed", str(common["seed"]),
            "--lr", str(stage["lr"]), "--warmup-steps", str(stage["warmup_steps"]),
            "--eval-every", str(stage["eval_every"]), "--save-every", str(stage["save_every"]),
            "--eval-batches", str(stage["eval_batches"]), "--validation-sampling", "fixed-shuffle",
            "--log-every", "20", "--persona-dropout", "0", "--no-persona-shuffle"]
    if branch == "language":
        args.extend(["--language-window-policy", "full-span"])
    else:
        args.extend(["--retrieval-index", stage["retrieval_index"], "--retrieval-cache", data["retrieval_cache"],
                     "--retrieval-dropout", str(stage["retrieval_dropout"])])
        if data.get("source_weights"):
            args.extend(["--source-weights", json.dumps(data["source_weights"], sort_keys=True)])
    last = output / "last.pt"
    args.extend(["--resume", str(last)] if last.exists() else ["--init-from", str(init_checkpoint)])
    return args


def review_signature(report):
    inputs = {key: value for key, value in report["inputs"].items() if key != "checkpoint_sha256"}
    return {"inputs": inputs, "selection": report["selection"],
            "generation_settings": report["generation_settings"],
            "generated_ids": [row["id"] for row in report["generated_samples"]]}


def review_checkpoint(config, run_dir, checkpoint, label, device, baseline_signature=None):
    review, common = config["review"], config["common"]
    digest = sha256(checkpoint)
    directory = Path(run_dir) / "reviews" / f"{label}-{digest[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    raw, conversation = directory / "raw.json", directory / "conversation.json"
    report = generate_report(["arena.generation_review", "--checkpoint", str(checkpoint),
                "--tokenizer", common["tokenizer"], "--train-data", review["train_data"],
                "--val-data", review["val_data"], "--index", review["index"], "--cache", review["cache"],
                "--device", device, "--seed", str(common["seed"]),
                "--max-per-source", str(review["max_per_source"]),
                "--generation-examples", str(review["generation_examples"]),
                "--batch-size", str(review["batch_size"]), "--output", str(raw)],
               raw, Path(run_dir) / "logs" / f"review-{label}.log")
    if not report.get("review_finished") or report["inputs"]["checkpoint_sha256"] != digest:
        raise ValueError(f"Incomplete or mismatched checkpoint review: {raw}")
    signature = review_signature(report)
    if baseline_signature is not None and signature != baseline_signature:
        raise ValueError(f"Review is not comparable with baseline: {raw}")
    generate_report(["arena.evaluation", "run", "--checkpoint", str(checkpoint),
                "--tokenizer", common["tokenizer"], "--index", review["index"],
                "--rounds", review["rounds"], "--persona", review["persona"],
                "--device", device, "--output", str(conversation)],
               conversation, Path(run_dir) / "logs" / f"conversation-{label}.log")
    manifest = directory / "manifest.json"
    expected = {"checkpoint_sha256": digest, "checkpoint": str(Path(checkpoint).resolve()),
                "raw_sha256": sha256(raw), "conversation_sha256": sha256(conversation),
                "note": "Raw generator and runtime are separate evaluations; no automatic quality verdict."}
    if manifest.exists():
        if read_json(manifest) != expected:
            raise ValueError(f"Existing review files changed: {directory}")
    else:
        write_json(manifest, expected)
    return signature


def snapshot_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    digest = sha256(source)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f"A different checkpoint already exists at {destination}")
        return destination
    with atomic_output(destination) as temporary:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != digest:
            raise IOError("Checkpoint snapshot copy checksum mismatch")
    return destination


def collect_results(run_dir):
    """Package only named reports/configs/logs, never model weights or corpora."""
    run_dir = Path(run_dir)
    allowed_top = {"settings.json", "status.json", "environment.json", "dialogue_initialization.json", "language_initial_review.json"}
    selected = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_dir)
        if len(relative.parts) == 1 and path.name in allowed_top:
            selected.append(path)
        elif relative.parts[0] in ("language", "control", "candidate") and len(relative.parts) == 2 and path.name in ("config.json", "metrics.jsonl"):
            selected.append(path)
        elif relative.parts[0] == "reviews" and len(relative.parts) == 3 and path.name in ("raw.json", "conversation.json", "manifest.json"):
            selected.append(path)
        elif relative.parts[0] == "logs" and len(relative.parts) == 2 and path.suffix == ".log":
            selected.append(path)
    archive = run_dir / "training_v2_results.zip"
    with atomic_output(archive) as temporary:
        manifest = {"version": 1, "files": {}, "weights_included": False,
                    "note": "Reports only. Checkpoints remain on the training computer. No candidate was automatically promoted."}
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for path in selected:
                name = path.relative_to(run_dir).as_posix()
                data = path.read_bytes()
                manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                zipped.writestr(name, data)
            zipped.writestr("RESULTS_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(temporary) as zipped:
            if zipped.testzip() is not None:
                raise IOError("Results archive CRC verification failed")
    return archive


def _run(config, run_dir, *, resume=False, device="cuda", engineering=False):
    hashes, _ = preflight(config, device=device, engineering=engineering)
    freeze_settings(run_dir, config, resume=resume, device=device, engineering=engineering, hashes=hashes)
    run_dir = Path(run_dir)
    status_path = run_dir / "status.json"
    initial_paths = (config["baseline_checkpoint"], config["language"]["init_checkpoint"])
    started = time.monotonic()
    try:
        if not (run_dir / "environment.json").exists():
            args = ["arena.doctor", "--output", str(run_dir / "environment.json")]
            if device == "cuda":
                args.append("--require-cuda")
            invoke(args, run_dir / "logs" / "environment.log")
        signature = review_checkpoint(config, run_dir, initial_paths[0], "baseline", device)
        review_initial_language(config, run_dir, device)
        language = config["language"]
        last = run_dir / "language" / "last.pt"
        for stop_at in segment_ends(language["steps"], language["segment_steps"]):
            state = checkpoint_info(last)["training_state"] if last.exists() else {"step": 0}
            if state["step"] < stop_at:
                write_json(status_path, {"state": "running", "stage": "language", "next_stop": stop_at, "engineering": engineering})
                invoke(training_arguments(config, "language", run_dir, stop_at, initial_paths[1], device), run_dir / "logs" / "language.log")
                state = checkpoint_info(last)["training_state"]
                if state["step"] != stop_at:
                    raise RuntimeError(f"Language training stopped at unexpected step {state['step']}; expected {stop_at}")
        language_best = run_dir / "language" / "best.pt"
        initialization = {"checkpoint": str(language_best.resolve()), "sha256": sha256(language_best),
                          "note": "Both dialogue branches initialize from these identical weights and reset optimizer state."}
        initialization_path = run_dir / "dialogue_initialization.json"
        if initialization_path.exists():
            if read_json(initialization_path) != initialization:
                raise ValueError("Language initialization changed after dialogue branches were started")
        else:
            write_json(initialization_path, initialization)
        dialogue = config["dialogue"]
        for branch, budget in (("control", dialogue["control_steps"]), ("candidate", dialogue["steps"])):
            last = run_dir / branch / "last.pt"
            for stop_at in segment_ends(budget, dialogue["segment_steps"]):
                snapshot = run_dir / branch / "snapshots" / f"step_{stop_at:06d}.pt"
                state = checkpoint_info(last)["training_state"] if last.exists() else {"step": 0}
                if state["step"] < stop_at:
                    if sha256(language_best) != initialization["sha256"]:
                        raise ValueError("Language initialization changed during the paired experiment")
                    write_json(status_path, {"state": "running", "stage": branch, "next_stop": stop_at, "engineering": engineering})
                    invoke(training_arguments(config, branch, run_dir, stop_at, language_best, device), run_dir / "logs" / f"{branch}.log")
                    state = checkpoint_info(last)["training_state"]
                    if state["step"] != stop_at:
                        raise RuntimeError(f"{branch} stopped at unexpected step {state['step']}; expected {stop_at}")
                if not snapshot.exists():
                    if state["step"] != stop_at:
                        raise ValueError(f"Missing historical snapshot at {branch} step {stop_at}; refusing to substitute a later model")
                    snapshot_checkpoint(last, snapshot)
                if checkpoint_info(snapshot)["training_state"]["step"] != stop_at:
                    raise ValueError(f"Wrong training step in snapshot: {snapshot}")
                review_checkpoint(config, run_dir, snapshot, f"{branch}-{stop_at:06d}", device, signature)
                collect_results(run_dir)
            review_checkpoint(config, run_dir, run_dir / branch / "best.pt", f"{branch}-best", device, signature)
        complete = {"state": "complete", "engineering": engineering,
                    "language_steps": language["steps"], "control_steps": dialogue["control_steps"],
                    "candidate_steps": dialogue["steps"], "promoted": False,
                    "note": "Training finished. Inspect matched raw answers and runtime conversations before choosing weights."}
        if not status_path.exists() or read_json(status_path) != complete:
            write_json(status_path, complete)
    except BaseException as error:
        try:
            write_json(status_path, {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                    "error": f"{type(error).__name__}: {error}", "engineering": engineering})
        except BaseException as reporting_error:
            error.add_note(f"Could not record failure status: {reporting_error}")
        try:
            collect_results(run_dir)
        except BaseException as reporting_error:
            error.add_note(f"Could not collect failure reports: {reporting_error}")
        raise
    archive = collect_results(run_dir)
    print(f"Finished in {time.monotonic() - started:.1f}s. Results: {archive}. Original main weights were not replaced.", flush=True)
    return archive


def run(config, run_dir, *, resume=False, device="cuda", engineering=False):
    with run_lock(run_dir), keep_awake():
        return _run(config, run_dir, resume=resume, device=device, engineering=engineering)


def main(argv=None):
    from .console import configure_utf8
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_v2.json")
    parser.add_argument("--run-name", default="training_v2")
    parser.add_argument("--resume", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--collect-only", action="store_true")
    action.add_argument("--check-only", action="store_true", help="Validate inputs/cache/checkpoint stages/CUDA without creating a training run")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--engineering", action="store_true", help="Explicitly label a CPU/tiny-budget code check")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--language-checkpoint")
    for name in ("language-steps", "language-segment-steps", "dialogue-steps", "segment-steps", "control-steps", "batch-size", "accumulation", "review-per-source", "generation-examples"):
        parser.add_argument("--" + name, type=int, help="Engineering-only override; saved and checked on resume")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("run-name must contain only letters, digits, _ or -")
    run_dir = Path("runs") / args.run_name
    if args.collect_only:
        if not (run_dir / "settings.json").is_file():
            parser.error("Run has no settings.json to collect")
        print(collect_results(run_dir))
        return
    config = copy.deepcopy(read_json(args.config))
    if args.baseline_checkpoint:
        config["baseline_checkpoint"] = args.baseline_checkpoint
    if args.language_checkpoint:
        config["language"]["init_checkpoint"] = args.language_checkpoint
    overrides = {"language_steps": ("language", "steps"), "language_segment_steps": ("language", "segment_steps"),
                 "dialogue_steps": ("dialogue", "steps"), "segment_steps": ("dialogue", "segment_steps"),
                 "control_steps": ("dialogue", "control_steps"), "batch_size": ("common", "batch_size"),
                 "accumulation": ("common", "accumulation"), "review_per_source": ("review", "max_per_source"),
                 "generation_examples": ("review", "generation_examples")}
    for name, (section, key) in overrides.items():
        value = getattr(args, name)
        if value is not None:
            if not args.engineering:
                parser.error("Tiny-budget overrides require --engineering")
            config[section][key] = value
    if args.engineering:
        for stage in ("language", "dialogue"):
            config[stage].update(eval_every=1, save_every=1, eval_batches=1, warmup_steps=1)
        config["review"]["batch_size"] = config["common"]["batch_size"]
    if args.check_only:
        return preflight(config, device=args.device, engineering=args.engineering)[1]
    return run(config, run_dir, resume=args.resume, device=args.device, engineering=args.engineering)


if __name__ == "__main__":
    main()
