"""Build the single reviewed update after the new corpus/cache are complete."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena.bundle import sha_file, verify_bundle
from arena.cache_validation import validate_cache


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserve the existing package: {output}")
    cache_report = validate_cache(ROOT / "data/curated/dialogue_v2/train.jsonl", ROOT / "data/processed/val.jsonl",
                                  ROOT / "artifacts/retrieval", ROOT / "artifacts/training_v2/retrieval_cache.sqlite3")
    audit = ROOT / "artifacts/next_training_research"
    (audit / "final_cache_validation.json").write_text(json.dumps(cache_report, ensure_ascii=False, indent=2), encoding="utf-8")
    files = {ROOT / name for name in ("README.md", "TRAINING_V2.txt", "requirements.txt", "pyproject.toml")}
    for directory in ("arena", "tests"):
        files.update((ROOT / directory).glob("*.py"))
    for name in ("train_v2_lenovo.ps1", "chat.ps1", "install_training_v2.py", "build_training_v2_bundle.py"):
        files.add(ROOT / "scripts" / name)
    for name in ("training_v2.json", "conversation_review_rounds.json", "conversation_v2_rounds.json", "persona.json"):
        files.add(ROOT / "configs" / name)
    for directory in (ROOT / "data/language_v2", ROOT / "data/curated/dialogue_v2"):
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "validation.json",
                     "oasst2_val.jsonl", "oasst2_test.jsonl", "curation_audit.json", "manual_exclusions.json"):
            path = directory / name
            if path.is_file():
                files.add(path)
    for directory in (ROOT / "data/curated/dialogue_v2_stage1", ROOT / "data/curated/dialogue_v2_broad", ROOT / "data/expansion/20261121"):
        path = directory / "manifest.json"
        if path.is_file():
            files.add(path)
    for path in (ROOT / "data/raw").rglob("*"):
        if "candidate_research" in path.parts:
            continue
        if path.is_file() and path.stat().st_size < 2_000_000 and any(word in path.name.lower() for word in ("readme", "license")):
            files.add(path)
    files.add(ROOT / "artifacts/persona_pilot_audit/RESULT.txt")
    for name in ("training_depth.txt", "training_depth.json", "generator_design.txt", "generator_data_profile.json",
                 "language_expansion_validation.json", "language_target_coverage_simulation.json",
                 "corpus_sources.txt", "corpus_selection.txt", "corpus_validation.json", "corpus_precision_audit.json",
                 "corpus_manual_exclusions.json", "additional_sources.txt", "conversation_holdout_check.json",
                 "final_cache_validation.json", "runner_smoke.json", "runner_final_smoke.json", "tests.xml"):
        path = audit / name
        if path.is_file():
            files.add(path)
    files.update(audit.glob("corpus_stage*_source.py"))
    files.add(ROOT / "artifacts/training_v2/retrieval_cache.sqlite3")
    base = []
    for name in ("data/processed/train.jsonl", "data/processed/val.jsonl", "data/processed/test.jsonl",
                 "artifacts/tokenizer.json", "artifacts/retrieval/index.sqlite3", "artifacts/retrieval/reranker.json",
                 "artifacts/retrieval_cache.sqlite3"):
        path = ROOT / name
        base.append({"path": name, "bytes": path.stat().st_size, "sha256": sha_file(path)})
    manifest = {"format_version": 1, "package_kind": "training_v2_update", "includes_data": True,
                "includes_retrieval_cache": True, "weights_included": False, "required_base_inputs": base,
                "original_checkpoint_reference": {"path": "runs/main/dialogue/best.pt", "sha256": "932be8b4d5c6a3a9b8ce3c820a8f8b21d54dae4a58540a57c5adce7794eeba86",
                    "note": "Reference from received reports; checkpoint stages, tokenizer and actual hashes are checked and recorded by the training runner."},
                "note": "Install over the existing project using scripts/install_training_v2.py; original weights/data are preserved.",
                "files": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for number, path in enumerate(sorted(files), 1):
            path = path.resolve()
            name = path.relative_to(ROOT).as_posix()
            if not path.is_file() or name.startswith("runs/") or path.suffix in (".pt", ".tmp"):
                raise ValueError(f"Invalid transfer file: {path}")
            item = {"path": name, "bytes": path.stat().st_size, "sha256": sha_file(path)}
            z.write(path, name)
            manifest["files"].append(item)
            if item["bytes"] > 20_000_000:
                print(f"Packed {name}: {item['bytes']:,} bytes", flush=True)
        z.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    verification = verify_bundle(temporary)
    os.replace(temporary, output)
    digest = sha_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "  " + output.name + "\n", encoding="ascii")
    report = {"archive": str(output), "bytes": output.stat().st_size, "sha256": digest, **verification}
    (audit / "package_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=ROOT / "dist/training-v2-kit.zip")
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
