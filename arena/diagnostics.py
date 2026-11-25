"""Compare a trained checkpoint under controlled runtime ablations, offline."""

import argparse
from pathlib import Path

from arena.evaluation import read_json, run_rounds, write_json
from arena.runtime import ArenaRuntime, IsolatedGenerator, load_persona


def compare_runtime(generator, index, persona, rounds):
    variants = [
        ("baseline", {}),
        ("without_persona", {"persona": []}),
        ("without_retrieval", {"use_retrieval": False}),
        ("without_persona_or_retrieval", {"persona": [], "use_retrieval": False}),
        ("retrieval_only", {"generator": None}),
        ("with_repeat_guard", {"reject_repeated_responses": True}),
    ]
    reports = {}
    for name, overrides in variants:
        options = dict(persona=persona.get("facts", []), generator=generator,
                       max_characters=persona.get("max_characters", 280),
                       deadline_seconds=persona.get("deadline_seconds", 30),
                       max_source_tokens=persona.get("max_source_tokens", 512),
                       max_new_tokens=persona.get("max_new_tokens", 96),
                       temperature=0.0, reject_repeated_responses=False)
        options.update(overrides)
        runtime = ArenaRuntime(index, **options)
        try:
            print(f"Checking {name}...", flush=True)
            reports[name] = run_rounds(runtime, rounds)
        finally:
            # The caller owns the shared model process; histories are per runtime.
            runtime.generator = None
            runtime.close()
    return reports


def main():
    from arena.console import configure_utf8
    from arena.training import file_sha256
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--index", default="artifacts/retrieval")
    parser.add_argument("--persona", default="configs/persona.json")
    parser.add_argument("--rounds", default="configs/technical_rounds.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="artifacts/generator_diagnostics.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new filename to preserve the comparison")
    generator = IsolatedGenerator(args.checkpoint, args.tokenizer, args.device)
    try:
        if generator.error:
            raise RuntimeError(f"Generator failed to load: {generator.error}")
        reports = compare_runtime(generator, args.index, load_persona(args.persona), read_json(args.rounds))
        report = {"checkpoint": str(Path(args.checkpoint).resolve()),
                  "checkpoint_sha256": file_sha256(args.checkpoint),
                  "tokenizer_sha256": file_sha256(args.tokenizer),
                  "variants": reports,
                  "note": "Greedy decoding. Repetition guard disabled except in with_repeat_guard. Inspect actual answers; no automatic quality pass or causal diagnosis."}
        write_json(args.output, report)
        print(f"Saved {args.output}", flush=True)
    finally:
        generator.close()


if __name__ == "__main__":
    main()
