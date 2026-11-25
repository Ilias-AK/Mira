"""Local chat. Questions on stdin, answers on stdout, diagnostics on stderr."""

import argparse
import json
import sys
from pathlib import Path

from arena.runtime import ArenaRuntime, IsolatedGenerator, load_persona


def main():
    from arena.console import configure_utf8
    configure_utf8()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default="artifacts/retrieval")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--checkpoint")
    p.add_argument("--persona", default="configs/persona.json")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--question")
    p.add_argument("--json", action="store_true", help="Return machine-readable diagnostics as well")
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repeat-guard", action="store_true", help="Reject replies repeated across turns; may select an off-topic retrieval fallback")
    p.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Block repeated token n-grams during generation (0 disables)")
    args = p.parse_args()
    persona = load_persona(args.persona)
    generator = None
    if args.checkpoint:
        if not Path(args.checkpoint).is_file():
            p.error(f"Checkpoint not found: {args.checkpoint}")
        generator = IsolatedGenerator(args.checkpoint, args.tokenizer, args.device)
        if generator.error:
            print(f"Generator unavailable: {generator.error}. Retrieval reserve remains active.", file=sys.stderr)
    runtime = ArenaRuntime(args.index, persona=persona.get("facts", []), generator=generator,
        max_characters=persona.get("max_characters", 280),
        deadline_seconds=persona.get("deadline_seconds", 30),
        max_source_tokens=persona.get("max_source_tokens", 512),
        max_new_tokens=persona.get("max_new_tokens", 96),
        use_retrieval=not args.no_retrieval, use_memory=not args.no_memory,
        temperature=args.temperature, reject_repeated_responses=args.repeat_guard,
        no_repeat_ngram_size=args.no_repeat_ngram_size)

    def reply(text):
        result = runtime.answer(text)
        if args.json:
            print(json.dumps(result.__dict__, ensure_ascii=False), flush=True)
        else:
            print(result.text, flush=True)

    try:
        if args.question:
            reply(args.question)
        else:
            if sys.stdin.isatty():
                print("Готово. /reset — новый раунд, /quit — выход.", file=sys.stderr)
            for line in sys.stdin:
                text = line.strip()
                if text == "/quit":
                    break
                if text == "/reset":
                    runtime.reset()
                    continue
                if text:
                    try:
                        reply(text)
                    except ValueError as error:
                        print(str(error), file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

