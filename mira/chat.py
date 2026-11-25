"""Chat with Mira. Questions on stdin, answers on stdout. /reset — new round, /quit — exit."""
from __future__ import annotations

import argparse
import random
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def build(args):
    from mira.runtime import MiraRuntime
    if args.config_dir:   # a whole runtime profile, e.g. config/candidate_v2
        base = Path(args.config_dir)
        args.persona, args.bank, args.fallbacks = str(base / "persona.json"), str(base / "bank.json"), str(base / "fallbacks.json")
    return MiraRuntime(args.persona, args.bank, args.fallbacks, args.index if args.retrieval else None,
                       use_retrieval=args.retrieval, mode="chat" if args.chat else "hall",
                       seed=args.seed if args.seed is not None else random.randrange(10 ** 6),
                       typo_rate=0.0 if args.no_typos else args.typo_rate,
                       intent_path=None if args.no_intent else args.intent)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default="artifacts/retrieval")
    p.add_argument("--persona", default=str(HERE / "config" / "persona.json"))
    p.add_argument("--bank", default=str(HERE / "config" / "bank.json"))
    p.add_argument("--fallbacks", default=str(HERE / "config" / "fallbacks.json"))
    p.add_argument("--retrieval", action="store_true",
                   help="also search the dialogue index (off by default: its matches are often off-topic)")
    p.add_argument("--no-retrieval", action="store_true", help="kept for old launch commands; retrieval is off anyway")
    p.add_argument("--json", action="store_true", help="print route and diagnostics too")
    p.add_argument("--typing-delay", action="store_true",
                   help="wait like a person typing: 6 s + 5 characters per second, at most 35 s (limit is 45 s)")
    p.add_argument("--show-time", action="store_true", help="print how long each answer took (for the demo video)")
    p.add_argument("--seed", type=int, default=None, help="fix the random choices (default: new every launch)")
    p.add_argument("--typo-rate", type=float, default=0.04, help="share of answers with one keyboard typo (default 0.04)")
    p.add_argument("--no-typos", action="store_true", help="never make typos")
    p.add_argument("--intent", default="runs/mira_intent/intent.pt", help="own neural intent model (optional)")
    p.add_argument("--no-intent", action="store_true", help="rules only")
    p.add_argument("--chat", action="store_true", help="casual messenger style instead of the contest (hall) style")
    p.add_argument("--no-log-misses", action="store_true", help="do not write runs/mira/misses.jsonl")
    p.add_argument("--config-dir", help="profile folder with persona/bank/team_qa/fallbacks json (default: config/)")
    p.add_argument("--question")
    return p


def main():
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args()
    runtime = build(args)

    def reply(text):
        started = time.monotonic()
        result = runtime.answer(text)
        if result.route.startswith("fallback") and not args.no_log_misses:
            try:  # questions Mira had no good answer for: material for the team's bank
                path = Path("runs/mira/misses.jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"question": text, "answer": result.text, "route": result.route}, ensure_ascii=False) + "\n")
            except OSError:
                pass
        if args.typing_delay:
            time.sleep(max(0.0, min(35.0, 6.0 + len(result.text) / 5.0) - result.seconds))
        if args.json:
            print(json.dumps(result.__dict__, ensure_ascii=False), flush=True)
        else:
            print(result.text, flush=True)
            if args.show_time:
                waited = time.monotonic() - started
                print(f"  [ответ сформирован за {result.seconds:.2f} с"
                      + (f", выдан через {waited:.1f} с" if args.typing_delay else "") + f", {len(result.text)} символов]",
                      file=sys.stderr, flush=True)

    try:
        if args.question:
            reply(args.question)
            return
        if sys.stdin.isatty():
            print(f"Мира готова. Нейросеть тем: {runtime.intent_status}. /reset — новый раунд, /quit — выход.", file=sys.stderr)
        for line in sys.stdin:
            text = line.strip()
            if text == "/quit":
                break
            if text == "/reset":
                runtime.reset()
                print("(новый раунд)", file=sys.stderr)
                continue
            if text:
                reply(text)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
