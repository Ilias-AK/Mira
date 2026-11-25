"""Run linked dev rounds through Mira and write a readable transcript.

python -m mira.eval_dev --rounds night_v4/configs/night_v4_dev_rounds.json --out runs/mira/dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from mira.chat import HERE


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", required=True)
    p.add_argument("--out", default="runs/mira/dev")
    p.add_argument("--index", default="artifacts/retrieval")
    p.add_argument("--persona", default=str(HERE / "config" / "persona.json"))
    p.add_argument("--bank", default=str(HERE / "config" / "bank.json"))
    p.add_argument("--fallbacks", default=str(HERE / "config" / "fallbacks.json"))
    p.add_argument("--retrieval", action="store_true")
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--chat", action="store_true", help="messenger style instead of contest style")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from mira.runtime import MiraRuntime
    rounds = json.loads(Path(args.rounds).read_text(encoding="utf-8-sig"))["rounds"]
    runtime = MiraRuntime(args.persona, args.bank, args.fallbacks, args.index if args.retrieval else None,
                          use_retrieval=args.retrieval, seed=args.seed, mode="chat" if args.chat else "hall")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    routes, lines, records, slow = Counter(), ["# Мира — dev-раунды", ""], [], 0.0
    try:
        for item in rounds:
            runtime.reset()
            lines += [f"## {item['id']}", ""]
            for number, question in enumerate(item["turns"], 1):
                result = runtime.answer(question)
                route = result.route.split(":")[0]
                routes[route] += 1
                slow = max(slow, result.seconds)
                lines += [f"**Собеседник:** {question}  ", f"**Мира:** {result.text}  ", f"<sub>{result.route} · {result.seconds:.2f} с</sub>", ""]
                records.append({"round": item["id"], "turn": number, "question": question, **result.__dict__})
    finally:
        runtime.close()
    total = sum(routes.values())
    misses = [r["question"] for r in records if r["route"].startswith("fallback")]
    summary = {"turns": total, "routes": dict(routes), "max_seconds": round(slow, 2), "fallback_questions": misses}
    lines[1:1] = ["Маршруты: " + ", ".join(f"{k} {v}" for k, v in routes.most_common()) + f" · максимум {slow:.2f} с", ""]
    (out / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
    with (out / "turns.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print("Транскрипт:", out / "transcript.md")


if __name__ == "__main__":
    main()
