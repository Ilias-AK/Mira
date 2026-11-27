"""Прогон закрытого теста (data/blind_gen/rounds.txt) с нейросетью: ответы и маршруты в файл.

    python -m mira.blind_run                      # -> runs/blind_run.json и сводка
    python -m mira.blind_run --out runs/blind_after.json

Каждый раунд — заново (rt.reset()), режим зала, без опечаток. Этим файлом не учат.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main(argv=None):
    from mira.runtime import MiraRuntime
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rounds", default=str(HERE / "data/blind_gen/rounds.txt"))
    p.add_argument("--intent", default=str(HERE / "runs/mira_intent/intent.pt"))
    p.add_argument("--out", default=str(HERE / "runs/blind_run.json"))
    args = p.parse_args(argv)
    intent = args.intent if args.intent != "none" and Path(args.intent).is_file() else None
    rt = MiraRuntime(str(HERE / "config/persona.json"), str(HERE / "config/bank.json"), str(HERE / "config/fallbacks.json"),
                     None, use_retrieval=False, mode="hall", seed=1, typo_rate=0.0, intent_path=intent)
    rounds, current = [], []
    for line in Path(args.rounds).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("#"):
            if current:
                rounds.append(current)
            current = []
        elif line:
            current.append(line)
    if current:
        rounds.append(current)
    rows = []
    for n, questions in enumerate(rounds, 1):
        rt.reset()
        for q in questions:
            a = rt.answer(q)
            rows.append({"round": n, "q": q, "a": a.text, "route": a.route, "seconds": round(a.seconds, 3)})
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    routes = Counter(r["route"].split(":")[0].split("+")[0] for r in rows)
    fb = sum(r["route"].startswith("fallback") for r in rows)
    print(f"нейросеть: {'да' if intent else 'нет'}; раундов {len(rounds)}, вопросов {len(rows)}; заглушек {fb} ({fb / len(rows):.1%})")
    print("маршруты:", ", ".join(f"{k} {v}" for k, v in routes.most_common(10)))
    print("ответы ->", args.out)


if __name__ == "__main__":
    main()
