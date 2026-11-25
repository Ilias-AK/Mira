"""Ask Mira single questions from test_questions_*.txt (one question per line) and report.

Each question is a fresh round, like a new audience question. Writes runs/mira/questions/report.md.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

from mira.chat import HERE


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    pattern = sys.argv[1] if len(sys.argv) > 1 else "data/team_raw/test_questions_*.txt"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/mira/questions")
    files = sorted(p for p in glob.glob(pattern) if not p.endswith("template.txt"))
    questions = []
    for path in files:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)
    from mira.runtime import MiraRuntime
    runtime = MiraRuntime(HERE / "config" / "persona.json", HERE / "config" / "bank.json", HERE / "config" / "fallbacks.json",
                          use_retrieval=False)
    routes, lines, misses = Counter(), [], []
    model_seen, model_conf, model_errors = 0, 0, Counter()
    for q in questions:
        runtime.reset()
        a = runtime.answer(q)
        route = a.route.split(":")[0].split("+")[0]
        routes[route] += 1
        if route == "fallback":
            misses.append(q)
        guess = a.diagnostics.get("intent_model")
        if guess:
            model_seen += 1
            if guess[0] == "error":
                model_errors[guess[1]] += 1
            elif guess[1] >= 0.6:
                model_conf += 1
        lines.append(f"- **{q}**  \n  {a.text}  <sub>{a.route}{' · модель: ' + str(guess) if guess else ''}</sub>")
    out.mkdir(parents=True, exist_ok=True)
    total = len(questions) or 1
    head = [f"# Проверочные вопросы: {len(questions)} из {len(files)} файлов",
            f"Заглушек: {routes['fallback']} ({100 * routes['fallback'] / total:.0f}%). Нейросеть тем: {runtime.intent_status}",
            "Маршруты: " + ", ".join(f"{k} {v}" for k, v in routes.most_common()),
            f"Нейросеть спрашивали: {model_seen} раз, уверена (>=0.6): {model_conf}"
            + (f", ошибки: {dict(model_errors)}" if model_errors else ""), ""]
    (out / "report.md").write_text("\n".join(head + lines), encoding="utf-8")
    (out / "misses.txt").write_text("\n".join(misses), encoding="utf-8")
    print("\n".join(head[:4]))
    print("Отчёт:", out / "report.md", "| вопросы без ответа:", out / "misses.txt")


if __name__ == "__main__":
    main()
