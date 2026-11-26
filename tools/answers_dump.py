"""Ответы Миры на вопросы людей и на зонд; запуск из корня репо (старого или нового кода): python tools/answers_dump.py out.json. Вопросы зонда написаны агентом (agent_probe): только для проверки, не для обучения"""
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
REPO = os.getcwd().replace("\\", "/")
from mira.runtime import MiraRuntime

rt = MiraRuntime("config/persona.json", "config/bank.json", "config/fallbacks.json", None,
                 use_retrieval=False, mode="hall", seed=1, typo_rate=0.0,
                 intent_path=REPO + "/runs/mira_intent/intent.pt", tokenizer_path=REPO + "/artifacts/tokenizer.json")
out = []
for path in sorted(glob.glob(REPO + "/data/team_raw/test_questions_*.txt")):
    if "template" in path:
        continue
    for line in open(path, encoding="utf-8-sig"):
        q = line.strip()
        if not q or q.startswith("#"):
            continue
        rt.reset()
        a = rt.answer(q)
        out.append({"set": os.path.basename(path), "q": q, "a": a.text, "route": a.route})
probe = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answers_dump_rounds.json"), encoding="utf-8"))
for name, questions in probe.items():
    rt.reset()
    for q in questions:
        a = rt.answer(q)
        out.append({"set": "probe:" + name, "q": q, "a": a.text, "route": a.route})
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(rt.intent_status, len(out))
