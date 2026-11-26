"""Оценка профиля Миры (рабочего или кандидата) на одинаковых наборах; результат — JSON с метриками.

python -m mira.eval_candidate --name baseline  --config-dir config              --intent runs/mira_intent/intent.pt
python -m mira.eval_candidate --name candidate --config-dir config/candidate_v2 --intent runs/mira_intent_v2/intent.pt
python -m mira.eval_candidate --compare runs/eval_v2/baseline.json runs/eval_v2/candidate.json

Наборы:
  intent  — data/delivery_v2/intent/val.jsonl (перефразы validation из архива; synthetic);
  traps   — data/delivery_v2/eval/traps_validation.jsonl (ловушки validation; synthetic);
  dialog  — data/delivery_v2/eval/dialogues_validation.jsonl (диалоги validation, проигрываются целиком);
  regress — mira/eval_sets/regression.json (вопросы, на которых Мира ошибалась раньше; ожидаемые шаблоны);
  memory  — mira/eval_sets/memory.json (сценарий «рассказал о себе — спросил»);
  human   — data/team_raw/test_questions_*.txt (вопросы, написанные людьми; без эталонов: только доля заглушек,
            длина и канон). Настоящего независимого blind-теста в репозитории нет.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from mira.canon_check import violations  # noqa: E402

TOPIC_ROUTES = ("bank:", "examples:", "followup:", "model:", "repeat_topic:")


def norm(text):
    return re.sub(r"[^\w]+", " ", str(text).lower().replace("ё", "е")).strip()


def grams(text):
    t = f"  {norm(text)}  "
    return collections.Counter(t[i:i + 3] for i in range(len(t) - 2))


def cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / ((sum(v * v for v in a.values()) ** .5) * (sum(v * v for v in b.values()) ** .5))


def similarity(text, golds):
    g = grams(text)
    return max((cosine(g, grams(x)) for x in golds), default=0.0)


def topic_of(route):
    for prefix in TOPIC_ROUTES:
        if route.startswith(prefix):
            return route[len(prefix):].split("+")[0]
    return None


def cyrillic_share(text):
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    return 1.0 if not letters else sum(1 for c in letters if re.match(r"[А-Яа-яЁё]", c)) / len(letters)


def macro_f1(pairs):
    labels = {g for g, _ in pairs}
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


class Probe:
    """Обёртка над рантаймом: каждый ответ проверяется на общие критические условия."""

    def __init__(self, args):
        from mira.runtime import MiraRuntime
        base = Path(args.config_dir)
        intent = None if args.no_intent else args.intent
        self.rt = MiraRuntime(str(base / "persona.json"), str(base / "bank.json"), str(base / "fallbacks.json"), None,
                              use_retrieval=False, mode="hall", seed=args.seed, typo_rate=0.0, intent_path=intent)
        self.times, self.lengths, self.canon, self.non_russian, self.over_limit, self.total = [], [], [], [], 0, 0
        self.intent_status = self.rt.intent_status

    def reset(self):
        self.rt.reset()

    def ask(self, question, where):
        started = time.perf_counter()
        answer = self.rt.answer(question)
        took = time.perf_counter() - started
        self.total += 1
        self.times.append(took)
        self.lengths.append(len(answer.text))
        if len(answer.text) > 300:
            self.over_limit += 1
        bad = violations(answer.text)
        if bad:
            self.canon.append({"set": where, "q": question, "a": answer.text, "rule": bad[0][0]})
        if cyrillic_share(answer.text) < 0.7:
            self.non_russian.append({"set": where, "q": question, "a": answer.text})
        return answer

    def summary(self):
        times = sorted(self.times)
        return {"answers": self.total, "over_300_chars": self.over_limit, "canon_violations": len(self.canon),
                "canon_examples": self.canon[:10], "non_russian": len(self.non_russian), "non_russian_examples": self.non_russian[:5],
                "len_mean": round(statistics.mean(self.lengths), 1) if self.lengths else 0,
                "len_max": max(self.lengths, default=0),
                "time_ms_mean": round(1000 * statistics.mean(times), 1) if times else 0,
                "time_ms_p95": round(1000 * times[int(.95 * (len(times) - 1))], 1) if times else 0,
                "time_ms_max": round(1000 * times[-1], 1) if times else 0,
                "over_45s": sum(1 for t in times if t > 45)}


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def eval_intent(probe, path, team_answers):
    rows = load_jsonl(path)
    pairs, fallback, answer_match, errors = [], 0, 0, []
    for r in rows:
        probe.reset()
        a = probe.ask(r["text"], "intent")
        predicted = topic_of(a.route)
        pairs.append((r["label"], predicted))
        fallback += a.route.startswith("fallback")
        ok_answer = similarity(a.text, team_answers.get(r["label"], [])) >= 0.6
        answer_match += ok_answer
        if predicted != r["label"] and len(errors) < 25:
            errors.append({"q": r["text"], "gold": r["label"], "route": a.route, "a": a.text, "answer_ok": ok_answer})
    n = len(rows)
    return {"n": n, "accuracy": round(sum(g == p for g, p in pairs) / n, 4), "macro_f1": round(macro_f1(pairs), 4),
            "answer_match": round(answer_match / n, 4), "fallback_rate": round(fallback / n, 4), "errors": errors}


def eval_traps(probe, path):
    rows = load_jsonl(path)
    by_cat = collections.defaultdict(lambda: {"n": 0, "fallback": 0, "sim": []})
    sims, fallback, worst = [], 0, []
    for r in rows:
        probe.reset()
        a = probe.ask(r["question"], "traps")
        s = similarity(a.text, r["answers"])
        sims.append(s)
        c = by_cat[r["category"]]
        c["n"] += 1
        c["sim"].append(s)
        if a.route.startswith("fallback"):
            fallback += 1
            c["fallback"] += 1
        worst.append((s, r["category"], r["question"], a.text, a.route))
    worst.sort()
    return {"n": len(rows), "fallback_rate": round(fallback / len(rows), 4), "gold_similarity_mean": round(statistics.mean(sims), 4),
            "by_category": {k: {"n": v["n"], "fallback_rate": round(v["fallback"] / v["n"], 3),
                                "gold_similarity": round(statistics.mean(v["sim"]), 3)} for k, v in sorted(by_cat.items())},
            "lowest_similarity": [{"sim": round(s, 2), "cat": c, "q": q, "a": a, "route": rt} for s, c, q, a, rt in worst[:15]]}


def eval_dialogues(probe, path):
    rows = load_jsonl(path)
    sims, fallback, turns, repeats, examples = [], 0, 0, 0, []
    for d in rows:
        probe.reset()
        said = []
        pending_user = None
        for t in d["turns"]:
            if t["role"] == "user":
                pending_user = t["text"]
                continue
            if pending_user is None:
                continue
            a = probe.ask(pending_user, "dialog")
            turns += 1
            s = similarity(a.text, [t["text"]])
            sims.append(s)
            fallback += a.route.startswith("fallback")
            if norm(a.text) in said:
                repeats += 1
            said.append(norm(a.text))
            if s < 0.2 and len(examples) < 12:
                examples.append({"dialog": d["dialogue_id"], "q": pending_user, "gold": t["text"], "a": a.text, "route": a.route})
            pending_user = None
    return {"dialogues": len(rows), "turns": turns, "gold_similarity_mean": round(statistics.mean(sims), 4),
            "fallback_rate": round(fallback / turns, 4), "repeated_replies": repeats, "low_similarity_examples": examples}


def eval_regression(probe, path):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    failed = []
    for case in cases:
        probe.reset()
        a = None
        for q in case["dialog"]:
            a = probe.ask(q, "regression")
        ok = re.search(case["expect"], a.text, re.IGNORECASE) is not None
        bad = case.get("not") and re.search(case["not"], a.text, re.IGNORECASE)
        if not ok or bad:
            failed.append({"dialog": case["dialog"], "expect": case["expect"], "a": a.text, "route": a.route})
    return {"n": len(cases), "passed": len(cases) - len(failed), "failed": failed}


def eval_human(probe, pattern):
    files = sorted(f for f in glob.glob(pattern) if not f.endswith("template.txt"))
    questions = [l.strip() for f in files for l in Path(f).read_text(encoding="utf-8-sig").splitlines()
                 if l.strip() and not l.startswith("#")]
    fallback, examples = 0, []
    for q in questions:
        probe.reset()
        a = probe.ask(q, "human")
        if a.route.startswith("fallback"):
            fallback += 1
            if len(examples) < 15:
                examples.append({"q": q, "a": a.text})
    return {"files": files, "n": len(questions), "fallback_rate": round(fallback / max(1, len(questions)), 4),
            "fallback_examples": examples}


def run(args):
    probe = Probe(args)
    team = json.loads((Path(args.config_dir) / "team_qa.json").read_text(encoding="utf-8-sig"))
    team_answers = {i["id"]: i["answers"] for i in team["intents"]}
    data = Path(args.data)
    result = {"name": args.name, "config_dir": args.config_dir, "intent": None if args.no_intent else args.intent,
              "intent_status": probe.intent_status, "seed": args.seed}
    result["intent_val"] = eval_intent(probe, data / "intent" / "val.jsonl", team_answers)
    result["traps_val"] = eval_traps(probe, data / "eval" / "traps_validation.jsonl")
    result["dialogues_val"] = eval_dialogues(probe, data / "eval" / "dialogues_validation.jsonl")
    result["regression"] = eval_regression(probe, HERE / "mira" / "eval_sets" / "regression.json")
    result["memory"] = eval_regression(probe, HERE / "mira" / "eval_sets" / "memory.json")
    result["human_questions"] = eval_human(probe, str(HERE / "data" / "team_raw" / "test_questions_*.txt"))
    result["critical"] = probe.summary()
    out = Path(args.out_dir) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Сохранено: {out}")
    return result


ROWS = [
    ("Перефразы validation: точность темы", ("intent_val", "accuracy"), "up"),
    ("Перефразы validation: macro-F1", ("intent_val", "macro_f1"), "up"),
    ("Перефразы validation: ответ по теме", ("intent_val", "answer_match"), "up"),
    ("Перефразы validation: заглушки", ("intent_val", "fallback_rate"), "down"),
    ("Ловушки validation: похожесть на эталон", ("traps_val", "gold_similarity_mean"), "up"),
    ("Ловушки validation: заглушки", ("traps_val", "fallback_rate"), "down"),
    ("Диалоги validation: похожесть на эталон", ("dialogues_val", "gold_similarity_mean"), "up"),
    ("Диалоги validation: заглушки", ("dialogues_val", "fallback_rate"), "down"),
    ("Диалоги validation: повторы реплик", ("dialogues_val", "repeated_replies"), "down"),
    ("Регрессия: пройдено", ("regression", "passed"), "up"),
    ("Память: пройдено", ("memory", "passed"), "up"),
    ("Вопросы людей: заглушки", ("human_questions", "fallback_rate"), "down"),
    ("Нарушения канона (все наборы)", ("critical", "canon_violations"), "down"),
    ("Ответы > 300 символов", ("critical", "over_300_chars"), "down"),
    ("Ответы не на русском", ("critical", "non_russian"), "down"),
    ("Длина ответа, среднее", ("critical", "len_mean"), None),
    ("Время ответа p95, мс", ("critical", "time_ms_p95"), "down"),
    ("Время ответа max, мс", ("critical", "time_ms_max"), "down"),
]
CRITICAL = [("critical", "canon_violations"), ("critical", "over_300_chars"), ("critical", "non_russian"),
            ("critical", "over_45s"), ("regression", "passed"), ("memory", "passed")]


def compare(base_path, cand_path):
    a = json.loads(Path(base_path).read_text(encoding="utf-8"))
    b = json.loads(Path(cand_path).read_text(encoding="utf-8"))
    get = lambda d, k: d[k[0]][k[1]]
    lines = [f"| Метрика | {a['name']} | {b['name']} | |", "|---|---|---|---|"]
    for title, key, better in ROWS:
        x, y = get(a, key), get(b, key)
        mark = ""
        if better and x != y:
            mark = "лучше" if (y > x) == (better == "up") else "хуже"
        lines.append(f"| {title} | {x} | {y} | {mark} |")
    worse_critical = []
    for key in CRITICAL:
        x, y = get(a, key), get(b, key)
        if (key[1] == "passed" and y < x) or (key[1] != "passed" and y > x):
            worse_critical.append(f"{key[0]}.{key[1]}: {x} -> {y}")
    gains = sum(1 for _, key, better in ROWS[:12] if better and get(a, key) != get(b, key)
                and (get(b, key) > get(a, key)) == (better == "up"))
    losses = sum(1 for _, key, better in ROWS[:12] if better and get(a, key) != get(b, key)
                 and (get(b, key) > get(a, key)) != (better == "up"))
    if worse_critical:
        verdict = "НЕ выпускать: кандидат хуже по критическим проверкам: " + "; ".join(worse_critical)
    elif gains > losses:
        verdict = (f"Кандидат лучше по {gains} метрикам качества, хуже по {losses}; критические проверки не хуже. "
                   "Validation архива synthetic — перед выпуском нужен человеческий blind-тест.")
    else:
        verdict = f"Оставить рабочую версию: улучшений {gains}, ухудшений {losses}."
    table = "\n".join(lines)
    print(table)
    print("\nВердикт:", verdict)
    return table, verdict


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="baseline")
    p.add_argument("--config-dir", default="config")
    p.add_argument("--intent", default="runs/mira_intent/intent.pt")
    p.add_argument("--no-intent", action="store_true")
    p.add_argument("--data", default="data/delivery_v2")
    p.add_argument("--out-dir", default="runs/eval_v2")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "CANDIDATE_JSON"))
    args = p.parse_args(argv)
    if args.compare:
        table, verdict = compare(*args.compare)
        out = Path(args.compare[1]).with_name("comparison.md")
        out.write_text(table + "\n\nВердикт: " + verdict + "\n", encoding="utf-8")
        print(f"Сохранено: {out}")
        return
    run(args)


if __name__ == "__main__":
    main()
