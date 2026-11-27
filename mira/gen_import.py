"""Сборка сгенерированных тем (data/team_raw/hall_qa_gen_*.txt) в config/team_qa_gen.json.

    python -m mira.gen_import            # собрать и показать, что отфильтровано
    python -m mira.gen_import --report runs/gen_import.md

Сгенерированные данные лежат отдельно от данных команды: свой файл исходников (gen в имени), свой
собранный файл (team_qa_gen.json), пометка source_kind="generated" в каждой теме. Удалить их —
удалить config/team_qa_gen.json.

Что выбрасывается (исходники не меняются):
- формулировки, совпадающие с вопросами независимых проверок (зонд, вопросы людей, закрытый тест);
- формулировки, которые уже есть у тем команды или у другой сгенерированной темы;
- ответы длиннее 300 символов, со смайликами, преимущественно латиницей, с казахскими буквами,
  с нарушением канона, «как бот», с матом;
- темы и уточнения, у которых не осталось вопросов или ответов.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from mira.canon_check import violations  # noqa: E402
from mira.check_gen import BOTTY, EMOJI, KAZAKH, SWEAR, norm  # noqa: E402
from mira.runtime import latin_dominant  # noqa: E402
from mira.team_qa import _core, parse  # noqa: E402


def held_out_questions():
    """Вопросы проверок: ими нельзя учить, иначе замер перестаёт быть честным."""
    out = set()
    probe = HERE / "data/hall_probe/probe_questions.jsonl"
    if probe.is_file():
        out |= {norm(json.loads(x)["question"]) for x in probe.read_text(encoding="utf-8").splitlines() if x.strip()}
    for path in glob.glob(str(HERE / "data/team_raw/test_questions_*.txt")) + glob.glob(str(HERE / "data/blind_gen/*.txt")):
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(norm(line))
    return out


def team_questions():
    out = set()
    for name in ("bank.json", "team_qa.json"):
        p = HERE / "config" / name
        if p.is_file():
            for item in json.loads(p.read_text(encoding="utf-8-sig")).get("intents", []):
                out |= {norm(x) for x in item.get("examples", [])}
    return out


def bad_answer(text):
    if len(text) > 300:
        return "длина"
    if EMOJI.search(text):
        return "смайлик"
    if latin_dominant(text):
        return "латиница"
    if KAZAKH.search(text):
        return "казахские буквы"
    if BOTTY.search(text):
        return "как бот"
    if SWEAR.search(text):
        return "мат"
    if violations(text):
        return "канон"
    return None


def load_fixes():
    """data/team_raw/factcheck_gen.json: {file: {старый ответ: новый ответ или None (удалить)}}."""
    path = HERE / "data/team_raw/factcheck_gen.json"
    fixes = {}
    if path.is_file():
        for fix in json.loads(path.read_text(encoding="utf-8")).get("fixes", []):
            fixes.setdefault(fix["file"], {})[fix["before"]] = fix.get("after")
    return fixes


def build(paths):
    held, taken = held_out_questions(), team_questions()
    all_fixes = load_fixes()
    dropped = Counter()
    examples_dropped = []
    intents = []

    def clean_questions(questions, follow=False):
        kept = []
        for q in questions:
            key = norm(q)
            if key in held:
                dropped["вопрос из проверки"] += 1
                examples_dropped.append(("проверка", q))
            elif follow:
                # «А почему?» — общий вопрос-уточнение у многих тем; его различает предыдущая тема (after)
                if key not in {norm(k) for k in kept}:
                    kept.append(q)
            elif key in taken:
                dropped["вопрос уже есть"] += 1
            else:
                taken.add(key)
                kept.append(q)
        return kept

    def clean_answers(answers):
        kept = []
        for a in answers:
            if a in fixes:
                dropped["исправлено по проверке фактов" if fixes[a] else "удалено по проверке фактов"] += 1
                a = fixes[a]
                if not a:
                    continue
            why = bad_answer(a)
            if why:
                dropped["ответ: " + why] += 1
                examples_dropped.append((why, a))
            else:
                kept.append(a)
        return kept

    for path in paths:
        # ответ-заготовка, повторённый в трёх и более темах файла («Сохранила бы вежливый тон…»), — не ответ
        # на конкретную ситуацию: убираем его, а тему, где остались только заготовки, не берём совсем
        counts = Counter(a for b in parse(path) for a in set(b["answers"]))
        stock = {a for a, n in counts.items() if n >= 3}
        stem = re.sub(r"\W+", "_", Path(path).stem.replace("hall_qa_", "")).strip("_")
        fixes = all_fixes.get(Path(path).name, {})
        previous = None
        for number, block in enumerate(parse(path), 1):
            ident = f"qa_{stem}_{number:03d}"
            specific = [a for a in block["answers"] if a not in stock]
            if len(specific) < len(block["answers"]):
                dropped["ответ-заготовка (повтор в 3+ темах)"] += len(block["answers"]) - len(specific)
            questions, answers = clean_questions(block["questions"]), clean_answers(specific)
            if not questions or not answers:
                dropped["тема целиком"] += 1
                continue
            cores = {_core(q) for q in questions}
            item = {"id": ident, "topic": ident, "examples": questions, "answers": answers, "patterns": [],
                    "priority": 1, "source": "team_qa", "source_kind": "generated", "file": Path(path).name}
            if len(cores) == 1 and len(questions) > 1 and previous is not None:
                item["after"] = [previous]
            intents.append(item)
            parent = ident
            for k, follow in enumerate(block.get("follows", []), 1):
                fq, fa = clean_questions(follow["questions"], follow=True), clean_answers(follow["answers"])
                if not fq or not fa:
                    dropped["уточнение целиком"] += 1
                    continue
                child = f"{ident}_f{k}"
                intents.append({"id": child, "topic": child, "examples": fq, "answers": fa, "patterns": [],
                                "priority": 1, "source": "team_qa", "source_kind": "generated",
                                "file": Path(path).name, "after": [parent]})
                parent = child
            previous = ident
    return intents, dropped, examples_dropped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pattern", default=str(HERE / "data/team_raw/hall_qa_gen_*.txt"))
    p.add_argument("--out", default=str(HERE / "config/team_qa_gen.json"))
    p.add_argument("--report", default=None)
    args = p.parse_args(argv)
    paths = sorted(glob.glob(args.pattern))
    intents, dropped, examples = build(paths)
    Path(args.out).write_text(json.dumps({
        "_note": "Сгенерированные темы (разрешение организаторов, 26.11.2026). Собирается из data/team_raw/hall_qa_gen_*.txt "
                 "командой python -m mira.gen_import. Чтобы убрать — удалите этот файл.",
        "intents": intents}, ensure_ascii=False, indent=1), encoding="utf-8")
    per_file = Counter(i["file"] for i in intents if not re.search(r"_f\d+$", i["id"]))
    lines = [f"Файлов: {len(paths)}, тем: {len(intents)} (уточнений: {sum('after' in i for i in intents)}) -> {args.out}"]
    lines += [f"  {f}: {n}" for f, n in sorted(per_file.items())]
    lines += ["Отфильтровано:"] + [f"  {k}: {v}" for k, v in dropped.most_common()]
    lines += ["Примеры выброшенного:"] + [f"  [{why}] {text}" for why, text in examples[:40]]
    text = "\n".join(lines)
    print(text)
    if args.report:
        Path(args.report).write_text("```\n" + text + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
