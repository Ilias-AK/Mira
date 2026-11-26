"""Проверка сгенерированных файлов для Миры перед импортом.

    python -m mira.check_gen data/team_raw/hall_qa_gen_*.txt
    python -m mira.check_gen data/team_raw/paraphrases_gen_*.txt
    python -m mira.check_gen config/gen/fallbacks_gen.json

Ошибки (ERROR) нужно исправить, предупреждения (WARN) — посмотреть. Код выхода 1, если есть ошибки.
Проверяет формат, длину, язык, смайлики, канон, дубли внутри файлов и с банком, утечку вопросов
независимого зонда (data/hall_probe/probe_questions.jsonl) в обучающие данные.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mira.canon_check import violations  # noqa: E402
from mira.team_qa import parse  # noqa: E402

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]|\)\)|\(\(|:\)|:\(|\bахах\w*|\bхаха\w*|\bлол\b|\bкек\b")
KAZAKH = re.compile(r"[әіңғүұқөһӘІҢҒҮҰҚӨҺ]")
BOTTY = re.compile(r"(?i)отличный\s+вопрос|безусловно|важно\s+отметить|надеюсь,?\s+(?:это\s+)?помо|как\s+(?:ии|искусственный|языковая\s+модель)"
                   r"|я\s+(?:бот|модель|нейросеть|программа)\b|chatgpt|claude|openai|во-первых")
SWEAR = re.compile(r"(?i)\b(?:бля\w*|хуй\w*|пизд\w*|еба\w*|ебу\w*|сука|нахуй)\b")


def norm(text: str) -> str:
    return re.sub(r"[^а-яёa-z0-9 ]", "", text.lower().replace("ё", "е")).strip()


def latin_share(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def bank_texts():
    questions, answers = set(), set()
    for path in ("config/bank.json", "config/team_qa.json"):
        p = ROOT / path
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        for item in data.get("intents", []):
            answers |= {norm(a) for a in item.get("answers", [])}
            questions |= {norm(e) for e in item.get("examples", [])}
    return questions, answers


def probe_questions():
    p = ROOT / "data/hall_probe/probe_questions.jsonl"
    if not p.is_file():
        return set()
    return {norm(json.loads(line)["question"]) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def check_answer(text, where, report):
    if len(text) > 300:
        report("ERROR", where, f"ответ длиннее 300 символов ({len(text)})")
    elif len(text) > 200:
        report("WARN", where, f"длинный ответ ({len(text)} символов)")
    if EMOJI.search(text):
        report("ERROR", where, "смайлик или «ахах»: в зале их нет")
    if KAZAKH.search(text):
        report("ERROR", where, "казахские буквы в ответе: фильтр языка их заблокирует")
    if latin_share(text.replace("{x}", "").replace("{X}", "")) > 0.3:
        report("ERROR", where, "много латиницы: ответы только на русском")
    if BOTTY.search(text):
        report("ERROR", where, f"звучит как бот: «{BOTTY.search(text).group(0)}»")
    if SWEAR.search(text):
        report("ERROR", where, "мат")
    for rule, fragment in violations(text):
        report("ERROR", where, f"противоречит канону ({rule}): «{fragment}»")


def check_qa(path, bank_q, bank_a, probe, seen_q, report):
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = parse(path)
    if not blocks:
        report("ERROR", path, "нет ни одного блока ### с Вопрос:/Мира:")
        return 0
    if raw.count("###") != len(blocks):
        report("WARN", path, f"блоков ### {raw.count('###')}, а полных (с вопросом и ответом) {len(blocks)}")
    for n, block in enumerate(blocks, 1):
        where = f"{Path(path).name} блок {n} «{block['questions'][0][:40]}»"
        qs = block["questions"]
        if len(qs) < 5:
            report("WARN", where, f"мало формулировок вопроса ({len(qs)}, нужно 6–10)")
        if len({norm(q) for q in qs}) < len(qs):
            report("WARN", where, "одинаковые формулировки вопроса")
        if not 1 <= len(block["answers"]) <= 4:
            report("WARN", where, f"ответов {len(block['answers'])}, нужно 2–3")
        firsts = Counter(norm(a).split(" ")[0] for a in block["answers"] if norm(a))
        if len(block["answers"]) >= 3 and firsts.most_common(1)[0][1] == len(block["answers"]):
            report("WARN", where, "все ответы начинаются одинаково")
        for q in qs:
            key = norm(q)
            if key in probe:
                report("ERROR", where, f"вопрос из независимого зонда, его нельзя учить: «{q}»")
            if key in seen_q:
                report("WARN", where, f"вопрос уже есть в другом блоке/файле: «{q}»")
            seen_q[key] = where
            if key in bank_q:
                report("WARN", where, f"такой вопрос уже есть в банке: «{q}»")
        for follow in block.get("follows", []):
            if not follow["answers"]:
                report("ERROR", where, "строки Дальше: без ответа Мира:")
            for a in follow["answers"]:
                check_answer(a, where + " (Дальше)", report)
        for a in block["answers"]:
            if norm(a) in bank_a:
                report("WARN", where, f"ответ слово в слово уже есть в банке: «{a[:50]}»")
            check_answer(a, where, report)
    return len(blocks)


def check_paraphrases(path, probe, report):
    known = set()
    for p in ("config/bank.json", "config/team_qa.json"):
        f = ROOT / p
        if f.is_file():
            known |= {i["id"] for i in json.loads(f.read_text(encoding="utf-8-sig")).get("intents", [])}
    topic, count, rows = None, Counter(), 0
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("###"):
            topic = line[3:].strip()
            if topic not in known:
                report("ERROR", path, f"неизвестная тема «{topic}» (строку ### менять нельзя)")
            continue
        if not line or line.startswith("#") or topic is None:
            continue
        rows += 1
        count[topic] += 1
        if norm(line) in probe:
            report("ERROR", path, f"вопрос из независимого зонда: «{line}»")
        if len(line) > 200:
            report("WARN", path, f"очень длинный вопрос: «{line[:50]}…»")
    thin = [t for t, n in count.items() if n < 5]
    if thin:
        report("WARN", path, f"меньше 5 перефраз у {len(thin)} тем: {', '.join(thin[:8])}")
    return rows


def check_fallbacks(path, report):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    total = 0
    for kind, options in data.items():
        if kind.startswith("_"):
            continue
        if not isinstance(options, list):
            report("ERROR", path, f"{kind}: нужен список строк")
            continue
        if len({norm(o) for o in options}) < len(options):
            report("WARN", path, f"{kind}: есть повторы")
        for o in options:
            total += 1
            check_answer(o, f"{Path(path).name} [{kind}]", report)
            if kind in ("tell", "make", "define", "fragment") and "{x}" not in o and "{X}" not in o:
                report("WARN", path, f"{kind}: нет подстановки {{x}}/{{X}}: «{o[:50]}»")
    return total


def main(argv=None):
    paths = []
    for pattern in (argv if argv is not None else sys.argv[1:]) or ["data/team_raw/hall_qa_gen_*.txt"]:
        paths += sorted(glob.glob(pattern)) or [pattern]
    errors, warnings = [], []

    def report(level, where, message):
        (errors if level == "ERROR" else warnings).append(f"{level} {where}: {message}")

    bank_q, bank_a = bank_texts()
    probe = probe_questions()
    seen_q = {}
    totals = Counter()
    for path in paths:
        if not Path(path).is_file():
            report("ERROR", path, "файл не найден")
            continue
        name = Path(path).name
        if name.endswith(".json"):
            totals["заглушек"] += check_fallbacks(path, report)
        elif name.startswith("paraphrases"):
            totals["перефраз"] += check_paraphrases(path, probe, report)
        else:
            totals["блоков"] += check_qa(path, bank_q, bank_a, probe, seen_q, report)
    for line in warnings[:200] + errors:
        print(line)
    if len(warnings) > 200:
        print(f"... и ещё {len(warnings) - 200} предупреждений")
    print(f"\nФайлов: {len(paths)}; " + ", ".join(f"{k}: {v}" for k, v in totals.items())
          + f"; ошибок: {len(errors)}, предупреждений: {len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
