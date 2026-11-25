"""Import team-written Q&A blocks (data/team_raw/hall_qa_*.txt) as bank topics.

Format (see hall_qa_template.txt):
    ###
    Вопрос: one way to ask
    Вопрос: another way to ask
    Мира: an answer in Mira's voice
A block whose questions are one question with "Слушай, " / "Кстати, " in front is a
follow-up: it only answers right after the block before it.

These topics have example questions instead of regex patterns. They are matched by
similarity to the examples (mira.bank.Bank.nearest) and by the team's own neural
intent model, which is trained on the same examples (mira/train_intent.py).

python -m mira.team_qa            -> writes mira/config/team_qa.json
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PREFIX = re.compile(r"^(?:слушай|кстати|а|и|ну)\s*[,:]?\s*", re.IGNORECASE)


def _core(question: str) -> str:
    return PREFIX.sub("", question.strip()).lower().strip(" ?!.")


def parse(path):
    blocks, current = [], None
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("###"):
            current = {"questions": [], "answers": []}
            blocks.append(current)
            continue
        if current is None or not line or line.startswith("#"):
            continue
        low = line.lower()
        text = line.split(":", 1)[1].strip() if ":" in line else ""
        if low.startswith("вопрос:"):
            if text:
                current["questions"].append(text)
        elif low.startswith("дальше:"):
            # a follow-up asked right after the block above ("какой клуб?" -> "а игрок?")
            follows = current.setdefault("follows", [])
            if not follows or follows[-1]["answers"]:
                follows.append({"questions": [], "answers": []})
            if text:
                follows[-1]["questions"].append(text)
        elif low.startswith("мира:"):
            if text:
                target = current["follows"][-1] if current.get("follows") else current
                target["answers"].append(text)
    return [b for b in blocks if b["questions"] and b["answers"]]


def to_intents(paths):
    intents = []
    for path in paths:
        stem = re.sub(r"\W+", "_", Path(path).stem.replace("hall_qa_", "")).strip("_") or "team"
        previous = None
        for number, block in enumerate(parse(path), 1):
            ident = f"qa_{stem}_{number:03d}"
            cores = {_core(q) for q in block["questions"]}
            follow_up = len(cores) == 1 and len(block["questions"]) > 1 and previous is not None
            item = {"id": ident, "topic": ident, "examples": block["questions"], "answers": block["answers"],
                    "patterns": [], "priority": 1, "source": "team_qa", "file": Path(path).name}
            if follow_up:
                item["after"] = [previous]
            intents.append(item)
            parent = ident
            for k, follow in enumerate(block.get("follows", []), 1):
                if follow["questions"] and follow["answers"]:
                    child = f"{ident}_f{k}"
                    intents.append({"id": child, "topic": child, "examples": follow["questions"],
                                    "answers": follow["answers"], "patterns": [], "priority": 1,
                                    "source": "team_qa", "file": Path(path).name, "after": [parent]})
                    parent = child
            previous = ident
    return intents


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    pattern = sys.argv[1] if len(sys.argv) > 1 else "data/team_raw/hall_qa_*.txt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "config" / "team_qa.json"
    paths = sorted(p for p in glob.glob(pattern) if not p.endswith("template.txt"))
    intents = to_intents(paths)
    out.write_text(json.dumps({"_note": "Темы из файлов команды hall_qa_*.txt (создаётся автоматически, правьте исходные файлы).",
                               "intents": intents}, ensure_ascii=False, indent=1), encoding="utf-8")
    follow = sum(1 for i in intents if "after" in i)
    print(f"Файлов: {len(paths)}, тем: {len(intents)} (из них уточняющих: {follow}) -> {out}")


if __name__ == "__main__":
    main()
