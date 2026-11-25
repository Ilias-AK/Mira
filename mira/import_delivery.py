"""Импорт пакета данных команды (mira_delivery_clean_v2*.zip) в формат проекта.

Воспроизводимо: читает ZIP только на чтение (исходный архив не меняется), пишет результат в
--out (по умолчанию data/delivery_v2) и кандидатный runtime-профиль в --config-out
(по умолчанию config/candidate_v2). Рабочий профиль config/ не трогается.

Что куда идёт (по договорённости с командой):
  runtime/bank.clean.json, team_qa.clean.json, fallbacks*.clean.json -> config/candidate_v2/
  train/paraphrases_train.jsonl            -> intent/train.jsonl (обучение классификатора тем)
  train/context_followups.jsonl            -> intent/train.jsonl + примеры уточнений в team_qa кандидата
  train/new_topics_retrieval_clean.jsonl   -> только поиск: элементы team_qa с source=retrieval_v2
  train/new_topics_intent_candidates.jsonl -> НЕ используется, только отчёт (intent_candidates_review.jsonl)
  train/dialogues_{chat,hall}_train.jsonl  -> generator/train.jsonl (одна реплика Миры = один пример)
  train/traps_train.jsonl                  -> generator/train.jsonl (source mira_team_v2_traps)
  validation/*                             -> только оценка (intent/val.jsonl, generator/val*.jsonl, eval/*)
  excluded/, reports/, canon_extensions.json, generator_ready/ -> не используются

python -m mira.import_delivery --zip path/to/mira_delivery_clean_v2.1.zip
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from mira.canon_check import CANON, violations  # noqa: E402

MAX_CONTEXT = 8
FORBIDDEN_PREFIXES = ("excluded/", "reports/", "generator_ready/")
FORBIDDEN_FILES = ("canon_extensions.json",)
USED = {
    "runtime/bank.clean.json", "runtime/team_qa.clean.json", "runtime/fallbacks.clean.json",
    "runtime/fallbacks_hall.clean.json", "train/paraphrases_train.jsonl", "train/context_followups.jsonl",
    "train/new_topics_retrieval_clean.jsonl", "train/new_topics_intent_candidates.jsonl",
    "train/dialogues_chat_train.jsonl", "train/dialogues_hall_train.jsonl", "train/traps_train.jsonl",
    "validation/paraphrases_validation.jsonl", "validation/dialogues_chat_validation.jsonl",
    "validation/dialogues_hall_validation.jsonl", "validation/traps_validation.jsonl",
}


def norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", str(text).lower().replace("ё", "е")).strip()


def grams(text: str):
    t = f"  {norm(text)}  "
    return collections.Counter(t[i:i + 3] for i in range(len(t) - 2))


def cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / ((sum(v * v for v in a.values()) ** .5) * (sum(v * v for v in b.values()) ** .5))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Archive:
    def __init__(self, path: Path):
        self.path = path
        self.zip = zipfile.ZipFile(path)
        names = [n for n in self.zip.namelist() if not n.endswith("/")]
        roots = {n.split("/", 1)[0] for n in names}
        self.root = roots.pop() + "/" if len(roots) == 1 else ""
        self.read_log = {}

    def raw(self, rel: str) -> bytes:
        if rel.startswith(FORBIDDEN_PREFIXES) or rel in FORBIDDEN_FILES:
            raise PermissionError(f"{rel}: этот файл нельзя использовать как данные")
        data = self.zip.read(self.root + rel)
        self.read_log[rel] = {"sha256": sha256_bytes(data), "bytes": len(data)}
        return data

    def json(self, rel: str):
        return json.loads(self.raw(rel).decode("utf-8-sig"))

    def jsonl(self, rel: str):
        rows = []
        for number, line in enumerate(io.StringIO(self.raw(rel).decode("utf-8-sig")), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{rel}:{number}: битая строка JSON ({error})")
        return rows


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def dialogue_examples(dialogues, source: str, split: str, stats: collections.Counter):
    """Одна реплика Миры = один пример; контекст — до восьми предыдущих реплик; общий group_id."""
    out = []
    for dialogue in dialogues:
        group = f"mira_v2:{dialogue['dialogue_id']}"
        turns = [{"role": "user" if t["role"] == "user" else "assistant", "text": str(t["text"]).strip()}
                 for t in dialogue["turns"] if str(t.get("text", "")).strip()]
        for index, turn in enumerate(turns):
            if turn["role"] != "assistant" or index == 0:
                continue
            context = turns[max(0, index - MAX_CONTEXT):index]
            if context[-1]["role"] != "user":
                stats["skipped_no_user_before"] += 1
                continue
            bad = violations(turn["text"])
            if bad:
                stats["skipped_canon"] += 1
                continue
            if len(turn["text"]) > 300:
                stats["skipped_too_long"] += 1
                continue
            out.append({"id": f"{group}:{index}", "group_id": group, "source": source, "kind": "dialogue",
                        "context": context, "response": turn["text"], "persona": CANON, "split": split})
    return out


def trap_examples(traps, split: str, stats: collections.Counter):
    out = []
    for trap in traps:
        group = f"mira_v2_trap:{trap['id']}"
        for n, answer in enumerate(trap["answers"]):
            if violations(answer) or len(answer) > 300:
                stats["skipped_trap_answer"] += 1
                continue
            out.append({"id": f"{group}:{n}", "group_id": group, "source": "mira_team_v2_traps", "kind": "dialogue",
                        "context": [{"role": "user", "text": trap["question"]}], "response": answer,
                        "persona": CANON, "split": split})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True, help="путь к mira_delivery_clean_v2*.zip (только чтение)")
    p.add_argument("--out", default=str(HERE / "data" / "delivery_v2"))
    p.add_argument("--config-out", default=str(HERE / "config" / "candidate_v2"))
    p.add_argument("--persona", default=str(HERE / "config" / "persona.json"))
    p.add_argument("--keep-dir", default=str(HERE / "config" / "candidate_v2_keep"),
                   help="второй кандидат: правила и новые темы пакета, но тексты ответов существующих тем — из рабочего профиля")
    p.add_argument("--near-dup", type=float, default=0.92,
                   help="вопрос из train с похожестью >= порога к вопросу validation той же темы удаляется из train")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    zip_path = Path(args.zip)
    zip_sha = sha256_bytes(zip_path.read_bytes())
    archive = Archive(zip_path)
    out, cfg = Path(args.out), Path(args.config_out)
    stats = collections.Counter()
    report = {"zip": str(zip_path.name), "zip_sha256": zip_sha, "notes": []}

    # --- 1. проверка контрольных сумм архива --------------------------------------------
    sums = {}
    try:
        for line in archive.zip.read(archive.root + "provenance/checksums.sha256").decode().splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                sums[name.strip().lstrip("*").removeprefix("./")] = digest
    except KeyError:
        report["notes"].append("в архиве нет provenance/checksums.sha256")
    for rel in sorted(USED):
        data = archive.raw(rel)
        if rel in sums and sums[rel] != sha256_bytes(data):
            raise SystemExit(f"{rel}: SHA-256 не совпадает с provenance/checksums.sha256")
    report["checksums_verified"] = sum(1 for rel in USED if rel in sums)

    # --- 2. runtime-профиль кандидата ---------------------------------------------------
    bank = archive.json("runtime/bank.clean.json")
    team_qa = archive.json("runtime/team_qa.clean.json")
    fallbacks = archive.json("runtime/fallbacks.clean.json")
    fallbacks_hall = archive.json("runtime/fallbacks_hall.clean.json")
    current_hall = json.loads((HERE / "config" / "fallbacks_hall.json").read_text(encoding="utf-8-sig"))
    current_fb = json.loads((HERE / "config" / "fallbacks.json").read_text(encoding="utf-8-sig"))
    for name, new, cur in (("fallbacks", fallbacks, current_fb), ("fallbacks_hall", fallbacks_hall, current_hall)):
        missing = [k for k in cur if not k.startswith("_") and k not in new]
        for k in missing:   # runtime ждёт эти типы; без них упадёт на отсутствующем ключе
            new[k] = cur[k]
        if missing:
            report["notes"].append(f"{name}: в пакете нет типов {missing}, взяты из рабочего профиля")
    canon_bad = [(i["id"], a) for i in bank["intents"] + team_qa["intents"] for a in i["answers"] if violations(a)]
    report["runtime_canon_violations"] = canon_bad[:20]
    stats["runtime_canon_violations"] = len(canon_bad)

    # уточнения из context_followups: дополнительные формулировки к темам team_qa
    followups = archive.jsonl("train/context_followups.jsonl")
    by_id = {i["id"]: i for i in team_qa["intents"]}
    added_follow = 0
    for row in followups:
        item = by_id.get(row["topic_id"])
        if item is None:
            stats["followup_unknown_topic"] += 1
            continue
        if row["question"] not in item.setdefault("examples", []):
            item["examples"].append(row["question"])
            added_follow += 1
        prev = row["context"]["previous_topic_id"]
        if item.get("after") and prev in by_id and prev not in item["after"]:
            item["after"].append(prev)   # только для тем, которые и так контекстные
    stats["followup_examples_added"] = added_follow

    # новые темы — только поиск (по примерам), не классы нейросети
    retrieval = archive.jsonl("train/new_topics_retrieval_clean.jsonl")
    kept_retrieval = 0
    for row in retrieval:
        answers = [a for a in row["answers"] if not violations(a) and len(a) <= 300]
        if not answers or not row["questions"]:
            stats["retrieval_skipped"] += 1
            continue
        team_qa["intents"].append({"id": f"rv2_{row['topic_id']}", "topic": f"rv2_{row['topic_id']}",
                                   "examples": row["questions"], "answers": answers, "patterns": [],
                                   "priority": 0, "source": "retrieval_v2", "category": row["category"]})
        kept_retrieval += 1
    stats["retrieval_topics"] = kept_retrieval
    team_qa["_note"] = (team_qa.get("_note", "") + " Кандидат v2: + формулировки уточнений из context_followups, "
                        "+ темы new_topics (source=retrieval_v2) только для поиска по примерам.").strip()

    cfg.mkdir(parents=True, exist_ok=True)
    write_json(cfg / "bank.json", bank)
    write_json(cfg / "team_qa.json", team_qa)
    write_json(cfg / "fallbacks.json", fallbacks)
    write_json(cfg / "fallbacks_hall.json", fallbacks_hall)
    persona = json.loads(Path(args.persona).read_text(encoding="utf-8-sig"))
    write_json(cfg / "persona.json", persona)

    # второй кандидат: ответы, уже утверждённые командой в рабочем профиле, не переписываются пакетом
    keep = Path(args.keep_dir)
    current_bank = json.loads((HERE / "config" / "bank.json").read_text(encoding="utf-8-sig"))
    current_answers = {i["id"]: i["answers"] for i in current_bank["intents"]}
    kept_bank = json.loads(json.dumps(bank))
    restored = 0
    for item in kept_bank["intents"]:
        if item["id"] in current_answers and item["answers"] != current_answers[item["id"]]:
            item["answers"] = current_answers[item["id"]]
            restored += 1
    stats["keep_profile_answers_restored"] = restored
    keep.mkdir(parents=True, exist_ok=True)
    write_json(keep / "bank.json", kept_bank)
    for name in ("team_qa.json", "fallbacks.json", "fallbacks_hall.json", "persona.json"):
        (keep / name).write_bytes((cfg / name).read_bytes())

    # кандидаты в новые классы — только на ревью
    candidates = archive.jsonl("train/new_topics_intent_candidates.jsonl")
    write_jsonl(out / "intent_candidates_review.jsonl", candidates)

    # --- 3. intent: train/validation без утечек ----------------------------------------
    para_train = archive.jsonl("train/paraphrases_train.jsonl")
    para_val = archive.jsonl("validation/paraphrases_validation.jsonl")
    val_rows = [{"text": r["question"], "label": r["topic_id"], "source": "paraphrases_validation"} for r in para_val]
    val_norm = {norm(r["text"]) for r in val_rows}
    val_by_label = collections.defaultdict(list)
    for r in val_rows:
        val_by_label[r["label"]].append(grams(r["text"]))
    train_rows, dropped_exact, dropped_near = [], 0, 0
    source_rows = [{"text": r["question"], "label": r["topic_id"], "source": "paraphrases_train"} for r in para_train]
    source_rows += [{"text": r["question"], "label": r["topic_id"], "source": "context_followups"} for r in followups]
    seen = set()
    for r in source_rows:
        key = (norm(r["text"]), r["label"])
        if key in seen:
            continue
        seen.add(key)
        if key[0] in val_norm:
            dropped_exact += 1
            continue
        g = grams(r["text"])
        if any(cosine(g, v) >= args.near_dup for v in val_by_label.get(r["label"], [])):
            dropped_near += 1
            continue
        train_rows.append(r)
    # примеры team_qa кандидата тоже идут в обучение — убираем из них совпадения с validation
    leaked_examples = 0
    for item in team_qa["intents"]:
        if item.get("source") == "team_qa":
            before = len(item.get("examples", []))
            item["examples"] = [e for e in item.get("examples", []) if norm(e) not in val_norm]
            leaked_examples += before - len(item["examples"])
    if leaked_examples:
        write_json(cfg / "team_qa.json", team_qa)
        write_json(keep / "team_qa.json", team_qa)
    stats.update({"intent_train": len(train_rows), "intent_val": len(val_rows),
                  "intent_train_dropped_exact_val": dropped_exact, "intent_train_dropped_near_val": dropped_near,
                  "team_qa_examples_removed_as_val": leaked_examples})
    write_jsonl(out / "intent" / "train.jsonl", train_rows)
    write_jsonl(out / "intent" / "val.jsonl", val_rows)

    # --- 4. генератор: реплики Миры с контекстом ----------------------------------------
    gen_train = (dialogue_examples(archive.jsonl("train/dialogues_chat_train.jsonl"), "mira_team_v2_chat", "train", stats)
                 + dialogue_examples(archive.jsonl("train/dialogues_hall_train.jsonl"), "mira_team_v2_hall", "train", stats)
                 + trap_examples(archive.jsonl("train/traps_train.jsonl"), "train", stats))
    gen_val = (dialogue_examples(archive.jsonl("validation/dialogues_chat_validation.jsonl"), "mira_team_v2_chat", "validation", stats)
               + dialogue_examples(archive.jsonl("validation/dialogues_hall_validation.jsonl"), "mira_team_v2_hall", "validation", stats)
               + trap_examples(archive.jsonl("validation/traps_validation.jsonl"), "validation", stats))
    overlap = {r["group_id"] for r in gen_train} & {r["group_id"] for r in gen_val}
    if overlap:
        raise SystemExit(f"утечка: {len(overlap)} group_id одновременно в train и validation")
    # validation «без повторов»: ни пара (вопрос, ответ), ни сам ответ Миры не встречались в train
    train_pairs = {(norm(r["context"][-1]["text"]), norm(r["response"])) for r in gen_train}
    train_responses = {norm(r["response"]) for r in gen_train}
    unseen_pair = [r for r in gen_val if (norm(r["context"][-1]["text"]), norm(r["response"])) not in train_pairs]
    unseen_resp = [r for r in gen_val if norm(r["response"]) not in train_responses]
    stats.update({"generator_train": len(gen_train), "generator_val": len(gen_val),
                  "generator_val_pair_also_in_train": len(gen_val) - len(unseen_pair),
                  "generator_val_unseen_pair": len(unseen_pair), "generator_val_unseen_response": len(unseen_resp),
                  "generator_groups_train": len({r["group_id"] for r in gen_train}),
                  "generator_groups_val": len({r["group_id"] for r in gen_val})})
    write_jsonl(out / "generator" / "train.jsonl", gen_train)
    # dev для выбора чекпойнта — 5 % групп из train (детерминированно по group_id); validation архива не трогаем
    is_dev = lambda g: int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 100 < 5
    fit = [r for r in gen_train if not is_dev(r["group_id"])]
    dev = [dict(r, split="validation") for r in gen_train if is_dev(r["group_id"])]
    write_jsonl(out / "generator" / "train_fit.jsonl", fit)
    write_jsonl(out / "generator" / "dev_from_train.jsonl", dev)
    stats.update({"generator_train_fit": len(fit), "generator_dev_from_train": len(dev)})
    write_jsonl(out / "generator" / "val.jsonl", gen_val)
    write_jsonl(out / "generator" / "val_unseen_pair.jsonl", unseen_pair)
    write_jsonl(out / "generator" / "val_unseen_response.jsonl", unseen_resp)
    by_source = collections.Counter((r["split"], r["source"]) for r in gen_train + gen_val)
    report["generator_by_source"] = {f"{k[0]}/{k[1]}": v for k, v in sorted(by_source.items())}

    # --- 5. наборы для оценки runtime ---------------------------------------------------
    write_jsonl(out / "eval" / "traps_validation.jsonl", archive.jsonl("validation/traps_validation.jsonl"))
    dialogues_val = [dict(d, source="chat") for d in archive.jsonl("validation/dialogues_chat_validation.jsonl")]
    dialogues_val += [dict(d, source="hall") for d in archive.jsonl("validation/dialogues_hall_validation.jsonl")]
    write_jsonl(out / "eval" / "dialogues_validation.jsonl", dialogues_val)

    # --- 6. manifest ---------------------------------------------------------------------
    outputs = {}
    for path in sorted(list(out.rglob("*")) + list(cfg.rglob("*")) + list(keep.rglob("*"))):
        if path.is_file() and path.name != "manifest.json":
            outputs[str(path.relative_to(HERE)).replace("\\", "/")] = {"sha256": sha256_bytes(path.read_bytes()),
                                                                        "bytes": path.stat().st_size}
    manifest = {
        "importer": "mira/import_delivery.py",
        "input_zip": {"name": zip_path.name, "sha256": zip_sha, "bytes": zip_path.stat().st_size},
        "inputs_used": dict(sorted(archive.read_log.items())),
        "inputs_not_used": "excluded/, reports/, generator_ready/, canon_extensions.json (карантин), new_topics_intent_candidates (только ревью)",
        "outputs": outputs,
        "stats": dict(sorted(stats.items())),
        "report": report,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"stats": manifest["stats"], "notes": report["notes"]}, ensure_ascii=False, indent=1))
    print("Готово:", out, "и", cfg)


if __name__ == "__main__":
    main()
