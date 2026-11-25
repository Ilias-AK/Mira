"""Train Mira's intent model on team-written questions.

Data (all written by the team):
  1. data/team_raw/rounds_*.txt — the team's rounds (### / Вопрос: / Мира:); each question
     is labelled with the bank topic the rules assign to it;
  2. data/team_raw/paraphrases_*.txt — team paraphrases, grouped under "### <topic id>".
Augmentation adds spelling noise and fillers only.

python -m mira.train_intent --checkpoint runs/main/dialogue/best.pt --tokenizer artifacts/tokenizer.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def read_rounds(paths):
    questions = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("Вопрос:"):
                questions.append(line[len("Вопрос:"):].strip())
    return questions


def read_paraphrases(paths, known):
    rows, unknown = [], Counter()
    for path in paths:
        label = None
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("###"):
                label = line.strip("# ").split()[0] if line.strip("# ") else None
                continue
            if not line or line.startswith("#") or label is None:
                continue
            text = line.lstrip("-–—•* ").strip()
            if len(text) < 2:
                continue
            if label not in known:
                unknown[label] += 1
                continue
            rows.append((text, label, Path(path).name))
    return rows, unknown


def label_by_rules(bank, questions):
    import re
    rows = []
    for q in questions:
        if "?" not in q and re.search(r"(?i)\b(?:я|меня|мой|моя|моего|мою|запомни)\b", q):
            continue   # the asker's own facts are handled by memory, not by topics
        matched = bank.match([q], [], 1, "hall")
        team = [m for m in matched if m.get("source") in ("team", "hall_draft")] or matched
        if team:
            rows.append((q, team[0]["id"], "rounds"))
    return rows


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/main/dialogue/best.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--bank", default=str(HERE / "config" / "bank.json"))
    p.add_argument("--rounds", default="data/team_raw/rounds_*.txt")
    p.add_argument("--paraphrases", default="data/team_raw/paraphrases_*.txt")
    p.add_argument("--out", default="runs/mira_intent/intent.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    import torch
    from torch.nn import functional as F
    from arena.training import load_model_checkpoint, tokenizer_hash
    from mira.bank import Bank
    from mira.intent_model import augment, batch, make_net, save

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    bank = Bank(args.bank)
    known = {item["id"] for item in bank.intents}
    round_files = sorted(glob.glob(args.rounds))
    para_files = sorted(f for f in glob.glob(args.paraphrases) if not f.endswith("template.txt"))
    rows = label_by_rules(bank, read_rounds(round_files))
    qa_rows = [(x, item["id"], item.get("file", "team_qa")) for item in bank.intents
               if item.get("source") == "team_qa" for x in item.get("examples", [])]
    rows += qa_rows
    print(f"Темы из hall_qa: {len({r[1] for r in qa_rows})}, вопросов в них: {len(qa_rows)}")
    para_rows, unknown = read_paraphrases(para_files, known)
    rows += para_rows
    print(f"Файлы раундов: {len(round_files)}, файлы перефразов: {len(para_files)}")
    print(f"Примеров: {len(rows)} (из раундов {len(rows) - len(para_rows)}, перефразов {len(para_rows)})")
    if unknown:
        print("Неизвестные темы в перефразах (проверьте id):", dict(unknown.most_common(10)))
    rows = list(dict.fromkeys(rows))
    by_label = defaultdict(list)
    for text, label, source in rows:
        by_label[label].append(text)
    labels = sorted(l for l, texts in by_label.items() if len(set(texts)) >= 1)
    if len(labels) < 2:
        raise SystemExit("Слишком мало данных: нужно хотя бы 2 темы с примерами")
    index = {l: i for i, l in enumerate(labels)}

    # one held-out example per topic that has at least 3 different questions
    train, val = [], []
    for label, texts in by_label.items():
        texts = list(dict.fromkeys(texts))
        rng.shuffle(texts)
        if len(texts) >= 3:
            val.append((texts[0], index[label]))
            texts = texts[1:]
        for t in texts:
            for variant in augment(t, rng):
                train.append((variant, index[label]))
    print(f"Тем: {len(labels)}, обучающих строк (с шумом): {len(train)}, проверочных: {len(val)}")

    base, tokenizer, checkpoint = load_model_checkpoint(args.checkpoint, device="cpu", tokenizer_path=args.tokenizer)
    net = make_net(base, len(labels)).to(device)
    encoder_params = [p for n, p in net.named_parameters() if n.startswith(("base.embedding", "base.source_position", "base.encoder"))]
    for n, p in net.named_parameters():
        if not n.startswith(("base.embedding", "base.source_position", "base.encoder", "head")):
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 5e-5},
        {"params": net.head.parameters(), "lr": 2e-3},
    ], weight_decay=0.01)

    def evaluate(items):
        if not items:
            return None, []
        net.eval()
        probs_all = []
        with torch.no_grad():
            for start in range(0, len(items), 128):
                chunk = items[start:start + 128]
                ids, mask = batch(tokenizer, [t for t, _ in chunk], device)
                probs_all += torch.softmax(net(ids, mask).float(), -1).cpu().tolist()
        correct = sum(int(max(range(len(p)), key=p.__getitem__) == y) for p, (_, y) in zip(probs_all, items))
        return correct / len(items), probs_all

    started, best = time.time(), (-1.0, None)
    for epoch in range(1, args.epochs + 1):
        net.train()
        rng.shuffle(train)
        total = 0.0
        for start in range(0, len(train), args.batch_size):
            chunk = train[start:start + args.batch_size]
            ids, mask = batch(tokenizer, [t for t, _ in chunk], device)
            target = torch.tensor([y for _, y in chunk], device=device)
            loss = F.cross_entropy(net(ids, mask).float(), target, label_smoothing=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(chunk)
        acc, _ = evaluate(val)
        train_acc, _ = evaluate(train[:2000])
        score = acc if acc is not None else train_acc
        if score >= best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"эпоха {epoch:3d}  потеря {total / len(train):.3f}  точность обуч. {train_acc:.2f}"
                  + (f"  проверка {acc:.2f}" if acc is not None else "") + f"  {time.time() - started:.0f} с", flush=True)
    net.load_state_dict(best[1])

    # confidence threshold: lowest value that keeps held-out precision >= 90 %
    acc, probs = evaluate(val)
    threshold = 0.55
    if len(val) < 30:
        threshold = 0.6   # too few held-out questions to calibrate: stay careful
    elif probs:
        scored = sorted(((max(p), int(max(range(len(p)), key=p.__getitem__) == y)) for p, (_, y) in zip(probs, val)), reverse=True)
        for t in [x / 100 for x in range(30, 91, 5)]:
            kept = [c for conf, c in scored if conf >= t]
            if kept and sum(kept) / len(kept) >= 0.9:
                threshold = max(0.4, t)
                break
    report = {"examples": len(rows), "topics": len(labels), "train_rows": len(train), "val_rows": len(val),
              "val_accuracy": acc, "threshold": threshold, "checkpoint": args.checkpoint,
              "round_files": round_files, "paraphrase_files": para_files, "seconds": round(time.time() - started)}
    save(args.out, net, labels, checkpoint["model_config"], tokenizer_hash(args.tokenizer), threshold, report)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Сохранено:", args.out)


if __name__ == "__main__":
    main()
