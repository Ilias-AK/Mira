"""Точность самой нейросети тем (без правил) на независимой проверке.

python -m mira.eval_intent_model --model runs/mira_intent/intent.pt --name intent_baseline
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--val-file", default="data/delivery_v2/intent/val.jsonl")
    ap.add_argument("--out-dir", default="runs/eval_v2")
    args = ap.parse_args(argv)
    from mira.intent_model import IntentModel
    model = IntentModel(args.model, args.tokenizer)
    rows = [json.loads(l) for l in Path(args.val_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    known = set(model.labels)
    covered = [r for r in rows if r["label"] in known]
    pairs, confident, errors = [], [], []
    for r in covered:
        label, prob, margin = model.predict(r["text"])
        pairs.append((r["label"], label))
        if prob >= model.threshold:
            confident.append(label == r["label"])
        if label != r["label"] and len(errors) < 20:
            errors.append({"q": r["text"], "gold": r["label"], "pred": label, "prob": round(prob, 3)})
    f1s = []
    for lab in {g for g, _ in pairs}:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        pr, rc = (tp / (tp + fp) if tp + fp else 0), (tp / (tp + fn) if tp + fn else 0)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    res = {"name": args.name, "model": args.model, "threshold": model.threshold, "labels_in_model": len(known),
           "n": len(rows), "covered": len(covered),
           "accuracy_on_covered": round(sum(g == p for g, p in pairs) / max(1, len(pairs)), 4),
           "accuracy_on_all": round(sum(g == p for g, p in pairs) / max(1, len(rows)), 4),
           "macro_f1_on_covered": round(sum(f1s) / max(1, len(f1s)), 4),
           "share_above_threshold": round(len(confident) / max(1, len(covered)), 4),
           "precision_above_threshold": round(sum(confident) / max(1, len(confident)), 4), "errors": errors}
    print(json.dumps({k: v for k, v in res.items() if k != "errors"}, ensure_ascii=False))
    out = Path(args.out_dir) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
