"""Оценка собственного генератора (arena, обучен с нуля) на validation архива.

python -m mira.eval_generator --checkpoint runs/main/dialogue/best.pt --name gen_baseline
python -m mira.eval_generator --checkpoint runs/generator_v2/best.pt  --name gen_candidate
Метрики: похожесть на эталон (символьные триграммы и токен-F1), разнообразие ответов, доля самого частого ответа,
нарушения канона, длина > 300, не-русский текст, время генерации. Нужен torch (запуск на ноутбуке).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from mira.canon_check import violations  # noqa: E402
from mira.eval_candidate import cyrillic_share, norm, similarity  # noqa: E402


def token_f1(a, b):
    x, y = norm(a).split(), norm(b).split()
    if not x or not y:
        return 0.0
    common = sum((collections.Counter(x) & collections.Counter(y)).values())
    if not common:
        return 0.0
    p, r = common / len(x), common / len(y)
    return 2 * p * r / (p + r)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--sets", nargs="+", default=["data/delivery_v2/generator/val_unseen_response.jsonl",
                                                  "data/delivery_v2/generator/val.jsonl"])
    ap.add_argument("--limit", type=int, default=300, help="примеров из каждого набора (случайная выборка, seed)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default="runs/eval_v2")
    args = ap.parse_args(argv)

    from arena.runtime import IsolatedGenerator
    gen = IsolatedGenerator(args.checkpoint, args.tokenizer, args.device)
    if gen.error:
        raise SystemExit(f"генератор не запустился: {gen.error}")
    result = {"name": args.name, "checkpoint": args.checkpoint, "seed": args.seed, "sets": {}}
    for path in args.sets:
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.limit]
        outs, sims, f1s, times, canon, long_, nonru, examples = [], [], [], [], [], 0, 0, []
        for r in rows:
            record = {k: r[k] for k in ("id", "group_id", "kind", "context", "persona") if k in r}
            record["response"] = ""
            started = time.perf_counter()
            resp = gen.generate(seconds=40, record=record, retrieved=[], memory=[], tools=[],
                                max_source_tokens=512, max_new_tokens=96, temperature=0.0, no_repeat_ngram_size=3)
            times.append(time.perf_counter() - started)
            text = resp.get("text", "") if resp.get("status") == "ok" else ""
            outs.append(norm(text))
            sims.append(similarity(text, [r["response"]]))
            f1s.append(token_f1(text, r["response"]))
            if violations(text):
                canon.append({"q": r["context"][-1]["text"], "a": text, "rule": violations(text)[0][0]})
            long_ += len(text) > 300
            nonru += cyrillic_share(text) < 0.7
            if len(examples) < 12:
                examples.append({"q": r["context"][-1]["text"], "gold": r["response"], "generated": text})
        top = collections.Counter(outs).most_common(1)
        result["sets"][Path(path).name] = {
            "n": len(rows), "gold_similarity_mean": round(statistics.mean(sims), 4), "token_f1_mean": round(statistics.mean(f1s), 4),
            "distinct_ratio": round(len(set(outs)) / max(1, len(outs)), 4),
            "most_common_share": round(top[0][1] / len(outs), 4) if top else 0, "most_common": top[0][0] if top else "",
            "canon_violations": len(canon), "canon_examples": canon[:8], "over_300": long_, "non_russian": nonru,
            "time_s_mean": round(statistics.mean(times), 3), "time_s_max": round(max(times), 3), "examples": examples}
        print(args.name, Path(path).name, json.dumps({k: v for k, v in result["sets"][Path(path).name].items()
                                                      if k not in ("examples", "canon_examples")}, ensure_ascii=False))
    gen.close()
    out = Path(args.out_dir) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Сохранено:", out)


if __name__ == "__main__":
    main()
