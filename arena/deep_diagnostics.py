"""Model-level diagnostics of repeated generator replies. Offline and read-only.

Loads the trained checkpoint without changing weights, data, tokenizer, index or
cache, and writes a new JSON + TXT report. Each section runs independently; an
error in one section is recorded and the others still run.

Sections
  input_check      decoded model input with special tokens, section sizes, shift check
  val_nll          teacher-forced NLL on a wide fixed val sample, per source, under
                   input ablations (shuffled dialogue/persona, no persona, ...)
  val_generation   free greedy generation on val examples vs references
  new_questions    new questions under greedy / seeded sampling / persona / retrieval /
                   copy-off ablations, exact question -> answer pairs
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import random
import re
import time
import traceback
from pathlib import Path

import torch

from arena.formatting import collate_examples, format_example, question_and_history, text_value
from arena.memory import ConversationMemory
from arena.training import IndexedJsonlDataset, file_sha256, load_model_checkpoint

ALT_PERSONA = ["Я работаю поваром.", "У меня есть собака.", "Я болею за футбол."]

NEW_QUESTIONS = [
    "Какой фильм ты недавно смотрел?",
    "Что ты ел на завтрак?",
    "Ты любишь собак или кошек?",
    "Куда бы ты поехал летом?",
    "Что тебя сегодня порадовало?",
    "Какая у тебя любимая книга?",
    "Ты умеешь готовить?",
    "Чем ты занимаешься по выходным?",
    "Тебе нравится зима?",
    "Какой у тебя любимый цвет?",
    "Ты бот?",
    "Сколько тебе лет?",
    "Что посоветуешь посмотреть вечером?",
    "Ты играешь в компьютерные игры?",
    "Как ты относишься к спорту?",
    "В каком городе ты живёшь?",
]

# Same final question with and without a short fixed history.
MULTI_TURN = [
    [("user", "Я вчера ходил в кино."), ("assistant", "Здорово! Понравилось?"),
     ("user", "Да, очень. Угадай, какой был жанр?")],
    [("user", "Я очень устал после работы."), ("assistant", "Сочувствую. Тяжёлый день?"),
     ("user", "Да. Что посоветуешь, чтобы отдохнуть?")],
]

SECTION_TAGS = ("<question>", "<memory>", "<tool>", "<persona>", "<retrieved>", "<history>")


def norm(text):
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def contains_fact(answer, facts, min_chars=10):
    a = norm(answer)
    return any(len(norm(f)) >= min_chars and norm(f) in a for f in facts)


class Diagnostics:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        self.model, self.tok, self.meta = load_model_checkpoint(args.checkpoint, device=self.device,
                                                                tokenizer_path=args.tokenizer)
        self.model.eval()
        self.cfg = self.model.config
        self.amp_dtype = None
        if self.device == "cuda":
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        persona_data = json.loads(Path(args.persona).read_text(encoding="utf-8-sig"))
        self.persona = list(persona_data.get("facts", persona_data if isinstance(persona_data, list) else []))
        self.special = {self.tok.token_id(t): t for t in SECTION_TAGS}

    # ---------- helpers ----------
    def amp(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    @contextlib.contextmanager
    def copy_disabled(self, disabled=True):
        old = self.cfg.copy_enabled
        if disabled:
            self.cfg.copy_enabled = False
        try:
            yield
        finally:
            self.cfg.copy_enabled = old

    @staticmethod
    def memory_for(persona, context):
        """Same construction as IndexedJsonlDataset and ArenaRuntime."""
        state = ConversationMemory(persona=list(persona or []))
        for turn in context:
            if turn.get("role") == "assistant":
                state.add_assistant(text_value(turn))
            else:
                state.add_user(text_value(turn))
        question = text_value(context[-1]) if context else ""
        return state.retrieve(question, limit=6)

    def fmt(self, row, retrieved, memory):
        return format_example(row, self.tok, self.cfg.max_source_length, self.cfg.max_target_length,
                              retrieved=retrieved, memory=memory)

    def section_sizes(self, ids):
        sizes, current = collections.OrderedDict(), None
        for token in ids[1:]:
            if token in self.special:
                current = self.special[token]
                sizes[current] = sizes.get(current, 0)
            elif current:
                sizes[current] += 1
        return dict(sizes)

    def forward_stats(self, examples):
        """Per-example summed target NLL, token count, mean copy share and gate."""
        batch = collate_examples(examples, self.tok)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode(), self.amp():
            out = self.model(input_ids=batch["input_ids"], decoder_input_ids=batch["decoder_input_ids"],
                             attention_mask=batch["attention_mask"], copy_mask=batch["copy_mask"],
                             retrieval_mask=batch["retrieval_mask"])
        logp = out["log_probs"].float()
        labels = batch["labels"]
        mask = labels.ne(-100)
        tok_nll = -logp.gather(-1, labels.clamp_min(0)[..., None]).squeeze(-1)
        tok_nll = tok_nll * mask
        p_gen = out["p_generate"].float().squeeze(-1)
        copy_share = ((1 - p_gen) * mask).sum(1) / mask.sum(1).clamp_min(1)
        result = []
        for i in range(len(examples)):
            result.append({"nll_sum": float(tok_nll[i].sum()), "tokens": int(mask[i].sum()),
                           "copy_share": float(copy_share[i]), "gate": float(out["gate"][i].float())})
        return result

    def generate(self, row, retrieved, memory, temperature=0.0, seed=None, copy_off=False, top_k=40):
        ex = self.fmt(row, retrieved, memory)
        kwargs = {k: torch.tensor([ex[k]], device=self.device)
                  for k in ("input_ids", "attention_mask", "copy_mask", "retrieval_mask")}
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)
        start = time.monotonic()
        with self.copy_disabled(copy_off), torch.inference_mode(), self.amp():
            ids = self.model.generate(**kwargs, max_new_tokens=self.cfg.max_target_length,
                                      bos_id=self.tok.bos_id, eos_id=self.tok.eos_id,
                                      temperature=temperature, top_k=top_k if temperature > 0 else 0)
        elapsed = time.monotonic() - start
        tokens = ids[0].tolist()
        if self.tok.eos_id in tokens:
            tokens = tokens[:tokens.index(self.tok.eos_id)]
        text = self.tok.decode(tokens)
        copy_share = None
        if tokens:
            scored = dict(ex)
            scored["labels"] = tokens[:self.cfg.max_target_length - 1] + [self.tok.eos_id]
            with self.copy_disabled(copy_off):
                copy_share = self.forward_stats([scored])[0]["copy_share"]
        return {"answer": text, "seconds": round(elapsed, 3),
                "copy_share": None if copy_share is None else round(copy_share, 4),
                "source_tokens": len(ex["input_ids"])}

    # ---------- sections ----------
    def input_check(self, retriever):
        question = "Привет! Как бы ты провёл свободный вечер?"
        context = [{"role": "user", "text": question}]
        retrieved = retriever.search(question, history=[], persona=self.persona, k=1, timeout_seconds=5) if retriever else []
        memory = self.memory_for(self.persona, context)
        row = {"id": "live", "group_id": "live", "kind": "dialogue", "context": context,
               "response": "", "persona": self.persona}
        ex = self.fmt(row, retrieved, memory)
        decoded = self.tok.tokenizer.decode(ex["input_ids"], skip_special_tokens=False)
        # Shift check on a real collate.
        sample = dict(ex)
        sample["labels"] = self.tok.encode("Проверка сдвига ответа.") + [self.tok.eos_id]
        batch = collate_examples([sample], self.tok)
        labels = batch["labels"][0].tolist()
        dec = batch["decoder_input_ids"][0].tolist()
        n = len(sample["labels"])
        shift_ok = dec[0] == self.tok.bos_id and dec[1:n] == labels[:n - 1]
        persona_tokens = len(self.tok.encode(text_value(self.persona)))
        return {"question": question, "decoded_source": decoded,
                "source_tokens": len(ex["input_ids"]), "max_source_length": self.cfg.max_source_length,
                "section_tokens": self.section_sizes(ex["input_ids"]),
                "persona_raw_tokens": persona_tokens,
                "persona_section_cap": self.cfg.max_source_length * 5 // 64,
                "memory_items": [text_value(m) for m in memory],
                "retrieved_top1": [r.get("response") for r in retrieved],
                "copy_mask_true": int(sum(ex["copy_mask"])),
                "decoder_shift_ok": bool(shift_ok)}

    def load_val(self, cache):
        ds = IndexedJsonlDataset(self.args.val_data, self.tok, split="validation", stage="dialogue",
                                 seed=42, max_source_length=self.cfg.max_source_length,
                                 max_target_length=self.cfg.max_target_length,
                                 retriever=None, retrieval_dropout=0.0, retrieval_cache=cache)
        rows = [ds.record(i) for i in range(len(ds))]
        ds.close()
        return rows

    def build(self, row, cache):
        retrieved = cache.get(row["id"], row["group_id"]) if cache is not None else []
        return retrieved or [], self.memory_for(row.get("persona"), row["context"])

    def val_nll(self, rows, cache):
        rng = random.Random(20261121)
        by_source = collections.defaultdict(list)
        for r in rows:
            by_source[r.get("source", "?")].append(r)
        sample = []
        for source, items in sorted(by_source.items()):
            items = list(items)
            rng.shuffle(items)
            limit = self.args.val_per_source
            sample.extend(items[:limit] if limit else items)
        # partner = next example of the same source (derangement within source)
        partners = {}
        for source in by_source:
            idx = [i for i, r in enumerate(sample) if r.get("source") == source]
            for a, b in zip(idx, idx[1:] + idx[:1]):
                partners[a] = b
        base = [self.build(r, cache) for r in sample]

        def variant(name, i):
            row, (ret, mem) = sample[i], base[i]
            j = partners[i]
            other, (ret_j, _) = sample[j], base[j]
            if name in ("correct", "copy_off"):
                return row, ret, mem
            if name == "shuffled_dialogue":
                r2 = dict(row, context=other["context"])
                return r2, ret_j, self.memory_for(row.get("persona"), r2["context"])
            if name == "shuffled_persona":
                r2 = dict(row, persona=other.get("persona", []))
                return r2, ret, self.memory_for(r2["persona"], row["context"])
            if name == "no_persona":
                r2 = dict(row, persona=[])
                return r2, ret, self.memory_for([], row["context"])
            if name == "no_retrieved":
                return row, [], mem
            if name == "question_only":
                r2 = dict(row, context=row["context"][-1:], persona=[])
                return r2, [], []
            if name == "empty_input":
                r2 = dict(row, context=[{"role": "user", "text": ""}], persona=[])
                return r2, [], []
            raise ValueError(name)

        names = ["correct", "shuffled_dialogue", "shuffled_persona", "no_persona",
                 "no_retrieved", "question_only", "empty_input", "copy_off"]
        per = {n: [None] * len(sample) for n in names}
        bs = self.args.batch_size
        for name in names:
            for s in range(0, len(sample), bs):
                idx = list(range(s, min(len(sample), s + bs)))
                exs = []
                for i in idx:
                    r, ret, mem = variant(name, i)
                    exs.append(self.fmt(r, ret, mem))
                with self.copy_disabled(name == "copy_off"):
                    stats = self.forward_stats(exs)
                for i, st in zip(idx, stats):
                    per[name][i] = st
            print(f"  val_nll: {name} done", flush=True)
        truncated = sum(1 for i in range(len(sample))
                        if len(self.fmt(sample[i], *base[i])["input_ids"]) >= self.cfg.max_source_length)
        report = {"examples": len(sample), "per_source_count": dict(collections.Counter(r.get("source") for r in sample)),
                  "source_at_max_length": truncated, "variants": {}}
        for name in names:
            report["variants"][name] = {}
            for source in sorted(by_source) + ["all"]:
                idx = [i for i in range(len(sample)) if source == "all" or sample[i].get("source") == source]
                nll = sum(per[name][i]["nll_sum"] for i in idx)
                tokens = sum(per[name][i]["tokens"] for i in idx)
                entry = {"nll": round(nll / max(1, tokens), 4),
                         "perplexity": round(math.exp(min(50, nll / max(1, tokens))), 2),
                         "copy_share": round(sum(per[name][i]["copy_share"] for i in idx) / max(1, len(idx)), 4),
                         "gate": round(sum(per[name][i]["gate"] for i in idx) / max(1, len(idx)), 4)}
                if name != "correct":
                    worse = sum(per[name][i]["nll_sum"] > per["correct"][i]["nll_sum"] for i in idx)
                    delta = [(per[name][i]["nll_sum"] - per["correct"][i]["nll_sum"]) / max(1, per["correct"][i]["tokens"]) for i in idx]
                    delta.sort()
                    entry["share_examples_where_correct_is_better"] = round(worse / max(1, len(idx)), 4)
                    entry["median_delta_nll_per_token"] = round(delta[len(delta) // 2], 4) if delta else None
                report["variants"][name][source] = entry
        return report, sample, base

    def val_generation(self, sample, base):
        idx = [i for i, r in enumerate(sample) if r.get("source") == "toloka"][:self.args.val_generations]
        outputs, counter = [], collections.Counter()
        persona_hits = first_fact = ref_persona = 0
        for k, i in enumerate(idx):
            row, (ret, mem) = sample[i], base[i]
            g = self.generate(row, ret, mem)
            answer = g["answer"]
            counter[answer] += 1
            facts = row.get("persona", [])
            persona_hits += contains_fact(answer, facts)
            first_fact += bool(facts) and contains_fact(answer, facts[:1])
            ref_persona += contains_fact(row["response"], facts)
            if k < 40:
                question, _ = question_and_history(row)
                outputs.append({"question": question, "reference": row["response"],
                                "generated": answer, "copy_share": g["copy_share"],
                                "persona": facts})
        n = max(1, len(idx))
        return {"examples": len(idx), "distinct_generated": len(counter),
                "top_generated": counter.most_common(15),
                "generated_contains_own_persona_fact": round(persona_hits / n, 4),
                "generated_contains_first_persona_fact": round(first_fact / n, 4),
                "reference_contains_own_persona_fact": round(ref_persona / n, 4),
                "samples": outputs}

    def new_questions(self, retriever):
        rotated = self.persona[1:] + self.persona[:1]
        modes = [
            ("greedy", dict(persona=self.persona)),
            ("greedy_no_persona", dict(persona=[])),
            ("greedy_persona_rotated", dict(persona=rotated)),
            ("greedy_alt_persona", dict(persona=ALT_PERSONA)),
            ("greedy_no_retrieved", dict(persona=self.persona, no_retrieved=True)),
            ("greedy_no_persona_no_retrieved", dict(persona=[], no_retrieved=True)),
            ("greedy_copy_off", dict(persona=self.persona, copy_off=True)),
            ("sample_t0.7_seed1", dict(persona=self.persona, temperature=0.7, seed=1)),
            ("sample_t0.7_seed2", dict(persona=self.persona, temperature=0.7, seed=2)),
            ("sample_t0.7_seed3", dict(persona=self.persona, temperature=0.7, seed=3)),
            ("sample_t0.7_seed1_copy_off", dict(persona=self.persona, temperature=0.7, seed=1, copy_off=True)),
        ]
        inputs = [("single", [{"role": "user", "text": q}]) for q in NEW_QUESTIONS]
        for dialogue in MULTI_TURN:
            turns = [{"role": r, "text": t} for r, t in dialogue]
            inputs.append(("with_history", turns))
            inputs.append(("same_question_no_history", turns[-1:]))
        retrieved_cache = {}
        results = {name: [] for name, _ in modes}
        for name, opt in modes:
            persona = opt["persona"]
            for kind, context in inputs:
                question = context[-1]["text"]
                history = context[:-1]
                key = (question, json.dumps(history, ensure_ascii=False), json.dumps(persona, ensure_ascii=False))
                if opt.get("no_retrieved") or retriever is None:
                    retrieved = []
                else:
                    if key not in retrieved_cache:
                        retrieved_cache[key] = retriever.search(question, history=history, persona=persona,
                                                                k=1, timeout_seconds=5)
                    retrieved = retrieved_cache[key]
                memory = self.memory_for(persona, context)
                row = {"id": "live", "group_id": "live", "kind": "dialogue", "context": context,
                       "response": "", "persona": persona}
                g = self.generate(row, retrieved, memory, temperature=opt.get("temperature", 0.0),
                                  seed=opt.get("seed"), copy_off=opt.get("copy_off", False))
                results[name].append({"kind": kind, "history": [t["text"] for t in history],
                                      "question": question, "answer": g["answer"],
                                      "retrieved_top1": [r.get("response") for r in retrieved],
                                      "copy_share": g["copy_share"], "seconds": g["seconds"]})
            print(f"  new_questions: {name} done", flush=True)
        default_facts = self.persona
        summary = {}
        for name, items in results.items():
            answers = [x["answer"] for x in items]
            summary[name] = {
                "answers": len(answers), "distinct": len(set(answers)),
                "contains_default_persona_fact": sum(contains_fact(a, default_facts) for a in answers),
                "mentions_universitet": sum("университет" in a.lower() for a in answers),
                "mean_copy_share": round(sum(x["copy_share"] or 0 for x in items) / max(1, len(items)), 4),
            }
        return {"modes": [m for m, _ in modes], "summary": summary, "results": results,
                "note": "Automatic counts are surface signals only; read the answers."}


def write_text(report, path):
    lines = [f"checkpoint: {report['checkpoint']}", f"checkpoint_sha256: {report['checkpoint_sha256']}", ""]
    ic = report.get("input_check", {})
    if "decoded_source" in ic:
        lines += ["== INPUT CHECK ==", ic["decoded_source"], f"sections: {ic['section_tokens']}",
                  f"decoder_shift_ok: {ic['decoder_shift_ok']}", ""]
    vn = report.get("val_nll", {})
    if "variants" in vn:
        lines.append("== VAL NLL (teacher forcing) ==")
        for name, by_src in vn["variants"].items():
            lines.append(f"{name}: " + "; ".join(f"{s}: nll={e['nll']} copy={e['copy_share']}"
                                                + (f" better={e.get('share_examples_where_correct_is_better')}" if 'share_examples_where_correct_is_better' in e else "")
                                                for s, e in by_src.items()))
        lines.append("")
    vg = report.get("val_generation", {})
    if "samples" in vg:
        lines += ["== VAL GREEDY GENERATION ==", f"distinct {vg['distinct_generated']} of {vg['examples']}",
                  f"top: {vg['top_generated'][:8]}"]
        for s in vg["samples"][:25]:
            lines.append(f"Q: {s['question']}\n   REF: {s['reference']}\n   GEN: {s['generated']}")
        lines.append("")
    nq = report.get("new_questions", {})
    if "results" in nq:
        lines.append("== NEW QUESTIONS ==")
        for name in nq["modes"]:
            lines.append(f"--- {name}  {nq['summary'][name]}")
            for x in nq["results"][name]:
                prefix = (" | ".join(x["history"]) + " || ") if x["history"] else ""
                lines.append(f"  [{x['kind']}] {prefix}{x['question']} -> {x['answer']}")
        lines.append("")
    for key, value in report.get("errors", {}).items():
        lines += [f"ERROR in {key}:", value]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    from arena.console import configure_utf8
    configure_utf8()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/main/dialogue/best.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--index", default="artifacts/retrieval")
    p.add_argument("--retrieval-cache", default="artifacts/retrieval_cache.sqlite3")
    p.add_argument("--persona", default="configs/persona.json")
    p.add_argument("--val-data", default="data/processed/val.jsonl")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--val-per-source", type=int, default=2000, help="0 = all val examples")
    p.add_argument("--val-generations", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="artifacts/deep_diagnostics.json")
    args = p.parse_args()
    out_json = Path(args.output)
    out_txt = out_json.with_suffix(".txt")
    if out_json.exists() or out_txt.exists():
        p.error("Output already exists; choose a new --output to keep the previous report")
    started = time.monotonic()
    d = Diagnostics(args)
    report = {"checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_sha256": file_sha256(args.checkpoint),
              "tokenizer_sha256": file_sha256(args.tokenizer),
              "loaded": {"model_config": d.meta.get("model_config"),
                         "training_state": d.meta.get("training_state"),
                         "stage": d.meta.get("training_config", {}).get("stage"),
                         "checkpoint_tokenizer_sha256": d.meta.get("tokenizer_sha256")},
              "device": d.device, "amp": str(d.amp_dtype), "persona": d.persona, "errors": {}}
    retriever = cache = None
    try:
        from arena.retrieval import HybridRetriever
        retriever = HybridRetriever.load(args.index)
    except Exception:
        report["errors"]["retriever"] = traceback.format_exc()
    try:
        from arena.retrieval_cache import RetrievalCache
        cache = RetrievalCache(args.retrieval_cache, readonly=True, require_complete=True)
    except Exception:
        report["errors"]["retrieval_cache"] = traceback.format_exc()

    def section(name, fn, *a):
        print(f"Running {name}...", flush=True)
        try:
            return fn(*a)
        except Exception:
            report["errors"][name] = traceback.format_exc()
            print(f"  {name} failed; continuing", flush=True)
            return None

    report["input_check"] = section("input_check", d.input_check, retriever)
    rows = section("load_val", d.load_val, cache)
    sample = base = None
    if rows:
        res = section("val_nll", d.val_nll, rows, cache)
        if res:
            report["val_nll"], sample, base = res
    if sample:
        report["val_generation"] = section("val_generation", d.val_generation, sample, base)
    report["new_questions"] = section("new_questions", d.new_questions, retriever)
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text(report, out_txt)
    for closer in (retriever, cache):
        close = getattr(closer, "close", None)
        if close:
            close()
    print(f"Saved {out_json} and {out_txt}" + (f"; errors in: {list(report['errors'])}" if report["errors"] else ""), flush=True)


if __name__ == "__main__":
    main()
