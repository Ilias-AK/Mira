"""Stage additional dialogue data without changing the frozen training bundle.

The Telegram collection is an experimental, single-turn training candidate.
It has no thread IDs, so no conversation-disjoint validation claim is possible.
OASST2 retains complete source trees and excludes every tree from OASST1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import time

from .data import (clean_text, download_file, normalized_fingerprint_text,
                   record_fingerprint, sha256_file, split_group)
from .quality import response_issues


DOWNLOADS = {
    "oasst2/messages.jsonl.gz": (
        "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/2023-11-05_oasst2_ready.messages.jsonl.gz",
        "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751", 54301900),
    "oasst2/README.md": (
        "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/README.md",
        "52bd472cf3cae04a4b21ecd5316f929741c47d991d4763b041c3a94de072dc31", 10606),
    "telegram/0000.parquet": (
        "https://huggingface.co/datasets/Den4ikAI/russian_dialogues/resolve/3191133ca40dcb1fa88bc5f771f041bed5029690/default/train/0000.parquet",
        "d268b34343388ad5a4d86997aca48e884b863a7696ead11c1fbe407c3cf2fd70", 274847433),
    "telegram/0001.parquet": (
        "https://huggingface.co/datasets/Den4ikAI/russian_dialogues/resolve/3191133ca40dcb1fa88bc5f771f041bed5029690/default/train/0001.parquet",
        "a0f0e2966c996fdc54d181800c2707a4c42a98d948076e9275fa861bc856eb6d", 1131199),
    "telegram/README.md": (
        "https://huggingface.co/datasets/Den4ikAI/russian_dialogues/resolve/94415c507af3bb878bd62448780209c8681deaaa/README.md",
        "a516c4aa8ba17daba7a6ed13fa765863bea19a712f399b39428e8fc64a3d3443", 951),
}

SOURCE_NOTES = {
    "oasst2": {
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst2",
        "publisher_license": "Apache-2.0",
        "selection": "Russian, synthetic=false, no model_name, deleted=false, review_result=true throughout the parent chain; exclude every OASST1 tree.",
        "limits": "Source flags describe the release; authorship of every utterance has not been independently verified. New source trees are kept together.",
    },
    "telegram": {
        "url": "https://huggingface.co/datasets/Den4ikAI/russian_dialogues",
        "publisher_license": "MIT as declared in the dataset card",
        "selection": "Only relevance=1. Negative shuffled pairs are never training responses. Deterministic lexical/length/quality filters, deduplication, held-out overlap blocking.",
        "limits": "No chat/thread IDs or individual consent records are provided. The publisher license is not independent verification of rights in each source message. Single-turn, training-only experimental candidate; no reconstructed conversations or independent test split. Relevance labels do not guarantee a useful answer.",
    },
}

_ADULT = re.compile(r"\b(?:секс\w*|порн\w*|оргазм\w*|эрот\w*|минет\w*|член(?:а|ом|е)?|письк\w*|пиписк\w*|вагин\w*|аналь\w*|проститут\w*)\b", re.I)
_NOISY = re.compile(r"(?:https?://|www\.|@\w|```|[<>]|\ufffd|(?:[!?.,]){4,})", re.I)
_HIGH_RISK = re.compile(r"\b(?:суицид\w*|наркот\w*|героин\w*|кокаин\w*|дозиров\w*|таблет\w*|препарат\w*|лекарств\w*|нацист\w*|фашист\w*)\b", re.I)


def _text_hash(text):
    return hashlib.sha256(normalized_fingerprint_text(text).encode()).digest()


def candidate_problem(question, answer, *, strict_chat=False):
    """Selection heuristics, not a semantic relevance or safety guarantee."""
    if not isinstance(question, str) or not isinstance(answer, str):
        return "not_text"
    if not question or not answer:
        return "empty"
    if len(question) > (384 if strict_chat else 1500) or len(answer) > (280 if strict_chat else 512):
        return "length"
    if normalized_fingerprint_text(question) == normalized_fingerprint_text(answer):
        return "echo"
    for text, minimum_words in ((question, 3), (answer, 4)):
        issues = response_issues(text)
        if issues:
            return "quality_" + issues[0]
        if _NOISY.search(text):
            return "noisy_markup"
        if _ADULT.search(text):
            return "adult_topic"
        if strict_chat and _HIGH_RISK.search(text):
            return "high_risk_topic"
        letters = sum(character.isalpha() for character in text)
        russian = len(re.findall(r"[А-Яа-яЁё]", text))
        if letters < (10 if strict_chat else 2) or russian / max(letters, 1) < .8:
            return "russian_ratio"
        words = re.findall(r"[А-Яа-яЁё]+", text.casefold())
        if strict_chat and len(words) < minimum_words:
            return "short_fragment"
        if len(words) >= 6 and len(set(words)) / len(words) < .5:
            return "repetition"
        if re.search(r"(.)\1{4,}", text):
            return "repeated_characters"
    return None


class FrozenProtection:
    """Exact normalized pair and question protection against all frozen splits.

    Held-out multiword utterances are also blocked when reused as new text.
    This cannot detect every paraphrase or an undocumented shared source thread.
    """
    def __init__(self, frozen_dir):
        self.pairs, self.questions, self.heldout_text = set(), set(), set()
        self.files = {}
        for split in ("train", "val", "test"):
            path = Path(frozen_dir) / f"{split}.jsonl"
            rows = 0
            with path.open(encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    rows += 1
                    if record.get("kind") != "dialogue":
                        continue
                    self.pairs.add(record_fingerprint(record))
                    self.questions.add(_text_hash(record["context"][-1]["text"]))
                    if split != "train":
                        for text in [record["response"], *[turn["text"] for turn in record["context"]]]:
                            if len(normalized_fingerprint_text(text)) >= 16:
                                self.heldout_text.add(_text_hash(text))
            self.files[split] = {"path": str(path), "records": rows, "sha256": sha256_file(path)}

    def reason(self, record):
        if record_fingerprint(record) in self.pairs:
            return "frozen_pair"
        if _text_hash(record["context"][-1]["text"]) in self.questions:
            return "frozen_question"
        if any(_text_hash(text) in self.heldout_text for text in
               [record["response"], *[turn["text"] for turn in record["context"]]]):
            return "heldout_utterance"
        return None

    def add_heldout(self, record):
        """Protect a new source's held-out rows before importing the next source."""
        if record.get("split") not in ("val", "test"):
            raise ValueError("Only validation/test records may extend held-out protection")
        self.pairs.add(record_fingerprint(record))
        self.questions.add(_text_hash(record["context"][-1]["text"]))
        for text in [record["response"], *[turn["text"] for turn in record["context"]]]:
            if len(normalized_fingerprint_text(text)) >= 16:
                self.heldout_text.add(_text_hash(text))


def old_tree_ids(path):
    result = set()
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            node = json.loads(line)
            tree_id = node.get("message_tree_id")
            if tree_id:
                result.add(tree_id)
            elif node.get("parent_id") is None:
                result.add(node["message_id"])
    return result


def iter_new_oasst(path, previous_trees, counts):
    nodes = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            node = json.loads(line)
            counts["input_messages"] += 1
            if node.get("lang") == "ru":
                counts["russian_messages"] += 1
                nodes[node["message_id"]] = node
    for node in sorted(nodes.values(), key=lambda value: value["message_id"]):
        if node.get("role") != "assistant":
            continue
        counts["assistant_candidates"] += 1
        if node.get("message_tree_id") in previous_trees:
            counts["rejected_existing_oasst1_tree"] += 1
            continue
        chain, seen, cursor, problem = [], set(), node, None
        while cursor is not None:
            if cursor["message_id"] in seen:
                problem = "cycle"
                break
            seen.add(cursor["message_id"])
            if (cursor.get("synthetic") is not False or cursor.get("model_name")
                    or cursor.get("deleted") is not False or cursor.get("review_result") is not True):
                problem = "unreviewed_or_nonhuman_chain"
                break
            if cursor.get("message_tree_id") != node.get("message_tree_id"):
                problem = "tree_mismatch"
                break
            if cursor.get("role") not in ("prompter", "assistant") or (chain and cursor["role"] == chain[-1]["role"]):
                problem = "roles"
                break
            chain.append(cursor)
            parent = cursor.get("parent_id")
            if parent is None:
                break
            cursor = nodes.get(parent)
            if cursor is None:
                problem = "missing_or_nonrussian_parent"
        if problem or len(chain) < 2 or chain[-1]["role"] != "prompter":
            counts["rejected_" + (problem or "incomplete_chain")] += 1
            continue
        tree_id = node.get("message_tree_id") or chain[-1]["message_id"]
        if tree_id in previous_trees:
            counts["rejected_existing_oasst1_tree"] += 1
            continue
        context = [{"role": "user" if item["role"] == "prompter" else "assistant", "text": clean_text(item["text"])}
                   for item in reversed(chain[1:])]
        answer = clean_text(node["text"])
        problem = candidate_problem(context[-1]["text"], answer)
        if not problem:
            for turn in context:
                if response_issues(turn["text"]) or _NOISY.search(turn["text"]) or _ADULT.search(turn["text"]):
                    problem = "earlier_context_quality"
                    break
        if problem:
            counts["rejected_" + problem] += 1
            continue
        yield {"id": "oasst2:" + node["message_id"], "group_id": "oasst2:" + tree_id,
               "source": "oasst2", "kind": "dialogue", "context": context[-7:],
               "response": answer, "persona": []}


def iter_telegram(paths, counts):
    import pyarrow.parquet as pq
    for path in paths:
        row_number = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["question", "answer", "relevance"]):
            for row in batch.to_pylist():
                row_number += 1
                counts["input_pairs"] += 1
                if row.get("relevance") != 1:
                    counts["rejected_negative_pair"] += 1
                    continue
                counts["positive_pairs"] += 1
                if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
                    counts["rejected_not_text"] += 1
                    continue
                question, answer = clean_text(row["question"]), clean_text(row["answer"])
                problem = candidate_problem(question, answer, strict_chat=True)
                if problem:
                    counts["rejected_" + problem] += 1
                    continue
                identity = f"telegram:{Path(path).stem}:{row_number}"
                # A pair ID is not a source conversation ID. These rows are
                # staged for training only; source conversation links are absent.
                yield {"id": identity, "group_id": identity, "source": "telegram_pairs",
                       "kind": "dialogue", "context": [{"role": "user", "text": question}],
                       "response": answer, "persona": [], "split": "train"}


def fetch(raw_dir):
    root = Path(raw_dir)
    results = {}
    for relative, (url, digest, size) in DOWNLOADS.items():
        results[relative] = download_file(url, root / relative, digest, size)
    (root / "download_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def prepare(raw_dir, output_dir, frozen_dir, previous_oasst, sources=("oasst2", "telegram")):
    if not sources or set(sources) - set(SOURCE_NOTES):
        raise ValueError("Select known nonempty sources")
    # The source with real conversation boundaries is split first; downstream
    # training-only sources must respect these new held-out boundaries too.
    sources = [source for source in ("oasst2", "telegram") if source in sources]
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Preserve existing stage and use a new output directory: {output}")
    raw = Path(raw_dir)
    started = time.monotonic()
    protection = FrozenProtection(frozen_dir)
    previous_trees = old_tree_ids(previous_oasst) if "oasst2" in sources else set()
    output.mkdir(parents=True)
    manifest = {"version": 1, "complete": False, "status": "experimental_staging_not_added_to_training",
                "frozen_inputs": protection.files, "sources": {},
                "preparation_code_sha256": sha256_file(__file__),
                "quality_code_sha256": sha256_file(Path(__file__).with_name("quality.py")),
                "limitations": "Lexical heuristics are not semantic quality certification. Telegram has no original conversation IDs. No new synthetic dialogues were generated."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    seen = set()
    for source in sources:
        counts = Counter()
        folder = output / source
        folder.mkdir()
        inputs = [raw / "oasst2/messages.jsonl.gz"] if source == "oasst2" else sorted((raw / "telegram").glob("*.parquet"))
        if not inputs or any(not path.is_file() for path in inputs):
            raise FileNotFoundError(f"Download source first: {source}")
        for path in inputs:
            relative = path.relative_to(raw).as_posix()
            if relative not in DOWNLOADS or sha256_file(path) != DOWNLOADS[relative][1]:
                raise ValueError(f"Source hash differs from the pinned release: {path}")
        iterator = iter_new_oasst(inputs[0], previous_trees, counts) if source == "oasst2" else iter_telegram(inputs, counts)
        handles = {split: (folder / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") for split in ("train", "val", "test")}
        samples, randomizer, groups, question_counts = [], random.Random(20261121), set(), Counter()
        try:
            for record in iterator:
                reason = protection.reason(record)
                if reason:
                    counts["rejected_" + reason] += 1
                    continue
                fingerprint = record_fingerprint(record)
                if fingerprint in seen:
                    counts["rejected_duplicate_new_pair"] += 1
                    continue
                seen.add(fingerprint)
                # Bound repeated prompt variants in single-turn data to keep a
                # popular forum prompt from dominating the proposed expansion.
                question_hash = _text_hash(record["context"][-1]["text"])
                if source == "telegram" and question_counts[question_hash] >= 2:
                    counts["rejected_prompt_cap"] += 1
                    continue
                question_counts[question_hash] += 1
                split = record.get("split") or split_group(record["group_id"], seed=20261121)
                record["split"] = split
                handles[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["retained"] += 1
                counts["retained_" + split] += 1
                counts["retained_multi_turn"] += int(len(record["context"]) > 1)
                counts["answer_characters"] += len(record["response"])
                groups.add(record["group_id"])
                sample = {"id": record["id"], "question": record["context"][-1]["text"], "answer": record["response"]}
                if len(samples) < 100:
                    samples.append(sample)
                else:
                    index = randomizer.randrange(counts["retained"])
                    if index < 100:
                        samples[index] = sample
                if counts["retained"] % 50000 == 0:
                    print(f"Staged {source}: {counts['retained']:,}", flush=True)
        finally:
            for handle in handles.values():
                handle.close()
        for split in ("val", "test"):
            with (folder / f"{split}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    protection.add_heldout(json.loads(line))
        (folder / "review_sample.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["sources"][source] = {**SOURCE_NOTES[source], "counts": dict(counts),
            "groups": len(groups), "old_oasst1_trees_excluded": len(previous_trees) if source == "oasst2" else None,
            "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in inputs],
            "outputs": {split: {"path": str(folder / f"{split}.jsonl"), "sha256": sha256_file(folder / f"{split}.jsonl"),
                        "records": counts["retained_" + split]} for split in handles}}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Finished {source}: {counts['retained']:,} records", flush=True)
    manifest["seconds"] = round(time.monotonic() - started, 3)
    manifest["complete"] = True
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "prepare"))
    parser.add_argument("--raw-dir", default="data/raw/expansion")
    parser.add_argument("--output-dir", default="data/expansion/20261121")
    parser.add_argument("--frozen-dir", default="data/processed")
    parser.add_argument("--previous-oasst", default="data/raw/oasst/2023-04-12_oasst_ready.messages.jsonl.gz")
    parser.add_argument("--sources", nargs="+", choices=("oasst2", "telegram"), default=["oasst2", "telegram"])
    args = parser.parse_args(argv)
    result = fetch(args.raw_dir) if args.command == "fetch" else prepare(args.raw_dir, args.output_dir, args.frozen_dir, args.previous_oasst, args.sources)
    print(json.dumps({"complete": True, "sources": list(result.get("sources", result))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
