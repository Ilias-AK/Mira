"""Shared training/runtime formatting with explicit source and copy boundaries."""
from __future__ import annotations

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<question>",
                  "<history>", "<persona>", "<retrieved>", "<memory>", "<tool>",
                  "<user>", "<assistant>", "<mask>")


def _token_id(tokenizer, text):
    if hasattr(tokenizer, "token_id"):
        return tokenizer.token_id(text)
    encoded = tokenizer.encode(text)
    if len(encoded) != 1:
        raise ValueError(f"Tokenizer lacks special token {text}")
    return encoded[0]


def text_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "answer", "response", "content", "fact"):
            if key in value:
                return text_value(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(part for item in value if (part := text_value(item)))
    return str(value)


def question_and_history(record):
    context = record.get("context", [])
    if isinstance(context, str):
        return str(record.get("question", context)), []
    turns = [dict(turn) if isinstance(turn, dict) else {"role": "user", "text": str(turn)} for turn in context]
    if "question" in record:
        return str(record["question"]), turns
    if turns and turns[-1].get("role", turns[-1].get("speaker", "user")) in ("user", "human", "person1"):
        return text_value(turns[-1]), turns[:-1]
    return "", turns


def _history_text(turns):
    return "\n".join(("<assistant>" if turn.get("role", turn.get("speaker")) in ("assistant", "bot") else "<user>")
                     + text_value(turn) for turn in turns)


def _retrieved_text(retrieved):
    if isinstance(retrieved, str):
        return retrieved
    if isinstance(retrieved, dict):
        retrieved = [retrieved]
    pieces = []
    for candidate in retrieved or []:
        if isinstance(candidate, str):
            pieces.append(candidate)
            continue
        context = candidate.get("context", [])
        if isinstance(context, list):
            context = _history_text(context)
        response = text_value(candidate.get("response", candidate.get("text", "")))
        pieces.append(str(context) + "\n<assistant>" + response)
    return "\n<sep>\n".join(pieces)


def format_example(record, tokenizer, max_source_length=512, max_target_length=96,
                   retrieved=None, memory=None, tools=None):
    """Return unpadded lists; context includes the latest user question.

    The source always retains BOS and a question section. Oversized questions keep
    both their beginning and end; factual memory/tools precede optional examples.
    This is a token budget, not a claim that truncation preserves arbitrary logic.
    """
    if max_source_length < 8 or max_target_length < 1:
        raise ValueError("Source length must be >=8 and target length >=1")
    question, history = question_and_history(record)
    special_ids = {_token_id(tokenizer, item) for item in SPECIAL_TOKENS}
    source = [tokenizer.bos_id]
    copies, retrieval_bits = [False], [False]
    remaining = max_source_length - 1

    def add_section(tag, value, cap, tail=False, keep_ends=False):
        nonlocal remaining
        ids = tokenizer.encode(value) if value else []
        if remaining < 2 or (not ids and tag != "<question>"):
            return
        allowance = min(cap, remaining - 1)
        if len(ids) > allowance:
            if keep_ends and allowance > 1:
                left = (allowance + 1) // 2
                ids = ids[:left] + ids[-(allowance-left):]
            else:
                ids = ids[-allowance:] if tail and allowance else ids[:allowance]
        values = [_token_id(tokenizer, tag)] + ids
        source.extend(values)
        copies.extend(token not in special_ids for token in values)
        retrieval_bits.extend([tag == "<retrieved>"] * len(values))
        remaining -= len(values)

    add_section("<question>", question, max(1, max_source_length // 2), keep_ends=True)
    add_section("<memory>", text_value(record.get("memory") if memory is None else memory), max_source_length // 8, tail=True)
    add_section("<tool>", text_value(record.get("tools") if tools is None else tools), max_source_length // 16)
    add_section("<persona>", text_value(record.get("persona", [])), max_source_length * 5 // 64)
    retrieval_value = record.get("retrieved", []) if retrieved is None else retrieved
    add_section("<retrieved>", _retrieved_text(retrieval_value), max_source_length // 4, tail=True)
    add_section("<history>", _history_text(history), remaining, tail=True)
    target = tokenizer.encode(text_value(record.get("response", "")))[:max_target_length - 1] + [tokenizer.eos_id]
    result = {
        "input_ids": source, "attention_mask": [True] * len(source),
        "copy_mask": copies, "retrieval_mask": retrieval_bits, "labels": target,
        "id": record.get("id", ""), "group_id": record.get("group_id", ""),
        "retrieval_target": float(record.get("retrieval_applicable", -1)),
    }
    return result


def collate_examples(examples, tokenizer):
    import torch
    if not examples:
        raise ValueError("Empty batch")
    b, s, t = len(examples), max(len(x["input_ids"]) for x in examples), max(len(x["labels"]) for x in examples)
    batch = {
        "input_ids": torch.full((b, s), tokenizer.pad_id, dtype=torch.long),
        "attention_mask": torch.zeros(b, s, dtype=torch.bool),
        "copy_mask": torch.zeros(b, s, dtype=torch.bool),
        "retrieval_mask": torch.zeros(b, s, dtype=torch.bool),
        "labels": torch.full((b, t), -100, dtype=torch.long),
        "decoder_input_ids": torch.full((b, t), tokenizer.pad_id, dtype=torch.long),
        "retrieval_targets": torch.tensor([x.get("retrieval_target", -1) for x in examples], dtype=torch.float),
    }
    for i, example in enumerate(examples):
        for key in ("input_ids", "attention_mask", "copy_mask", "retrieval_mask"):
            batch[key][i, :len(example[key])] = torch.tensor(example[key], dtype=batch[key].dtype)
        labels = example["labels"]
        batch["labels"][i, :len(labels)] = torch.tensor(labels)
        batch["decoder_input_ids"][i, :len(labels)] = torch.tensor([tokenizer.bos_id] + labels[:-1])
    return batch
