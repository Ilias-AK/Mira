"""Mira's own neural intent model: which bank topic is this question about?

The encoder is the team's own transformer (arena.model.ArenaModel, trained from
random initialisation on the team's data; no external weights). On top of the
encoder's mean-pooled states sits a small linear classifier over bank topics.
Both are fine-tuned on team-written questions (mira/train_intent.py).

At runtime the model is optional: if torch, the checkpoint or the tokenizer is
missing, Mira works on rules alone.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from mira.hall import to_ty

MAX_LEN = 64


def normalize(text: str) -> str:
    return to_ty(text)


def encode_text(tokenizer, text: str, max_len: int = MAX_LEN):
    ids = [tokenizer.bos_id, tokenizer.token_id("<user>")] + tokenizer.encode(normalize(text))
    return ids[:max_len]


def make_net(base, n_labels: int):
    import torch
    from torch import nn

    class IntentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            d = base.config.d_model
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_labels))

        def embed(self, ids, mask):
            memory, valid, *_ = self.base.encode(ids, mask)
            weight = valid.unsqueeze(-1).to(memory.dtype)
            return (memory * weight).sum(1) / weight.sum(1).clamp_min(1)

        def forward(self, ids, mask):
            return self.head(self.embed(ids, mask))

    return IntentNet()


def batch(tokenizer, texts, device):
    import torch
    encoded = [encode_text(tokenizer, t) for t in texts]
    width = max(len(e) for e in encoded)
    ids = torch.full((len(encoded), width), tokenizer.pad_id, dtype=torch.long)
    for row, e in enumerate(encoded):
        ids[row, :len(e)] = torch.tensor(e, dtype=torch.long)
    mask = ids.ne(tokenizer.pad_id)
    return ids.to(device), mask.to(device)


KEEP_PREFIXES = ("base.embedding.", "base.source_position.", "base.encoder.", "head.")


def save(path, net, labels, model_config, tokenizer_sha, threshold, report):
    import torch
    state = {k: v.detach().cpu() for k, v in net.state_dict().items() if k.startswith(KEEP_PREFIXES)}
    payload = {"format": "mira-intent-1", "labels": labels, "model_config": model_config,
               "tokenizer_sha256": tokenizer_sha, "threshold": threshold, "state": state, "report": report}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


class IntentModel:
    """Loads runs/mira_intent/intent.pt; predict(text) -> (label, probability, margin)."""

    def __init__(self, path, tokenizer_path, device="cpu"):
        import torch
        from arena.model import ArenaModel, ModelConfig
        from arena.tokenization import ArenaTokenizer
        from arena.training import tokenizer_hash
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "mira-intent-1":
            raise ValueError("Not a Mira intent checkpoint")
        if payload["tokenizer_sha256"] != tokenizer_hash(tokenizer_path):
            raise ValueError("Tokenizer does not match the intent model")
        self.tokenizer = ArenaTokenizer.load(tokenizer_path)
        base = ArenaModel(ModelConfig(**payload["model_config"]))
        self.labels = payload["labels"]
        self.threshold = float(payload.get("threshold", 0.6))
        self.net = make_net(base, len(self.labels))
        missing, unexpected = self.net.load_state_dict(payload["state"], strict=False)
        if unexpected or any(k.startswith(KEEP_PREFIXES) for k in missing):
            raise ValueError("Intent checkpoint is incomplete")
        self.net.to(device).eval()
        self.device = device
        self.report = payload.get("report", {})

    def predict(self, text: str):
        import torch
        with torch.no_grad():
            ids, mask = batch(self.tokenizer, [text], self.device)
            probs = torch.softmax(self.net(ids, mask).float(), -1)[0]
            top = torch.topk(probs, 2)
        p1, p2 = float(top.values[0]), float(top.values[1])
        return self.labels[int(top.indices[0])], p1, p1 - p2


# ------------------------------------------------------------------ augmentation
FILLERS = ["", "", "", "а ", "слушай, ", "кстати, ", "скажи, ", "ну ", "мира, "]


def typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.random()
    if kind < 0.4:
        return word[:i] + word[i + 1:]                       # missing letter
    if kind < 0.7:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]  # swapped letters
    return word[:i] + word[i] + word[i:]                     # doubled letter


def augment(text: str, rng: random.Random, n: int = 4):
    """Spelling noise, fillers, punctuation; the meaning stays the team's."""
    out = {text}
    for _ in range(n * 3):
        t = rng.choice(FILLERS) + text
        if rng.random() < 0.5:
            t = re.sub(r"[?!.,]", "", t)
        if rng.random() < 0.5:
            words = t.split()
            if words:
                k = rng.randrange(len(words))
                words[k] = typo(words[k], rng)
                t = " ".join(words)
        if rng.random() < 0.3:
            t = t.lower()
        out.add(t.strip())
        if len(out) > n:
            break
    return list(out)
