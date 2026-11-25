"""Random-initialized, retrieval-aware sequence generator. No external weights."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_heads: int = 6
    encoder_layers: int = 4
    decoder_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_source_length: int = 512
    max_target_length: int = 96
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    copy_enabled: bool = True
    gate_loss_weight: float = 0.1

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.vocab_size, self.encoder_layers, self.decoder_layers,
               self.max_source_length, self.max_target_length) < 1:
            raise ValueError("Model dimensions must be positive")

    @classmethod
    def from_preset(cls, name="main", vocab_size=8192, **kwargs):
        presets = {
            "small": dict(d_model=256, n_heads=4, encoder_layers=4, decoder_layers=4),
            "main": dict(d_model=384, n_heads=6, encoder_layers=4, decoder_layers=4),
            "large": dict(d_model=512, n_heads=8, encoder_layers=5, decoder_layers=5),
        }
        if name not in presets:
            raise ValueError(f"Unknown model preset: {name}")
        return cls(vocab_size=vocab_size, **(presets[name] | kwargs))

    def to_dict(self):
        return asdict(self)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.self_attn = nn.MultiheadAttention(d, config.n_heads, config.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, config.n_heads, config.dropout, batch_first=True)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * config.ff_mult), nn.GELU(),
                                nn.Dropout(config.dropout), nn.Linear(d * config.ff_mult, d))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, memory, causal_mask, cross_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0])
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, attn_mask=cross_mask, need_weights=False)[0])
        return x + self.dropout(self.ff(self.norm3(x)))


class ArenaModel(nn.Module):
    """Encoder-decoder with tied vocabulary and a normalized pointer mixture.

    Encoder blocks prevent primary context positions from reading retrieved tokens.
    The learned per-example gate therefore controls their use at cross attention and
    copying, without a hidden path through primary context states.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_id)
        self.source_position = nn.Embedding(config.max_source_length, d)
        self.target_position = nn.Embedding(config.max_target_length + 1, d)
        layer = nn.TransformerEncoderLayer(d, config.n_heads, d * config.ff_mult,
                                          config.dropout, activation="gelu", batch_first=True,
                                          norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.encoder_layers,
                                             norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.decoder = nn.ModuleList(DecoderBlock(config) for _ in range(config.decoder_layers))
        self.decoder_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        self.retrieval_gate = nn.Sequential(nn.Linear(4 * d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.copy_query = nn.Linear(d, d, bias=False)
        self.copy_key = nn.Linear(d, d, bias=False)
        self.generate_gate = nn.Linear(2 * d, 1)
        self.apply(self._init_weights)
        # MultiheadAttention's packed projections are not nn.Linear modules.
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, mean=0, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        with torch.no_grad():
            self.embedding.weight[config.pad_id].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _mean(states, mask):
        weight = mask.to(states.dtype).unsqueeze(-1)
        return (states * weight).sum(1) / weight.sum(1).clamp_min(1)

    def encode(self, input_ids, attention_mask=None, retrieval_mask=None):
        b, s = input_ids.shape
        if s > self.config.max_source_length:
            raise ValueError("Source exceeds configured context")
        valid = input_ids.ne(self.config.pad_id) if attention_mask is None else attention_mask.bool()
        if not valid.any(1).all():
            raise ValueError("Every example needs at least one non-padding source token")
        retrieved = torch.zeros_like(valid) if retrieval_mask is None else retrieval_mask.bool() & valid
        # A BOS/section token in every formatted example guarantees primary context.
        primary = valid & ~retrieved
        if not primary.any(1).all():
            raise ValueError("Retrieved material cannot be the entire source")
        pos = torch.arange(s, device=input_ids.device)
        embedded = self.dropout(self.embedding(input_ids) + self.source_position(pos)[None])
        blocked = (primary[:, :, None] & retrieved[:, None, :]) | ~valid[:, None, :]
        blocked = blocked[:, None].expand(b, self.config.n_heads, s, s).reshape(b * self.config.n_heads, s, s)
        memory = self.encoder(embedded, mask=blocked)
        p, r = self._mean(memory, primary), self._mean(memory, retrieved)
        gate_logits = self.retrieval_gate(torch.cat([p, r, p * r, (p - r).abs()], -1)).squeeze(-1)
        gate = torch.sigmoid(gate_logits) * retrieved.any(1).to(gate_logits.dtype)
        return memory, valid, retrieved, gate, gate_logits

    def decode(self, decoder_input_ids, memory, valid, retrieved, gate,
               input_ids, copy_mask=None):
        b, t = decoder_input_ids.shape
        s = memory.shape[1]
        if t > self.config.max_target_length + 1:
            raise ValueError("Target exceeds configured context")
        x = self.dropout(self.embedding(decoder_input_ids) +
                         self.target_position(torch.arange(t, device=input_ids.device))[None])
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        bias = torch.zeros(b, s, dtype=x.dtype, device=x.device)
        bias = bias + retrieved.to(x.dtype) * gate.float().clamp_min(1e-6).log().to(x.dtype)[:, None]
        bias = bias.masked_fill(~valid, float("-inf"))
        cross_mask = bias[:, None, None, :].expand(b, self.config.n_heads, t, s).reshape(b * self.config.n_heads, t, s)
        for block in self.decoder:
            x = block(x, memory, causal, cross_mask)
        hidden = self.decoder_norm(x)
        # Accumulate probabilities in FP32 even under AMP.
        vocab_logits = F.linear(hidden, self.embedding.weight).float()
        vocab_probs = F.softmax(vocab_logits, dim=-1)
        allowed = valid if copy_mask is None else copy_mask.bool() & valid
        if not self.config.copy_enabled:
            allowed = torch.zeros_like(allowed)
        has_copy = allowed.any(1)
        scores = torch.bmm(self.copy_query(hidden).float(), self.copy_key(memory).float().transpose(1, 2)) / math.sqrt(self.config.d_model)
        scores = scores + retrieved.float()[:, None, :] * gate.float().clamp_min(1e-6).log()[:, None, None]
        scores = scores.masked_fill(~allowed[:, None, :], -1e9)
        pointer = F.softmax(scores, -1) * has_copy.float()[:, None, None]
        context = torch.bmm(pointer.to(memory.dtype), memory)
        p_generate = torch.sigmoid(self.generate_gate(torch.cat([hidden, context], -1)).float())
        p_generate = torch.where(has_copy[:, None, None], p_generate, torch.ones_like(p_generate))
        probs = p_generate * vocab_probs
        probs = probs.scatter_add(-1, input_ids[:, None, :].expand(b, t, s), (1 - p_generate) * pointer)
        # clamp only at log boundary: the mixed distribution remains normalized.
        return probs.clamp_min(1e-30).log(), p_generate

    def forward(self, input_ids, decoder_input_ids, attention_mask=None,
                copy_mask=None, retrieval_mask=None, labels=None, retrieval_targets=None):
        memory, valid, retrieved, gate, gate_logits = self.encode(input_ids, attention_mask, retrieval_mask)
        log_probs, p_generate = self.decode(decoder_input_ids, memory, valid, retrieved, gate, input_ids, copy_mask)
        result = {"log_probs": log_probs, "gate": gate, "p_generate": p_generate}
        if labels is not None:
            nll = F.nll_loss(log_probs.reshape(-1, self.config.vocab_size), labels.reshape(-1), ignore_index=-100)
            loss = nll
            if retrieval_targets is not None:
                known = (retrieval_targets >= 0) & retrieved.any(1)
                if known.any():
                    gate_loss = F.binary_cross_entropy_with_logits(gate_logits[known].float(), retrieval_targets[known].float())
                    loss = loss + self.config.gate_loss_weight * gate_loss
                    result["gate_loss"] = gate_loss
            result.update(loss=loss, nll=nll)
        return result

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, copy_mask=None,
                 retrieval_mask=None, max_new_tokens=None, bos_id=None, eos_id=None,
                 temperature=0.0, top_k=0, deadline_seconds: Optional[float] = None,
                 return_info=False, no_repeat_ngram_size: int = 0):
        """Decode with an optional generated-token n-gram repetition constraint.

        The constraint applies to tokenizer IDs, not whole words, and never
        examines the source. Zero preserves unconstrained decoding. It prevents
        exact loops but does not establish that an answer is relevant or fluent.
        """
        start = time.monotonic()
        if type(no_repeat_ngram_size) is not int or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
        limit = self.config.max_target_length if max_new_tokens is None else min(max_new_tokens, self.config.max_target_length)
        if limit < 1:
            raise ValueError("max_new_tokens must be positive")
        bos = self.config.bos_id if bos_id is None else bos_id
        eos = self.config.eos_id if eos_id is None else eos_id
        was_training = self.training
        self.eval()
        timed_out = False
        outputs = []
        try:
            encoded = self.encode(input_ids, attention_mask, retrieval_mask)
            memory, valid, retrieved, gate, _ = encoded
            sequence = torch.full((input_ids.shape[0], 1), bos, dtype=torch.long, device=input_ids.device)
            finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            for _ in range(limit):
                if deadline_seconds is not None and time.monotonic() - start >= deadline_seconds:
                    timed_out = True
                    break
                logp, _ = self.decode(sequence, memory, valid, retrieved, gate, input_ids, copy_mask)
                scores = logp[:, -1].clone()
                scores[:, self.config.pad_id] = -float("inf")
                scores[:, bos] = -float("inf")
                if no_repeat_ngram_size:
                    # BOS is an instruction to start decoding, not part of any
                    # generated n-gram. Keep each batch member independent.
                    n = no_repeat_ngram_size
                    for row, history in enumerate(sequence[:, 1:].tolist()):
                        if len(history) + 1 < n:
                            continue
                        prefix = history[-(n - 1):] if n > 1 else []
                        banned = {history[index + n - 1]
                                  for index in range(len(history) - n + 1)
                                  if history[index:index + n - 1] == prefix}
                        if banned:
                            scores[row, list(banned)] = -float("inf")
                # An exhausted candidate set must terminate rather than choose
                # padding in argmax or feed NaNs to multinomial sampling.
                scores = scores.masked_fill(~torch.isfinite(scores), -float("inf"))
                exhausted = ~torch.isfinite(scores).any(dim=-1)
                scores[exhausted, eos] = 0.0
                if temperature > 0:
                    scores = scores / temperature
                    if top_k > 0:
                        cutoff = torch.topk(scores, min(top_k, scores.shape[-1]), -1).values[:, -1:]
                        scores = scores.masked_fill(scores < cutoff, -float("inf"))
                    token = torch.multinomial(F.softmax(scores, -1), 1).squeeze(1)
                else:
                    token = scores.argmax(-1)
                token = torch.where(finished, torch.full_like(token, eos), token)
                outputs.append(token)
                finished |= token.eq(eos)
                sequence = torch.cat([sequence, token[:, None]], 1)
                if finished.all():
                    break
            tokens = torch.stack(outputs, 1) if outputs else input_ids.new_empty((input_ids.shape[0], 0))
            if return_info:
                return {"tokens": tokens, "elapsed_seconds": time.monotonic() - start,
                        "deadline_reached": timed_out, "gate": gate.detach()}
            return tokens
        finally:
            self.train(was_training)
