"""Score a saved language checkpoint on the same full-span validation as v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .formatting import collate_examples
from .training import (EpochBatchSampler, IndexedJsonlDataset, evaluate,
                       file_sha256, load_model_checkpoint, tokenizer_hash)


def review(checkpoint, data, tokenizer_path, *, device="cpu", max_batches=500,
           batch_size=4, seed=42):
    if max_batches < 1 or batch_size < 1:
        raise ValueError("Validation batch size and budget must be positive")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    model, tokenizer, payload = load_model_checkpoint(checkpoint, device, tokenizer_path)
    stage = payload.get("training_config", {}).get("stage")
    if stage != "language":
        raise ValueError("Expected a language-stage checkpoint")
    state = payload.get("training_state", {})
    del payload
    config = model.config
    dataset = IndexedJsonlDataset(data, tokenizer, stage="language", split="validation", seed=seed,
                                   max_source_length=config.max_source_length,
                                   max_target_length=config.max_target_length,
                                   language_window_policy="full-span")
    try:
        sampler = EpochBatchSampler(len(dataset), batch_size, seed + 100000, shuffle=True)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                            collate_fn=lambda rows: collate_examples(rows, tokenizer),
                            generator=torch.Generator().manual_seed(seed))
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None
        metrics = evaluate(model, loader, device, max_batches, amp_dtype, return_diagnostics=True)
        return {"review_finished": True, "checkpoint_sha256": file_sha256(checkpoint),
                "tokenizer_sha256": tokenizer_hash(tokenizer_path), "data_sha256": file_sha256(data),
                "training_state": state, "metrics": metrics,
                "settings": {"language_window_policy": "full-span", "validation_sampling": "fixed-shuffle",
                             "seed": seed, "batch_size": batch_size, "max_batches": max_batches,
                             "max_source_length": config.max_source_length,
                             "max_target_length": config.max_target_length},
                "note": "Compare only with identical full-span validation settings. This is language NLL, not conversational quality."}
    finally:
        dataset.close()


def main(argv=None):
    from .console import configure_utf8
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(f"Preserve the existing review: {path}")
    torch.set_num_threads(4)
    report = review(args.checkpoint, args.data, args.tokenizer, device=args.device,
                    max_batches=args.max_batches, batch_size=args.batch_size, seed=args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
