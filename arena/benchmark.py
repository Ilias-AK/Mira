"""Benchmark the actual copy generator; synthetic IDs measure capacity, not language quality."""

import argparse
import contextlib
import gc
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from arena.model import ArenaModel, ModelConfig


def benchmark_size(name, args):
    cuda = not args.cpu_smoke
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(2026)
    config = ModelConfig.from_preset(name)
    if args.cpu_smoke:
        config = ModelConfig(vocab_size=512, d_model=32, n_heads=4,
            encoder_layers=1, decoder_layers=1, max_source_length=32, max_target_length=8)
    model = ArenaModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else None
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    batch_size = 1 if args.cpu_smoke else args.batch_size
    source_length, target_length = config.max_source_length, config.max_target_length
    src = torch.randint(14, config.vocab_size, (batch_size, source_length), device=device)
    labels = torch.randint(14, config.vocab_size, (batch_size, target_length), device=device)
    labels[:, ::2] = src[:, :target_length:2]
    retrieved = torch.zeros_like(src, dtype=torch.bool)
    retrieved[:, -source_length // 4:] = True
    inputs = {"input_ids": src, "attention_mask": torch.ones_like(retrieved),
              "copy_mask": torch.ones_like(retrieved), "retrieval_mask": retrieved}
    decoder = torch.cat([torch.full((batch_size, 1), config.bos_id, device=device), labels[:, :-1]], 1)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def amp():
        return torch.autocast("cuda", dtype=dtype) if cuda else contextlib.nullcontext()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with amp():
            loss = model(**inputs, decoder_input_ids=decoder, labels=labels)["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        return loss

    model.train()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for _ in range(1 if args.cpu_smoke else 10):
        step()
    sync()
    start = time.monotonic()
    timings = []
    budget = .2 if args.cpu_smoke else args.minutes_per_size * 60
    report_at = 60
    while not timings or time.monotonic() - start < budget:
        tick = time.monotonic()
        loss = step()
        sync()
        timings.append(time.monotonic() - tick)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite benchmark loss")
        elapsed = time.monotonic() - start
        if elapsed >= report_at:
            print(f"{name}: {elapsed:.0f}s, {len(timings)} updates", flush=True)
            report_at += 60
    elapsed = time.monotonic() - start
    each = batch_size * (source_length + target_length)
    quarter = max(1, len(timings) // 4)
    result = {"preset": name, "parameters": model.parameter_count(), "precision": str(dtype),
        "source_length": source_length, "target_length": target_length, "batch_size": batch_size,
        "seconds": elapsed, "steps": len(timings),
        "all_tokens_per_second": len(timings) * each / elapsed,
        "target_tokens_per_second": len(timings) * batch_size * target_length / elapsed,
        "early_all_tokens_per_second": each / statistics.mean(timings[:quarter]),
        "late_all_tokens_per_second": each / statistics.mean(timings[-quarter:]),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if cuda else None}
    del optimizer, scaler
    model.zero_grad(set_to_none=True)
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    model.eval()
    result["generation"] = []
    with torch.inference_mode():
        for candidates in ([1] if args.cpu_smoke else [1, 2]):
            query = {key: value[:1].expand(candidates, -1).contiguous() for key, value in inputs.items()}
            latencies = []
            for _ in range(1 if args.cpu_smoke else args.decode_runs):
                sync()
                tick = time.monotonic()
                with amp():
                    # eos=-1 deliberately forces the complete token budget.
                    model.generate(**query, max_new_tokens=target_length, eos_id=-1)
                sync()
                latencies.append(time.monotonic() - tick)
            latencies.sort()
            result["generation"].append({"candidates": candidates, "tokens_each": target_length,
                "samples": len(latencies), "seconds_p50": statistics.median(latencies),
                "seconds_p95": latencies[min(len(latencies) - 1, int(.95 * len(latencies)))]})
    return result


def main():
    from arena.console import configure_utf8
    configure_utf8()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", nargs="+", choices=["small", "main", "large"], default=["small", "main", "large"])
    p.add_argument("--minutes-per-size", type=float, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--decode-runs", type=int, default=10)
    p.add_argument("--cpu-smoke", action="store_true")
    p.add_argument("--output", default="artifacts/lenovo_benchmark.json")
    args = p.parse_args()
    if args.minutes_per_size <= 0 or args.batch_size <= 0 or args.decode_runs <= 0:
        p.error("Durations and counts must be positive")
    if not args.cpu_smoke and not torch.cuda.is_available():
        p.error("CUDA unavailable. Run this on the Lenovo; --cpu-smoke verifies code only.")
    if args.cpu_smoke:
        torch.set_num_threads(2)
        args.sizes = ["small"]
    report = {"mode": "cpu_smoke_not_gpu_measurement" if args.cpu_smoke else "cuda_capacity",
        "torch": torch.__version__, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name() if not args.cpu_smoke else None,
        "note": "Random tokens, full sequences. Real loader speed and language quality must be measured separately.",
        "results": []}
    for name in args.sizes:
        try:
            result = benchmark_size(name, args)
        except torch.cuda.OutOfMemoryError:
            result = {"preset": name, "error": "CUDA OOM; retry with --batch-size 2 or 1"}
        report["results"].append(result)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
