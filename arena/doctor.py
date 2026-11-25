"""Read-only environment diagnostics. Never downloads models or data."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def diagnose(root=None):
    root = Path(root or Path.cwd()).resolve()
    packages = {}
    for package in ["torch", "tokenizers", "pyarrow", "requests", "numpy"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "project": str(root),
        "cpu_count": os.cpu_count(), "free_disk_gib": round(shutil.disk_usage(root).free / 2**30, 2),
        "packages": packages, "cuda_available": False, "gpus": [],
        "files": {p: (root / p).exists() for p in [
            "data/processed/train.jsonl", "data/processed/val.jsonl",
            "artifacts/tokenizer.json", "artifacts/retrieval",
        ]},
    }
    if importlib.util.find_spec("torch"):
        import torch
        result["cuda_runtime"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            result["gpus"].append({"index": i, "name": p.name,
                "vram_gib": round(p.total_memory / 2**30, 2), "compute_capability": [p.major, p.minor]})
    try:
        result["nvidia_smi"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu",
            "--format=csv,noheader"], text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).strip()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    return result


def main():
    from arena.console import configure_utf8
    configure_utf8()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = diagnose(args.root)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.require_cuda and not result["cuda_available"]:
        print("CUDA unavailable. Use the NVIDIA laptop with a CUDA-enabled PyTorch wheel and compatible driver.", file=sys.stderr)
        raise SystemExit(2)
    if any(v is None for v in result["packages"].values()):
        raise SystemExit(3)


if __name__ == "__main__":
    main()

