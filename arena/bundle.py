"""Create and verify a portable Lenovo kit with explicit file selection."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files(root, with_data, with_cache=False):
    if with_cache and not with_data:
        raise ValueError("--with-cache requires --with-data")
    files = set()
    for name in ["README.md", "pyproject.toml", "requirements.txt"]:
        path = root / name
        if path.is_file():
            files.add(path)
    if with_cache:
        from .cache_validation import validate_cache
        cache = root / "artifacts" / "retrieval_cache.sqlite3"
        validate_cache(root / "data/processed/train.jsonl", root / "data/processed/val.jsonl",
                       root / "artifacts/retrieval", cache)
        files.add(cache)
    for directory, pattern in [("arena", "*.py"), ("tests", "*.py"), ("scripts", "*.ps1"), ("configs", "*.json")]:
        files.update((root / directory).rglob(pattern))
    for name in ["tokenizer.json", "tokenizer.meta.json", "verification.json"]:
        path = root / "artifacts" / name
        if path.is_file():
            files.add(path)
    index = root / "artifacts" / "retrieval"
    if index.is_dir():
        if (index / "index.building.sqlite3").exists():
            raise RuntimeError("Retrieval index is still being built; wait before packaging")
        files.update(index / name for name in ("index.sqlite3", "manifest.json", "reranker.json")
                     if (index / name).is_file())
    # Keep the original accompanying license/readme files, not large raw archives.
    raw = root / "data" / "raw"
    if raw.is_dir():
        files.update(path for path in raw.rglob("*") if path.is_file() and
                     ("license" in path.name.lower() or "readme" in path.name.lower()) and path.stat().st_size < 2_000_000)
    if with_data:
        for name in ("artifacts/tokenizer.json", "artifacts/retrieval/index.sqlite3",
                     "artifacts/retrieval/manifest.json", "artifacts/retrieval/reranker.json"):
            if not (root / name).is_file():
                raise FileNotFoundError(f"Training kit requires the prepared artifact: {name}")
        for name in ["train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"]:
            path = root / "data" / "processed" / name
            if not path.is_file():
                raise FileNotFoundError(f"Prepared data missing: {path}")
            files.add(path)
    for path in sorted(files):
        path.resolve().relative_to(root)
        yield path


def build_bundle(root, output, with_data=False, with_cache=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Bundle already exists; choose a new output name: {output}")
    temporary = output.with_suffix(output.suffix + ".part")
    manifest = {"format_version": 1, "includes_data": with_data,
                "includes_retrieval_cache": with_cache, "files": []}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in selected_files(root, with_data, with_cache):
            name = path.relative_to(root).as_posix()
            item = {"path": name, "bytes": path.stat().st_size, "sha256": sha_file(path)}
            archive.write(path, name)
            manifest["files"].append(item)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    os.replace(temporary, output)
    checksum = sha_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(checksum + "  " + output.name + "\n", encoding="ascii")
    return {"archive": str(output), "bytes": output.stat().st_size, "files": len(manifest["files"]), "sha256": checksum}


def verify_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        expected = {item["path"] for item in manifest["files"]} | {"bundle_manifest.json"}
        if set(names) != expected:
            raise ValueError("Unexpected or missing ZIP members")
        for item in manifest["files"]:
            member = Path(item["path"])
            if member.is_absolute() or ".." in member.parts or "\\" in item["path"]:
                raise ValueError("Unsafe ZIP path")
            digest, size = hashlib.sha256(), 0
            with archive.open(item["path"]) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Corrupt ZIP member: {item['path']}")
    return {"verified": True, "files": len(manifest["files"]), "includes_data": manifest["includes_data"],
            "includes_retrieval_cache": manifest.get("includes_retrieval_cache", False)}


def main():
    from arena.console import configure_utf8
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", default=Path(__file__).resolve().parents[1])
    build.add_argument("--output", required=True)
    build.add_argument("--with-data", action="store_true")
    build.add_argument("--with-cache", action="store_true", help="Include a complete, fully verified retrieval cache (requires --with-data)")
    verify = sub.add_parser("verify")
    verify.add_argument("path")
    args = parser.parse_args()
    result = build_bundle(args.root, args.output, args.with_data, args.with_cache) if args.command == "build" else verify_bundle(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
