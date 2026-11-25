"""Install the reviewed update into an existing project, preserving checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import uuid


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def within(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe package path: {name}")
    path = root.joinpath(*relative.parts).resolve()
    path.relative_to(root)
    return path


def atomic_install(src, dst, expected_digest):
    fd, temporary_name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".install.tmp", dir=dst.parent)
    os.close(fd)
    temporary, complete = Path(temporary_name), False
    try:
        shutil.copyfile(src, temporary)
        if digest(temporary) != expected_digest:
            raise IOError(f"Installed copy checksum mismatch: {src}")
        complete = True
        delays = (.05, .1, .2, .4, .8, 1.6)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(temporary, dst)
                return
            except OSError as error:
                if getattr(error, "winerror", None) not in (32, 33) or attempt == len(delays):
                    raise
                time.sleep(delays[attempt])
    except BaseException as error:
        if complete:
            error.add_note(f"Previous destination was not replaced. Verified copy retained: {temporary}")
        else:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(f"Incomplete install copy could not be removed: {temporary}: {cleanup_error}")
        raise


def install(source, target, check_only=False):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Extract the update into a separate folder first")
    if not (target / "arena/training.py").is_file():
        raise ValueError("Target is not the existing turing-arena project")
    manifest = json.loads((source / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("package_kind") != "training_v2_update":
        raise ValueError("This is not a training_v2 update package")
    for item in manifest["required_base_inputs"]:
        path = within(target, item["path"])
        if not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError(f"Required original file differs or is missing: {item['path']}")
    planned, backups = [], []
    names = set()
    for item in manifest["files"]:
        name = item["path"]
        src, dst = within(source, name), within(target, name)
        canonical = PurePosixPath(name).as_posix().casefold()
        destination_name = dst.relative_to(target).as_posix().casefold()
        if destination_name in names or any(
            value.startswith("runs/") or value.endswith((".pt", ".tmp", ".part"))
            for value in (canonical, destination_name)
        ):
            raise ValueError(f"Unexpected package member: {name}")
        names.add(destination_name)
        if not src.is_file() or src.stat().st_size != item["bytes"] or digest(src) != item["sha256"]:
            raise ValueError(f"Extracted package file is damaged: {name}")
        if dst.exists():
            if not dst.is_file():
                raise ValueError(f"Destination is not a file: {name}")
            if digest(dst) == item["sha256"]:
                continue
            if any(value.startswith(("data/", "artifacts/training_v2/")) for value in (canonical, destination_name)):
                raise ValueError(f"New prepared data already exists with different contents: {name}. Preserve it and inspect before installing.")
            backups.append((dst, name))
        planned.append((src, dst, item))
    needed = sum(item["bytes"] for _, _, item in planned) + sum(path.stat().st_size for path, _ in backups)
    if shutil.disk_usage(target).free < needed + 1024 ** 3:
        raise OSError(f"Need at least {needed / 1024**3 + 1:.2f} GiB free for installation and backups")
    if check_only:
        return {"checked": True, "files_to_install": len(planned), "files_to_back_up": len(backups), "weights_unchanged": True}
    backup = target / "backups" / ("training_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    backup.mkdir(parents=True, exist_ok=False)
    # Finish all old-file backups before changing any destination.
    for path, name in backups:
        saved = within(backup, name)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
        if digest(path) != digest(saved):
            raise IOError(f"Backup verification failed: {name}")
    shutil.copy2(source / "bundle_manifest.json", backup / "installed_bundle_manifest.json")
    for number, (src, dst, item) in enumerate(planned, 1):
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            atomic_install(src, dst, item["sha256"])
        except BaseException as error:
            error.add_note(f"Installation is incomplete. Old files are backed up in {backup}. "
                           "Resolve the reported file problem and rerun this installer to finish; do not train with a partial installation.")
            raise
        if number % 20 == 0 or number == len(planned):
            print(f"Installed {number}/{len(planned)} files", flush=True)
    return {"installed": len(planned), "backup": str(backup), "weights_unchanged": True,
            "next": "Run scripts/train_v2_lenovo.ps1 -CheckOnly, then the same script without -CheckOnly"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install(Path(__file__).resolve().parents[1], args.target, args.check_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
