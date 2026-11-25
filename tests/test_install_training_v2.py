import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("installer", Path(__file__).resolve().parents[1] / "scripts/install_training_v2.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def package(tmp_path):
    source, target = tmp_path / "update", tmp_path / "project"
    (source / "arena").mkdir(parents=True)
    (target / "arena").mkdir(parents=True)
    (source / "arena/training.py").write_text("new", encoding="utf-8")
    (target / "arena/training.py").write_text("old", encoding="utf-8")
    (target / "base.txt").write_text("frozen", encoding="utf-8")
    (target / "runs").mkdir()
    (target / "runs/best.pt").write_bytes(b"original weights")
    manifest = {"package_kind": "training_v2_update", "required_base_inputs": [{"path": "base.txt", "sha256": hashlib.sha256(b"frozen").hexdigest()}],
                "files": [{"path": "arena/training.py", "bytes": 3, "sha256": hashlib.sha256(b"new").hexdigest()}]}
    (source / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source, target


def test_install_checks_backups_and_preserves_weights(tmp_path):
    source, target = package(tmp_path)
    checked = installer.install(source, target, check_only=True)
    assert checked["files_to_install"] == checked["files_to_back_up"] == 1
    assert (target / "arena/training.py").read_text() == "old"
    installed = installer.install(source, target)
    assert (Path(installed["backup"]) / "arena/training.py").read_text() == "old"
    assert (target / "arena/training.py").read_text() == "new"
    assert (target / "runs/best.pt").read_bytes() == b"original weights"
    assert installer.install(source, target)["installed"] == 0


def test_install_rejects_wrong_base_before_changing_code(tmp_path):
    source, target = package(tmp_path)
    (target / "base.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Required original"):
        installer.install(source, target)
    assert (target / "arena/training.py").read_text() == "old"


@pytest.mark.parametrize("name", ["../outside", "C:/outside", "runs/best.pt", "Runs/best.PT", "./runs/best.pt"])
def test_install_rejects_unsafe_manifest_paths(tmp_path, name):
    source, target = package(tmp_path)
    path = source / "bundle_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0]["path"] = name
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        installer.install(source, target)
    assert (target / "arena/training.py").read_text() == "old"


def test_atomic_install_retries_transient_lock(tmp_path, monkeypatch):
    source, target = package(tmp_path)
    src, dst = source / "arena/training.py", target / "arena/training.py"
    real_replace = installer.os.replace
    attempts, sleeps = [], []
    def replace(temporary, destination):
        attempts.append(temporary)
        if len(attempts) < 3:
            error = PermissionError("locked")
            error.winerror = 32
            raise error
        real_replace(temporary, destination)
    monkeypatch.setattr(installer.os, "replace", replace)
    monkeypatch.setattr(installer.time, "sleep", sleeps.append)
    installer.atomic_install(src, dst, installer.digest(src))
    assert dst.read_text() == "new" and len(set(attempts)) == 1 and sleeps == [.05, .1]


def test_atomic_install_retains_verified_copy_after_persistent_lock(tmp_path, monkeypatch):
    source, target = package(tmp_path)
    src, dst = source / "arena/training.py", target / "arena/training.py"
    error = PermissionError("locked")
    error.winerror = 33
    def replace(*args):
        raise error
    monkeypatch.setattr(installer.os, "replace", replace)
    monkeypatch.setattr(installer.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError) as caught:
        installer.atomic_install(src, dst, installer.digest(src))
    assert caught.value is error and dst.read_text() == "old"
    copies = list(dst.parent.glob("*.install.tmp"))
    assert len(copies) == 1 and copies[0].read_text() == "new"
    assert str(copies[0]) in error.__notes__[0]


def test_atomic_install_cleanup_preserves_original_copy_error(tmp_path, monkeypatch):
    source, target = package(tmp_path)
    src, dst = source / "arena/training.py", target / "arena/training.py"
    original = OSError("disk full")
    def fail_copy(*args):
        raise original
    def fail_cleanup(*args, **kwargs):
        raise PermissionError("locked cleanup")
    monkeypatch.setattr(installer.shutil, "copyfile", fail_copy)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(OSError) as caught:
        installer.atomic_install(src, dst, installer.digest(src))
    assert caught.value is original and dst.read_text() == "old" and original.__notes__
