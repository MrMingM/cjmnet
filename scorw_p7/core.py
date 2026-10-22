"""Small immutable transactions and data guards; no Git or research data imports."""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import platform

from shared_weather_benchmark.provenance import atomic_json, atomic_text, atomic_bytes, digest, identity, seed_for

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def read_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    value = json.loads(Path(path).read_text("utf-8"), parse_constant=reject)
    def finite(data):
        if isinstance(data,float) and not math.isfinite(data):
            raise ValueError("Nonfinite JSON number")
        if isinstance(data,dict):
            for v in data.values():
                finite(v)
        if isinstance(data,list):
            for v in data:
                finite(v)
    finite(value)
    return value


def ensure_json(path, data):
    path = Path(path)
    if path.exists():
        if read_json(path) != data:
            raise ValueError("Frozen identity changed: " + str(path))
    else:
        atomic_json(path, data)


def child(root, name):
    root = Path(root).resolve()
    rel = Path(name)
    p = root / rel
    if rel.is_absolute() or ".." in rel.parts or root not in p.resolve().parents:
        raise ValueError("Artifact path escapes root: " + str(name))
    return p


@contextmanager
def lock(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Output lock directory cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".p7.lock").open("a+b") as stream:
        if os.name == "posix":
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            stream.seek(0); stream.write(b"0"); stream.flush(); stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(stream, fcntl.LOCK_UN)
            else:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def begin(root, request, resume):
    root = Path(root)
    data = dict(identity=identity(request), request=request)
    path = root / "CONTRACT.json"
    if path.exists():
        if read_json(path) != data:
            raise ValueError("Source/config/runtime/input contract changed; old output cannot be reused")
        if not resume:
            raise ValueError("Existing run requires RESUME=1")
    else:
        if any(p.name != ".p7.lock" for p in root.iterdir()):
            raise ValueError("Unrelated nonempty output directory")
        atomic_json(path, data)
    return data["identity"]


def inventory(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != ".p7.lock" and p != root/"COMMIT.json" and not p.name.endswith(".partial")}


def verify_inventory(root, hashes):
    for name, expected in hashes.items():
        p = child(root, name)
        if not p.is_file() or digest(p) != expected:
            raise ValueError("Committed artifact missing/changed: " + str(p))


def transaction(root, request, callback):
    root = Path(root)
    with lock(root):
        code = begin(root, request, (root / "CONTRACT.json").exists())
        commit = root / "COMMIT.json"
        if commit.exists():
            c = read_json(commit)
            if c.get("contract") != code or c.get("status") != "passed" or inventory(root) != c["hashes"]:
                raise ValueError("Stage inventory/contract changed: " + str(root))
            verify_inventory(root, c["hashes"])
            return c
        callback(code)
        c = dict(status="passed", contract=code, hashes=inventory(root))
        atomic_json(commit, c)
        return c


def save_arrays(path, **arrays):
    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def server_train(path):
    if platform.system() != "Linux":
        raise RuntimeError("Real data are Linux-server only")
    from shared_weather_benchmark.dataset_io import approved_path
    root = approved_path(path)
    expected = Path("/data/scd/datasets/opv2v_official_data_dumping/train").resolve(strict=True)
    if root != expected:
        raise ValueError("Only original OPV2V train is permitted; test/validation and aliases forbidden")
    return root


def close(a, b, tolerance):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(
            float(a), float(b), abs_tol=tolerance["absolute"], rel_tol=tolerance["relative"]):
        raise ValueError(f"Strict numerical equality gate failed: {a} / {b}")
