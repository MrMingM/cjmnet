"""Strict identities and isolated stage transactions. No Git, models or data IO."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import statistics

from shared_weather_benchmark.provenance import (
    atomic_bytes, atomic_json, atomic_text, digest, identity, seed_for,
    inspect_sources, verify_sources, install_runtime_source_guard)
from shared_weather_benchmark.run import lock_output

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LEGACY_CONTRACT = "ad3c9bd3a5c566d0d1c4e65decf3193d311f972f99e35af7dd98526fa4a40083"


def finite_tree(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite value: " + path)
    if isinstance(value, dict):
        for k, v in value.items():
            finite_tree(v, path + "/" + str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            finite_tree(v, path + "/" + str(i))
    return value


def read_json(path):
    return finite_tree(json.loads(Path(path).read_text("utf-8")))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Empty JSONL record: " + str(path))
            yield finite_tree(json.loads(line))


def mean(values):
    values = [v for v in values if v is not None]
    finite_tree(values)
    return statistics.fmean(values) if values else None


def stats(values):
    values = list(values)
    valid = [v for v in values if v is not None]
    return dict(total=len(values), valid=len(valid), missing=len(values)-len(valid),
                mean=mean(valid), median=statistics.median(valid) if valid else None,
                minimum=min(valid) if valid else None, maximum=max(valid) if valid else None,
                std=statistics.pstdev(valid) if valid else None)


def safe_child(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest path escapes root: " + str(name))
    path = root / relative
    if root not in path.resolve().parents:
        raise ValueError("Manifest link escapes root: " + str(path))
    return path


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        path = safe_child(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Missing/changed committed artifact: " + str(path))


def ensure_json(path, value):
    """Idempotent commit; never replace a different prior identity."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Existing frozen JSON differs: " + str(path))
    else:
        atomic_json(path, value)


def begin(root, request, resume):
    root = Path(root)
    contract = dict(identity=identity(request), request=request)
    path = root / "CONTRACT.json"
    if path.exists():
        if read_json(path) != contract:
            raise ValueError("Contract changed; use a new output directory")
        if not resume:
            raise ValueError("Existing run requires RESUME=1")
    else:
        if any(p.name != ".scor-w.lock" for p in root.iterdir()):
            raise ValueError("Unrelated nonempty output directory")
        atomic_json(path, contract)
    return contract["identity"]


def assert_close(a, b, absolute=1e-10, relative=1e-8, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError("Schema mismatch: " + path)
        for key in a:
            assert_close(a[key], b[key], absolute, relative, path+"/"+str(key))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("Length mismatch: " + path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, absolute, relative, path+"/"+str(i))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if isinstance(a, int) and isinstance(b, int):
            ok = a == b
        else:
            ok = math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=absolute, rel_tol=relative)
        if not ok:
            raise ValueError(f"Recomputed value differs: {path}: {a} / {b}")
    elif a != b:
        raise ValueError("Recomputed identity/value differs: " + path)


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in rows))


def write_csv(path, rows):
    rows = list(rows)
    stream = io.StringIO(newline="")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    atomic_text(path, stream.getvalue())


def verify_code():
    legacy = inspect_sources()
    verify_sources(legacy)
    snapshot = read_json(HERE / "source_snapshot.json")
    actual = {name: digest(REPO/name) for name in snapshot["sources"]}
    if actual != snapshot["sources"]:
        changed = [f"{name}: expected={snapshot['sources'][name]} actual={actual[name]}"
                   for name in actual if actual[name] != snapshot["sources"][name]]
        raise ValueError("P6 source/dependency hash mismatch; sync complete delivery:\n"+"\n".join(changed))
    inventory = {p.relative_to(REPO).as_posix() for p in HERE.rglob("*") if p.is_file()
                 and p.suffix in {".py", ".sh", ".md", ".json", ".txt"}
                 and not {"local_validation", "__pycache__"}.intersection(p.relative_to(HERE).parts)
                 and p.name != "source_snapshot.json"}
    if inventory != {n for n in actual if n.startswith(HERE.name+"/")}:
        raise ValueError("Unregistered or missing P6 source")
    install_runtime_source_guard()
    return identity(dict(p6=actual, snapshot=digest(HERE/"source_snapshot.json"), legacy=legacy["source_identity"]))


def stage_manifest(root, names):
    return {str(Path(n).relative_to(root)): digest(n) for n in names}


def verify_external(entries):
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError("Read-only input changed: " + str(path))


def bin_name(value, edges):
    if value is None:
        return "missing"
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            return f"[{low},{high})"
    raise ValueError("Value outside preregistered strata: " + str(value))
