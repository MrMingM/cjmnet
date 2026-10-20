"""Content identities, durable atomic commits, and fail-closed recovery; no Git."""
from __future__ import annotations

import hashlib
import ast
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# This file was read to understand the historical loader, but SCOR-W reads
# original YAML/PCD through dataset_io.py. Never execute the unreviewed server
# variant. All other previously pinned dependencies retain their strict gate.
REVIEWED_ONLY = {"opencood/data_utils/datasets/basedataset.py": "opencood.data_utils"}


def _blocked_import(name):
    return any(name == prefix or name.startswith(prefix + ".")
               for prefix in REVIEWED_ONLY.values())


class _ReviewedSourceImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if _blocked_import(fullname):
            raise RuntimeError("SCOR-W forbids executing reviewed-only dataset code: " + fullname)
        return None


def install_runtime_source_guard():
    """Refuse cached modules too: sys.meta_path alone cannot block those."""
    loaded = sorted(name for name in sys.modules if _blocked_import(name))
    if loaded:
        raise RuntimeError("Reviewed-only dataset code already imported: " + ", ".join(loaded))
    if not any(isinstance(finder, _ReviewedSourceImportGuard) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ReviewedSourceImportGuard())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False,
                                   ensure_ascii=False) + "\n").encode("utf-8"))


def atomic_text(path, value):
    atomic_bytes(path, value.encode("utf-8"))


def seed_for(seed, *keys):
    return int(identity([seed, *keys])[:16], 16)


def inspect_sources(repo=None, package=None):
    """Read-only diagnostics. Format hints never relax the exact-byte gate."""
    repo = Path(repo) if repo is not None else REPO
    package = Path(package) if package is not None else HERE
    snapshot = read_json(package / "source_snapshot.json")
    actual, changed, missing, unreadable = {}, [], [], []
    reviewed, reviewed_actual, policy_errors = [], {}, []
    for name, expected in sorted(snapshot.get("reviewed_only_sources", {}).items()):
        if name not in REVIEWED_ONLY or name in snapshot["sources"]:
            policy_errors.append("Invalid reviewed-only source classification: " + name)
            continue
        path = repo / name
        row = dict(path=name, reviewed_sha256=expected, actual_sha256=None,
                   blocked_import_prefix=REVIEWED_ONLY[name],
                   role="historical review only; forbidden at runtime")
        if not path.is_file():
            missing.append(name)
            row["status"] = "missing"
        else:
            try:
                row["actual_sha256"] = digest(path)
                reviewed_actual[name] = row["actual_sha256"]
                row["status"] = ("matches_reviewed" if row["actual_sha256"] == expected
                                 else "different_not_executed")
            except OSError as error:
                unreadable.append(dict(path=name,error=str(error)))
                row["status"] = "unreadable"
        reviewed.append(row)
    for name, expected in sorted(snapshot["sources"].items()):
        path = repo / name
        if not path.is_file():
            missing.append(name)
            continue
        try:
            observed = digest(path)
            actual[name] = observed
            if observed == expected:
                continue
            hint = "content_differs"
            content = path.read_bytes()
            if hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest() == expected:
                hint = "CRLF_conversion_only; exact-byte gate still fails"
            elif (content.startswith(b"\xef\xbb\xbf") and
                  hashlib.sha256(content[3:].replace(b"\r\n", b"\n")).hexdigest() == expected):
                hint = "UTF8_BOM_or_BOM_plus_CRLF_only; exact-byte gate still fails"
            changed.append(dict(path=name, expected_sha256=expected,
                                actual_sha256=observed, hint=hint))
        except OSError as error:
            unreadable.append(dict(path=name,error=str(error)))
    files = {p.relative_to(repo).as_posix() for p in package.rglob("*")
             if p.is_file() and p.suffix in {".py", ".sh", ".json", ".md", ".txt"}
             and "__pycache__" not in p.relative_to(package).parts and p.name != "source_snapshot.json"
             and "local_validation" not in p.relative_to(package).parts}
    prefix = package.relative_to(repo).as_posix() + "/"
    expected_files = {n for n in snapshot["sources"] if n.startswith(prefix)}
    unexpected = sorted(files - expected_files)
    missing_registered = sorted(expected_files - files)
    # Catch accidental direct imports before P0; the runtime finder also blocks
    # indirect/dynamic imports, including attempts through external libraries.
    for name in sorted(files):
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse((repo / name).read_text("utf-8"), filename=name)
            for node in ast.walk(tree):
                imported = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                            [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
                if any(_blocked_import(n) for n in imported):
                    policy_errors.append(f"Forbidden dataset import: {name}:{node.lineno}")
        except (OSError, SyntaxError, UnicodeError) as error:
            policy_errors.append(f"Cannot inspect imports: {name}: {error}")
    passed = not (changed or missing or unreadable or unexpected or missing_registered or policy_errors)
    return dict(status="passed" if passed else "failed", expected_files=len(snapshot["sources"]),
                matched_files=sum(actual.get(n) == h for n,h in snapshot["sources"].items()),
                changed=changed, missing=missing, unreadable=unreadable,
                unexpected=unexpected, missing_registered=missing_registered,
                reviewed_only=reviewed,policy_errors=policy_errors,
                runtime_guard=dict(blocked_import_prefixes=sorted(set(REVIEWED_ONLY.values())),
                                   required_by="run.execute before any stage or dataset loading"),
                source_identity=identity(dict(strict_sources=actual,reviewed_only_sources=reviewed_actual,
                    snapshot_sha256=digest(package / "source_snapshot.json"))) if passed else None,
                policy="strict original-byte SHA256 for pinned sources; reviewed-only actual bytes recorded, imports forbidden; no file modified")


def source_diagnostic_text(report):
    lines = [f"Source check: {report['status']}; {report['matched_files']}/{report['expected_files']} files match."]
    for row in report["changed"]:
        lines.extend(["CHANGED " + row["path"],
                      "  expected=" + row["expected_sha256"],
                      "  actual=" + row["actual_sha256"], "  hint=" + row["hint"]])
    lines.extend("MISSING " + name for name in report["missing"])
    lines.extend("UNREADABLE " + row["path"] + ": " + row["error"] for row in report["unreadable"])
    lines.extend("UNREGISTERED " + name for name in report["unexpected"])
    lines.extend("POLICY ERROR " + error for error in report["policy_errors"])
    for row in report["reviewed_only"]:
        if row["status"] != "matches_reviewed":
            lines.extend(["REVIEWED ONLY " + row["path"] + " (" + row["status"] + ")",
                          "  reviewed=" + row["reviewed_sha256"],
                          "  actual=" + str(row["actual_sha256"]),
                          "  runtime import forbidden: " + row["blocked_import_prefix"]])
    if report["status"] != "passed":
        lines.append("No gate was bypassed. Inspect the listed files; do not rewrite the snapshot on the server.")
    return "\n".join(lines)


def verify_sources(report=None):
    report = inspect_sources() if report is None else report
    if report["status"] != "passed":
        raise ValueError("Source hash mismatch.\n" + source_diagnostic_text(report))
    return report["source_identity"]


def check_hashes(root, hashes):
    for name, expected in hashes.items():
        path = Path(root) / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Committed file missing/changed: " + str(path))


def point_identity(scene, stamp, cav, source_sha256, index):
    """PCD decoder order, before shuffle/filter, bound to original bytes."""
    return identity([scene, stamp, cav, source_sha256, int(index)])
