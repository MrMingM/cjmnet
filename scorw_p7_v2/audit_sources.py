"""Strict independent delivery identity; no edits to any historical snapshot."""
from pathlib import Path

from scorw_p7.core import REPO, child, read_json, digest, identity
from scorw_p7.audit_sources import source_audit as original_source_audit

HERE = Path(__file__).resolve().parent


def source_audit():
    original = original_source_audit()
    snapshot = HERE/"source_snapshot.json"
    pins = read_json(snapshot)["sources"]
    errors = [name for name, sha in pins.items()
              if not child(REPO, name).is_file() or digest(child(REPO, name)) != sha]
    actual = {p.relative_to(REPO).as_posix() for p in HERE.rglob("*") if p.is_file()
              and p.suffix in {".py", ".sh", ".json", ".md", ".txt"}
              and not {"__pycache__", "local_validation"}.intersection(p.relative_to(HERE).parts)
              and p != snapshot}
    expected = {name for name in pins if name.startswith("scorw_p7_v2/")}
    if errors or actual != expected:
        raise ValueError("P7 v2 delivery mismatch; synchronize the complete directory and frozen dependencies, "
                         "never rewrite a server snapshot: " + str(dict(changed_or_missing=errors,
                         inventory=sorted(actual ^ expected))))
    code = dict(original_source_identity=original["source_identity"],
                v2_snapshot_sha256=digest(snapshot), strict_sources=pins)
    return dict(status="passed", source_identity=identity(code), **code,
                original_p7=original, git_required=False,
                policy="All historical executable/config pins preserved; v2 owns a separate protocol and output contract")
