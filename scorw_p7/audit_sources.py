"""Exact-byte code checks and narrowly scoped historical metadata audit."""
from pathlib import Path

from scorw_p7.core import HERE, REPO, child, read_json, digest, identity


def verify_runtime_legacy(report):
    """P7 does not execute AI_CONTEXT.md. Never relax an executable/config pin."""
    blockers = {key: report.get(key,[]) for key in
                ("missing","unreadable","unexpected","missing_registered","policy_errors")}
    blockers["changed"] = [r for r in report.get("changed",[]) if r["path"] != "AI_CONTEXT.md"]
    if any(blockers.values()):
        raise ValueError("Frozen v1 executable/config identity mismatch: " + str(blockers))


def source_audit():
    from shared_weather_benchmark.provenance import inspect_sources, install_runtime_source_guard
    legacy = inspect_sources()
    verify_runtime_legacy(legacy)
    snap = read_json(HERE / "source_snapshot.json")
    errors = []
    for name, expected in snap["sources"].items():
        p = child(REPO, name)
        if not p.is_file() or digest(p) != expected:
            errors.append(name)
    actual = {p.relative_to(REPO).as_posix() for p in HERE.rglob("*") if p.is_file()
              and p.suffix in {".py", ".sh", ".json", ".md", ".txt"}
              and not {"__pycache__", "local_validation"}.intersection(p.relative_to(HERE).parts)
              and p.name != "source_snapshot.json"}
    expected = {n for n in snap["sources"] if n.startswith("scorw_p7/")}
    if errors or actual != expected:
        raise ValueError("P7 delivery mismatch; synchronize complete files, never rewrite a server snapshot: "
                         + str(dict(changed_or_missing=errors, inventory=sorted(actual ^ expected))))
    p6 = REPO / "shared_weather_benchmark_p6"
    p6_report = dict(status="absent_not_imported", missing=[], changed=[], runtime_dependency=False)
    if (p6 / "source_snapshot.json").is_file():
        pins = read_json(p6 / "source_snapshot.json")["sources"]
        missing = [n for n in pins if not (REPO / n).is_file()]
        changed = [n for n, h in pins.items() if (REPO / n).is_file() and digest(REPO/n) != h]
        p6_report.update(status="complete_matches_snapshot" if not missing and not changed else "partial_or_changed_not_imported",
                         missing=missing, changed=changed, snapshot_sha256=digest(p6/"source_snapshot.json"))
    elif p6.exists():
        p6_report.update(status="partial_snapshot_missing_not_imported", missing=["shared_weather_benchmark_p6/source_snapshot.json"])
    install_runtime_source_guard()
    context = REPO/"AI_CONTEXT.md"
    report = dict(status="passed", p7_snapshot_sha256=digest(HERE/"source_snapshot.json"),
                  strict_sources=snap["sources"], v1=legacy, p6_source_inventory=p6_report,
                  historical_report_review=snap["reviewed_reports"], git_required=False,
                  context_document=dict(role="review-only; never executed; original v1 status preserved above",
                      actual_sha256=digest(context),reviewed_sha256=snap["reviewed_reports"]["AI_CONTEXT.md"],
                      original_v1_expected_sha256=read_json(REPO/"shared_weather_benchmark/source_snapshot.json")["sources"]["AI_CONTEXT.md"]))
    runtime_id = identity(dict(strict_sources=snap["sources"],snapshot_sha256=report["p7_snapshot_sha256"],
                              blocked_reviewed_code=legacy["reviewed_only"]))
    return dict(report, source_identity=runtime_id)


def historical_metadata(config):
    """Validate only the metadata actually used; never claim a full P6 replay."""
    p5, p6 = (Path(config[k]).resolve(strict=True) for k in ("legacy_root", "p6_root"))
    artifacts = []

    def take(root, name, expected=None):
        p = child(root, name)
        actual = digest(p)
        if expected is not None and expected != actual:
            raise ValueError("Historical metadata hash mismatch: " + str(p))
        data = read_json(p)
        if digest(p) != actual:
            raise ValueError("Historical input changed during read")
        artifacts.append(dict(path=str(p), sha256=actual, bytes=p.stat().st_size))
        return data

    old_contract = take(p5, "CONTRACT.json")
    if old_contract["identity"] != config["legacy_contract"] or identity(old_contract["request"]) != config["legacy_contract"]:
        raise ValueError("P5 historical contract mismatch")
    old_gate = take(p5, "VALIDATION_GATES.json")
    if old_gate["status"] != "complete" or old_gate["contract"] != config["legacy_contract"] or set(old_gate["stages"]) != {"P0","P1","P2","P3","P4","P5"}:
        raise ValueError("P5 stage identity incomplete")
    old_registry = {}
    for stage in old_gate["stages"].values():
        if stage["status"] != "passed":
            raise ValueError("P5 historical gate failed")
        for name, sha in stage["hashes"].items():
            if name in old_registry and old_registry[name] != sha:
                raise ValueError("Conflicting P5 artifact hash")
            old_registry[name] = sha
    if "SELECTION.json" not in old_registry:
        raise ValueError("P5 selection not registered in passed gates")
    old = take(p5, "SELECTION.json", old_registry["SELECTION.json"])
    contract = take(p6, "CONTRACT.json")
    if identity(contract["request"]) != contract["identity"]:
        raise ValueError("P6 output contract is inconsistent")
    commit = take(p6, "COMMIT.json")
    gates = take(p6, "VALIDATION_GATES.json")
    if (commit["status"] != "passed" or commit["contract"] != contract["identity"] or gates["status"] != "complete"
            or any(gates["stages"].get(s, {}).get("status") != "passed" for s in ("P6-A","P6-B","P6-C"))
            or gates["stages"]["P6-C"]["commit_sha256"] != digest(p6/"COMMIT.json")):
        raise ValueError("Formal P6 not committed; SMOKE cannot supply the P7 history")
    registry = {p["path"]: p["sha256"] for p in commit["products"]}
    used = {}
    for name in ("P6C_SELECTION_FREEZE.json", "P6C_FROZEN_CONFIG.json", "SUMMARY.json"):
        expected = registry.get(str(child(p6, name)))
        if expected is None:
            raise ValueError("P6 metadata not registered: " + name)
        used[name] = take(p6, name, expected)
    freeze, frozen, summary = (used[n] for n in ("P6C_SELECTION_FREEZE.json", "P6C_FROZEN_CONFIG.json", "SUMMARY.json"))
    if (summary["status"] != "complete" or summary["smoke"] or summary["independent_scenes"] != 12
            or summary["scene_seed_pairs"] != 36 or frozen["smoke"]
            or freeze["legacy_selection_sha256"] != digest(p5/"SELECTION.json")
            or frozen["config"] != contract["request"]["config"]
            or contract["request"]["smoke"]):
        raise ValueError("Historical P6 scale/source/selection identity differs")
    if Path(frozen["config"]["train_root"]).resolve(strict=True) != Path(config["train_root"]).resolve(strict=True):
        raise ValueError("P6 and P7 original train roots differ")
    selected = freeze["selected"]
    ids = [s for values in selected.values() for s in values]
    if (set(selected) != {"generator_development","physics_validation"}
            or len(ids) != 12 or len(set(ids)) != 12 or len(freeze["frames"]) != 24
            or {f["scene"] for f in freeze["frames"]} != set(ids)
            or len({(f["scene"],f["timestamp"]) for f in freeze["frames"]}) != 24
            or any(sum(f["scene"] == s for f in freeze["frames"]) != 2 for s in ids)
            or frozen["selection"]["frames"] != freeze["frames"]
            or frozen["weather_seeds"] != freeze["weather_seeds"]
            or set(freeze["weather_seeds"]) != {"0","1","2"}
            or freeze["config_sha256"] != identity(frozen["config"])):
        raise ValueError("P6 scene/frame/seed freeze inventory is inconsistent")
    return dict(old=old, p6=freeze, p6_summary=summary, artifacts=artifacts,
                identity=identity(artifacts), audit_scope="only used historical selection/contract/summary metadata; no full P6 PCD replay",
                p6_sidecar_reuse="disabled: rebuild rays from hash-checked original Clean; no unverified sidecar cache trusted")
