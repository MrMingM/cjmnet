"""Explicit v2-to-budgetfix import; old contracts/results remain read-only."""
from contextlib import contextmanager
import os
from pathlib import Path

import numpy as np

from scorw_repair.core import (HERE,REPO,Blocked,read_json,digest,identity,atomic_json,atomic_bytes,transaction)
from scorw_p7.core import inventory,verify_inventory

LEGACY_SNAPSHOT="070d1d301105f22fa756244ad6f5302c2a9cfa579c5430f5370652fd44d0bff4"
STAGES={
    "coordinates":["INPUT_MANIFEST.json","CLEAN_GT.json","FIELD_DOMAINS.json","REAL_COORDINATE_AUDIT.json","COORDINATE_AUDIT.md"],
    "R0":["R0_SENSOR_AUDIT.json","R0_RESULTS.md"],
    "hahner_regression":["R1_HAHNER_REGRESSION.json","R1_HAHNER_REGRESSION.md"],
    "R1":["R1_PHYSICS_AUDIT.json","R1_RESULTS.md","R1_FIELD_POOL_FREEZE.json"]}


def verified_stage(root):
    contract=read_json(root/"CONTRACT.json");commit=read_json(root/"COMMIT.json")
    if (identity(contract["request"])!=contract["identity"] or commit.get("contract")!=contract["identity"]
            or commit.get("status")!="passed" or inventory(root)!=commit["hashes"]):
        raise Blocked("LEGACY_STAGE_INVENTORY_CHANGED: "+str(root))
    verify_inventory(root,commit["hashes"])
    return contract,commit


@contextmanager
def readonly_previous(config,mode):
    root=Path(config["reuse_previous_root"])/mode
    if mode=="preflight" or not root.exists():
        yield None;return
    # Never create or alter an old lock file. An old writer must be finished.
    with (root/".p7.lock").open("rb") as stream:
        if os.name=="posix":
            import fcntl
            try:fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:raise Blocked("OLD_RUN_STILL_ACTIVE")
        try:yield root
        finally:
            if os.name=="posix":fcntl.flock(stream,fcntl.LOCK_UN)


def audit(root,config,selection,model,runtime):
    if root is None:return None
    pin=HERE/"compat/v2_snapshot.json"
    if digest(pin)!=LEGACY_SNAPSHOT:raise Blocked("LEGACY_DELIVERY_PIN_CHANGED")
    oldpins=read_json(pin)["sources"]
    # Only orchestration, diagnostics, tests/docs and budget config may differ.
    allowed={"scorw_repair/detection.py","scorw_repair/run.py","scorw_repair/config.json",
        "scorw_repair/run_all.sh","scorw_repair/tests/test_repair.py","scorw_repair/README.md"}
    changed=[n for n,h in oldpins.items() if n not in allowed and (not (REPO/n).is_file() or digest(REPO/n)!=h)]
    if changed:raise Blocked("LEGACY_PHYSICS_OR_MODEL_SOURCE_CHANGED: "+str(changed))
    old_config=read_json(HERE/"compat/v2_config.json")
    if digest(HERE/"compat/v2_config.json")!=oldpins["scorw_repair/config.json"]:
        raise Blocked("LEGACY_CONFIG_PIN_CHANGED")
    # Resource limits and output paths have a new contract; every scientific setting stays identical.
    old_science=dict(old_config);new_science=dict(config)
    for c in (old_science,new_science):
        for key in ("revision","output_root","logs_root","reuse_previous_root"):c.pop(key,None)
        c["r2"]={k:v for k,v in c["r2"].items() if k not in {"maximum_ray_field_pairs","maximum_ray_field_pairs_per_frame"}}
    if old_science!=new_science:raise Blocked("LEGACY_SCIENTIFIC_PROTOCOL_CHANGED")
    old=read_json(root/"CONTRACT.json");request=old["request"]
    if (identity(request)!=old["identity"] or request["source_identity"]!=identity(oldpins)
            or request["snapshot"]!=LEGACY_SNAPSHOT or request["config"]!=old_config
            or request["opv2v_selection"]!=selection or request["model"]!=model or request["runtime"]!=runtime):
        raise Blocked("LEGACY_CONTRACT_IDENTITY_MISMATCH")
    gates=read_json(root/"VALIDATION_GATES.json")
    if gates.get("contract")!=old["identity"]:raise Blocked("LEGACY_GATE_CONTRACT_MISMATCH")
    completed=gates.get("status")=="SMOKE_UNCALIBRATED_EXPLORATORY_COMPLETE"
    budget_stop=(gates.get("status")=="blocked" and gates.get("blocked_stage")=="R2"
        and gates.get("reason")=="R2_COMPUTE_BUDGET_EXCEEDED_BEFORE_RESPONSE_GENERATION")
    if not (completed or budget_stop):raise Blocked("LEGACY_RUN_NOT_AN_APPROVED_BUDGET_RECOVERY")
    report_manifest=read_json(root/"REPORT_MANIFEST.json")
    for name,row in report_manifest.items():
        if Path(name).name!=name or not (root/name).is_file() or digest(root/name)!=row["sha256"]:
            raise Blocked("LEGACY_PUBLISHED_REPORT_CHANGED: "+name)
    records={}
    for name in STAGES:
        stage=root/"stages"/name
        contract,commit=verified_stage(stage)
        if contract["request"]!=dict(parent=old["identity"],stage=name):raise Blocked("LEGACY_STAGE_PARENT_CHANGED")
        if gates["stages"][name].get("commit_sha256")!=digest(stage/"COMMIT.json"):
            raise Blocked("LEGACY_GATE_COMMIT_CHANGED")
        records[name]=dict(root=str(stage),commit_sha256=digest(stage/"COMMIT.json"))
    r2=read_json(root/"stages/R2/CONTRACT.json")
    if r2["request"]!=dict(parent=old["identity"],stage="R2") or r2["identity"]!=identity(r2["request"]):
        raise Blocked("LEGACY_R2_PARENT_CHANGED")
    return dict(root=str(root),contract=old["identity"],r2_contract=r2["identity"],stages=records,
        report_manifest_sha256=digest(root/"REPORT_MANIFEST.json"),snapshot_sha256=LEGACY_SNAPSHOT,
        policy="new contract; read-only verified upstream reports and committed per-ray response caches only; old detector predictions are not reused")


def verify(info):
    if info is None:return
    root=Path(info["root"])
    if (digest(root/"REPORT_MANIFEST.json")!=info["report_manifest_sha256"]
            or read_json(root/"CONTRACT.json")["identity"]!=info["contract"]):
        raise Blocked("LEGACY_RUN_CHANGED_DURING_RECOVERY")
    for row in info["stages"].values():
        path=Path(row["root"])
        if digest(path/"COMMIT.json")!=row["commit_sha256"]:raise Blocked("LEGACY_COMMIT_CHANGED_DURING_RECOVERY")
        verified_stage(path)


def import_stage(info,name,target):
    if info is None or name not in STAGES:return False
    source=Path(info["stages"][name]["root"])
    verified_stage(source)
    for filename in STAGES[name]:atomic_bytes(target/filename,(source/filename).read_bytes())
    atomic_json(target/"LEGACY_IMPORT.json",dict(source=str(source),commit_sha256=digest(source/"COMMIT.json"),
        source_contract=info["contract"],status="verified_read_only_import",executed_in_current_run=False))
    print("REUSE completed "+name+": "+str(source),flush=True)
    return True


def cached(raw,pose,field,config,key,indices,path,parent,info,relative):
    """Adopt a verified numerical array into a NEW transaction, never its old contract."""
    from scorw_repair.generate import cached_response
    from scorw_repair.response import validate
    old=None if info is None else Path(info["root"])/"stages/R2/cloud_cache"/relative
    if old is not None and (old/"COMMIT.json").is_file() and not (path/"COMMIT.json").exists():
        request=dict(parent=parent,field=field.manifest(),indices=indices.tolist(),receiver=config["receiver"])
        old_contract,_=verified_stage(old)
        expected=dict(request,parent=info["r2_contract"])
        if old_contract["request"]!=expected:raise Blocked("LEGACY_RESPONSE_PARAMETERS_CHANGED")
        with np.load(old/"response.npz",allow_pickle=False) as a:
            data={k:a[k] for k in a.files}
        validate(data)
        expected_ids=np.asarray([identity([*key,int(i)]) for i in indices])
        if (not np.array_equal(data["original_xyzi"],raw) or not np.array_equal(data["original_index"],indices)
                or not np.array_equal(data["point_identity"],expected_ids) or data["frame_identity"].tolist()!=key):
            raise Blocked("LEGACY_RESPONSE_ORIGINAL_IDENTITY_CHANGED")
        # Bound old pose to the immutable coordinate audit (the old cache request did not store it).
        rows=read_json(Path(info["stages"]["coordinates"]["root"])/"INPUT_MANIFEST.json")
        row=next(r for r in rows if [r["scene"],r["timestamp"],r["cav"],r["pcd_sha256"]]==key)
        if row["pose"]!=pose:raise Blocked("LEGACY_RESPONSE_POSE_CHANGED")
        def copy(code):
            for filename in ("response.npz","STATS.json"):
                atomic_bytes(path/filename,(old/filename).read_bytes())
            atomic_json(path/"LEGACY_RESPONSE.json",dict(source=str(old),old_contract=old_contract["identity"],
                response_sha256=digest(old/"response.npz"),status="verified_response_import"))
        transaction(path,request,copy)
        print("REUSE response: "+str(relative),flush=True)
    return cached_response(raw,pose,field,config,key,indices,path,parent)
