"""Reviewed v1 -> argsfix cache adoption; never change an old run or its contract."""
from contextlib import contextmanager
from pathlib import Path
import os

from scorw_r3.core import HERE,REPO,Blocked,read_json,digest,identity,atomic_json,atomic_bytes,check_inputs
from scorw_repair.reuse import verified_stage

V1_SNAPSHOT="ef04c08d5fff5acbeb115a8b1296bcc8d8051ee77d55a036178b94b6e89a9c63"
V1_CONFIG="2c4f8f6274cc1a30d887cb7993997c2a109d4c513a52498f3667b2399104e3f1"
ALLOWED_CHANGED={"scorw_r3/runtime.py","scorw_r3/run.py","scorw_r3/run_all.sh","scorw_r3/config.json",
    "scorw_r3/README.md","scorw_r3/VALIDATION.md"}
STAGES=("R3A","selection","waves")


@contextmanager
def readonly_previous(c,mode):
    root=Path(c["reuse_r3_root"])/mode
    if mode=="preflight" or not (root/"CONTRACT.json").is_file():
        yield None;return
    with (root/".p7.lock").open("rb") as stream:
        if os.name=="posix":
            import fcntl
            try:fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:raise Blocked("R3_V1_PROCESS_STILL_ACTIVE")
        try:yield root
        finally:
            if os.name=="posix":fcntl.flock(stream,fcntl.LOCK_UN)


def compatible_config(c):
    p=HERE/"compat/r3_v1_config.json"
    if digest(p)!=V1_CONFIG:raise Blocked("R3_V1_COMPAT_CONFIG_CHANGED")
    previous=read_json(p)
    def science(value):
        return {k:v for k,v in value.items() if k not in {"revision","output_root","logs_root","reuse_r3_root"}}
    if science(c)!=science(previous):raise Blocked("R3_ARGS_FIX_SCIENTIFIC_PROTOCOL_CHANGED")
    return previous


def _reports(root,expected=None):
    manifest=root/"REPORT_MANIFEST.json"
    if expected is not None and digest(manifest)!=expected:raise Blocked("R3_V1_REPORT_MANIFEST_CHANGED")
    for name,row in read_json(manifest).items():
        if Path(name).name!=name or digest(root/name)!=row["sha256"]:raise Blocked("R3_V1_REPORT_CHANGED: "+name)


def _stage(root,name,parent):
    folder=root/"stages"/name
    contract,commit=verified_stage(folder)
    if contract["request"]!=dict(parent=parent,stage=name):raise Blocked("R3_V1_STAGE_PARENT_CHANGED: "+name)
    return folder,contract,commit


def _wave_paths(folder,commit):
    index=read_json(folder/"R3B_WAVE_INDEX.json")
    if index.get("status")!="passed" or read_json(folder/"R3B_BALANCE_CHECK.json").get("status")!="passed":
        raise Blocked("R3_V1_WAVE_GATE_NOT_PASSED")
    for row in index["responses"]:
        for key in ("response","pcd"):
            p=Path(row[key]).resolve(strict=True)
            try:relative=p.relative_to(folder.resolve(strict=True)).as_posix()
            except ValueError:raise Blocked("R3_V1_WAVE_PATH_ESCAPES_COMMITTED_STAGE")
            if commit["hashes"].get(relative)!=row[key+"_sha256"]:raise Blocked("R3_V1_WAVE_INDEX_HASH_CHANGED")


def audit(root,c,new_request):
    compatible_config(c)
    if root is None:return None
    pin=HERE/"compat/r3_v1_snapshot.json"
    if digest(pin)!=V1_SNAPSHOT:raise Blocked("R3_V1_SOURCE_SNAPSHOT_CHANGED")
    pins=read_json(pin)["sources"]
    if any(not (REPO/n).is_file() or digest(REPO/n)!=h for n,h in pins.items() if n not in ALLOWED_CHANGED):
        raise Blocked("R3_V1_SCIENTIFIC_SOURCE_CHANGED")
    _reports(root)
    parent=read_json(root/"CONTRACT.json")
    request=parent["request"]
    if identity(request)!=parent["identity"] or request["config"]!=compatible_config(c) or request["config_sha256"]!=V1_CONFIG:
        raise Blocked("R3_V1_RUN_CONTRACT_CHANGED")
    if request["source_snapshot_sha256"]!=V1_SNAPSHOT or request["sources"]!=identity(pins):
        raise Blocked("R3_V1_RUN_SOURCE_IDENTITY_CHANGED")
    source=read_json(root/"SOURCE_AUDIT.json")
    if source["files"]!=pins or source["identity"]!=identity(pins) or source["snapshot_sha256"]!=V1_SNAPSHOT:
        raise Blocked("R3_V1_SOURCE_AUDIT_CHANGED")
    excluded={"config","config_sha256","sources","source_snapshot_sha256"}
    if {k:v for k,v in request.items() if k not in excluded}!={k:v for k,v in new_request.items() if k not in excluded}:
        raise Blocked("R3_V1_RUNTIME_OR_EXTERNAL_INPUT_IDENTITY_CHANGED")
    gates=read_json(root/"VALIDATION_GATES.json")
    failure="AttributeError: 'list' object has no attribute 'get'"
    locations=gates.get("locations",[])
    if (gates.get("contract")!=parent["identity"] or gates.get("status")!="failed" or gates.get("blocked_stage")!="detector"
            or gates.get("reason")!=failure or not locations or locations[-1].get("file")!="runtime.py"
            or locations[-1].get("function")!="__init__" or (root/"stages/detector/COMMIT.json").exists()
            or (root/"stages/detector/R3B_MODEL_AUDIT.json").exists()):
        raise Blocked("R3_V1_REUSE_REQUIRES_THE_REVIEWED_PRE_INFERENCE_ARGS_FAILURE")
    records={}
    for name in STAGES:
        folder,contract,commit=_stage(root,name,parent["identity"])
        gate=gates["stages"].get(name,{})
        if gate.get("status")!="passed" or gate.get("commit_sha256")!=digest(folder/"COMMIT.json"):
            raise Blocked("R3_V1_STAGE_GATE_CHANGED: "+name)
        if name=="R3A" and read_json(folder/"R3A_GATE.json").get("status")!="PASS_WITH_DISCLOSED_APPROXIMATIONS":
            raise Blocked("R3_V1_R3A_NOT_PASSED")
        if name=="selection":
            selected=read_json(folder/"R3B_SELECTION_FREEZE.json")
            check_inputs(selected.get("consulted_inputs",[]));check_inputs(read_json(folder/"coordinates/INPUT_MANIFEST.json"))
            if read_json(folder/"R3B_COMPUTE_PLAN.json").get("status")!="passed":raise Blocked("R3_V1_BUDGET_NOT_PASSED")
        if name=="waves":_wave_paths(folder,commit)
        records[name]=dict(root=str(folder),contract=contract["identity"],commit_sha256=digest(folder/"COMMIT.json"))
    return dict(root=str(root),contract=parent["identity"],manifest_sha256=digest(root/"REPORT_MANIFEST.json"),
        stages=records,status="verified_read_only_adoption",detector_predictions_reused=False,
        numerical_tests="rerun in the corrected version",scientific_protocol_changed=False)


def verify(info):
    if info is None:return
    root=Path(info["root"]);_reports(root,info["manifest_sha256"])
    for name,row in info["stages"].items():
        folder,contract,commit=_stage(root,name,info["contract"])
        if digest(folder/"COMMIT.json")!=row["commit_sha256"] or contract["identity"]!=row["contract"]:
            raise Blocked("R3_V1_IMPORTED_STAGE_CHANGED: "+name)
        if name=="selection":
            check_inputs(read_json(folder/"coordinates/INPUT_MANIFEST.json"))
            check_inputs(read_json(folder/"R3B_SELECTION_FREEZE.json").get("consulted_inputs",[]))


def import_stage(info,name,target):
    if info is None or name not in STAGES:return False
    source=Path(info["stages"][name]["root"])
    _,commit=verified_stage(source)
    for relative in commit["hashes"]:
        p=Path(relative)
        if relative=="CONTRACT.json":continue
        if name=="waves" and relative not in {"R3B_WAVE_INDEX.json","R3B_BALANCE_CHECK.json"}:continue
        if name!="waves" and p.suffix not in {".json",".md"}:raise Blocked("R3_V1_UNEXPECTED_IMPORT_ARTIFACT")
        atomic_bytes(target/p,(source/p).read_bytes())
    atomic_json(target/"LEGACY_IMPORT.json",dict(source=str(source),commit_sha256=info["stages"][name]["commit_sha256"],
        source_contract=info["contract"],executed_in_current_run=False,
        status="verified_read_only_adoption",arrays_copied=False,
        waveform_cache_location="original v1 waves directory" if name=="waves" else None))
    print("REUSE completed R3 "+name+": "+str(source),flush=True)
    return True
