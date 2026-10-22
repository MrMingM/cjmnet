"""Read original server artifacts and follow explicit immutable legacy-import receipts."""
from pathlib import Path
from contextlib import contextmanager
import os

from scorw_r3.core import HERE,REPO,Blocked,read_json,digest,identity,check_inputs
from scorw_repair.reuse import verified_stage
from scorw_repair.detection import model_identity
from scorw_p7.core import inventory


def audit_historical_r2(folder):
    """Only absent forensic PCDs may be disclosed as unavailable; altered bytes always stop."""
    contract=read_json(folder/"CONTRACT.json");commit=read_json(folder/"COMMIT.json")
    expected=commit["hashes"];actual=inventory(folder)
    missing=sorted(set(expected)-set(actual))
    absent_optional=[n for n in missing if n.startswith("diagnostic_pcd/") and n.endswith(".pcd")]
    if (identity(contract["request"])!=contract["identity"] or commit.get("contract")!=contract["identity"]
            or commit.get("status")!="passed" or set(actual)-set(expected) or missing!=absent_optional
            or any(actual[n]!=expected[n] for n in actual if n in expected)):
        raise Blocked("R3_OLD_R2_REQUIRED_CACHE_OR_EXISTING_BYTES_CHANGED")
    return absent_optional


class OldRun:
    def __init__(self,config,deep=False):
        self.root=Path(config["legacy_root"])
        expected=read_json(HERE/"inputs/R2_IDENTITY.json")
        self.expected_reports=expected["files"]
        self.optional_missing={}
        if digest(self.root/"REPORT_MANIFEST.json")!=expected["report_manifest_sha256"]:
            raise Blocked("R3_OLD_REPORT_MANIFEST_CHANGED")
        for n,row in expected["files"].items():
            if Path(n).name!=n or digest(self.root/n)!=row["sha256"]:raise Blocked("R3_OLD_REPORT_CHANGED: "+n)
        self.request=read_json(self.root/"CONTRACT.json")
        if self.request["identity"]!=expected["contract"] or identity(self.request["request"])!=expected["contract"]:
            raise Blocked("R3_OLD_CONTRACT_CHANGED")
        self.config=self.request["request"]["config"]
        if any(config[k]!=self.config[k] for k in ("receiver","field","b1","model")):
            raise Blocked("R3_FORMAL_RECEIVER_FIELD_OR_MODEL_CHANGED")
        self.source=read_json(self.root/"SOURCE_AUDIT.json")
        if self.source["identity"]!=expected["source_identity"] or identity(self.source["files"])!=self.source["identity"]:
            raise Blocked("R3_OLD_SOURCE_IDENTITY_CHANGED")
        for n,h in self.source["files"].items():
            if digest(REPO/n)!=h:raise Blocked("R3_FROZEN_R0_R2_SOURCE_CHANGED: "+n)
        self.gates=read_json(self.root/"VALIDATION_GATES.json")
        if self.gates["status"]!="FULL_UNCALIBRATED_EXPLORATORY_COMPLETE":raise Blocked("R3_OLD_R2_NOT_COMPLETE")
        self.coords=self.stage("coordinates",deep)
        self.r1=self.stage("R1",deep)
        self.r2=self.stage("R2",deep)
        self.rows=read_json(self.coords/"INPUT_MANIFEST.json")
        self.frames=read_json(self.coords/"CLEAN_GT.json")
        self.domains=read_json(self.coords/"FIELD_DOMAINS.json")
        check_inputs(self.rows)
        self.model=model_identity(config)
        if self.model!=self.request["request"]["model"] or self.model["status"]!="verified":
            raise Blocked("R3_FROZEN_MODEL_IDENTITY_FAILED")
        self.report=read_json(self.root/"R2_TARGET_DIAGNOSIS.json")
        self.manifest=digest(self.root/"REPORT_MANIFEST.json")

    def stage(self,name,deep):
        folder=self.root/"stages"/name
        expected=self.gates["stages"][name]["commit_sha256"]
        visited=set()
        while True:
            if str(folder) in visited:raise Blocked("R3_LEGACY_IMPORT_CYCLE")
            visited.add(str(folder))
            if digest(folder/"COMMIT.json")!=expected:raise Blocked("R3_OLD_STAGE_COMMIT_CHANGED")
            if deep:
                if name=="R2":
                    missing=audit_historical_r2(folder);key=str(folder)
                    if key in self.optional_missing and self.optional_missing[key]!=missing:
                        raise Blocked("R3_OLD_FORENSIC_CACHE_AVAILABILITY_CHANGED_DURING_RUN")
                    self.optional_missing[key]=missing
                else:verified_stage(folder)
            receipt=folder/"LEGACY_IMPORT.json"
            if not receipt.is_file():return folder
            r=read_json(receipt);folder=Path(r["source"]);expected=r["commit_sha256"]

    def verify(self):
        if digest(self.root/"REPORT_MANIFEST.json")!=self.manifest:raise Blocked("R3_OLD_RUN_CHANGED")
        for name,row in self.expected_reports.items():
            if digest(self.root/name)!=row["sha256"]:raise Blocked("R3_OLD_REPORT_CHANGED_DURING_RUN: "+name)
        check_inputs(self.rows)
        for name in ("coordinates","R1","R2"):self.stage(name,True)

    def cache_info(self):
        return dict(root=str(self.root),r2_contract=read_json(self.r2/"CONTRACT.json")["identity"],
            stages=dict(coordinates=dict(root=str(self.coords))))


@contextmanager
def readonly_old(root):
    with (Path(root)/".p7.lock").open("rb") as f:
        if os.name=="posix":
            import fcntl
            try:fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:raise Blocked("R3_OLD_PROCESS_STILL_ACTIVE")
        try:yield
        finally:
            if os.name=="posix":fcntl.flock(f,fcntl.LOCK_UN)
