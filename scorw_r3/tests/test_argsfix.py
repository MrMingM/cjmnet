"""Server acceptance for legacy YAML args and hash-checked cache adoption; no detector/GPU."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import unittest

from scorw_r3.core import HERE,Blocked,read_json,identity,digest,atomic_json,atomic_bytes,transaction
from scorw_r3.runtime import validate_project_first
from scorw_r3 import recovery
from scorw_r3.tests.test_r3 import scratch
from scorw_repair.reuse import verified_stage

C=read_json(HERE/"config.json")


def previous_fixture(root):
    """Small synthetic commit tree with byte proxies; never an experimental result."""
    config=recovery.compatible_config(C)
    pins=read_json(HERE/"compat/r3_v1_snapshot.json")["sources"]
    current=dict(config=C,config_sha256=digest(HERE/"config.json"),sources="new",
        source_snapshot_sha256="new",mode="smoke",runtime=dict(python="synthetic",threads=1),model="fixed-model-proxy")
    request=dict(current,config=config,config_sha256=recovery.V1_CONFIG,sources=identity(pins),
        source_snapshot_sha256=recovery.V1_SNAPSHOT)
    parent=identity(request)
    atomic_json(root/"CONTRACT.json",dict(identity=parent,request=request))
    atomic_json(root/"SOURCE_AUDIT.json",dict(files=pins,identity=identity(pins),snapshot_sha256=recovery.V1_SNAPSHOT))
    stages={}
    for name in recovery.STAGES:
        folder=root/"stages"/name
        def work(code):
            if name=="R3A":atomic_json(folder/"R3A_GATE.json",dict(status="PASS_WITH_DISCLOSED_APPROXIMATIONS"))
            elif name=="selection":
                atomic_json(folder/"R3B_SELECTION_FREEZE.json",dict(consulted_inputs=[]))
                atomic_json(folder/"coordinates/INPUT_MANIFEST.json",[])
                atomic_json(folder/"R3B_COMPUTE_PLAN.json",dict(status="passed"))
            else:
                response=folder/"responses/test/response.npz";pcd=folder/"encoded/test/input.pcd"
                atomic_bytes(response,b"synthetic array-byte proxy");atomic_bytes(pcd,b"synthetic codec-byte proxy")
                atomic_json(folder/"R3B_WAVE_INDEX.json",dict(status="passed",responses=[dict(response=str(response),
                    response_sha256=digest(response),pcd=str(pcd),pcd_sha256=digest(pcd))]))
                atomic_json(folder/"R3B_BALANCE_CHECK.json",dict(status="passed"))
        transaction(folder,dict(parent=parent,stage=name),work)
        stages[name]=dict(status="passed",commit_sha256=digest(folder/"COMMIT.json"))
    atomic_json(root/"VALIDATION_GATES.json",dict(status="failed",contract=parent,blocked_stage="detector",
        reason="AttributeError: 'list' object has no attribute 'get'",stages=stages,
        locations=[dict(file="runtime.py",function="__init__",line=26)]))
    refresh_manifest(root)
    return current


def refresh_manifest(root):
    rows={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in root.iterdir()
          if p.is_file() and p.suffix==".json" and p.name!="REPORT_MANIFEST.json"}
    atomic_json(root/"REPORT_MANIFEST.json",rows)


class FusionArgs(unittest.TestCase):
    def test_legacy_empty_list_and_missing_option_use_original_default(self):
        for fusion in (dict(args=[]),dict(args={}),{},dict(args=dict(proj_first=True))):
            self.assertTrue(validate_project_first(dict(fusion=fusion))["proj_first"])

    def test_false_or_ambiguous_args_stop_with_a_clear_error(self):
        for args in (dict(proj_first=False),dict(proj_first=0),dict(proj_first="true"),["proj_first"],None,""):
            with self.assertRaises(Blocked):validate_project_first(dict(fusion=dict(args=args)))

    def test_yaml_empty_sequence_is_accepted_without_model_initialization(self):
        import yaml
        h=yaml.safe_load("fusion:\n  core_method: IntermediateFusionDataset\n  args: []\n")
        result=validate_project_first(h)
        self.assertEqual(result["args_type"],"list")


class CacheAdoption(unittest.TestCase):
    def test_only_engineering_configuration_changes_allowed(self):
        recovery.compatible_config(C)
        modified=deepcopy(C);modified["field"]["mean_alpha"]+=.001
        with self.assertRaises(Blocked):recovery.compatible_config(modified)

    def test_verified_adoption_preserves_old_bytes_and_does_not_copy_arrays(self):
        with scratch() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)/"v1";request=previous_fixture(root)
            info=recovery.audit(root,C,request)
            before=digest(root/"REPORT_MANIFEST.json")
            out=Path(tmp)/"new"
            for name in recovery.STAGES:
                target=out/name
                transaction(target,dict(parent="new-contract",stage=name),lambda code:recovery.import_stage(info,name,target))
                verified_stage(target)
            recovery.verify(info)
            self.assertEqual(digest(root/"REPORT_MANIFEST.json"),before)
            self.assertFalse((out/"waves/responses").exists())
            self.assertEqual(read_json(out/"waves/R3B_WAVE_INDEX.json"),read_json(root/"stages/waves/R3B_WAVE_INDEX.json"))
            self.assertFalse(recovery.import_stage(info,"detector",out/"detector"))
            self.assertFalse(recovery.import_stage(info,"numerical",out/"numerical"))

    def test_changed_existing_array_or_runtime_is_rejected(self):
        with scratch() as tmp:
            root=Path(tmp)/"v1";request=previous_fixture(root)
            changed=dict(request,runtime=dict(python="different"))
            with self.assertRaises(Blocked):recovery.audit(root,C,changed)
            info=recovery.audit(root,C,request)
            atomic_bytes(root/"stages/waves/responses/test/response.npz",b"modified")
            with self.assertRaises(Blocked):recovery.verify(info)

    def test_failed_or_previously_executed_detector_cannot_be_adopted(self):
        with scratch() as tmp:
            root=Path(tmp)/"v1";request=previous_fixture(root)
            atomic_json(root/"stages/detector/R3B_MODEL_AUDIT.json",dict(status="passed"))
            with self.assertRaises(Blocked):recovery.audit(root,C,request)

    def test_new_contract_still_rejects_protocol_changes_on_resume(self):
        with scratch() as tmp:
            root=Path(tmp)/"new";transaction(root,dict(source="argsfix"),lambda code:None)
            with self.assertRaises(ValueError):transaction(root,dict(source="different"),lambda code:None)


if __name__=="__main__":unittest.main()
