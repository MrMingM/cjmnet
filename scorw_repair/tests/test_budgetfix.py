"""Synthetic workload/geometry/cache fixtures only; never real OPV2V or inference."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from scorw_repair.core import (HERE,Blocked,read_json,atomic_json,atomic_text,save_arrays,identity,digest,lock,begin,transaction)
from scorw_repair.tests.test_repair import temporary,config
from scorw_repair.budget import plan
from scorw_repair.iou_audit import IoUAudit
from scorw_repair import reuse
from scorw_repair.response import simulate,summary
from shared_weather_benchmark.world_field import WorldField


def workload(points=60000,scenes=2):
    frames=[];sources=[]
    for i in range(scenes):
        scene="synthetic-"+str(i)
        frames.append(dict(scene=scene,timestamp="1",ego="a",active_cavs=["b","a"]))
        for cav in ("a","b"):
            sources.append(dict(scene=scene,timestamp="1",cav=cav,points=points,pcd_sha256="synthetic",yaml_sha256="synthetic"))
    return frames,sources


class BudgetTests(unittest.TestCase):
    def test_two_frames_fit_explicit_total_without_changing_fields(self):
        frames,rows=workload()
        selected,r=plan(config(),frames,rows)
        self.assertEqual(len(selected),2)
        self.assertEqual(r["required_ray_field_pairs"],5040000)
        self.assertTrue(all(f["ray_field_pairs"]==2520000 and f["field_responses_per_ray"]==21 for f in r["frames"]))
        self.assertTrue(all(f["cavs"]==["a","b"] for f in r["frames"]))

    def test_global_failure_is_reported_before_model_or_response(self):
        from scorw_repair.detection import evaluate
        c=config();c["r2"]["maximum_ray_field_pairs"]=4000000
        frames,rows=workload()
        with temporary() as temp:
            root=Path(temp);coords=root/"coords";out=root/"out";coords.mkdir();out.mkdir()
            atomic_json(coords/"INPUT_MANIFEST.json",rows);atomic_json(coords/"CLEAN_GT.json",frames)
            with patch("scorw_repair.detection.model_identity",side_effect=AssertionError("Model must not be used")),self.assertRaises(Blocked):
                evaluate(c,coords,out,{},"UNCALIBRATED_EXPLORATORY","synthetic")
            r=read_json(out/"R2_COMPUTE_PLAN.json")
            self.assertEqual(r["status"],"blocked");self.assertEqual(len(r["frames"]),2)
            self.assertEqual(r["problems"],[dict(scope="run",required=5040000,limit=4000000)])

    def test_single_frame_cap_still_stops_even_below_total_cap(self):
        frames,rows=workload(points=110000,scenes=1)
        with self.assertRaises(Blocked):plan(config(),frames,rows)

    def test_duplicate_source_and_missing_neighbor_stop(self):
        frames,rows=workload(scenes=1)
        with self.assertRaises(Blocked):plan(config(),frames,rows+[rows[0]])
        frames[0]["active_cavs"]=["a"]
        with self.assertRaises(Blocked):plan(config(),frames,rows)


class IoUTests(unittest.TestCase):
    def polygon(self):return SimpleNamespace(is_valid=True,is_empty=False,area=1.)

    def test_finite_warning_is_recorded_without_changing_values(self):
        value=np.array([.6],dtype=np.float32);audit=IoUAudit()
        def calculate(box,boxes):
            warnings.warn("invalid value encountered in intersection",RuntimeWarning)
            return value
        out=audit.call(calculate,self.polygon(),[self.polygon()])
        self.assertIs(out,value)
        self.assertEqual(audit.report["intersection_warnings_with_finite_result"],1)
        self.assertFalse(audit.report["values_changed"])

    def test_nan_stops_and_guard_is_restored(self):
        audit=IoUAudit();original=lambda b,bs:np.array([np.nan])
        module=SimpleNamespace(compute_iou=original)
        with self.assertRaises(Blocked):
            with audit.guard(module):module.compute_iou(self.polygon(),[self.polygon()])
        self.assertIs(module.compute_iou,original)
        self.assertEqual(audit.report["status"],"failed")

    def test_invalid_geometry_is_not_repaired_or_skipped(self):
        audit=IoUAudit();bad=self.polygon();bad.area=0
        with self.assertRaises(Blocked):audit.call(lambda b,bs:np.zeros(len(bs)),bad,[])
        self.assertFalse(audit.report["geometry_repairs"])


class ReuseTests(unittest.TestCase):
    def fixture(self):
        t=temporary();self.addCleanup(t.cleanup);root=Path(t.name)
        c=config();pose=[0]*6
        field=WorldField(kind="constant",mean_alpha=.06)
        raw=np.array([[10,0,0,.5],[50,0,0,.8]],dtype=np.float32);idx=np.arange(2,dtype=np.int64)
        key=["synthetic","1","a","synthetic-sha"]
        old=root/"old";relative=Path("synthetic/1/a/constant")
        path=old/"stages/R2/cloud_cache"/relative
        request=dict(parent="old-r2",field=field.manifest(),indices=idx.tolist(),receiver=c["receiver"])
        def create(code):
            data=simulate(raw,pose,field,c["receiver"],key,idx)
            save_arrays(path/"response.npz",**data)
            atomic_json(path/"STATS.json",summary(data,c["distance_bins_m"]))
        transaction(path,request,create)
        coords=old/"stages/coordinates";coords.mkdir(parents=True)
        atomic_json(coords/"INPUT_MANIFEST.json",[dict(scene=key[0],timestamp=key[1],cav=key[2],pcd_sha256=key[3],pose=pose)])
        info=dict(root=str(old),r2_contract="old-r2",stages=dict(coordinates=dict(root=str(coords))))
        return root,c,field,raw,idx,key,pose,relative,path,info

    def test_committed_response_is_imported_without_recomputation(self):
        root,c,f,raw,idx,key,pose,relative,old,info=self.fixture()
        target=root/"new-cache";old_bytes=(old/"COMMIT.json").read_bytes()
        with patch("scorw_repair.generate.simulate",side_effect=AssertionError("No recomputation")):
            a=reuse.cached(raw,pose,f,c,key,idx,target,"new-r2",info,relative)
            b=reuse.cached(raw,pose,f,c,key,idx,target,"new-r2",info,relative)
        np.testing.assert_array_equal(a["weather_xyzi"],b["weather_xyzi"])
        self.assertEqual(old_bytes,(old/"COMMIT.json").read_bytes())
        self.assertNotEqual(read_json(old/"CONTRACT.json")["identity"],read_json(target/"CONTRACT.json")["identity"])
        self.assertTrue((target/"LEGACY_RESPONSE.json").is_file())

    def test_corrupt_or_misidentified_cache_is_rejected(self):
        root,c,f,raw,idx,key,pose,relative,old,info=self.fixture()
        wrong=list(key);wrong[2]="different-CAV"
        with self.assertRaises(Blocked):reuse.cached(raw,pose,f,c,wrong,idx,root/"wrong","new-r2",info,relative)
        atomic_text(old/"STATS.json","corrupt")
        with self.assertRaises(Blocked):reuse.cached(raw,pose,f,c,key,idx,root/"corrupt","new-r2",info,relative)

    def test_interrupted_import_resumes_without_recomputing_waveform(self):
        root,c,f,raw,idx,key,pose,relative,old,info=self.fixture()
        target=root/"interrupted"
        request=dict(parent="new-r2",field=f.manifest(),indices=idx.tolist(),receiver=c["receiver"])
        with lock(target):begin(target,request,False)
        with patch("scorw_repair.generate.simulate",side_effect=AssertionError("No recomputation")):
            data=reuse.cached(raw,pose,f,c,key,idx,target,"new-r2",info,relative)
        self.assertTrue((target/"COMMIT.json").is_file())
        np.testing.assert_array_equal(data["original_xyzi"],raw)

    def test_changed_receiver_or_pose_is_rejected(self):
        root,c,f,raw,idx,key,pose,relative,old,info=self.fixture()
        c["receiver"]["gamma"]*=2
        with self.assertRaises(Blocked):reuse.cached(raw,pose,f,c,key,idx,root/"receiver","new-r2",info,relative)
        c["receiver"]["gamma"]/=2
        with self.assertRaises(Blocked):reuse.cached(raw,[1,0,0,0,0,0],f,c,key,idx,root/"pose","new-r2",info,relative)

    def test_full_legacy_contract_import_and_science_change_rejection(self):
        t=temporary();self.addCleanup(t.cleanup);root=Path(t.name)/"legacy"
        c=config();oldconfig=read_json(HERE/"compat/v2_config.json")
        oldpins=read_json(HERE/"compat/v2_snapshot.json")["sources"]
        selection=dict(fixture_only=True);model=dict(status="verified",fixture_only=True);runtime=dict(fixture_only=True)
        request=dict(config=oldconfig,source_identity=identity(oldpins),snapshot=reuse.LEGACY_SNAPSHOT,
            opv2v_selection=selection,model=model,runtime=runtime)
        with lock(root):
            parent=begin(root,request,False)
        gates=dict(contract=parent,status="blocked",blocked_stage="R2",
            reason="R2_COMPUTE_BUDGET_EXCEEDED_BEFORE_RESPONSE_GENERATION",stages={})
        for name,files in reuse.STAGES.items():
            stage=root/"stages"/name
            def work(code,stage=stage,files=files):
                for filename in files:atomic_text(stage/filename,"{}\n" if filename.endswith(".json") else "Synthetic fixture only\n")
            transaction(stage,dict(parent=parent,stage=name),work)
            gates["stages"][name]=dict(commit_sha256=digest(stage/"COMMIT.json"))
        r2=root/"stages/R2"
        with lock(r2):begin(r2,dict(parent=parent,stage="R2"),False)
        atomic_json(root/"VALIDATION_GATES.json",gates)
        manifest={"VALIDATION_GATES.json":{"sha256":digest(root/"VALIDATION_GATES.json")}}
        atomic_json(root/"REPORT_MANIFEST.json",manifest)
        info=reuse.audit(root,c,selection,model,runtime)
        self.assertEqual(info["contract"],parent)
        reuse.verify(info)
        target=Path(t.name)/"imported";target.mkdir()
        self.assertTrue(reuse.import_stage(info,"R1",target))
        self.assertTrue((target/"LEGACY_IMPORT.json").is_file())
        c["field"]["mean_alpha"]*=2
        with self.assertRaises(Blocked):reuse.audit(root,c,selection,model,runtime)


if __name__=="__main__":unittest.main()
