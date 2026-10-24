"""Server-only small engineering acceptance; no local research execution."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import io
import tempfile
import unittest
from scorw_dataset.core import atomic_text,atomic_json,transaction,read_json,identity
from scorw_dataset.io import inventory_layout,pose_only


class Contracts(unittest.TestCase):
    def test_resume_never_accepts_a_different_source_or_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"record"
            transaction(p,dict(source="a",field="f"),lambda k:atomic_text(p/"done.txt","same"))
            transaction(p,dict(source="a",field="f"),lambda k:self.fail("recomputed committed result"))
            with self.assertRaises(ValueError):transaction(p,dict(source="a",field="new"),lambda k:None)
            atomic_text(p/"done.txt","altered")
            with self.assertRaises(ValueError):transaction(p,dict(source="a",field="f"),lambda k:None)

    def test_all_CAV_frames_and_orphans_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for cav in ("1","2","-1"):
                for stamp in ("0001","0002"):
                    atomic_text(p/"scene"/cav/(stamp+".yaml"),"lidar_pose: [0,0,0,0,0,0]")
                    atomic_text(p/"scene"/cav/(stamp+".pcd"),"byte proxy")
            layout=inventory_layout(p);self.assertEqual(layout[0]["cavs"],["1","2","-1"])
            (p/"scene/2/0002.pcd").unlink()
            with self.assertRaises(ValueError):inventory_layout(p)

    def test_interrupted_stage_restarts_without_changing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"record";request=dict(source="frozen",field="same")
            def interrupted(code):
                atomic_text(p/"cloud.txt","deterministic bytes")
                raise InterruptedError("synthetic interruption")
            with self.assertRaises(InterruptedError):transaction(p,request,interrupted)
            before=(p/"cloud.txt").read_bytes()
            transaction(p,request,lambda k:atomic_text(p/"cloud.txt","deterministic bytes"))
            self.assertEqual(before,(p/"cloud.txt").read_bytes())
            self.assertEqual(read_json(p/"COMMIT.json")["status"],"passed")

    def test_pose_reader_ignores_GT_contents_and_preserves_angles(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"frame.yaml"
            atomic_text(p,"lidar_pose: [1,2,3,4,90,6]\nvehicles: {example: arbitrary}\n")
            self.assertEqual(pose_only(p),[1.,2.,3.,4.,90.,6.])
            atomic_text(p,"lidar_pose: [1,2,3,4,90,6]\nvehicles: !!python/object:unavailable.Class {}\n")
            self.assertEqual(pose_only(p),[1.,2.,3.,4.,90.,6.])

    def test_split_scene_seed_is_independent_of_timestamp_and_CAV(self):
        from scorw_dataset.core import seed_for
        a=seed_for(20261023,"scorw-fog-scene-v1","train","scene")
        b=seed_for(20261023,"scorw-fog-scene-v1","test","scene")
        self.assertNotEqual(a,b)
        self.assertEqual(a,seed_for(20261023,"scorw-fog-scene-v1","train","scene"))

    def test_source_count_mismatch_is_not_automatically_fixed(self):
        from unittest.mock import patch
        from scorw_dataset.inventory import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/"out";out.mkdir()
            for s in ("train","validate","test"):(root/s).mkdir()
            c=dict(source_root=str(root),expected={s:dict(scenes=1,frames=1) for s in ("train","validate","test")})
            with patch("scorw_dataset.inventory.inventory_layout",return_value=[]):
                with self.assertRaises(ValueError):run(c,out)
            self.assertEqual(len(read_json(out/"SOURCE_LAYOUT.json")["count_errors"]),3)

    def test_timing_covers_distance_bin_absent_from_three_primary_clouds(self):
        from scorw_dataset.performance import coverage_plan
        def row(name,bins):
            h=[0]*9
            for b in bins:h[b]=1
            return dict(split="train",relative="train/"+name,distance_hist=h)
        primary=[row("primary"+str(i),[0,1]) for i in range(3)]
        train=dict(split="train",rows=[row("z_far",[7]),*primary,row("a_far",[7])])
        plan=coverage_plan(train,primary,[3,3,0,0,0,0,0,2,0])
        self.assertEqual(plan["missing_before_supplement"],[7])
        self.assertEqual(plan["no_train_candidate_bins"],[])
        self.assertEqual(plan["entries"][-1]["row"]["relative"],"train/a_far")
        self.assertEqual(plan["entries"][-1]["bins"],[7])
        train["rows"].reverse()
        self.assertEqual(plan,coverage_plan(train,primary,[3,3,0,0,0,0,0,2,0]))

    def test_missing_train_timing_source_remains_blocked_and_never_uses_test(self):
        from scorw_dataset.performance import coverage_plan
        r=dict(split="train",relative="train/primary",distance_hist=[1,0,0,0,0,0,0,0,0])
        plan=coverage_plan(dict(split="train",rows=[r]),[r],[1,0,0,0,0,0,0,1,0])
        self.assertEqual(plan["no_train_candidate_bins"],[7])
        self.assertEqual(len(plan["entries"]),1)
        with self.assertRaises(ValueError):coverage_plan(dict(split="test",rows=[]),[],[1]*9)

    def test_supplement_merges_missing_bins_and_keeps_point_budget(self):
        import numpy as np
        from scorw_dataset.performance import coverage_plan,sample_indices
        base=dict(split="train",relative="train/base",distance_hist=[1]+[0]*8)
        extra=dict(split="train",relative="train/far",distance_hist=[0]*7+[3,3])
        plan=coverage_plan(dict(split="train",rows=[extra,base]),[base],[1]+[0]*6+[3,3])
        self.assertEqual(len(plan["entries"]),2);self.assertEqual(plan["entries"][-1]["bins"],[7,8])
        c=dict(distance_edges_m=[0,3,10,20,40,60,80,120,200,1000000],performance=dict(rays_per_cloud=18))
        raw=np.asarray([[d,0,0,.5] for d in (1,130,150,190,210,250,300)],dtype=np.float32)
        ix=sample_indices(raw,c,[7,8]);self.assertEqual(ix.tolist(),[1,3,4,6])
        self.assertEqual(ix.tolist(),sample_indices(raw,c,[7,8]).tolist())


def run(out):
    stream=io.StringIO()
    with redirect_stdout(stream),redirect_stderr(stream):r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    report=dict(status="passed" if r.wasSuccessful() else "failed",tests=r.testsRun,errors=len(r.errors),failures=len(r.failures),log=stream.getvalue())
    atomic_json(out/"ENGINEERING_ACCEPTANCE.json",report)
    if not r.wasSuccessful():raise ValueError("Dataset engineering acceptance failed; no gate relaxed")
    return report
