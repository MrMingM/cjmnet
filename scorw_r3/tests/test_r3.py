"""Synthetic and static acceptance only; no OPV2V data IO or real detector run."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from scorw_r3.core import (HERE,REPO,Blocked,read_json,digest,identity,atomic_json,atomic_text,
    array_hash,save_arrays,transaction,begin,lock)
from scorw_r3.receiver_audit import intensity_test,differences,trial,sensitivity_variants,cache_diagnosis
from scorw_r3.metrics import match_matrix,ap,target_category
from scorw_r3.selection import allowed_candidates,budget
from scorw_r3.waves import pool_balance
from scorw_r3.analysis import group_ap,global_table,delta,analyze
from scorw_r3.run import validate_config
from scorw_r3.detection import memo_predict
from scorw_repair.response import single_response,simulate,validate
from shared_weather_benchmark.world_field import WorldField
from shared_weather_benchmark.geometry import pose_matrix,inverse
from scorw_p7.balanced_assignment import assignment
from scorw_p7.field_pool import make_pool

C=read_json(HERE/"config.json")


@contextmanager
def scratch():
    base=HERE/"local_validation" if os.name=="nt" else Path(tempfile.gettempdir())
    base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r3-unit-",dir=base) as name:yield name


class Physics(unittest.TestCase):
    def test_R3A_synthetic_artifact_chain(self):
        from scorw_r3.receiver_audit import run
        from scorw_repair.response import summary
        oldcfg=read_json(REPO/"scorw_repair/config.json")
        raw=np.array([[10,0,0,.3],[30,0,0,.8]],dtype=np.float32)
        d=simulate(raw,[0]*6,WorldField(kind="constant",mean_alpha=.06),C["receiver"],["synthetic","0","cav","sha"])
        with scratch() as tmp:
            root=Path(tmp);r1=root/"legacy1";r2=root/"legacy2";out=root/"new";out.mkdir()
            save_arrays(r2/"cloud_cache/test/response.npz",**d)
            save_arrays(r1/"responses/test/B2.npz",**d)
            ray=r1/"rays/test"
            save_arrays(ray/"rays.npz",point_id=d["point_identity"],origin=np.zeros(3),world_endpoints=raw[:,:3].astype(float),original_xyzi=raw)
            atomic_json(ray/"SAMPLING.json",dict(scene="synthetic"))
            pool=make_pool("synthetic",read_json(REPO/"scorw_p7/config/protocol.json"),None)
            atomic_json(root/"R1_FIELD_POOL_FREEZE.json",dict(pools=dict(synthetic=pool)))
            report=dict(echo_statistics=[dict(mode=m,stats=summary(d,oldcfg["distance_bins_m"])) for m in ("B0","B1","B2","B3","B4")])
            old=SimpleNamespace(config=oldcfg,r1=r1,r2=r2,root=root,report=report)
            gate=run(C,old,out,smoke=True)
            self.assertEqual(gate["status"],"PASS_WITH_DISCLOSED_APPROXIMATIONS")
            from scorw_r3.run import REQUIRED_A
            self.assertTrue(all((out/n).is_file() for n in REQUIRED_A))
            self.assertFalse(read_json(out/"R3A_SENSITIVITY.json")["formal_configuration_changed"])

    def test_positive_intensity_linearity_and_selection(self):
        report=intensity_test(C)
        self.assertEqual(len(report["cases"]),15)
        self.assertLess(max(r["maximum_absolute_linearity_error"] for r in report["cases"]),1e-12)

    def test_zero_input_special_rules(self):
        cfg=C["receiver"]
        a=single_response(WorldField(kind="constant",mean_alpha=0),[0]*3,[10,0,0],0.,cfg)
        b=single_response(WorldField(kind="constant",mean_alpha=.06),[0]*3,[10,0,0],0.,cfg)
        n=single_response(WorldField(kind="constant",mean_alpha=.06),[0]*3,[.5,0,0],0.,cfg)
        self.assertTrue(a.get("emitted",True));self.assertFalse(b["emitted"])
        self.assertTrue(n["unidentifiable_overlap"]);self.assertTrue(n.get("emitted",True))

    def test_weak_positive_return_zero_code_and_no_threshold(self):
        row=single_response(WorldField(kind="constant",mean_alpha=.06),[0]*3,[10,0,0],1/255,C["receiver"])
        self.assertGreater(row["waveform_peak"],0.)
        self.assertLess(row["waveform_peak"],C["receiver"]["intensity_quantum"]/2)
        self.assertFalse(row["emitted"])

    def test_sensitivity_never_changes_formal_config(self):
        before=identity(C)
        f=WorldField(kind="constant",mean_alpha=.06)
        variants=sensitivity_variants(C)
        self.assertEqual(len(variants),12)
        for v in variants:trial(f,[0]*3,[30,0,0],.4,C,v)
        self.assertEqual(identity(C),before)

    def test_no_ADC_counterfactual_preserves_observed_zero_fog(self):
        r=trial(WorldField(kind="constant",mean_alpha=0),[0]*3,[10,0,0],0,C,dict(factor="quantum",value=0))
        self.assertFalse(r["no_return"])

    def test_constant_hard_analytic(self):
        r=single_response(WorldField(kind="constant",mean_alpha=.06),[0]*3,[30,0,0],.8,C["receiver"],return_waveform=True)
        self.assertAlmostEqual(r["tau"],1.8,places=13)
        width=299792458*C["receiver"]["pulse_half_width_s"]
        grid=r["waveform_grid_m"]
        shape=np.where((grid>=30)&(grid<=30+width),np.sin(np.pi*(grid-30)/width)**2,0.)
        np.testing.assert_allclose(r["hard_waveform"],.8*np.exp(-3.6)*shape,rtol=1e-13,atol=1e-15)

    def test_numeric_and_original_Hahner_40_case_regression(self):
        from scorw_repair.regression import check
        cfg=read_json(REPO/"scorw_repair/config.json")
        r=check(cfg)
        d=differences(C,r)
        self.assertEqual(len(d["rows"]),40)
        self.assertEqual(r["behavior_totals"],{"hahner":{"replaced":19,"no_return":8},"new_engine":{"replaced":12,"no_return":15}})
        self.assertTrue(all(v=="passed" for v in r["strict_checks"].values()))

    def test_cache_replay_counts_including_zero_fog(self):
        raw=np.array([[.5,0,0,0],[3,0,0,1/255],[30,0,0,.7],[50,0,0,0]],dtype=np.float32)
        with scratch() as tmp:
            for alpha in (0.,.06):
                d=simulate(raw,[0]*6,WorldField(kind="constant",mean_alpha=alpha),C["receiver"],["synthetic","0","cav","sha"])
                p=Path(tmp)/"r.npz";save_arrays(p,**d)
                r=cache_diagnosis(p,C["receiver"]["intensity_quantum"])
                self.assertEqual(r["no_return"],int((~d["emitted"]).sum()))
                self.assertEqual(sum(int((d["emitted"]&(d["origin"]==k)).sum()) for k in range(3))+r["no_return"],4)

    def test_same_world_path_from_two_CAVs(self):
        field=WorldField(seed=14)
        a=np.array([10.,4.,2.]);b=np.array([35.,5.,3.])
        pose=[10,4,2,0,90,0];m=np.asarray(pose_matrix(pose));inv=np.asarray(inverse(m.tolist()))
        local=inv[:3,:3]@b+inv[:3,3];back=m[:3,:3]@local+m[:3,3]
        self.assertAlmostEqual(field.alpha(b),field.alpha(back),places=14)
        r1=single_response(field,a,b,.8,C["receiver"]);r2=single_response(field,m[:3,3],back,.8,C["receiver"])
        self.assertEqual(r1,r2)


class BalanceAndIdentity(unittest.TestCase):
    def test_full_20_fields_distinct_and_scene_isolation(self):
        c=read_json(REPO/"scorw_p7/config/protocol.json")
        a=make_pool("synthetic_a",c,None);b=make_pool("synthetic_b",c,None)
        self.assertEqual(len(set(a["seeds"])),20)
        self.assertFalse(set(a["seeds"])&set(b["seeds"]))
        changed=deepcopy(c);changed["weather_root_seed"]+=1
        self.assertNotEqual(identity(make_pool("synthetic_a",changed,None)),identity(a))

    def test_assignment_order_two_three_and_more_than20(self):
        for n in (2,3,20):
            cavs=[str(k) for k in range(n)]
            a=assignment(cavs,"0");b=assignment(cavs[::-1],"0")
            self.assertEqual(a,b)
            for v in cavs:
                for mode in ("E2","E3"):self.assertEqual(sorted(a["schedules"][mode][v]),list(range(20)))
            for j in range(20):self.assertEqual(len({a["schedules"]["E3"][v][j] for v in cavs}),n)
        with self.assertRaises(ValueError):assignment([str(k) for k in range(21)],"0")

    def test_full_response_marginals_and_shift(self):
        plan=assignment(["ego","b","a"],"ego");rows=[]
        for k,v in enumerate(plan["cavs"]):
            for j in range(20):
                rows.append(dict(cav=v,name=str(j),response_sha256=identity([v,j]),
                    moments=dict(mean_tau=k+j*.01,emitted_signal_sum=j+k*.7,hard_signal_sum=k+j*.2),
                    stats=dict(original=100,original_hard=50+j,no_return=j)))
        r=pool_balance(rows,plan)
        self.assertEqual(r["shared_joint_shift"],"passed")
        self.assertTrue(all(v==0 for p in r["per_cav"] for v in p["mean_differences"].values()))
        self.assertNotEqual(r["per_repeat"][0]["B3"],r["per_repeat"][0]["B4"])
        broken=deepcopy(plan);broken["schedules"]["E3"]["a"][0]=broken["schedules"]["E3"]["a"][1]
        with self.assertRaises(Blocked):pool_balance(rows,broken)

    def test_one_neighbor_mean_advantage_zero_but_variance_can_change(self):
        x=np.linspace(0,1,20);y=x*x;shift=np.roll(y,1)
        self.assertAlmostEqual(np.mean(y-x)-np.mean(shift-x),0.,places=14)
        self.assertNotAlmostEqual(np.var(y-x),np.var(shift-x),places=3)

    def test_history_reserves_and_unknown_never_recruited(self):
        h=read_json(HERE/"inputs/HISTORY_FREEZE.json")
        a,u=allowed_candidates(h,[*h["all_historical_scenes"],"unverified_new_scene"])
        self.assertEqual(a,sorted(h["eligible_known_unused"]))
        self.assertEqual(len(a),5)
        self.assertEqual(u,["unverified_new_scene"])
        self.assertFalse(set(a)&(set(h["used_or_reviewed_scenes"])|set(h["future_reserved_scenes"])))

    def test_budget_counts_all_fields_before_work(self):
        with scratch() as tmp:
            out=Path(tmp);coord=out/"coordinates";coord.mkdir()
            rows=[dict(scene="s",timestamp="1",cav=v,points=200000) for v in ("a","b","c")]
            atomic_json(coord/"INPUT_MANIFEST.json",rows)
            selected=dict(frames=[dict(scene="s",timestamp="1",selected_cavs=["a","b","c"],views=[dict(name="two_cav",cavs=["a","b"])])])
            r=budget(C,selected,coord,out,False)
            self.assertEqual(r["ray_field_pairs"],12600000)
            self.assertEqual(r["status"],"BUDGET_BLOCKED")
            self.assertIn("per_scene_ray_field_pairs",r["failures"])
            self.assertTrue(r["all_twenty_fields_preserved"])

    def test_formal_physics_changes_forbidden(self):
        self.assertEqual(validate_config(deepcopy(C)),C)
        c=deepcopy(C);c["receiver"]["gamma"]*=2
        with self.assertRaises(Blocked):validate_config(c)

    def test_original_train_only_guard(self):
        from shared_weather_benchmark.dataset_io import approved_path
        with scratch() as tmp:
            p=Path(tmp)/"test";p.mkdir()
            with self.assertRaises(ValueError):approved_path(p)


class Evaluation(unittest.TestCase):
    def test_synthetic_R3B_full_pool_cache_to_analysis_and_resume(self):
        """Exercise production orchestration with tiny arrays and an explicit fake detector/codec."""
        from contextlib import redirect_stdout
        import io
        from scorw_r3 import waves,detection
        from scorw_r3.runtime import Runtime
        cfg=read_json(REPO/"scorw_repair/config.json")
        class Fake:
            initial_state="synthetic_only";calls=0
            audit=None
            model_audit=dict(neighbor_frame="synthetic unit test, not a real inference result")
            pcd=SimpleNamespace(mask_ego_points=lambda pc:pc,mask_points_by_range=lambda pc,b:pc)
            h=dict(preprocess=dict(cav_lidar_range=[-140,-40,-3,140,40,1]))
            project=Runtime.project
            def predict(self,clouds,gt):
                self.calls+=1
                # Deterministic fake outputs exercise counts and identities, never detector validity.
                matched={str(t):match_matrix(np.eye(len(gt)),np.full(len(gt),.9),t) for t in (.3,.5,.7)}
                return dict(boxes=gt.tolist(),scores=[.9]*len(gt),matches=matched,diagnostics=[dict(reason="synthetic_detected")]*len(gt),
                    input_hash=identity([array_hash(p) for p in clouds]),output_hash=identity(gt.tolist()))
            def finish(self):pass
        def write(path,points):save_arrays(path,points=points)
        def read(path):
            with np.load(path,allow_pickle=False) as a:return a["points"]
        def b1(raw,c,key,ix):return simulate(raw,[0]*6,WorldField(mean_alpha=0,kind="constant"),c["receiver"],key,ix)
        with scratch() as tmp:
            root=Path(tmp);coord=root/"coordinates";out=root/"waves";coord.mkdir();out.mkdir()
            frame=dict(scene="synthetic",timestamp="1",ego="a",partition="independent_validation",selected_cavs=["a","b"],
                views=[dict(name="two_cav",cavs=["a","b"])],objects={"object":dict(location=[20,0,0],center=[0,0,0],angle=[0,0,0],extent=[11,2,2])})
            rows=[]
            for k,cav in enumerate(frame["selected_cavs"]):
                p=root/(cav+".pcd");y=root/(cav+".yaml");atomic_text(y,"synthetic")
                write(p,np.array([[12,0,0,.2],[15,0,0,.8],[30,0,0,.4]],dtype=np.float32))
                rows.append(dict(scene="synthetic",timestamp="1",cav=cav,pcd=str(p),yaml=str(y),pcd_sha256=digest(p),yaml_sha256=digest(y),points=3,pose=[k*2,0,0,0,0,0]))
            atomic_json(coord/"INPUT_MANIFEST.json",rows);atomic_json(coord/"FIELD_DOMAINS.json",dict(synthetic=None))
            selection=dict(frames=[frame],pools=dict(synthetic=make_pool("synthetic",read_json(REPO/"scorw_p7/config/protocol.json"),None)),
                assignments=dict(synthetic=assignment(["a","b"],"a")),selection_identity="synthetic-selection")
            with patch("reference_weather.common.write_pcd",side_effect=write),patch.object(waves,"read_points",side_effect=read),patch.object(waves,"original_b1",side_effect=b1),redirect_stdout(io.StringIO()):
                index=waves.build(C,SimpleNamespace(config=cfg),selection,coord,out,"synthetic-parent")
                again=waves.build(C,SimpleNamespace(config=cfg),selection,coord,out,"synthetic-parent")
                self.assertEqual(index,again);self.assertEqual(len(index["responses"]),46)
            detector=Fake();detected=root/"detector";detected.mkdir()
            with patch.object(detection,"read_points",side_effect=read),patch.object(detection,"iou_matrix",side_effect=lambda b,g,a:np.zeros((len(b),len(g)))),redirect_stdout(io.StringIO()):
                result=detection.evaluate(C,None,selection,coord,index,detector,detected,"synthetic-parent")
                previous=detector.calls
                resumed=detection.evaluate(C,None,selection,coord,index,detector,detected,"synthetic-parent")
            self.assertEqual(detector.calls,previous);self.assertEqual(result,resumed)
            self.assertEqual(len(result[0]["runs"]),43)
            self.assertEqual(result[0]["ego_B3_B4_identity"],"passed")
            stats=root/"analysis";stats.mkdir();decision=analyze(C,result,stats)
            self.assertEqual(decision["decision"],"INCONCLUSIVE")

    def test_global_AP_differs_from_mean_scene_AP(self):
        a=match_matrix([[.9]],[.1],.7)
        b=match_matrix([[0.],[.9]],[.9,.8],.7)
        self.assertAlmostEqual((ap([a])+ap([b]))/2,.75)
        self.assertAlmostEqual(ap([a,b]),2/3)

    def test_none_predictions_and_empty_GT(self):
        row=match_matrix(np.empty((0,2)),[],.7)
        self.assertEqual(row["FN"],2);self.assertEqual(ap([row]),0.)
        self.assertIsNone(ap([match_matrix(np.empty((0,0)),[],.7)]))

    def test_greedy_threshold_and_duplicate_gt(self):
        r=match_matrix([[.7],[.9],[.1]],[.9,.8,.1],.7)
        self.assertEqual(r["matched_gt"],[0]);self.assertEqual(r["TP"],1);self.assertEqual(r["FP"],2)

    def test_neighbor_categories_use_predictions(self):
        self.assertEqual(target_category(False,[True],False),"B_neighbor_detects_fusion_misses")
        self.assertEqual(target_category(False,[False],False),"C_all_miss")
        self.assertEqual(target_category(False,[True],True),"A_neighbor_detects_fusion_recovers")
        self.assertEqual(target_category(True,[False],False),"D_ego_detects_fusion_loses")

    def test_prediction_cache_and_exact_ego_repeat_gate(self):
        class Fake:
            initial_state="fixed";calls=0
            def predict(self,clouds,gt):
                self.calls+=1
                return dict(input_hash=identity([array_hash(v) for v in clouds]),output_hash="fixed")
        f=Fake();cloud=np.ones((2,4),dtype=np.float32)
        with scratch() as tmp:
            for _ in range(2):memo_predict(f,[cloud],np.ones((1,8,3)),Path(tmp),"contract")
            self.assertEqual(f.calls,1)
            memo_predict(f,[cloud],np.ones((1,8,3)),Path(tmp),"contract",True)
            self.assertEqual(f.calls,2)
            with patch.object(f,"predict",return_value=dict(input_hash="wrong",output_hash="fixed")):
                with self.assertRaises(Blocked):memo_predict(f,[cloud],np.ones((1,8,3)),Path(tmp),"contract",True)

    def test_unbalanced_ego_predictions_stop_analysis(self):
        def p(ok):return dict(matches={str(t):match_matrix([[.9 if ok else 0.]],[.9],t) for t in (.3,.5,.7)})
        s=dict(runs=[dict(view="two_cav",mode=m,repeat=j,predictions=dict(ego=p(m=="B3"),fusion=p(True))) for m in ("B3","B4") for j in range(20)])
        with self.assertRaises(Blocked):delta([s],"two_cav")

    def test_no_scene_means_inconclusive_not_no_effect(self):
        with scratch() as tmp:
            r=analyze(C,[],Path(tmp))
            self.assertEqual(r["decision"],"INCONCLUSIVE")
            self.assertFalse(r["no_effect_proven"])

    def test_twenty_repeats_do_not_inflate_scene_count(self):
        matches={str(t):match_matrix([[.9]],[.9],t) for t in (.3,.5,.7)}
        runs=[dict(view="two_cav",mode=m,repeat=j,predictions={s:dict(matches=matches) for s in ("ego","fusion")},new_cooperative_FP=0) for m in ("B3","B4") for j in range(20)]
        targets=[dict(scene="s",object_id="obj",view="two_cav",mode=m,repeat=j,category="ego_and_fusion_detect",ego_detected=True,
            fusion_detected=True,neighbor_detected={"n":True}) for m in ("B3","B4") for j in range(20)]
        s=dict(scene="s",independent=True,gt_ids=["obj"],views=[dict(name="two_cav")],runs=runs,targets=targets)
        with scratch() as tmp:
            r=analyze(C,[s],Path(tmp))
            stat=read_json(Path(tmp)/"R3B_STATISTICAL_ANALYSIS.json")
            self.assertEqual(r["independent_scenes"],1)
            self.assertEqual(stat["intervals"]["two_cav"]["bootstrap_replicates"],0)
            self.assertIsNone(stat["intervals"]["two_cav"]["scene_bootstrap_CI95"])
            self.assertEqual(r["decision"],"INCONCLUSIVE")

    def test_statistical_stop_requires_joint_AP_and_target_equivalence(self):
        c=deepcopy(C);c["statistics"]["bootstrap_replicates"]=12
        matches={str(t):match_matrix(np.eye(8),np.full(8,.9),t) for t in (.3,.5,.7)}
        scenes=[]
        for scene in range(6):
            runs=[dict(view="two_cav",mode=m,repeat=j,predictions={s:dict(matches=matches) for s in ("ego","fusion")},new_cooperative_FP=0)
                for m in ("B3","B4") for j in range(20)]
            targets=[dict(scene=str(scene),object_id=str(k),view="two_cav",mode=m,repeat=j,category="ego_and_fusion_detect",
                ego_detected=True,fusion_detected=True,neighbor_detected={"n":True}) for m in ("B3","B4") for j in range(20) for k in range(8)]
            scenes.append(dict(scene=str(scene),independent=True,gt_ids=list(map(str,range(8))),views=[dict(name="two_cav")],runs=runs,targets=targets))
        with scratch() as tmp:
            out=Path(tmp)
            r=analyze(c,scenes,out)
            self.assertEqual(r["decision"],"STOP_OR_REDESIGN")
            # Isolate the decision rule: equally precise AP intervals do not establish
            # negligible target-event structure. This is an artificial statistics fixture.
            for s in scenes:
                for t in s["targets"]:
                    if t["mode"]=="B4":t["category"]="B_neighbor_detects_fusion_misses"
            r=analyze(c,scenes,out)
            self.assertEqual(r["decision"],"INCONCLUSIVE")
            self.assertFalse(r["simultaneous_target_structure_equivalence"])


class RecoveryAndStatic(unittest.TestCase):
    def test_only_absent_old_forensic_PCD_is_optional(self):
        from scorw_r3.old import audit_historical_r2
        with scratch() as tmp:
            root=Path(tmp)/"old"
            transaction(root,dict(protocol="immutable"),lambda k:(atomic_text(root/"diagnostic_pcd/field_0.pcd","forensic"),atomic_text(root/"required.txt","array proxy")))
            self.assertEqual(audit_historical_r2(root),[])
            (root/"diagnostic_pcd/field_0.pcd").unlink()
            self.assertEqual(audit_historical_r2(root),["diagnostic_pcd/field_0.pcd"])
            atomic_text(root/"required.txt","altered")
            with self.assertRaises(Blocked):audit_historical_r2(root)

    def test_historical_root_report_cannot_change_midrun(self):
        from scorw_r3.old import OldRun
        with scratch() as tmp:
            out=Path(tmp);atomic_json(out/"REPORT_MANIFEST.json",{})
            atomic_json(out/"OLD.json",dict(result=1))
            old=OldRun.__new__(OldRun);old.root=out;old.manifest=digest(out/"REPORT_MANIFEST.json")
            old.expected_reports={"OLD.json":dict(sha256=digest(out/"OLD.json"))};old.rows=[]
            old.stage=lambda name,deep:None
            old.verify()
            atomic_json(out/"OLD.json",dict(result=2))
            with self.assertRaises(Blocked):old.verify()

    def test_blocked_run_cannot_publish_stale_positive_decision(self):
        from scorw_r3.run import finalize
        with scratch() as tmp:
            out=Path(tmp)/"out";out.mkdir()
            c=deepcopy(C);c["logs_root"]=str(Path(tmp)/"logs")
            atomic_json(out/"R3_PROJECT_DECISION.json",dict(decision="CONTINUE_DATASET_RESEARCH"))
            atomic_json(out/"R3A_GATE.json",dict(status="BLOCKED_PHYSICS_ERROR"))
            finalize(out,c,dict(mode="full",status="blocked",stages={},reason="synthetic math failure"))
            self.assertEqual(read_json(out/"R3_PROJECT_DECISION.json")["decision"],"STOP_OR_REDESIGN")
            self.assertTrue((out/"SMALL_RESULTS.tar.gz").is_file())
            for n,r in read_json(out/"REPORT_MANIFEST.json").items():self.assertEqual(digest(out/n),r["sha256"])

    def test_interruption_reuses_only_committed_subwork(self):
        with scratch() as tmp:
            out=Path(tmp)/"phase";calls=[]
            def first(code):
                transaction(out/"cache",dict(parent=code),lambda k:(calls.append(k),atomic_text(out/"cache/result.txt","fixed")))
                raise InterruptedError()
            with self.assertRaises(InterruptedError):transaction(out,dict(protocol="a"),first)
            transaction(out,dict(protocol="a"),lambda code:transaction(out/"cache",dict(parent=code),lambda k:calls.append("wrong")))
            self.assertEqual(len(calls),1)
            with self.assertRaises(ValueError):transaction(out,dict(protocol="changed"),lambda k:None)
            atomic_text(out/"cache/result.txt","modified")
            with self.assertRaises(ValueError):transaction(out,dict(protocol="a"),lambda k:None)

    def test_snapshot_does_not_rely_on_Git_or_track_private_data(self):
        text=(HERE/"run_all.sh").read_text("utf-8")
        self.assertTrue(text.startswith("#!/bin/sh\n"))
        self.assertNotIn(b"\r",(HERE/"run_all.sh").read_bytes())
        for token in ("pipefail","[[", "git ","curl ","wget "):
            self.assertNotIn(token,text)

    def test_python310_syntax_all_new_sources(self):
        for p in HERE.rglob("*.py"):
            if "local_validation" not in p.parts:ast.parse(p.read_text("utf-8"),filename=str(p),feature_version=(3,10))


if __name__=="__main__":unittest.main()
