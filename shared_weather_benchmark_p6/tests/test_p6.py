import copy
from contextlib import contextmanager, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from shared_weather_benchmark_p6.core import (HERE,read_json,atomic_json,atomic_bytes,digest,identity,
    begin,verify_hashes,bin_name)
from shared_weather_benchmark_p6.legacy import verify_contract
from shared_weather_benchmark_p6.metrics import signal,target_signal,check_denominators,pearson,summarize,cloud_rows
from shared_weather_benchmark_p6.selection import choose_scenes,build_freeze,scope_selection,repeat_seeds,contiguous_block
from shared_weather_benchmark_p6.comparison import matching_details,nested_summary,summarize_repeats,decision


@contextmanager
def temporary():
    parent = HERE/"local_validation"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="synthetic-",dir=parent) as name:
        path = Path(name).resolve()
        assert path.parent == parent.resolve()
        yield path


def config():
    return read_json(HERE/"config/protocol.json")


def row(cav="1",ego=True,clean=1.,weather=.2,mode="E2"):
    return dict(scene="scene",timestamp="0001",object_id="object",cav=cav,is_ego=ego,active=True,
        mode=mode,partition="physics_validation",repeat_index=0,clean_gt_sha256="GT",
        distance_m=10.,view_angle_rad=0.,clean_hard_count=4 if clean > 0 else 0,weather_hard_count=4 if clean > 0 else 0,
        clean_signal=clean,weather_signal=weather,serialized_signal=weather,R=weather/clean if clean else None,
        R_post=weather/clean if clean else None,retention_ratio=1 if clean else None)


class IdentityTests(unittest.TestCase):
    def fixture(self,root):
        req = dict(synthetic=True,source_identity="frozen")
        code = identity(req)
        atomic_json(root/"CONTRACT.json",dict(identity=code,request=req))
        atomic_bytes(root/"payload",b"original synthetic bytes\n")
        atomic_json(root/"VALIDATION_GATES.json",dict(contract=code,status="complete",stages={
            f"P{i}":dict(status="passed",hashes={"payload":digest(root/"payload")}) for i in range(6)}))
        return code

    def test_legacy_contract_and_all_registered_artifacts(self):
        with temporary() as root:
            code = self.fixture(root)
            self.assertEqual(verify_contract(root,code)[0]["identity"],code)

    def test_legacy_audit_is_read_only(self):
        with temporary() as root:
            code = self.fixture(root)
            before = {p.name:(digest(p),p.stat().st_mtime_ns) for p in root.iterdir()}
            verify_contract(root,code)
            self.assertEqual(before,{p.name:(digest(p),p.stat().st_mtime_ns) for p in root.iterdir()})

    def test_wrong_legacy_contract_rejected(self):
        with temporary() as root:
            self.fixture(root)
            with self.assertRaisesRegex(ValueError,"contract"):
                verify_contract(root,"different")

    def test_tampered_artifact_rejected(self):
        with temporary() as root:
            code = self.fixture(root)
            (root/"payload").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"artifact"):
                verify_contract(root,code)

    def test_missing_artifact_rejected(self):
        with temporary() as root:
            code = self.fixture(root)
            (root/"payload").unlink()
            with self.assertRaises(ValueError):
                verify_contract(root,code)

    def test_failed_legacy_gate_rejected(self):
        with temporary() as root:
            code = self.fixture(root)
            gates = read_json(root/"VALIDATION_GATES.json")
            gates["stages"]["P4"]["status"] = "failed"
            atomic_json(root/"VALIDATION_GATES.json",gates)
            with self.assertRaises(ValueError):
                verify_contract(root,code)

    def test_manifest_path_traversal_rejected(self):
        with temporary() as root:
            with self.assertRaises(ValueError):
                verify_hashes(root,{"../outside":"fake"})

    def test_nonfinite_json_rejected(self):
        with temporary() as root:
            for text in ('{"a": NaN}','{"a": Infinity}','{"a": 1e999}'):
                (root/"bad.json").write_text(text)
                with self.assertRaises(ValueError):
                    read_json(root/"bad.json")

    def test_protocol_change_requires_new_contract_before_write(self):
        with temporary() as root:
            begin(root,dict(revision=1,smoke=True),False)
            atomic_bytes(root/"weather.bin",b"committed")
            before = digest(root/"weather.bin")
            with self.assertRaisesRegex(ValueError,"Contract changed"):
                begin(root,dict(revision=2,smoke=True),True)
            self.assertEqual(before,digest(root/"weather.bin"))

    def test_smoke_and_formal_contracts_cannot_mix(self):
        with temporary() as root:
            begin(root,dict(smoke=True),False)
            with self.assertRaises(ValueError):
                begin(root,dict(smoke=False),True)


class SignalTests(unittest.TestCase):
    def calc(self,clean,weather,quant=None,tau=None):
        clean,weather = np.array(clean),np.array(weather)
        return signal(clean,weather,weather if quant is None else np.array(quant),
                      np.ones(len(clean),bool),np.zeros(len(clean)) if tau is None else np.array(tau),config()["thresholds"])

    def test_signal_is_sum_ratio_not_mean_point_ratios(self):
        result = self.calc([1.,.1],[.5,.01])
        self.assertAlmostEqual(result["R"],.51/1.1)
        self.assertNotAlmostEqual(result["R"],(.5+.1)/2)

    def test_zero_clean_signal_null_and_missing_reason(self):
        result = self.calc([0.,0.],[0.,0.])
        self.assertIsNone(result["R"])
        self.assertEqual(result["missing_reason"],"zero_clean_signal")

    def test_no_clean_points_null(self):
        result = self.calc([],[])
        self.assertIsNone(result["R"])
        self.assertEqual(result["missing_reason"],"no_clean_points")

    def test_zero_alpha_signal_ratio_one(self):
        self.assertEqual(self.calc([.1,.5],[.1,.5])["R"],1.)

    def test_retention_does_not_change_with_weather_strength(self):
        for alpha in (0.,.06,.1):
            result = self.calc([.1,.5],np.array([.1,.5])*math.exp(-2*alpha*30))
            self.assertEqual(result["raw_retention"],1.)
            self.assertEqual(result["retained_points"],2)

    def test_quantization_is_separate_and_zero_count_explicit(self):
        result = self.calc([.1,.2],[.0001,.01],[0.,3/255])
        self.assertNotEqual(result["R"],result["R_post"])
        self.assertEqual(result["quantization_only_zero_points"],1)

    def test_clean_already_below_is_not_weather_new_loss(self):
        result = self.calc([.001,.02],[.0001,.002],[0.,1/255])
        t = result["threshold"]["0.01"]
        self.assertEqual((t["clean_already_below"],t["newly_below_pre"]),(1,1))

    def test_nonfinite_arrays_stop(self):
        with self.assertRaises(ValueError):
            self.calc([1.],[float("nan")])

    def test_no_neighbor_C_null(self):
        t = target_signal([row()])
        self.assertIsNone(t["C"])
        self.assertEqual(t["valid_neighbor_count"],0)

    def test_neighbor_zero_clean_signal_is_not_valid(self):
        t = target_signal([row(),row("2",False,clean=0.,weather=0.)])
        self.assertEqual(t["neighbor_count"],1)
        self.assertEqual(t["valid_neighbor_count"],0)
        self.assertIsNone(t["C"])

    def test_missing_ego_C_null(self):
        t = target_signal([row(clean=0.,weather=0.),row("2",False)])
        self.assertIsNone(t["C"])
        self.assertEqual(t["C_missing_reason"],"ego_signal_missing")

    def test_neighbor_count_changes_maximum(self):
        a = target_signal([row(weather=.1),row("2",False,weather=.2)])
        b = target_signal([row(weather=.1),row("2",False,weather=.2),row("3",False,weather=.9)])
        self.assertEqual((a["valid_neighbor_count"],b["valid_neighbor_count"]),(1,2))
        self.assertGreater(b["C"],a["C"])

    def test_constant_field_different_distances_different_ratios(self):
        from shared_weather_benchmark.world_field import WorldField
        from shared_weather_benchmark.ray_integrator import integrate
        field = WorldField(kind="constant",mean_alpha=.06)
        ratios = [math.exp(-2*integrate(field,[0,0,0],[d,0,0])) for d in (10.,20.)]
        self.assertGreater(ratios[0],ratios[1])

    def test_shared_field_can_have_no_neighbor_advantage(self):
        t = target_signal([row(weather=math.exp(-1.2)),row("2",False,weather=math.exp(-2.4))])
        self.assertLess(t["C"],0)

    def test_absolute_clean_signal_is_reported(self):
        t = target_signal([row(clean=100.,weather=10.),row("2",False,clean=1.,weather=.9)])
        self.assertGreater(t["C"],0)
        self.assertGreater(t["ego_weather_signal"],t["best_neighbor_weather_signal"])

    def test_same_GT_CAV_denominator_with_different_signals(self):
        self.assertTrue(check_denominators([row(mode="E2"),row(mode="E3",weather=.5)],["E2","E3"]))

    def test_different_GT_CAV_denominator_stops(self):
        bad = row(mode="E3");bad["object_id"] = "other"
        with self.assertRaises(ValueError):
            check_denominators([row(),bad],["E2","E3"])

    def test_correlation_missing_and_constant_not_faked(self):
        self.assertIsNone(pearson([1,1,1],[1,2,3])["value"])
        self.assertIsNone(pearson([None,1],[1,2])["value"])
        self.assertAlmostEqual(pearson([1,2,3],[3,2,1])["value"],-1)

    def test_A_B_count_labels_inherit_clean_even_with_weak_signal(self):
        from shared_weather_benchmark.observability import classify
        for quality in (.8,.02,0.):
            rows = [dict(is_ego=True,weather_hard_count=0,effective_intensity_ratio=None),
                    dict(is_ego=False,weather_hard_count=5,effective_intensity_ratio=quality)]
            self.assertTrue(classify(rows).startswith("B_"))
            rows[1]["weather_hard_count"] = 0
            self.assertTrue(classify(rows).startswith("A_"))

    def test_fixed_world_boxes_to_full_scene_summary_on_synthetic_arrays(self):
        objects = {str(i):dict(location=[x,0.,0.],center=[0.,0.,0.],angle=[0.,0.,0.],extent=[1.,1.,1.])
                   for i,x in enumerate((10.,20.,30.))}
        frame = dict(ego="1",active_cavs=["1","2"],objects=objects,clean_gt_sha256=identity(objects))
        rows = []
        for mode in ("E1","E2","E3"):
            for cav,offset in (("1",0.),("2",2.)):
                raw = np.array([[x-offset,0.,0.,j/10] for j,x in enumerate((10.,20.,30.),start=1)],dtype=np.float32)
                tau = np.linalg.norm(raw[:,:3].astype(float),axis=1)*.06
                weather = raw.copy();weather[:,3] = raw[:,3]*np.exp(-2*tau)
                quantized = weather.copy();quantized[:,3] = np.rint(weather[:,3]*255)/255
                data = dict(original_xyzi=raw,weather_xyzi=weather,serialized_xyzi=quantized,integral=tau,
                    emitted=np.ones(3,bool),retained=np.ones(3,bool),origin=np.zeros(3,np.uint8))
                source = dict(scene="s",timestamp="0001",cav=cav,pose=[offset,0,0,0,0,0],partition="physics_validation")
                rows.extend(cloud_rows(data,source,frame,mode,0,config()["thresholds"]))
        result = summarize(rows,config(),["E1","E2","E3"])
        self.assertEqual(len(rows),18)
        self.assertEqual(result["by_scene"]["s"]["E2"]["C"]["valid"],3)
        self.assertEqual(result["by_scene"]["s"]["E2"]["R"]["valid"],6)
        self.assertEqual(result["pair_analysis"]["s"]["E2"]["all"]["R_correlation"]["paired"],3)


class SelectionTests(unittest.TestCase):
    def fixture(self):
        c = config()
        dev = ["d"+str(i) for i in range(7)]
        val = ["v"+str(i) for i in range(11)]
        future = ["f0","f1","f2"]
        old = dict(all_scene_partition=dict(generator_development=dev,physics_validation=val,future_model_development=future),
            selected=dict(future_model_development=["f0","f1"]),frames=[dict(scene=s) for s in ("d0","v0")],
            profiles={s:dict(cav_count=2,initial_xy_span_m=40.) for s in dev+val+future})
        c["historical_scenes"] = ["d0","v0"]
        inventory = {s:dict(cavs=["1","2"],timestamps=["0001","0002","0003","0004"])
                     for s in dev+val}
        return c,old,inventory

    def test_historical_and_all_future_partition_excluded(self):
        c,old,_ = self.fixture()
        plan = choose_scenes(old,c)
        selected = {s for names in plan["selected"].values() for s in names}
        self.assertEqual(len(selected),12)
        self.assertFalse(selected & {"d0","v0","f0","f1","f2"})

    def test_insufficient_candidates_stop_without_future_fallback(self):
        c,old,_ = self.fixture()
        old["all_scene_partition"]["physics_validation"] = ["v0","v1"]
        old["profiles"] = {s:p for s,p in old["profiles"].items() if s in sum(old["all_scene_partition"].values(),[])}
        with self.assertRaisesRegex(ValueError,"Insufficient"):
            choose_scenes(old,c)

    def test_scene_timestamp_seed_freeze_reproducible(self):
        c,old,inventory = self.fixture()
        a = build_freeze(old,c,inventory,"source")
        b = build_freeze(old,c,inventory,"source")
        self.assertEqual(a,b)
        self.assertEqual(len(a["frames"]),24)

    def test_repeat_seeds_cannot_depend_on_results(self):
        c = config()
        seeds = repeat_seeds(c)
        c["unmatched_results_forbidden_input"] = {"anything":"worse"}
        self.assertEqual(seeds,repeat_seeds(c))
        self.assertEqual(len(set(seeds.values())),3)

    def test_smoke_is_declared_subset_of_complete_formal_plan(self):
        c,old,inventory = self.fixture()
        freeze = build_freeze(old,c,inventory,"source")
        smoke,full = scope_selection(freeze,c,True),scope_selection(freeze,c,False)
        self.assertEqual(len(smoke["frames"]),4)
        self.assertEqual(smoke["full_plan"],full["full_plan"])
        self.assertTrue({f["scene"] for f in smoke["frames"]} <= {f["scene"] for f in full["frames"]})

    def test_noncontinuous_time_stops_no_reselection(self):
        with self.assertRaises(ValueError):
            contiguous_block(["1","2","4"],3,2)


class AnalysisTests(unittest.TestCase):
    def physical(self,t,alpha=.06):
        return dict(mean_transmittance=t,hard_retention=1.,mean_integral=alpha*10,
                    bins=[dict(clean=10,mean_transmittance=t,retention=1.)])

    def test_matching_reports_each_failure(self):
        c = config()
        result = matching_details(self.physical(.3),self.physical(.1),.07,.06,c["matching"])
        self.assertFalse(result["matched"])
        self.assertIn("alpha_mean",result["failed_conditions"])
        self.assertIn("distance_bin_T/0",result["failed_conditions"])

    def test_empty_bin_explicitly_not_tested(self):
        a,b = self.physical(.1),self.physical(.1)
        for v in (a,b):
            v["bins"] = [dict(clean=0,mean_transmittance=None,retention=None)]
        result = matching_details(a,b,.06,.06,config()["matching"])
        self.assertIsNone(result["checks"]["distance_bin_T/0"]["passed"])

    def test_seeds_average_inside_scene_not_independent_units(self):
        result = nested_summary([dict(scene="a",x=x) for x in (0.,0.,3.)]+[dict(scene="b",x=0.) for _ in range(3)],"x",config()["bootstrap"])
        self.assertEqual(result["independent_units"],2)
        self.assertEqual(result["count_scene_seed_results"],6)
        self.assertAlmostEqual(result["aggregate"]["mean"],.5)
        self.assertIsNone(result["aggregate"]["ci95"])

    def test_scene_bootstrap_deterministic(self):
        rows = [dict(scene=str(i),x=i/100) for i in range(8) for _ in range(3)]
        self.assertEqual(nested_summary(rows,"x",config()["bootstrap"]),nested_summary(rows,"x",config()["bootstrap"]))

    def test_strata_edges_do_not_silently_drop_data(self):
        self.assertEqual(bin_name(20,[0,20,40]),"[20,40)")
        with self.assertRaises(ValueError):
            bin_name(80,[0,20,40])

    def test_unmatched_all_retained_and_repeat_inventory_enforced(self):
        c = config()
        rows = []
        for scene in ("a","b"):
            for i in range(3):
                rows.append(dict(scene=scene,repeat_index=i,partition="physics_validation" if scene == "a" else "generator_development",
                    matching=dict(matched=False,mean_transmittance_delta=.1,alpha_delta=.01,failed_conditions=["mean_T"]),
                    delta_C=.01,delta_R=.02,delta_intensity_weighted_T=.02,delta_pair_R_correlation=None,
                    delta_weather_added_absolute_gap=.3,strata={}))
        result = summarize_repeats(rows,c,{"a","b"},[0,1,2])
        self.assertEqual(len(result["unmatched"]),6)
        self.assertEqual(result["partitions"]["all_scenes"]["scene_count"],2)
        self.assertEqual(decision(result,c,True)["decision"],"INCONCLUSIVE")
        with self.assertRaises(ValueError):
            summarize_repeats(rows[:-1],c,{"a","b"},[0,1,2])


class GenerationResumeTests(unittest.TestCase):
    def test_synthetic_array_backend_resume_preserves_weather_and_rejects_new_contract(self):
        # No real PCD/Open3D/OPV2V: storage adapter uses NumPy .npy bytes only.
        import shared_weather_benchmark.generator as generator
        from reference_weather import common
        with temporary() as root:
            source = root/"synthetic_clean.bin"
            yaml = root/"synthetic_yaml.bin"
            source.write_bytes(b"synthetic input identity")
            yaml.write_bytes(b"synthetic metadata identity")
            raw = np.array([[5.,1.,0.,.2],[15.,2.,0.,.5],[30.,3.,0.,1.]],dtype=np.float32)
            item = dict(scene="s",timestamp="1",cav="1",pose=[0,0,0,0,0,0],partition="generator_development",
                pcd=str(source),yaml=str(yaml),pcd_sha256=digest(source),yaml_sha256=digest(yaml))
            atomic_json(root/"INPUT_MANIFEST.json",[item]);atomic_json(root/"CLEAN_GT.json",[])
            atomic_json(root/"FIELD_DOMAINS.json",dict(s=[-100,-100,-100,100,100,100]))
            c = dict(seed=3,protocols=["E1","E2","E3"],field=config()["field"],integration_for_generation=dict(step_m=.5,batch=8),
                return_policy=dict(name="attenuation_only"),distance_bins_m=[0,20,1000])
            def read(path):
                if Path(path) == source:
                    return raw.copy()
                with Path(path).open("rb") as stream:
                    return np.load(stream,allow_pickle=False)
            def write(path,points):
                path.parent.mkdir(parents=True,exist_ok=True)
                stream = io.BytesIO();np.save(stream,points,allow_pickle=False)
                atomic_bytes(path,stream.getvalue())
            def metadata(src,target):
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    target.write_bytes(src.read_bytes())
            with patch.object(generator,"read_points",side_effect=read),patch.object(common,"write_pcd",side_effect=write),patch.object(generator,"ensure_link",side_effect=metadata),redirect_stdout(io.StringIO()):
                generator.generate(c,root,"contract",False)
                files = list(root.rglob("*.pcd"))+list(root.rglob("*.npz"))
                before = {str(p):(digest(p),p.stat().st_mtime_ns) for p in files}
                generator.generate(c,root,"contract",True)
                self.assertEqual(before,{str(p):(digest(p),p.stat().st_mtime_ns) for p in files})
                for mode in c["protocols"]:
                    with np.load(root/f"sidecars/{mode}/train/s/1/1.npz",allow_pickle=False) as data:
                        self.assertTrue(data["retained"].all())
                with self.assertRaises(ValueError):
                    generator.generate(c,root,"new-contract",True)


class ExportTests(unittest.TestCase):
    def test_export_only_registered_small_files_and_idempotent(self):
        from shared_weather_benchmark_p6.export_audit import export,NAMES
        with temporary() as root:
            source = root/"audit";source.mkdir()
            destination = root/"export"
            for name in NAMES:
                (source/name).write_bytes(b"synthetic report\n")
            (source/"large.pcd").write_bytes(b"must not copy")
            products = [dict(path=str((source/n).resolve()),sha256=digest(source/n)) for n in NAMES]
            atomic_json(source/"COMMIT.json",dict(status="passed",contract="synthetic",products=products))
            result = export(source,destination)
            self.assertEqual(result,export(source,destination))
            self.assertFalse((destination/"large.pcd").exists())
            with self.assertRaises(ValueError):
                export(source,root/"too-small",1)


class StageTests(unittest.TestCase):
    def test_committed_stage_inventory_tamper_stops(self):
        from shared_weather_benchmark_p6.run import phase
        with temporary() as root:
            stage = root/"stage"
            request = dict(stage="synthetic")
            writer = lambda contract:atomic_json(stage/"result.json",dict(status="complete"))
            first = phase(stage,request,writer)
            self.assertEqual(first,phase(stage,request,lambda code:self.fail("Committed work re-executed")))
            commit = read_json(stage/"COMMIT.json")
            commit["products"] = []
            atomic_json(stage/"COMMIT.json",commit)
            with self.assertRaisesRegex(ValueError,"inventory"):
                phase(stage,request,writer)

    def test_failed_legacy_identity_is_recorded_as_stopped_P6A(self):
        from argparse import Namespace
        import shared_weather_benchmark_p6.run as runner
        with temporary() as root:
            c = config();c["legacy_root"] = str(root/"old")
            path = root/"config.json";atomic_json(path,c)
            args = Namespace(config=path,mode="audit",smoke=False,resume=False,check_output=None)
            with patch.object(runner,"verify_code",return_value="synthetic"),patch.object(runner,"runtime_identity",return_value={"synthetic":True}),patch.object(runner,"server_paths",return_value=(root/"data",root/"logs")),patch("shared_weather_benchmark_p6.legacy.verify_legacy",side_effect=ValueError("synthetic missing artifact")):
                with self.assertRaisesRegex(ValueError,"missing artifact"):
                    runner.execute(args)
            state = read_json(root/"logs/state-full/VALIDATION_GATES.json")
            self.assertEqual(state["status"],"stopped")
            self.assertEqual(state["failure"]["stage"],"P6-A")

    def test_full_report_includes_all_scene_seed_rows_and_allows_negative_effect(self):
        from shared_weather_benchmark_p6.comparison import write_results
        c = config()
        rows = []
        for s in range(12):
            for i in range(3):
                rows.append(dict(scene=str(s),repeat_index=i,partition="generator_development" if s < 4 else "physics_validation",
                    cav_count=2,matching=dict(matched=True,mean_transmittance_delta=0.,alpha_delta=0.,failed_conditions=[],
                    checks={},mean_optical_depth_delta=0.,retention_delta=0.,distance_bin_transmittance_delta=[0.],distance_bin_retention_delta=[0.]),
                    delta_C=-.02,delta_R=0.,delta_intensity_weighted_T=0.,delta_pair_R_correlation=-.02,
                    delta_weather_added_absolute_gap=-.01,strata={}))
        result = summarize_repeats(rows,c,{str(i) for i in range(12)},[0,1,2])
        self.assertEqual(decision(result,c,False)["decision"],"CONTINUE")
        with temporary() as root:
            write_results(root,result,c,False)
            summary = read_json(root/"SUMMARY.json")
            self.assertEqual((summary["independent_scenes"],summary["scene_seed_pairs"]),(12,36))
            self.assertEqual(len((root/"P6C_SCENE_SEED_RESULTS.csv").read_text().splitlines()),37)
        for r in rows:
            r["delta_C"] = r["delta_pair_R_correlation"] = 0.
        null_result = summarize_repeats(rows,c,{str(i) for i in range(12)},[0,1,2])
        self.assertEqual(decision(null_result,c,False)["decision"],"STOP_OR_REDESIGN")


if __name__ == "__main__":
    unittest.main()
