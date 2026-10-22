import copy
import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scorw_p7 import core
from scorw_p7.balanced_assignment import assignment
from scorw_p7.field_pool import make_pool, restore_field
from scorw_p7.integrate_pool import strict_balance, integrate_cloud
from scorw_p7.extract_rays import sample_cloud
from scorw_p7.signal_metrics import calculate, analyze_scene, centered_pair
from scorw_p7.select_scenes import choose
from scorw_p7.statistics import analyze
from scorw_p7.decision import decide
from shared_weather_benchmark.ray_integrator import integrate, integrate_numpy, transmittance
from shared_weather_benchmark.world_field import WorldField
from shared_weather_benchmark.geometry import numerical_audit


def config():
    return core.read_json(core.HERE/"config/protocol.json")


def rows(cavs=("ego","b"), targets=2, scene="synthetic-scene"):
    result = []
    phase = np.arange(20)*2*np.pi/20
    for oid in range(targets):
        for j,cav in enumerate(cavs):
            signal = .4 + .15*np.sin(phase+.2*j+.03*oid)
            tau = -np.log(signal)/2
            result.append(dict(scene=scene,partition="generator_development",timestamp="000100",object_id=str(oid),
                cav=cav,ego="ego",is_ego=cav=="ego",active=True,clean_gt_sha256="fixed-synthetic-gt",
                clean_points=8,sampled_points=8,clean_signal=4.,sample_fraction=1.,view_angle_rad=0.,distance_m=20.+oid,
                fixed_weighted_clean_denominator=4.,missing_reason=None,R=signal.tolist(),T=signal.tolist(),tau=tau.tolist()))
    return result


def history(available=10):
    dev=[f"dev{i}" for i in range(4)]
    val=[f"oldval{i}" for i in range(8)]
    old=["old0","old1"]
    fresh=[f"fresh{i}" for i in range(available)]
    future=["reserved0","reserved1"]
    all_names=dev+val+old+fresh+future
    return dict(identity="synthetic-history",old=dict(
        all_scene_partition=dict(generator_development=dev+old+fresh,physics_validation=val,future_model_development=future),
        profiles={s:dict(cav_count=2,initial_xy_span_m=20.) for s in all_names},
        selected=dict(future_model_development=future[:1]),frames=[dict(scene=s) for s in old]),
        p6=dict(selected=dict(generator_development=dev,physics_validation=val),
                frames=[dict(scene=s) for s in dev+val],excluded_future=future))


class AssignmentTests(unittest.TestCase):
    def test_only_context_document_is_review_only(self):
        from scorw_p7.audit_sources import verify_runtime_legacy
        verify_runtime_legacy(dict(changed=[dict(path="AI_CONTEXT.md")]))
        with self.assertRaises(ValueError):
            verify_runtime_legacy(dict(changed=[dict(path="shared_weather_benchmark/world_field.py")]))
        with self.assertRaises(ValueError):
            verify_runtime_legacy(dict(missing=["shared_weather_benchmark/ray_integrator.py"]))

    def test_all_twenty_exactly_once_per_vehicle(self):
        for n in (1,2,3,7,20):
            cavs=["ego"]+[str(j) for j in range(n-1)]
            p=assignment(cavs,"ego")
            for mode,schedule in p["schedules"].items():
                for v in schedule.values():
                    self.assertEqual(sorted(v),list(range(20)))
            for j in range(20):
                self.assertEqual(len({p["schedules"]["E3"][c][j] for c in cavs}),n)

    def test_order_invariance(self):
        self.assertEqual(assignment(["4","ego","2"],"ego"),assignment(["2","4","ego"],"ego"))

    def test_more_than_twenty_stops(self):
        with self.assertRaises(ValueError):
            assignment(["ego"]+list(map(str,range(20))),"ego")

    def test_duplicate_and_wrong_ego(self):
        for cavs,ego in ((["a","a"],"a"),(["a"],"b")):
            with self.assertRaises(ValueError):
                assignment(cavs,ego)

    def test_per_ray_marginals_and_single_repeat_difference(self):
        p=assignment(["ego","b"],"ego")
        data=np.arange(60,dtype=float).reshape(20,3)/7
        strict_balance(data,p,config()["tolerance"])
        self.assertFalse(np.array_equal(data[p["schedules"]["E2"]["b"]][0],data[p["schedules"]["E3"]["b"]][0]))

    def test_corrupted_schedule_stops_balance(self):
        p=assignment(["ego","b"],"ego")
        p["schedules"]["E3"]["b"][0]=p["schedules"]["E3"]["b"][1]
        with self.assertRaises(ValueError):
            strict_balance(np.arange(20.)[:,None],p,config()["tolerance"])


class PhysicsTests(unittest.TestCase):
    def test_twenty_unique_fields_and_scene_separation(self):
        p=make_pool("a",config(),[-200]*3+[200]*3)
        self.assertEqual(len({v["sha256"] for v in p["fields"]}),20)
        self.assertNotEqual(p["seeds"],make_pool("b",config(),[-200]*3+[200]*3)["seeds"])
        self.assertEqual(p,make_pool("a",config(),[-200]*3+[200]*3))

    def test_changed_seed_changes_contract(self):
        c=config(); a=make_pool("a",c,None)
        c["weather_root_seed"]+=1
        b=make_pool("a",c,None)
        self.assertNotEqual(core.identity(a),core.identity(b))

    def test_exact_repeated_integration_and_marginal_means(self):
        p=make_pool("a",config(),None)
        end=np.array([[20.,0,0],[30,10,2]])
        values=[]
        for field in p["fields"]:
            f=restore_field(field)
            a=integrate_numpy(f,[0,0,0],end,.5,2)
            np.testing.assert_array_equal(a,integrate_numpy(f,[0,0,0],end,.5,2))
            values.append(a)
        a=np.stack(values); plan=assignment(["ego","b"],"ego")
        for v in (a,np.exp(-2*a),np.exp(-2*a)@np.array([.2,.8])):
            strict_balance(v[:,None] if v.ndim==1 else v,plan,config()["tolerance"])

    def test_zero_constant_layered_and_reverse(self):
        self.assertEqual(transmittance(integrate(WorldField(mean_alpha=0,kind="constant"),[0,0,0],[10,0,0])),1.)
        f=WorldField(mean_alpha=.06,kind="constant")
        self.assertAlmostEqual(integrate(f,[0,0,0],[10,0,0]),.6,places=13)
        f=WorldField(mean_alpha=0,kind="layered",layers=[[0,5,.1],[5,10,.2]])
        self.assertAlmostEqual(integrate(f,[0,0,0],[0,0,10],.25),1.5,places=13)
        self.assertAlmostEqual(integrate(f,[0,0,10],[0,0,0],.25),1.5,places=13)

    def test_smooth_convergence_and_shared_position(self):
        f=restore_field(make_pool("a",config(),None)["fields"][0])
        a,b=[0,0,0],[40,10,2]
        error=abs(integrate(f,a,b,.5)-integrate(f,a,b,.25))
        self.assertLess(error,.002)
        self.assertAlmostEqual(integrate(f,a,b,.5),integrate(f,b,a,.5),places=12)
        self.assertEqual(f.alpha([1,2,3]),restore_field(f.manifest()).alpha([1,2,3]))

    def test_coordinate_roundtrip(self):
        self.assertLess(numerical_audit()["roundtrip_max_m"],1e-10)


class MetricsTests(unittest.TestCase):
    def test_single_neighbor_zero_mean_but_nonzero_structure(self):
        out=calculate(rows(),assignment(["ego","b"],"ego"),config())
        self.assertLess(abs(out["metrics"]["delta_C"]),1e-12)
        self.assertGreater(abs(out["metrics"]["delta_C_std"]),.001)
        self.assertEqual(out["negative_control"]["status"],"passed")

    def test_multiple_neighbors_and_counts(self):
        cavs=["ego","b","c"]
        out=calculate(rows(cavs),assignment(cavs,"ego"),config())
        self.assertEqual(out["subgroups"]["multiple_neighbors"]["target_instances"],2)
        self.assertEqual(out["counts"]["candidate_target_cav_pairs"],6)

    def test_zero_signal_and_no_neighbor(self):
        rr=rows()
        for r in rr:
            if r["cav"]=="b":
                r.update(R=None,clean_signal=0.,fixed_weighted_clean_denominator=0.,missing_reason="zero_clean_signal")
        out=calculate(rr,assignment(["ego","b"],"ego"),config())
        self.assertEqual(out["counts"]["valid_target_cav_pairs"],0)
        self.assertIsNone(out["metrics"]["delta_C"])
        self.assertEqual(out["counts"]["missing_C"]["no_valid_neighbor"],2)

    def test_zero_weather_variance_is_null(self):
        p=centered_pair([.5]*20,[.4]*20,1e-20)
        self.assertIsNone(p["correlation"])
        self.assertEqual(p["reason"],"zero_or_near_zero_weather_variance")

    def test_no_cross_target_distance_correlation(self):
        rr=rows(targets=4)
        for r in rr:
            r["R"]=[.1+int(r["object_id"])*.1]*20
        out=calculate(rr,assignment(["ego","b"],"ego"),config())
        self.assertIsNone(out["metrics"]["delta_correlation"])
        self.assertEqual(out["counts"]["valid_target_cav_pairs"],0)

    def test_row_order_invariance(self):
        rr=rows(["ego","c","b"])
        p=assignment(["b","ego","c"],"ego")
        a=calculate(rr,p,config())
        b=calculate(list(reversed(rr)),p,config())
        self.assertEqual(a["metrics"],b["metrics"])
        self.assertEqual(a["targets"],b["targets"])

    def test_incomplete_pool_and_denominator_fail(self):
        rr=rows(); rr[0]["R"].pop()
        with self.assertRaises(ValueError):
            calculate(rr,assignment(["ego","b"],"ego"),config())
        rr=rows(); rr[0]["fixed_weighted_clean_denominator"]=0
        with self.assertRaises(ValueError):
            calculate(rr,assignment(["ego","b"],"ego"),config())

    def test_cross_scene_mixing_rejected(self):
        with self.assertRaises(ValueError):
            calculate(rows()+rows(scene="other"),assignment(["ego","b"],"ego"),config())

    def test_scene_bootstrap_never_uses_twenty_as_scenes(self):
        scene=analyze_scene(rows(),assignment(["ego","b"],"ego"),config())
        a=analyze([scene],config())
        p=a["partitions"]["generator_development"]
        self.assertEqual(p["independent_scene_count"],1)
        self.assertIsNone(p["metrics"]["delta_correlation"]["ci95"])
        self.assertEqual(len(scene["leave_one_field"]["results"]),20)


class SelectionTests(unittest.TestCase):
    def test_unused_scenes_and_future_isolation(self):
        h=history(); p=choose(h,config())
        chosen=set(p["selected"]["independent_validation"])
        self.assertEqual(len(chosen),8)
        self.assertFalse(chosen & set(p["excluded_future"]))
        self.assertFalse(chosen & set(p["excluded_historical"]))
        self.assertEqual(p,choose(h,config()))

    def test_insufficient_new_scenes_retains_development_only(self):
        p=choose(history(7),config())
        self.assertEqual(len(p["selected"]["generator_development"]),4)
        self.assertEqual(p["selected"]["independent_validation"],[])
        self.assertEqual(p["validation_status"],"INCONCLUSIVE_VALIDATION_UNAVAILABLE")

    def test_overlapping_partition_rejected(self):
        h=history(); h["old"]["all_scene_partition"]["future_model_development"].append("dev0")
        with self.assertRaises(ValueError):
            choose(h,config())

    def test_test_root_rejected_before_data_read(self):
        from scorw_p7.core import server_train
        with patch("scorw_p7.core.platform.system",return_value="Linux"):
            with self.assertRaises((ValueError,FileNotFoundError)):
                server_train("/data/scd/datasets/opv2v_official_data_dumping/test")


class TransactionTests(unittest.TestCase):
    def setUp(self):
        base=core.HERE/"local_validation"
        base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix=".unit-",dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_resume_and_corruption(self):
        root=self.root/"run"; calls=[]
        def work(code):
            calls.append(1); core.atomic_json(root/"value.json",dict(value=4))
        first=core.transaction(root,dict(seed=1),work)
        second=core.transaction(root,dict(seed=1),work)
        self.assertEqual(first,second); self.assertEqual(calls,[1])
        core.atomic_json(root/"value.json",dict(value=5))
        with self.assertRaises(ValueError):
            core.transaction(root,dict(seed=1),work)

    def test_changed_protocol_cannot_resume(self):
        root=self.root/"run"
        core.transaction(root,dict(seed=1),lambda code:None)
        with self.assertRaises(ValueError):
            core.transaction(root,dict(seed=2),lambda code:None)

    def test_interruption_recomputes_uncommitted_values(self):
        root=self.root/"run"
        def fail(code):
            core.atomic_json(root/"value.json",dict(value=3))
            raise RuntimeError("synthetic interruption")
        with self.assertRaises(RuntimeError):
            core.transaction(root,dict(seed=1),fail)
        core.transaction(root,dict(seed=1),lambda code:core.atomic_json(root/"value.json",dict(value=3)))
        self.assertEqual(core.read_json(root/"value.json"),dict(value=3))

    def test_nonfinite_json_rejected(self):
        p=self.root/"bad.json"
        for text in ('{"v": NaN}','{"v": 1e999}'):
            p.write_text(text,encoding="utf-8")
            with self.assertRaises(ValueError):
                core.read_json(p)

    def test_existing_test_directory_is_refused_before_file_read(self):
        from shared_weather_benchmark.dataset_io import approved_path
        forbidden=self.root/"test"
        forbidden.mkdir()
        with self.assertRaisesRegex(ValueError,"Only original OPV2V train"):
            approved_path(forbidden)

    def test_cached_ray_field_results_not_reintegrated(self):
        cloud=self.root/"cloud"; cloud.mkdir()
        core.save_arrays(cloud/"rays.npz",origin=np.zeros(3),world_endpoints=np.array([[10.,0,0]]),
            original_index=np.array([0]),original_xyzi=np.array([[10.,0,0,.5]]),point_id=np.array(["synthetic"]))
        rr=rows(targets=1)[:1]
        rr[0].update(ray_slots=[0],weights=[1.],fixed_weighted_clean_denominator=.5)
        core.atomic_json(cloud/"targets.json",rr)
        core.atomic_json(cloud/"SAMPLING.json",dict(scene="a",timestamp="1",cav="ego",rays_identity="synthetic"))
        pool=make_pool("a",config(),None); p=assignment(["ego","b"],"ego")
        out=self.root/"pool"
        a=integrate_cloud(cloud,out,pool,p,config(),"synthetic")
        with patch("scorw_p7.integrate_pool.integrate_numpy",side_effect=AssertionError("unexpected reintegration")):
            b=integrate_cloud(cloud,out,pool,p,config(),"synthetic")
        self.assertEqual(a,b)
        self.assertEqual(core.read_json(out/"MATRIX.json")["primary_ray_field_pairs"],20)

    def test_sampling_is_clean_only_fixed_and_deduplicated(self):
        raw=np.array([[10.,0,0,0.],[11,0,0,1.],[12,0,0,.5],[13,0,0,.2]])
        row=dict(scene="a",timestamp="1",cav="ego",pcd_sha256="fixed",pose=[0]*6)
        obj=dict(location=[12.,0,0],center=[0]*3,angle=[0]*3,extent=[5.,2.,2.])
        frame=dict(objects={"o":obj,"overlap":obj},ego="ego",active_cavs=["ego","b"],partition="generator_development",clean_gt_sha256="fixed")
        masks={"o":np.arange(4),"overlap":np.arange(4)}
        c=config(); c["sampling"]=dict(c["sampling"],max_per_stratum=1)
        a,rr,audit=sample_cloud(raw,raw[:,:3],row,frame,masks,{"o":2,"overlap":2},c)
        b,ss,other=sample_cloud(raw,raw[:,:3],row,frame,masks,{"o":2,"overlap":2},c)
        np.testing.assert_array_equal(a["original_index"],b["original_index"])
        self.assertEqual(rr,ss); self.assertEqual(audit,other)
        self.assertLessEqual(len(a["original_index"]),audit["sampled_memberships"])
        self.assertTrue(all(r["fixed_weighted_clean_denominator"]>0 for r in rr))

    def test_sampling_budget_insufficient_stops(self):
        raw=np.array([[10.,0,0,1.]])
        obj=dict(location=[10.,0,0],center=[0]*3,angle=[0]*3,extent=[1.]*3)
        c=config(); c["sampling"]=dict(c["sampling"],max_memberships_per_cloud=1)
        frame=dict(objects={"a":obj,"b":obj},ego="ego",active_cavs=["ego"],partition="generator_development",clean_gt_sha256="fixed")
        with self.assertRaises(ValueError):
            sample_cloud(raw,raw[:,:3],dict(scene="a",timestamp="1",cav="ego",pcd_sha256="x",pose=[0]*6),frame,
                         {"a":np.array([0]),"b":np.array([0])},{},c)

    def test_unavailable_validation_decision(self):
        scene=analyze_scene(rows(),assignment(["ego","b"],"ego"),config())
        result=decide(analyze([scene],config()),[scene],choose(history(7),config()),config())
        self.assertEqual(result["decision"],"INCONCLUSIVE_VALIDATION_UNAVAILABLE")

    def test_nested_commit_is_hash_protected(self):
        root=self.root/"outer"
        def work(code):
            core.transaction(root/"inner",dict(parent=code),lambda nested:None)
        core.transaction(root,dict(seed=1),work)
        core.atomic_json(root/"inner/COMMIT.json",dict(tampered=True))
        with self.assertRaises(ValueError):
            core.transaction(root,dict(seed=1),work)

    def test_full_orchestrator_mock_smoke_and_resume(self):
        # Exercise P0..P4 publication with synthetic substitutes for all real IO.
        from scorw_p7 import run
        output,logs=self.root/"smoke",self.root/"logs"
        frame=dict(scene="synthetic-scene",timestamp="000100",cavs=["ego","b"],ego="ego",partition="generator_development")
        selection=dict(selected=dict(generator_development=[frame["scene"]],independent_validation=[]),frames=[frame],
            inventories={frame["scene"]:dict(cavs=frame["cavs"],ego="ego")},available_validation_candidates=0,
            validation_status="INCONCLUSIVE_VALIDATION_UNAVAILABLE")
        cfg=config(); cfgfile=self.root/"config.json"; core.atomic_json(cfgfile,cfg)
        args=argparse.Namespace(mode="smoke",resume="0",config=str(cfgfile))
        def coordinates(config,selected,train,folder):
            core.atomic_json(folder/"FIELD_DOMAINS.json",{frame["scene"]:[-200.]*3+[200.]*3})
            core.atomic_json(folder/"INPUT_MANIFEST.json",[])
            core.atomic_json(folder/"CLEAN_GT.json",[dict(frame,active_cavs=frame["cavs"])])
        def extract(scene,coords,folder,effective,code):
            clouds=[]
            for cav in frame["cavs"]:
                cloud=folder/frame["timestamp"]/cav
                cloud.mkdir(parents=True)
                core.atomic_json(cloud/"COMMIT.json",dict(synthetic=True))
                clouds.append(cloud)
            return clouds
        calls=[]
        def integrate(cloud,target,pool,plan,effective,code):
            calls.append(cloud.name)
            data=[r for r in rows() if r["cav"]==cloud.name]
            core.atomic_json(target/"MATRIX.json",dict(rows=data,matrix_identity=core.identity(data),
                balance=dict(status="passed"),sampling=dict(scene=frame["scene"],cav=cloud.name),primary_ray_field_pairs=40))
        sources=dict(source_identity="synthetic-sources")
        with patch.object(run,"server_train",return_value=self.root), \
                patch.object(run,"output_paths",return_value=(output,logs)), \
                patch.object(run,"historical_metadata",return_value=dict(identity="synthetic-history")), \
                patch("scorw_p7.select_scenes.server_selection",return_value=selection), \
                patch.object(run,"input_fingerprint",return_value=[]), \
                patch.object(run,"runtime_identity",return_value={}), \
                patch.object(run,"source_audit",return_value=sources), \
                patch("shared_weather_benchmark.audit.real_coordinate_audit",side_effect=coordinates), \
                patch("scorw_p7.extract_rays.extract_scene",side_effect=extract), \
                patch("scorw_p7.integrate_pool.integrate_cloud",side_effect=integrate), \
                patch.object(run,"synthetic_checks",side_effect=lambda folder:core.atomic_json(folder/"CHECKS.json",dict(status="passed"))), \
                redirect_stdout(io.StringIO()):
            run.run_experiment(args,cfg,sources)
            archive=(output/"P7_SMALL_RESULTS.tar.gz").read_bytes()
            args.resume="1"
            run.run_experiment(args,cfg,sources)
            self.assertEqual(archive,(output/"P7_SMALL_RESULTS.tar.gz").read_bytes())
        self.assertEqual(len(calls),2)
        gates=core.read_json(output/"VALIDATION_GATES.json")
        self.assertEqual(gates["status"],"complete_smoke_only")
        self.assertEqual(gates["stages"]["P7-3"]["status"],"not_requested_smoke")
        self.assertEqual(gates["decision"],"INCONCLUSIVE")


class DecisionTests(unittest.TestCase):
    def dossier(self,correlation=.2,signal=.03):
        from scorw_p7.signal_metrics import METRICS
        scenes=[]
        values=dict(delta_correlation=correlation,delta_C=0.,delta_C_std=signal,delta_C_positive_probability=signal*3)
        for i in range(8):
            objects={m:{str(j):v for j in range(10)} for m,v in values.items()}
            scenes.append(dict(scene=f"v{i}",partition="independent_validation",pool_size=20,metrics=values,
                counts=dict(valid_target_instances=20,valid_unique_objects=10,valid_target_cav_pairs=20,
                    invalid_pair_fraction=0.,maximum_object_pair_share=.1),balance=dict(status="passed"),negative_control=dict(status="passed"),
                per_object_metrics=objects,leave_one_object=objects,
                leave_one_field=dict(results=[dict(metrics=values) for _ in range(20)])))
        return scenes,analyze(scenes,config())

    def test_continue_needs_correlation_and_complementarity(self):
        scenes,stats=self.dossier()
        verdict=decide(stats,scenes,dict(validation_status="available_8_new_scenes"),config())
        self.assertEqual(verdict["decision"],"CONTINUE_TO_DETECTOR")
        self.assertFalse(verdict["detector_started"])

    def test_nonsignificant_is_not_proof_of_small_effect(self):
        scenes,stats=self.dossier(0.,0.)
        for item in stats["partitions"]["independent_validation"]["metrics"].values():
            item["ci95"]=[-.5,.5]
        self.assertEqual(decide(stats,scenes,dict(validation_status="available_8_new_scenes"),config())["decision"],"INCONCLUSIVE")

    def test_stop_requires_intervals_inside_small_effect_bounds(self):
        scenes,stats=self.dossier(0.,0.)
        self.assertEqual(decide(stats,scenes,dict(validation_status="available_8_new_scenes"),config())["decision"],"STOP_OR_REDESIGN")

    def test_small_sample_cannot_stop_or_continue(self):
        scenes,stats=self.dossier(0.,0.)
        scenes[0]["counts"]["valid_unique_objects"]=1
        self.assertEqual(decide(stats,scenes,dict(validation_status="available_8_new_scenes"),config())["decision"],"INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
