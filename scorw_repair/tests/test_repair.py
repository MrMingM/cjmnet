from copy import deepcopy
import io
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.integrate import quad

from scorw_repair.core import HERE, REPO, Blocked, read_json, identity, transaction, digest
from scorw_repair.response import simulate, single_response, pulse, summary, C
from scorw_repair.detection import r2_gate
from shared_weather_benchmark.world_field import WorldField
from shared_weather_benchmark.geometry import numerical_audit, pose_matrix
from shared_weather_benchmark.ray_integrator import integrate, integrate_numpy
from scorw_p7.field_pool import make_pool, restore_field
from scorw_p7.balanced_assignment import assignment


def config(): return read_json(HERE/"config.json")


def temporary():
    base=Path(os.environ.get("TMPDIR",str(HERE/"local_validation")))/"synthetic_fixtures"
    base.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


class ResponseTests(unittest.TestCase):
    def test_equal_mean_alpha_does_not_imply_equal_soft_response(self):
        a=WorldField(kind="layered",mean_alpha=0,layers=[[0,15,.1],[15,30,0]])
        b=WorldField(kind="layered",mean_alpha=0,layers=[[0,15,0],[15,30,.1]])
        ra=single_response(a,[0]*3,[0,0,30],.8,config()["receiver"])
        rb=single_response(b,[0]*3,[0,0,30],.8,config()["receiver"])
        self.assertAlmostEqual(ra["tau"],rb["tau"],places=12)
        self.assertNotAlmostEqual(ra["waveform_peak"],rb["waveform_peak"],places=7)

    def test_high_optical_depth_is_finite_but_not_physically_validated(self):
        r=single_response(WorldField(kind="constant",mean_alpha=2),[0]*3,[100,0,0],1.,config()["receiver"])
        self.assertTrue(r["high_optical_depth_warning"])
        self.assertTrue(np.isfinite(r["waveform_peak"]))
    def test_zero_fog_exact_fallback(self):
        raw=np.asarray([[10,0,0,.3],[20,4,1,0],[.5,0,0,.3]],dtype=np.float32)
        d=simulate(raw,[0]*6,WorldField(kind="constant",mean_alpha=0),config()["receiver"],["s","t","c","sha"])
        np.testing.assert_array_equal(d["weather_xyzi"],raw)
        self.assertTrue(d["emitted"].all()); self.assertTrue(d["retained"].all())

    def test_constant_analytical_attenuation(self):
        c=config()["receiver"]
        r=single_response(WorldField(kind="constant",mean_alpha=.06),[0]*3,[30,0,0],.8,c)
        self.assertAlmostEqual(r["tau"],1.8,places=12)
        self.assertAlmostEqual(integrate(WorldField(kind="constant",mean_alpha=.06),[0]*3,[30,0,0]),1.8,places=12)

    def test_target_centred_hard_dominated_peak_retains_return(self):
        raw=np.asarray([[10,0,0,.8]],dtype=np.float32)
        d=simulate(raw,[0]*6,WorldField(kind="constant",mean_alpha=.001),config()["receiver"],["s","t","c","h"])
        self.assertEqual(int(d["origin"][0]),0)
        self.assertTrue(d["retained"][0]);self.assertGreater(d["hard_signal"][0],d["soft_signal"][0])
        np.testing.assert_array_equal(d["weather_xyzi"][:,:3],raw[:,:3])

    def test_uniform_soft_against_independent_quadrature(self):
        c=config()["receiver"]; alpha=.06; distance=30.; intensity=.8
        field=WorldField(kind="constant",mean_alpha=alpha)
        result=single_response(field,[0]*3,[distance,0,0],intensity,c)
        u=result["range_m"]+C*c["pulse_half_width_s"]/2
        lo=max(c["overlap_start_m"],u-C*c["pulse_half_width_s"]); hi=min(distance,u)
        scale=intensity*distance**2/(c["gamma"]/np.pi)*(c["beta_over_alpha"]*alpha)*(2/C)
        def fun(s):
            overlap=np.clip((s-c["overlap_start_m"])/(c["overlap_full_m"]-c["overlap_start_m"]),0,1)
            return float(overlap/s**2*np.exp(-2*alpha*s)*pulse(u-s,C*c["pulse_half_width_s"]))
        expected=scale*quad(fun,lo,hi,points=[v for v in [c["overlap_full_m"]] if lo<v<hi],epsabs=1e-12)[0] if hi>lo else 0.
        self.assertLessEqual(abs(result["soft_signal"]-expected),config()["tests"]["waveform_relative_tolerance"]*max(abs(expected),1e-12))

    def test_step_refinement_and_finite_peak(self):
        c=config()["receiver"]
        f=WorldField(kind="constant",mean_alpha=.06)
        rr=[]
        for step in (.1,.05,.025):
            r=single_response(f,[0]*3,[45,0,0],.8,c,step=step)
            self.assertTrue(np.isfinite(r["waveform_peak"]))
            rr.append(r)
        error=abs(rr[1]["waveform_peak"]-rr[2]["waveform_peak"])/max(rr[2]["waveform_peak"],1e-12)
        self.assertLessEqual(error,config()["tests"]["waveform_relative_tolerance"])
        self.assertLessEqual(abs(rr[1]["range_m"]-rr[2]["range_m"]),config()["tests"]["range_convergence_m"])

    def test_replacement_and_quantized_no_return_exist(self):
        c=config()["receiver"]; f=WorldField(kind="constant",mean_alpha=.06)
        scattered=single_response(f,[0]*3,[100,0,0],1.,c)
        self.assertTrue(scattered["emitted"])
        self.assertEqual(scattered["origin"],1)
        self.assertLess(scattered["range_m"],100)
        # This is an available positive 8-bit input code, not an invented tiny power.
        lost=single_response(f,[0]*3,[10,0,0],1/255,c)
        self.assertFalse(lost["emitted"])

    def test_origin_conservation_stability_and_reproducibility(self):
        raw=np.asarray([[10,0,0,1],[100,0,0,1],[100,0,0,1e-10]],dtype=np.float32)
        f=WorldField(kind="smooth",mean_alpha=.06,seed=4)
        a=simulate(raw,[0]*6,f,config()["receiver"],["s","t","c","h"],[1,4,9])
        b=simulate(raw,[0]*6,f,config()["receiver"],["s","t","c","h"],[1,4,9])
        for k in a: np.testing.assert_array_equal(a[k],b[k])
        s=summary(a,config()["distance_bins_m"])
        self.assertEqual(s["original_hard"]+s["replaced_by_weather"]+s["unknown"]+s["no_return"],3)
        self.assertEqual(a["original_index"].tolist(),[1,4,9])

    def test_near_field_unidentified_not_fabricated(self):
        r=single_response(WorldField(kind="constant",mean_alpha=.06),[0]*3,[.5,0,0],.3,config()["receiver"])
        self.assertEqual(r["origin"],2);self.assertTrue(r["unidentifiable_overlap"])

    def test_layered_reverse_and_cross_cav_coordinates(self):
        field=WorldField(kind="layered",mean_alpha=0,layers=[[0,5,.1],[5,10,.2]])
        self.assertAlmostEqual(integrate(field,[0,0,0],[0,0,10],.25),1.5,places=12)
        self.assertAlmostEqual(integrate(field,[0,0,10],[0,0,0],.25),1.5,places=12)
        self.assertLess(numerical_audit()["roundtrip_max_m"],1e-10)

    def test_same_world_segment_different_sensor_pose(self):
        field=WorldField(kind="smooth",mean_alpha=.06,seed=13)
        c=config()["receiver"]
        # Both local measurements define the same world start/end despite different yaw.
        a=simulate(np.array([[30,0,0,.5]],dtype=np.float32),[0]*6,field,c,["s","t","a","h"])
        b=simulate(np.array([[0,-30,0,.5]],dtype=np.float32),[0,0,0,0,90,0],field,c,["s","t","b","h"])
        np.testing.assert_allclose(a["integral"],b["integral"],rtol=1e-12,atol=1e-12)
        ma=np.asarray(pose_matrix([0]*6)); mb=np.asarray(pose_matrix([0,0,0,0,90,0]))
        np.testing.assert_allclose(a["weather_xyzi"][:,:3]@ma[:3,:3].T,b["weather_xyzi"][:,:3]@mb[:3,:3].T,atol=1e-6)

    def test_original_hahner_hard_quantization(self):
        from reference_weather.vendor.fog.fog_simulation import ParameterSet, P_R_fog_hard
        pc=np.array([[10.,0,0,80.],[30,0,0,255.]])
        ref=P_R_fog_hard(ParameterSet(alpha=.06),pc.copy())
        analytic=pc[:,3]*np.exp(-2*.06*np.linalg.norm(pc[:,:3],axis=1))
        np.testing.assert_array_equal(ref[:,3],np.round(analytic))

    def test_original_hahner_full_hard_soft_with_pinned_table(self):
        from reference_weather.vendor.fog import fog_simulation as fog
        name="integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_0.06.pickle"
        server=Path("/data/cjm/datasets/reference-weather-assets-v1/fog/integral_lookup_tables/original")/name
        local=REPO/"related_code/LiDAR_fog_sim/integral_lookup_tables/original"/name
        path=server if server.is_file() else local
        payload=path.read_bytes()
        lock=read_json(REPO/"reference_weather/assets_lock.json")
        entry=next(r for r in lock["assets"] if r["relative_path"].endswith(name))
        blob=hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()
        self.assertEqual(blob,entry["git_blob_sha1"]);self.assertEqual(len(payload),entry["size"])
        previous=fog.INTEGRAL_PATH
        try:
            fog.INTEGRAL_PATH=path.parent
            raw=np.asarray([[30,0,0,255,0],[80,0,0,255,1],[100,0,0,255,2]],dtype=float)
            out,soft,info=fog.simulate_fog(fog.ParameterSet(alpha=.06,gamma=1e-6),raw,noise=0)
            self.assertTrue(np.isfinite(out).all());self.assertGreater(info["num_fog_responses"],0)
            np.testing.assert_array_equal(out[:,4],raw[:,4])
            self.assertTrue((out[:,3]>=0).all());self.assertTrue((out[:,3]<=255).all())
        finally:
            fog.INTEGRAL_PATH=previous

    def test_b1_adapter_uses_frozen_32_bit_seed_convention(self):
        from scorw_repair.generate import original_b1
        observed=[];raw=np.asarray([[10,0,0,.5]],dtype=np.float32)
        def backend(points,legacy,seed):
            observed.append(seed)
            return points.copy(),np.ones(1,dtype=bool),np.ones(1,dtype=bool),np.zeros(1,dtype=np.uint8),np.zeros(1),{}
        with patch("scorw_repair.generate.hahner",side_effect=backend):
            out=original_b1(raw,config(),["s","t","c","sha"],np.asarray([7],dtype=np.int64))
        self.assertTrue(0<=observed[0]<2**32)
        self.assertEqual(out["original_index"].tolist(),[7])

    def test_pool_balances_complete_response_not_only_attenuation(self):
        c=read_json(REPO/"scorw_p7/config/protocol.json")
        pool=make_pool("synthetic-repair",c,None); plan=assignment(["ego","b"],"ego")
        raw=np.array([[30,0,0,.7]],dtype=np.float32)
        values=[]
        for manifest in pool["fields"]:
            d=simulate(raw,[0]*6,restore_field(manifest),config()["receiver"],["s","t","b","h"])
            values.append([float(d["integral"][0]),float(d["weather_xyzi"][0,3]),int(d["origin"][0]),int(d["emitted"][0])])
        a=np.asarray(values); b=a[plan["schedules"]["E3"]["b"]]
        np.testing.assert_array_equal(np.sort(a,axis=0),np.sort(b,axis=0))
        self.assertEqual(len({m["sha256"] for m in pool["fields"]}),20)


class GateTests(unittest.TestCase):
    def good(self):
        return (dict(status="SOURCE_AND_CLEAN_SENSOR_AUDIT_PASS",physical_calibration_passed=False),
                dict(status="numerical_engine_executed",balance=[dict(status="passed",fields=20)],
                     data_identity_check="passed",source_count_conservation="passed",physical_calibration_passed=False),
                dict(status="verified"),dict(status="passed"),dict(status="passed"))

    def test_authorized_exploration_without_real_reference(self):
        gate=r2_gate(*self.good())
        self.assertEqual(gate["status"],"eligible")
        self.assertEqual(gate["label"],"UNCALIBRATED_EXPLORATORY")
        self.assertFalse(gate["formal_physical_validation"])

    def test_each_failed_gate_blocks_detector(self):
        for index in (0,2,3,4):
            args=list(self.good());args[index]=dict(status="failed")
            self.assertNotEqual(r2_gate(*args)["status"],"eligible")
        for field,value in (("balance",[]),("balance",[dict(status="passed",fields=19)]),
                            ("data_identity_check","failed"),("source_count_conservation","failed")):
            args=list(self.good());args[1][field]=value
            self.assertNotEqual(r2_gate(*args)["status"],"eligible")

    def test_frozen_range_literal_does_not_import_loader(self):
        from scorw_repair.range_adapter import gt_range
        from scorw_repair.core import _DatasetImportGuard
        self.assertEqual(gt_range(),[-140,-40,-3,140,40,1])
        with self.assertRaises(Blocked):_DatasetImportGuard().find_spec("opencood.data_utils.datasets.basedataset")
        self.assertIsNone(_DatasetImportGuard().find_spec("opencood.data_utils.pre_processor"))

    def test_changed_protocol_resume_rejected(self):
        with temporary() as temp:
            root=Path(temp)/"run"
            transaction(root,dict(version=1),lambda code:None)
            with self.assertRaises(ValueError):transaction(root,dict(version=2),lambda code:None)

    def test_preflight_needs_no_reference_data_or_network(self):
        from scorw_repair.run import preflight
        c=config()
        with temporary() as temp:
            c["train_root"]=temp
            dep=dict(required_r0_r1=True,modules={n:True for n in ("torch","spconv","cumm")})
            with patch("scorw_repair.regression.table_resource",return_value=(Path(temp)/"table",dict(sha256="synthetic"))), \
                    patch("urllib.request.urlopen",side_effect=AssertionError("No network")):
                r=preflight(c,dep,dict(status="verified"))
        self.assertEqual(r["status"],"ready");self.assertEqual(r["network_requests"],0)
        self.assertEqual(r["real_validation_status"],"NOT_VALIDATED_NO_REAL_REFERENCE")


class RegressionTests(unittest.TestCase):
    def test_strict_uniform_and_original_kernel_regression(self):
        from scorw_repair.regression import check
        result=check(config())
        self.assertEqual(result["status"],"passed")
        self.assertEqual(len(result["mathematical_tests"]),40)
        self.assertTrue(all(v=="passed" for v in result["strict_checks"].values()))
        self.assertFalse(result["physical_calibration_passed"])
        self.assertTrue(all(not r["strict_selected_return_equivalence_required"] for r in result["descriptive_selected_return_comparison"]))

    def test_single_neighbor_average_delta_c_is_zero(self):
        pool=make_pool("synthetic-single-neighbor",read_json(REPO/"scorw_p7/config/protocol.json"),None)
        plan=assignment(["b","ego"],"ego")
        values={c:[] for c in ("ego","b")}
        for f in pool["fields"]:
            for c,xyz in (("ego",[25,0,0]),("b",[20,10,0])):
                d=simulate(np.array([[*xyz,.7]],dtype=np.float32),[0]*6,restore_field(f),config()["receiver"],["s","t",c,"h"])
                values[c].append(float(d["weather_xyzi"][0,3]*d["retained"][0]/.7))
        a=np.array(values["b"])-np.array(values["ego"])
        b=np.array(values["b"])[plan["schedules"]["E3"]["b"]]-np.array(values["ego"])
        self.assertAlmostEqual(float(np.mean(a)-np.mean(b)),0.,places=14)
        shifted=np.roll(a,3)
        np.testing.assert_array_equal(np.sort(shifted),np.sort(a))

    def test_waveform_positive_area_and_linear_intensity_scale(self):
        field=WorldField(kind="smooth",mean_alpha=.06,seed=12)
        c=config()["receiver"]
        a=single_response(field,[0]*3,[45,0,0],.2,c,return_waveform=True)
        b=single_response(field,[0]*3,[45,0,0],.4,c,return_waveform=True)
        self.assertAlmostEqual(a["soft_area"],a["soft_area_discrete_expected"],places=12)
        self.assertAlmostEqual(b["soft_area"],2*a["soft_area"],places=12)
        self.assertAlmostEqual(b["waveform_peak"],2*a["waveform_peak"],places=12)
        self.assertTrue((a["soft_waveform"]>=0).all())

    def test_sidecar_rejects_corrupt_component_or_duplicate_identity(self):
        from scorw_repair.response import validate
        raw=np.asarray([[10,0,0,.5],[20,0,0,.5]],dtype=np.float32)
        data=simulate(raw,[0]*6,WorldField(kind="constant",mean_alpha=.06),config()["receiver"],["s","t","c","h"])
        damaged={k:v.copy() for k,v in data.items()}
        damaged["point_identity"][1]=damaged["point_identity"][0]
        with self.assertRaises(ValueError):validate(damaged)
        damaged={k:v.copy() for k,v in data.items()};damaged["soft_signal"][0]=float("nan")
        with self.assertRaises(ValueError):validate(damaged)

    def test_source_audit_no_real_reference_and_optional_material(self):
        from scorw_repair.source_audit import audit
        from scorw_repair.core import atomic_json,atomic_text
        with temporary() as temp:
            root=Path(temp);coord=root/"coords";out=root/"R0";coord.mkdir();out.mkdir()
            pcd=root/"synthetic.pcd";yaml=root/"synthetic.yaml"
            atomic_text(pcd,"FIELDS x y z rgb\nPOINTS 0\nDATA ascii\n");atomic_text(yaml,"synthetic only\n")
            atomic_json(coord/"REAL_COORDINATE_AUDIT.json",dict(status="passed"))
            atomic_json(coord/"INPUT_MANIFEST.json",[dict(scene="synthetic",timestamp="0",cav="a",pcd=str(pcd),yaml=str(yaml),
                pcd_sha256=digest(pcd),yaml_sha256=digest(yaml),pose=[0]*6,points=0)])
            result=audit(config(),coord,out)
            self.assertEqual(result["real_validation_status"],"NOT_VALIDATED_NO_REAL_REFERENCE")
            self.assertFalse(result["private_url_files_read"])
            self.assertFalse(result["real_data_compared"])

    def test_coordinates_do_not_recursively_lock_stage(self):
        import inspect
        from scorw_repair.generate import coordinates
        self.assertNotIn("transaction(",inspect.getsource(coordinates))


class OrchestrationTests(unittest.TestCase):
    def fixture(self):
        temp=temporary();self.addCleanup(temp.cleanup)
        root=Path(temp.name);cfg=config()
        cfg.update(output_root=str(root/"data"),logs_root=str(root/"logs"),reuse_previous_root=str(root/"absent_previous"))
        return root,cfg

    def invoke(self,cfg,args,events,fail=None):
        from argparse import Namespace
        from contextlib import ExitStack,redirect_stdout
        from scorw_repair import run
        from scorw_repair.core import atomic_json,atomic_text
        def record(name,p,file,status):
            events.append(name)
            if fail==name:raise Blocked("SYNTHETIC_GATE_FAILURE_"+name)
            atomic_json(p/file,dict(status=status))
        def numbers(p):record("numerical",p,"NUMERICAL_TESTS.json","passed")
        def source(c,coords,p):
            record("R0",p,"R0_SENSOR_AUDIT.json","SOURCE_AND_CLEAN_SENSOR_AUDIT_PASS")
            atomic_text(p/"R0_RESULTS.md","Synthetic source-audit fixture only\n")
        def coords(c,f,p,parent):
            record("coordinates",p,"REAL_COORDINATE_AUDIT.json","passed")
            atomic_json(p/"INPUT_MANIFEST.json",[])
        def reg(c,p):record("hahner_regression",p,"R1_HAHNER_REGRESSION.json","passed")
        def response(c,f,coords,p,parent):
            record("R1",p,"R1_PHYSICS_AUDIT.json","numerical_engine_executed")
            atomic_json(p/"R1_PHYSICS_AUDIT.json",dict(status="numerical_engine_executed",
                balance=[dict(status="passed",fields=20)],rows=[],data_identity_check="passed",
                source_count_conservation="passed",physical_calibration_passed=False))
            atomic_text(p/"R1_RESULTS.md","Synthetic response fixture only\n")
        def detect(c,coords,p,pin,label,parent,reuse_info=None):
            record("R2",p,"R2_TARGET_DIAGNOSIS.json","detector_diagnostic_executed")
            atomic_json(p/"R2_TARGET_DIAGNOSIS.json",dict(status="detector_diagnostic_executed",label=label,
                fixture_only=True,real_detector_executed=False))
            atomic_text(p/"R2_RESULTS.md","Synthetic orchestration fixture, no actual detector\n")
        patches=[patch.object(run,"server_only"),patch.object(run,"load_config",return_value=cfg),
            patch.object(run,"sources",return_value=dict(identity="synthetic",snapshot_sha256="synthetic")),
            patch.object(run,"dependencies",return_value=dict(required_r0_r1=True,modules={n:True for n in ("torch","spconv","cumm")})),
            patch.object(run,"numerical_checks",side_effect=numbers),
            patch("scorw_repair.generate.opv2v_freeze",return_value=dict(frames=[])),
            patch("scorw_repair.detection.model_identity",return_value=dict(status="verified")),
            patch("scorw_repair.source_audit.audit",side_effect=source),
            patch("scorw_repair.regression.check",side_effect=reg),
            patch("scorw_repair.generate.coordinates",side_effect=coords),
            patch("scorw_repair.generate.execute_r1",side_effect=response),
            patch("scorw_repair.detection.evaluate",side_effect=detect),
            patch("urllib.request.urlopen",side_effect=AssertionError("No networking permitted"))]
        with ExitStack() as stack:
            for p in patches:stack.enter_context(p)
            stack.enter_context(redirect_stdout(io.StringIO()))
            return run.run(Namespace(**args))

    def test_order_strict_resume_and_deterministic_archive(self):
        root,cfg=self.fixture();events=[];args=dict(mode="full",download="0",resume="0")
        self.assertEqual(self.invoke(cfg,args,events),0)
        self.assertEqual(events,["numerical","coordinates","R0","hahner_regression","R1","R2"])
        output=root/"data/full";archive=(output/"SMALL_RESULTS.tar.gz").read_bytes()
        args["resume"]="1"
        self.assertEqual(self.invoke(cfg,args,events),0)
        self.assertEqual(len(events),6)
        self.assertEqual(archive,(output/"SMALL_RESULTS.tar.gz").read_bytes())
        evidence=read_json(output/"EVIDENCE_LEVELS.json")
        self.assertTrue(evidence["REAL_WORLD_VALIDATION_MISSING"])
        self.assertTrue(evidence["PHYSICS_NUMERICAL_PASS"])
        self.assertFalse(evidence["physical_calibration_passed"])

    def test_numerical_or_regression_failure_stops_downstream(self):
        for failed,expected in (("numerical",["numerical"]),
                ("hahner_regression",["numerical","coordinates","R0","hahner_regression"]),
                ("R1",["numerical","coordinates","R0","hahner_regression","R1"])):
            root,cfg=self.fixture();events=[]
            self.assertEqual(self.invoke(cfg,dict(mode="full",download="0",resume="0"),events,failed),3)
            self.assertEqual(events,expected)
            self.assertFalse(read_json(root/"data/full/EVIDENCE_LEVELS.json")["DETECTOR_EXPLORATORY_RESULT"])

    def test_smoke_includes_r2_with_separate_contract(self):
        root,cfg=self.fixture()
        for mode in ("smoke","full"):
            events=[];self.invoke(cfg,dict(mode=mode,download="0",resume="0"),events)
            self.assertIn("R2",events)
        self.assertNotEqual(read_json(root/"data/smoke/CONTRACT.json")["identity"],
                            read_json(root/"data/full/CONTRACT.json")["identity"])

    def test_download_request_rejected_without_reading_urls(self):
        _,cfg=self.fixture()
        with self.assertRaises(Blocked):self.invoke(cfg,dict(mode="full",download="1",resume="0"),[])

    def test_source_changed_protocol_cannot_reuse_contract(self):
        root,cfg=self.fixture();args=dict(mode="full",download="0",resume="0")
        self.invoke(cfg,args,[])
        changed=deepcopy(cfg);changed["receiver"]["gamma"]*=2;args["resume"]="1"
        with self.assertRaises(ValueError):self.invoke(changed,args,[])

    def test_interrupted_derived_report_publication_can_resume(self):
        from scorw_repair.core import atomic_json
        root,cfg=self.fixture();args=dict(mode="full",download="0",resume="0")
        self.invoke(cfg,args,[])
        output=root/"data/full";contract=read_json(output/"CONTRACT.json")["identity"]
        atomic_json(output/"PUBLICATION_IN_PROGRESS.json",dict(contract=contract))
        atomic_json(output/"VALIDATION_GATES.json",dict(status="running",contract=contract))
        args["resume"]="1";events=[]
        self.assertEqual(self.invoke(cfg,args,events),0)
        self.assertEqual(events,[])
        self.assertFalse((output/"PUBLICATION_IN_PROGRESS.json").exists())

    def test_committed_stage_corruption_never_skipped_on_resume(self):
        from scorw_repair.core import atomic_text
        root,cfg=self.fixture();args=dict(mode="full",download="0",resume="0")
        self.invoke(cfg,args,[])
        atomic_text(root/"data/full/stages/R0/R0_RESULTS.md","corrupted")
        args["resume"]="1"
        self.assertEqual(self.invoke(cfg,args,[]),1)
        self.assertEqual(read_json(root/"data/full/VALIDATION_GATES.json")["failed_stage"],"R0")


if __name__=="__main__":unittest.main()
