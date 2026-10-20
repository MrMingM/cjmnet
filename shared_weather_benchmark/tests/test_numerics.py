import math
import random
import unittest
import tempfile
from pathlib import Path
from contextlib import contextmanager, redirect_stdout
from unittest.mock import patch
import io
import json

from shared_weather_benchmark.geometry import pose_matrix, transform, inverse, numerical_audit, in_box, box_corners
from shared_weather_benchmark.world_field import WorldField
from shared_weather_benchmark.ray_integrator import integrate, transmittance
from shared_weather_benchmark.fog_physics import (attenuate_intensity,scattering_kernel_samples,
                                                simulate_nonuniform_soft_returns)
from shared_weather_benchmark.provenance import point_identity, identity, seed_for
from shared_weather_benchmark.observability import classify
from shared_weather_benchmark.compare import correlation, paired_scene_interval


class GeometryTests(unittest.TestCase):
    def test_roundtrip_cross_cav(self):
        report = numerical_audit()
        self.assertLess(report["roundtrip_max_m"],1e-10)
        self.assertLess(report["cross_cav_max_m"],1e-10)

    def test_mixed_rotation_order(self):
        p = [.2,-5,3]
        a = transform(p,pose_matrix([0,0,0,7,31,-13]))
        b = transform(transform(transform(p,pose_matrix([0,0,0,7,0,0])),
                                 pose_matrix([0,0,0,0,0,-13])),pose_matrix([0,0,0,0,31,0]))
        self.assertLess(math.dist(a,b),1e-12)

    def test_yaw_units_direction(self):
        self.assertLess(math.dist(transform([1,0,0],pose_matrix([0,0,0,0,90,0])),[0,1,0]),1e-12)
        self.assertLess(math.dist(transform([0,0,1],pose_matrix([0,0,0,0,0,90])),[-1,0,0]),1e-12)

    def test_box_local_world(self):
        obj = dict(location=[4,6,1],center=[.5,0,.2],angle=[7,37,-9],extent=[2,1,.7])
        self.assertTrue(in_box([4.5,6,1.2],obj))
        self.assertFalse(in_box([20,6,1.2],obj))
        m = pose_matrix([-4,7,2,9,-87,3])
        for corner in box_corners(obj):
            local = transform(corner,inverse(m))
            self.assertLess(math.dist(corner,transform(local,m)),1e-12)

    def test_invalid_pose(self):
        for pose in ([1,2,3], [0,0,0,0,float("nan"),0]):
            with self.assertRaises(ValueError):
                pose_matrix(pose)

    def test_stable_point_identity(self):
        a = point_identity("s","1","2","hash",5)
        self.assertEqual(a,point_identity("s","1","2","hash",5))
        self.assertNotEqual(a,point_identity("s","1","2","other",5))
        self.assertNotEqual(a,point_identity("s","1","3","hash",5))
        self.assertNotEqual(a,point_identity("s","1","2","hash",6))


class SourceDiagnosticTests(unittest.TestCase):
    @contextmanager
    def source_fixture(self):
        from shared_weather_benchmark.provenance import HERE,atomic_json,digest
        parent = HERE/"local_validation"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="numeric-sources-",dir=parent) as temp:
            self.assertEqual(Path(temp).resolve().parent,parent.resolve())
            repo = Path(temp)/"checkout"
            package = repo/"shared_weather_benchmark"
            package.mkdir(parents=True)
            dependency = repo/"reference_weather"/"frozen.py"
            dependency.parent.mkdir()
            core = package/"core.py"
            core.write_bytes(b"value = .06\n")
            dependency.write_bytes(b"kept = 1\n")
            sources = {p.relative_to(repo).as_posix():digest(p) for p in (core,dependency)}
            atomic_json(package/"source_snapshot.json",dict(sources=sources))
            yield repo,package,core,dependency,sources

    def test_exact_sources_pass_and_diagnostics_are_read_only(self):
        from shared_weather_benchmark.provenance import inspect_sources,digest,identity
        with self.source_fixture() as (repo,package,core,dependency,sources):
            files = [core,dependency,package/"source_snapshot.json"]
            before = [(digest(p),p.stat().st_mtime_ns) for p in files]
            report = inspect_sources(repo,package)
            self.assertEqual(report["status"],"passed")
            self.assertEqual(report["matched_files"],2)
            self.assertEqual(report["source_identity"],identity(dict(strict_sources=sources,
                reviewed_only_sources={},snapshot_sha256=digest(package/"source_snapshot.json"))))
            self.assertEqual(before,[(digest(p),p.stat().st_mtime_ns) for p in files])

    def test_changed_dependency_names_expected_and_actual_hashes(self):
        from shared_weather_benchmark import provenance
        with self.source_fixture() as (repo,package,core,dependency,sources):
            dependency.write_bytes(b"kept = 2\n")
            report = provenance.inspect_sources(repo,package)
            self.assertEqual(report["changed"][0]["path"],"reference_weather/frozen.py")
            self.assertEqual(report["changed"][0]["expected_sha256"],sources["reference_weather/frozen.py"])
            self.assertEqual(report["changed"][0]["actual_sha256"],provenance.digest(dependency))
            self.assertEqual(report["changed"][0]["hint"],"content_differs")
            with patch.multiple(provenance,REPO=repo,HERE=package):
                with self.assertRaisesRegex(ValueError,"reference_weather/frozen.py"):
                    provenance.verify_sources()

    def test_crlf_hint_preserves_strict_rejection(self):
        from shared_weather_benchmark import provenance
        with self.source_fixture() as (repo,package,core,dependency,sources):
            core.write_bytes(core.read_bytes().replace(b"\n",b"\r\n"))
            report = provenance.inspect_sources(repo,package)
            self.assertIn("CRLF_conversion_only",report["changed"][0]["hint"])
            self.assertEqual(report["status"],"failed")
            with patch.multiple(provenance,REPO=repo,HERE=package):
                with self.assertRaises(ValueError):
                    provenance.verify_sources()

    def test_bom_hint_preserves_strict_rejection(self):
        from shared_weather_benchmark import provenance
        for crlf in (False,True):
            with self.subTest(crlf=crlf), self.source_fixture() as (repo,package,core,dependency,sources):
                content = core.read_bytes()
                core.write_bytes(b"\xef\xbb\xbf"+(content.replace(b"\n",b"\r\n") if crlf else content))
                report = provenance.inspect_sources(repo,package)
                self.assertIn("UTF8_BOM",report["changed"][0]["hint"])
                with patch.multiple(provenance,REPO=repo,HERE=package):
                    with self.assertRaises(ValueError):
                        provenance.verify_sources()

    def test_missing_and_unregistered_files_are_both_listed(self):
        from shared_weather_benchmark.provenance import inspect_sources,source_diagnostic_text
        with self.source_fixture() as (repo,package,core,dependency,sources):
            dependency.unlink()
            (package/"extra.py").write_bytes(b"extra = True\n")
            report = inspect_sources(repo,package)
            self.assertEqual(report["missing"],["reference_weather/frozen.py"])
            self.assertEqual(report["unexpected"],["shared_weather_benchmark/extra.py"])
            self.assertIn("MISSING reference_weather/frozen.py",source_diagnostic_text(report))
            self.assertIn("UNREGISTERED shared_weather_benchmark/extra.py",source_diagnostic_text(report))

    def test_diagnostic_cli_returns_nonzero_json_without_bypassing_gate(self):
        from shared_weather_benchmark import provenance
        from shared_weather_benchmark.diagnose_sources import main
        with self.source_fixture() as (repo,package,core,dependency,sources):
            core.write_bytes(b"changed = True\n")
            stream = io.StringIO()
            with patch.multiple(provenance,REPO=repo,HERE=package), redirect_stdout(stream):
                code = main(["--json"])
            self.assertEqual(code,1)
            self.assertEqual(json.loads(stream.getvalue())["status"],"failed")

    def add_reviewed_source(self,repo,package,sources):
        from shared_weather_benchmark.provenance import atomic_json,digest
        reviewed = repo/"opencood/data_utils/datasets/basedataset.py"
        reviewed.parent.mkdir(parents=True)
        reviewed.write_bytes(b"historical_loader = 1\n")
        atomic_json(package/"source_snapshot.json",dict(sources=sources,
            reviewed_only_sources={reviewed.relative_to(repo).as_posix():digest(reviewed)}))
        return reviewed

    def test_reviewed_only_difference_records_actual_and_changes_contract_identity(self):
        from shared_weather_benchmark.provenance import inspect_sources,digest,source_diagnostic_text
        with self.source_fixture() as (repo,package,core,dependency,sources):
            reviewed = self.add_reviewed_source(repo,package,sources)
            before = inspect_sources(repo,package)
            expected = digest(reviewed)
            reviewed.write_bytes(b"server_variant = 2\n")
            report = inspect_sources(repo,package)
            self.assertEqual(report["status"],"passed")
            self.assertEqual(report["changed"],[])
            self.assertEqual(report["reviewed_only"][0]["reviewed_sha256"],expected)
            self.assertEqual(report["reviewed_only"][0]["actual_sha256"],digest(reviewed))
            self.assertEqual(report["reviewed_only"][0]["status"],"different_not_executed")
            self.assertNotEqual(before["source_identity"],report["source_identity"])
            self.assertIn("REVIEWED ONLY",source_diagnostic_text(report))

    def test_reviewed_only_policy_cannot_exempt_an_arbitrary_runtime_dependency(self):
        from shared_weather_benchmark.provenance import inspect_sources,atomic_json
        with self.source_fixture() as (repo,package,core,dependency,sources):
            atomic_json(package/"source_snapshot.json",dict(sources=sources,
                reviewed_only_sources={"reference_weather/frozen.py":sources["reference_weather/frozen.py"]}))
            report = inspect_sources(repo,package)
            self.assertEqual(report["status"],"failed")
            self.assertTrue(report["policy_errors"])

    def test_missing_reviewed_only_file_still_stops(self):
        from shared_weather_benchmark.provenance import inspect_sources
        with self.source_fixture() as (repo,package,core,dependency,sources):
            reviewed = self.add_reviewed_source(repo,package,sources)
            reviewed.unlink()
            report = inspect_sources(repo,package)
            self.assertEqual(report["status"],"failed")
            self.assertIn("opencood/data_utils/datasets/basedataset.py",report["missing"])

    def test_accidental_direct_import_of_reviewed_dataset_stops(self):
        from shared_weather_benchmark.provenance import inspect_sources,atomic_json,digest
        with self.source_fixture() as (repo,package,core,dependency,sources):
            core.write_bytes(b"from opencood.data_utils.datasets import basedataset\n")
            sources["shared_weather_benchmark/core.py"] = digest(core)
            atomic_json(package/"source_snapshot.json",dict(sources=sources))
            report = inspect_sources(repo,package)
            self.assertEqual(report["status"],"failed")
            self.assertIn("Forbidden dataset import",report["policy_errors"][0])

    def test_runtime_guard_blocks_dynamic_and_indirect_imports(self):
        import importlib
        import sys
        import types
        from shared_weather_benchmark.provenance import install_runtime_source_guard
        parent = types.ModuleType("opencood")
        parent.__path__ = []
        with patch.object(sys,"meta_path",list(sys.meta_path)), patch.dict(sys.modules,{"opencood":parent}):
            install_runtime_source_guard()
            size = len(sys.meta_path)
            install_runtime_source_guard()
            self.assertEqual(size,len(sys.meta_path))
            for name in ("opencood.data_utils","opencood.data_utils.datasets.basedataset"):
                with self.assertRaisesRegex(RuntimeError,"forbids executing"):
                    importlib.import_module(name)

    def test_runtime_guard_rejects_already_loaded_dataset_modules(self):
        import sys
        import types
        from shared_weather_benchmark.provenance import install_runtime_source_guard
        with patch.dict(sys.modules,{"opencood.data_utils":types.ModuleType("opencood.data_utils")}):
            with self.assertRaisesRegex(RuntimeError,"already imported"):
                install_runtime_source_guard()


class IntegrationTests(unittest.TestCase):
    def test_zero_fog(self):
        self.assertEqual(transmittance(integrate(WorldField(mean_alpha=0),[0,0,0],[100,5,2])),1)

    def test_constant_analytic(self):
        field = WorldField(kind="constant",mean_alpha=.06)
        for step in (.13,.5,7,200):
            tau = integrate(field,[3,-2,1],[13,2,7],step)
            self.assertAlmostEqual(tau,.06*math.sqrt(152),places=13)
            self.assertAlmostEqual(transmittance(tau),math.exp(-.12*math.sqrt(152)),places=13)

    def test_piecewise_layers_hand_integral(self):
        field = WorldField(kind="layered",mean_alpha=0,layers=[[2,4,.1],[6,8,.2]])
        self.assertAlmostEqual(integrate(field,[0,0,0],[0,0,10],.1),.6,places=13)

    def test_reverse_path(self):
        field = WorldField(seed=5)
        a,b = [3,-4,5],[-18,44,8]
        self.assertAlmostEqual(integrate(field,a,b,.27),integrate(field,b,a,.27),places=13)

    def test_convergence(self):
        field = WorldField(seed=41,length_m=4)
        a,b = [1,2,3],[52,-17,6]
        fine = integrate(field,a,b,.0125)
        errors = [abs(integrate(field,a,b,s)-fine) for s in (2,1,.5,.25)]
        self.assertTrue(all(b < a for a,b in zip(errors,errors[1:])),errors)
        self.assertLess(errors[-1],errors[0]/30)

    def test_zero_length(self):
        self.assertEqual(integrate(WorldField(),[1,2,3],[1,2,3]),0)

    def test_invalid_integrator(self):
        with self.assertRaises(ValueError):
            integrate(WorldField(),[0,0,0],[1,2,3],0)
        with self.assertRaises(ValueError):
            transmittance(-1)

    def test_attenuation_deterministic(self):
        f1,f2 = WorldField(seed=42),WorldField(seed=42)
        a,b = [1,2,3],[50,-20,8]
        self.assertEqual(attenuate_intensity(.7,integrate(f1,a,b)),attenuate_intensity(.7,integrate(f2,a,b)))

    def test_hahner_hard_component_rounding_bound(self):
        # Upstream hard kernel rounds x255. E1 is continuous before RGB encoding.
        for r in [0,1,10,50,200]:
            continuous = attenuate_intensity(.7,.06*r)
            upstream = round(255*.7*math.exp(-2*.06*r))/255
            self.assertLessEqual(abs(continuous-upstream),.5/255+1e-15)

    def test_layer_scattering_prefix(self):
        values = scattering_kernel_samples([0,1,2],[.1,.1,.1],[.02,.02,.02],[0,1,1])
        self.assertAlmostEqual(values[1],.02*math.exp(-.2))
        self.assertAlmostEqual(values[2],.02*math.exp(-.4)/4)

    def test_scattering_disabled(self):
        with self.assertRaises(NotImplementedError):
            simulate_nonuniform_soft_returns()
        with self.assertRaises(ValueError):
            scattering_kernel_samples([0,0],[.1,.1],[.02,.02],[0,1])


class FieldTests(unittest.TestCase):
    def test_nonnegative_bounded(self):
        f = WorldField(seed=9,contrast=.99)
        for i in range(200):
            a = f.alpha([i,-i*.3,math.sin(i)])
            self.assertGreaterEqual(a,f.mean*(1-f.contrast))
            self.assertLessEqual(a,f.mean*(1+f.contrast))

    def test_order_independent(self):
        points = [[i*3,-i*.5,1.] for i in range(20)]
        field = WorldField(seed=4)
        a = [field.alpha(p) for p in points]
        b = list(reversed([field.alpha(p) for p in reversed(points)]))
        self.assertEqual(a,b)

    def test_cross_cav_same_world_point(self):
        field = WorldField(seed=5)
        world = [10,20,3]
        for pose in ([1,2,3,4,5,6],[-3,5,1,-3,48,7]):
            m = pose_matrix(pose)
            recovered = transform(transform(world,inverse(m)),m)
            self.assertAlmostEqual(field.alpha(world),field.alpha(recovered),places=13)

    def test_field_reproduction_hash(self):
        self.assertEqual(WorldField(seed=5).manifest(),WorldField(seed=5).manifest())
        self.assertNotEqual(WorldField(seed=5).manifest()["sha256"],WorldField(seed=6).manifest()["sha256"])

    def test_continuity_and_frozen_time(self):
        f = WorldField(seed=8)
        self.assertLess(abs(f.alpha([1,2,3])-f.alpha([1.00001,2,3])),1e-6)
        self.assertEqual(f.alpha([1,2,3],0),f.alpha([1,2,3],500))

    def test_domain_error(self):
        f = WorldField(bounds=[0,0,0,10,10,10])
        self.assertGreaterEqual(f.alpha([10,5,1]),0)
        with self.assertRaises(ValueError):
            f.alpha([10.0001,5,1])

    def test_e2_e3_parameter_law(self):
        # Validate common marginal parameters, not realized sample equality.
        a = WorldField(seed=seed_for(10,"scene","shared"))
        b = WorldField(seed=seed_for(10,"scene","independent","1"))
        self.assertEqual((a.mean,a.contrast,a.length),(b.mean,b.contrast,b.length))
        self.assertNotEqual(a.coefficients,b.coefficients)

    def test_invalid_field(self):
        for kwargs in (dict(mean_alpha=-1),dict(length_m=0),dict(contrast=1),dict(modes=0),dict(mean_alpha=float("nan"))):
            with self.assertRaises(ValueError):
                WorldField(**kwargs)

    def test_layer_boundary(self):
        f = WorldField(kind="layered",mean_alpha=.01,layers=[[2,4,.2]])
        self.assertEqual(f.alpha([0,0,2]),.2)
        self.assertEqual(f.alpha([0,0,4]),.01)


class DiagnosticTests(unittest.TestCase):
    def rows(self,counts,ratios):
        return [dict(weather_hard_count=n,effective_intensity_ratio=q,is_ego=i==0)
                for i,(n,q) in enumerate(zip(counts,ratios))]

    def test_categories_and_residual(self):
        self.assertTrue(classify(self.rows([0,0],[None,None])).startswith("A_"))
        self.assertTrue(classify(self.rows([0,6],[None,.9])).startswith("B_"))
        self.assertTrue(classify(self.rows([5,6],[.1,.9])).startswith("C_"))
        self.assertTrue(classify(self.rows([5,6],[.8,.9])).startswith("D_"))
        self.assertEqual(classify(self.rows([5,6],[.1,.2])),"residual_mixed_or_weak")

    def test_threshold_sensitivity(self):
        rows = self.rows([2,4],[.8,.8])
        self.assertTrue(classify(rows,3).startswith("B_"))
        self.assertTrue(classify(rows,1).startswith("D_"))

    def test_degenerate_correlation(self):
        self.assertIsNone(correlation([1,1,1],[.1,.2,.3]))
        self.assertAlmostEqual(correlation([1,2,3],[3,2,1]),-1)

    def test_scene_interval_small_sample(self):
        result = paired_scene_interval([.1,.2],dict(seed=1,replicates=100,minimum_scenes_for_interval=5))
        self.assertIsNone(result["ci95"])
        self.assertEqual(result["scenes"],2)

    def test_scene_interval_deterministic(self):
        cfg = dict(seed=1,replicates=100,minimum_scenes_for_interval=5)
        self.assertEqual(paired_scene_interval([.1,.2,.3,.4,.5],cfg),paired_scene_interval([.1,.2,.3,.4,.5],cfg))

    def test_atomic_nan_preserves_previous_commit(self):
        from shared_weather_benchmark.provenance import atomic_json,read_json,HERE
        parent = HERE/"local_validation"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="numeric-atomic-",dir=parent) as temp:
            self.assertEqual(Path(temp).resolve().parent,parent.resolve())
            path = Path(temp)/"record.json"
            atomic_json(path,{"complete":True})
            with self.assertRaises(ValueError):
                atomic_json(path,{"tau":float("nan")})
            self.assertEqual(read_json(path),{"complete":True})

    def test_committed_hash_detects_modified_bytes(self):
        from shared_weather_benchmark.provenance import atomic_bytes,digest,check_hashes,HERE
        parent = HERE/"local_validation"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="numeric-hash-",dir=parent) as temp:
            self.assertEqual(Path(temp).resolve().parent,parent.resolve())
            path = Path(temp)/"payload"
            atomic_bytes(path,b"valid")
            hashes = {"payload":digest(path)}
            check_hashes(temp,hashes)
            atomic_bytes(path,b"changed")
            with self.assertRaises(ValueError):
                check_hashes(temp,hashes)

    def test_stage_resume_and_tamper_stop(self):
        from argparse import Namespace
        from shared_weather_benchmark.provenance import HERE,read_json
        from shared_weather_benchmark.run import execute
        parent = HERE/"local_validation"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="numeric-runner-",dir=parent) as temp:
            self.assertEqual(Path(temp).resolve().parent,parent.resolve())
            args = Namespace(config=HERE/"config"/"prototype.json",output=temp,
                             through="P0",from_stage="P0",smoke=True,resume=False)
            with redirect_stdout(io.StringIO()):
                execute(args)
                args.resume = True
                execute(args)
                payload = Path(temp)/"P0_NUMERICAL.json"
                payload.write_bytes(payload.read_bytes()+b" ")
                with self.assertRaises(ValueError):
                    execute(args)
            status = read_json(Path(temp)/"VALIDATION_GATES.json")
            self.assertEqual(status["status"],"stopped")
            self.assertEqual(status["failure"]["stage"],"P0")


if __name__ == "__main__":
    unittest.main()
