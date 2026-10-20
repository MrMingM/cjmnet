"""P0 -> P5 stage runner; numerical stages use only Python standard library."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import os
from pathlib import Path
import platform
import sys
import traceback
import unittest
import tempfile

from shared_weather_benchmark.geometry import numerical_audit
from shared_weather_benchmark.provenance import (HERE,REPO,atomic_bytes,atomic_json,atomic_text,digest,identity,
                                                read_json,verify_sources,inspect_sources,check_hashes,
                                                install_runtime_source_guard,source_diagnostic_text)

STAGES = ["P0","P1","P2","P3","P4","P5"]


def load_config(path):
    import math
    c = read_json(path)
    if c["schema"] != 1 or c["protocols"] != ["E0","E1","E2","E3"]:
        raise ValueError("All four frozen protocols required")
    if c["partitions"] != ["generator_development","physics_validation","future_model_development"]:
        raise ValueError("Three fixed disjoint scene partitions required")
    for key in ("scenes_per_partition","frames_per_scene","smoke_frames_per_scene"):
        if not isinstance(c[key],int) or c[key] < 1:
            raise ValueError("Positive scene/frame counts required")
    if c["frames_per_scene"] > 32 or c["scenes_per_partition"] > 5:
        raise ValueError("v1 explicitly limits generation scale; full generation is outside scope")
    if c["smoke_frames_per_scene"] > c["frames_per_scene"]:
        raise ValueError("SMOKE cannot use more frames than the small sample protocol")
    if c["e0"] != read_json(HERE/"config"/"prototype.json")["e0"]:
        raise ValueError("E0 must preserve frozen reference Fog control settings")
    policy = c["return_policy"]
    if policy["name"] not in {"attenuation_only","empirical_intensity_threshold"}:
        raise ValueError("Unknown return policy")
    if policy["name"] == "empirical_intensity_threshold" and (policy["status"] != "uncalibrated_approximation" or not 0 < policy["threshold"] < 1):
        raise ValueError("Empirical threshold must be explicitly uncalibrated")
    if policy["name"] == "attenuation_only" and policy["threshold"] != 0:
        raise ValueError("Attenuation-only cannot imply a return threshold")
    from shared_weather_benchmark.world_field import WorldField
    WorldField(**c["field"])
    if not math.isfinite(c["integration"]["step_m"]) or c["integration"]["step_m"] <= 0:
        raise ValueError("Invalid integration step")
    if c["integration"]["batch"] < 1 or c["integration"]["sample_error_tolerance"] <= 0:
        raise ValueError("Invalid integration batch/tolerance")
    if any(not math.isfinite(float(v)) or v <= 0 for v in c["audit"].values()):
        raise ValueError("Audit tolerances and limits must be positive and finite")
    edges = c["distance_bins_m"]
    if edges[0] != 0 or len(edges) < 2 or any(not math.isfinite(v) for v in edges) or any(b <= a for a,b in zip(edges,edges[1:])):
        raise ValueError("Distance bins must be finite, strictly increasing, and start at zero")
    obs = c["observability"]
    if (obs["min_returns"] < 1 or not 0 <= obs["good_ratio"] <= 1 or not 0 <= obs["disparity"] <= 1
            or not obs["sensitivity_min_returns"] or any(n < 1 for n in obs["sensitivity_min_returns"])
            or not obs["sensitivity_good_ratio"] or any(not 0 <= n <= 1 for n in obs["sensitivity_good_ratio"])
            or any(not 0 < n < 1 for n in obs["intensity_threshold_sensitivity"])):
        raise ValueError("Invalid pre-registered observability thresholds")
    if c["bootstrap"]["replicates"] < 100 or c["bootstrap"]["minimum_scenes_for_interval"] < 5:
        raise ValueError("Scene-bootstrap settings below protocol minimum")
    c["integration_for_generation"] = {k:c["integration"][k] for k in ("step_m","batch")}
    return c


@contextmanager
def lock_output(output):
    output.mkdir(parents=True,exist_ok=True)
    with (output/".scor-w.lock").open("a+b") as stream:
        if os.name == "posix":
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            import msvcrt
            stream.seek(0); stream.write(b"0"); stream.flush(); stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(stream,fcntl.LOCK_UN)
            else:
                stream.seek(0); msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def numerical_tests(classes):
    from shared_weather_benchmark.tests import test_numerics as tests
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(tests,c)) for c in classes)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():
        raise ValueError("Numerical gate failed")
    return dict(status="passed",tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                backend="Python standard library; synthetic numeric data only",log=stream.getvalue())


def preflight_generation(config,output):
    import numpy as np
    from importlib.metadata import version
    from shared_weather_benchmark.dataset_io import validate_roots,select_scenes
    from shared_weather_benchmark.e0_adapter import table_identity
    from shared_weather_benchmark.audit import real_coordinate_audit
    from shared_weather_benchmark.generator import materialize_fields
    from shared_weather_benchmark.compare import field_diagnostics
    source,_ = validate_roots(config)
    table = table_identity(config)
    from shared_weather_benchmark.tests.test_server_arrays import run_server_tests
    server_tests = run_server_tests(config)
    runtime = dict(python=platform.python_version(),numpy=np.__version__,
                   dependencies={n:version(n) for n in ("scipy","open3d","PyYAML")})
    # Audit into a temporary tree. Never overwrite committed GT/input identities
    # while discovering a resume mismatch.
    with tempfile.TemporaryDirectory(prefix=".preflight-",dir=output) as temporary:
        work = Path(temporary)
        selection = select_scenes(config,read_json(output/"CONTRACT.json")["request"]["smoke"])
        atomic_json(work/"SELECTION.json",selection)
        real_coordinate_audit(config,selection,source,work)
        domains = read_json(work/"FIELD_DOMAINS.json")
        fields = materialize_fields(config,selection,domains)
        atomic_json(work/"SCENE_WEATHER_FIELD_MANIFEST.json",fields)
        diagnostics = field_diagnostics(config,selection,domains)
        atomic_json(work/"PREGEN_FIELD_CHECKS.json",dict(server_tests=server_tests,scenes=diagnostics))
        frozen = dict(runtime=runtime,e0_table=table,selection_sha256=identity(selection),
            input_manifest_sha256=digest(work/"INPUT_MANIFEST.json"),clean_gt_sha256=digest(work/"CLEAN_GT.json"),
            field_manifest_sha256=digest(work/"SCENE_WEATHER_FIELD_MANIFEST.json"),
            config_sha256=identity(config),ap_used=False,parameter_fit="none",detectors_started=False)
        path = output/"GENERATION_FREEZE.json"
        if path.exists() and read_json(path) != frozen:
            raise ValueError("Generation freeze/input/runtime changed; use a new output root")
        for file in work.iterdir():
            dest = output/file.name
            if path.exists() and (not dest.is_file() or digest(dest) != digest(file)):
                raise ValueError("Committed preflight output changed: " + str(dest))
        for file in work.iterdir():
            atomic_bytes(output/file.name,file.read_bytes())
        atomic_json(path,frozen)
    return frozen


def run_stage(stage,config,output,contract,resume):
    if stage == "P0":
        import ast
        syntax_files = list(HERE.rglob("*.py"))
        for file in syntax_files:
            if "local_validation" not in file.parts:
                ast.parse(file.read_text("utf-8"),filename=str(file))
        shell = (HERE/"run_server.sh").read_bytes()
        if b"\r" in shell or b"pipefail" in shell or b"[[" in shell:
            raise ValueError("Shell LF/POSIX static gate failed")
        result = numerical_tests(["GeometryTests","SourceDiagnosticTests"])
        source_report = inspect_sources()
        source_id = verify_sources(source_report)
        if source_id != read_json(output/"CONTRACT.json")["request"]["source_identity"]:
            raise ValueError("Source identity changed during P0")
        atomic_json(output/"SOURCE_AUDIT.json",source_report)
        atomic_json(output/"P0_NUMERICAL.json",dict(result,geometry=numerical_audit(),
            ast_files=len(syntax_files),shell_static_lf_posix=True,source_identity=source_id))
        for name in ("PRIOR_ART_AND_SCOPE.md","PHYSICAL_MODEL.md","PHYSICS_ASSUMPTIONS.md"):
            atomic_text(output/name,(HERE/name).read_text("utf-8"))
        if not (output/"REAL_COORDINATE_AUDIT.json").exists():
            atomic_text(output/"COORDINATE_AUDIT.md",(HERE/"COORDINATE_AUDIT.md").read_text("utf-8"))
        return ["P0_NUMERICAL.json","CONFIG_INPUT.json","RESOLVED_CONFIG.json","SOURCE_SNAPSHOT.json","SOURCE_AUDIT.json",
                "PRIOR_ART_AND_SCOPE.md","PHYSICAL_MODEL.md","PHYSICS_ASSUMPTIONS.md"]
    if stage in {"P1","P2"}:
        result = numerical_tests(["IntegrationTests"] if stage == "P1" else ["FieldTests","DiagnosticTests"])
        atomic_json(output/(stage+"_NUMERICAL.json"),result)
        return [stage+"_NUMERICAL.json"]
    if stage == "P3":
        from shared_weather_benchmark.generator import generate
        preflight_generation(config,output)
        generate(config,output,contract,resume)
        return ["SELECTION.json","INPUT_MANIFEST.json","CLEAN_GT.json","REAL_COORDINATE_AUDIT.json",
                "FIELD_DOMAINS.json","SCENE_WEATHER_FIELD_MANIFEST.json","PREGEN_FIELD_CHECKS.json",
                "GENERATION_FREEZE.json","DATA_MANIFEST.json","COORDINATE_AUDIT.md"]
    if stage == "P4":
        from shared_weather_benchmark.observability import audit_observability
        from shared_weather_benchmark.compare import compare
        audit_observability(config,output,contract)
        compare(config,output)
        return ["OBSERVABILITY_ROWS.jsonl","OBSERVABILITY_TARGETS.json","OBSERVABILITY_SUMMARY.json",
                "COMPARISON.json","E0_E1_E2_E3_COMPARISON.md"]
    if stage == "P5":
        comparison = read_json(output/"COMPARISON.json")
        unmatched = [name for name,data in comparison["scenes"].items() if not data["matching"]["finite_realization_matched"]]
        atomic_json(output/"ISSUES.json",dict(unmatched_scenes=unmatched,
            physical_realism_validated=False,scan_timing_available=False,sensor_threshold_calibrated=False,
            nonuniform_scattering_implemented=False,object_surface_point_truth_available=False,
            statistical_power="small sample, descriptive",model_failure_modes="pending future frozen detector experiment"))
        atomic_text(output/"RESULTS.md","# SCOR-W 第一版结果\n\nP0–P4工程/数值/身份门禁通过；真实物理真实性尚未证明。\n\n"+
            f"运行合同：`{contract}`。未匹配scene：{unmatched}。\n\n"+
            "逐scene结果见E0_E1_E2_E3_COMPARISON.md；问题清单见ISSUES.json。\n\n"+
            "默认E1–E3仅衰减强度，点数保持，因此零回波比例主要反映Clean几何遮挡；新差异应首先检查连续强度证据。\n\n"+
            "扩大依据：坐标/GT/来源门禁全过；开发和物理验证分区均显示稳定的E2/E3互补差异；匹配残差足够小且效应超出scene变化；阈值敏感性保持；增加独立scene/天气seed；传感器参数有可核验依据。\n\n"+
            "如果差异只来自总衰减不匹配、少数scene或任意阈值，停止扩大并报告负结果。检测失败模式仍需以后冻结协议后验证；本轮未运行AttFuse/GSPR。\n")
        return ["RESULTS.md","ISSUES.json"]
    raise ValueError(stage)


def execute(args):
    source_report = inspect_sources()
    source_id = verify_sources(source_report)
    install_runtime_source_guard()
    print(source_diagnostic_text(source_report),flush=True)
    config = load_config(args.config)
    if args.output:
        config["output_root"] = str(Path(args.output).expanduser().resolve())
    output = Path(config["output_root"]).expanduser().resolve()
    # Validate server output roots before any directory creation for real phases.
    if STAGES.index(args.through) >= 3:
        from shared_weather_benchmark.dataset_io import validate_roots
        validate_roots(config)
    request = dict(config=config,config_file_sha256=digest(args.config),smoke=args.smoke,source_identity=source_id)
    contract = identity(request)
    with lock_output(output):
        path = output/"CONTRACT.json"
        if path.exists():
            if read_json(path) != dict(identity=contract,request=request):
                raise ValueError("Config/smoke/source contract differs; use a separate output directory")
            if not args.resume:
                raise ValueError("Existing output requires RESUME=1")
        else:
            if any(p.name != ".scor-w.lock" for p in output.iterdir()):
                raise ValueError("Refusing unrelated nonempty output")
            atomic_json(path,dict(identity=contract,request=request))
            atomic_bytes(output/"CONFIG_INPUT.json",Path(args.config).read_bytes())
            atomic_json(output/"RESOLVED_CONFIG.json",config)
            atomic_bytes(output/"SOURCE_SNAPSHOT.json",(HERE/"source_snapshot.json").read_bytes())
            atomic_json(output/"SOURCE_AUDIT.json",source_report)
        state_path = output/"VALIDATION_GATES.json"
        state = read_json(state_path) if state_path.exists() else dict(contract=contract,stages={},status="in_progress")
        if state["contract"] != contract:
            raise ValueError("Gate contract mismatch")
        state.pop("failure",None)
        state["status"] = "in_progress"
        try:
            for stage in STAGES[:STAGES.index(args.through)+1]:
                existing = state["stages"].get(stage)
                if existing and existing["status"] == "passed":
                    check_hashes(output,existing["hashes"])
                    if stage == "P3":
                        # Revalidate input/resource/runtime AND all committed cloud files.
                        preflight_generation(config,output)
                        from shared_weather_benchmark.generator import validate_record
                        inputs = {(r["scene"],r["timestamp"],r["cav"]):r for r in read_json(output/"INPUT_MANIFEST.json")}
                        manifest = read_json(output/"DATA_MANIFEST.json")
                        check_hashes(output,manifest["record_hashes"])
                        for name in manifest["records"]:
                            record = read_json(output/name)
                            validate_record(output,record,inputs[(record["scene"],record["timestamp"],record["cav"])],contract)
                    print(stage+" committed gate revalidated",flush=True)
                    continue
                if STAGES.index(stage) < STAGES.index(args.from_stage):
                    raise ValueError("Required earlier stage has not passed: " + stage)
                print(stage+" start",flush=True)
                products = run_stage(stage,config,output,contract,args.resume)
                state["stages"][stage] = dict(status="passed",hashes={p:digest(output/p) for p in products})
                state["status"] = "complete" if stage == "P5" else "in_progress"
                atomic_json(state_path,state)
            state["requested_through"] = args.through
            state["status"] = "complete" if state["stages"].get("P5",{}).get("status") == "passed" else "numerical_passed_pending_real_data"
            atomic_json(state_path,state)
        except BaseException as error:
            state["status"] = "stopped"
            state["failure"] = dict(stage=stage,error=str(error),type=type(error).__name__)
            atomic_json(state_path,state)
            atomic_text(output/"FAILURE.txt",traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(HERE/"config"/"prototype.json"))
    parser.add_argument("--output")
    parser.add_argument("--through",choices=STAGES,default="P2")
    parser.add_argument("--from-stage",choices=STAGES,default="P0")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if STAGES.index(args.from_stage) > STAGES.index(args.through):
        parser.error("from-stage must precede through")
    execute(args)


if __name__ == "__main__":
    main()
