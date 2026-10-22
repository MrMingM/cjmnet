"""P7-0 -> P7-1 -> P7-2 -> P7-3 -> P7-4, CPU-only, fail closed and resumable."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import platform
import signal
import sys
import traceback
import unittest

from scorw_p7.core import (HERE, REPO, read_json, digest, identity, atomic_json, atomic_text,
    atomic_bytes, ensure_json, transaction, lock, begin, server_train, verify_inventory)
from scorw_p7.audit_sources import source_audit, historical_metadata


def load_config(path):
    c = read_json(path)
    reference = read_json(HERE/"config/protocol.json")
    if any(c[k] != v for k,v in reference.items() if k not in {"output_root","logs_root"}) or set(c) != set(reference):
        raise ValueError("Protocol differs from the delivered preregistration; freeze a new delivery before a new run")
    if (c["pool_size"] != 20 or c["field"] != dict(mean_alpha=.06,length_m=25.,contrast=.8,modes=32)
            or c["integration"]["step_m"] != .5 or c["frames_per_scene"] != 2
            or c["detectors_enabled"] or c["weather_pcd_enabled"] or c["automatic_seed_expansion"]):
        raise ValueError("P7 physical/scope invariants violated")
    return c


def synthetic_checks(output):
    from scorw_p7.tests import test_p7
    for p in HERE.rglob("*"):
        if not p.is_file() or {"__pycache__","local_validation"}.intersection(p.relative_to(HERE).parts):
            continue
        if p.suffix in {".py",".sh",".json",".md"} and b"\r" in p.read_bytes():
            raise ValueError("CR/CRLF in P7 delivery: "+str(p))
        if p.suffix == ".py":
            tree = ast.parse(p.read_text("utf-8"),filename=str(p))
            for node in ast.walk(tree):
                imports = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ""] if isinstance(node,ast.ImportFrom) else []
                if any(n.startswith(("torch","attfuse","shared_weather_benchmark_p6","scorw_p6_pipeline","reference_weather")) for n in imports):
                    raise ValueError("Forbidden detector/history execution dependency: "+str(p))
        if p.suffix == ".json":
            read_json(p)
    shell = (HERE/"run_all.sh").read_text("utf-8")
    if "pipefail" in shell or "[[" in shell:
        raise ValueError("Non-POSIX shell")
    stream = io.StringIO()
    with redirect_stdout(stream),redirect_stderr(stream):
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_p7)
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    atomic_json(output/"CHECKS.json",dict(status="passed" if result.wasSuccessful() else "failed",
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=stream.getvalue(),
        scope="synthetic arrays/temp fixtures + static code checks only; no real data/model experiment"))
    if not result.wasSuccessful():
        raise ValueError("P7 synthetic gate failed; numerical tolerances unchanged")


def output_paths(config, smoke):
    data = Path(config["output_root"]+("-smoke" if smoke else ""))
    logs = Path(config["logs_root"]+("-smoke" if smoke else ""))
    for p,parent,prefix in ((data,Path("/data/cjm/datasets"),"scorw-p7-"),
                            (logs,Path("/data/cjm/datasets/logs"),"scorw-p7-")):
        if p.is_symlink() or not p.is_absolute() or p.resolve().parent != parent.resolve() or not p.name.startswith(prefix):
            raise ValueError("Unsafe P7 output namespace")
    protected = [REPO,Path(config["train_root"]),Path(config["legacy_root"]),Path(config["p6_root"])]
    if any(p.resolve() == q.resolve() or p.resolve() in q.resolve().parents or q.resolve() in p.resolve().parents for p in (data,logs) for q in protected):
        raise ValueError("P7 output overlaps frozen inputs")
    return data,logs


def input_fingerprint(config, frames):
    from shared_weather_benchmark.dataset_io import source_paths
    root = server_train(config["train_root"])
    return [dict(scene=f["scene"],timestamp=f["timestamp"],cav=c,
        pcd=str(p),yaml=str(y),pcd_sha256=digest(p),yaml_sha256=digest(y))
        for f in frames for c in f["cavs"] for p,y in [source_paths(root,f,c)]]


def runtime_identity():
    from importlib.metadata import version
    return dict(python=platform.python_version(),platform=platform.system(),machine=platform.machine(),
        packages={n:version(n) for n in ("numpy","open3d","PyYAML")},
        threads={n:os.environ.get(n) for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})


def run_experiment(args, config, sources):
    from shared_weather_benchmark.audit import real_coordinate_audit
    from scorw_p7.select_scenes import server_selection
    from scorw_p7.field_pool import make_pool
    from scorw_p7.balanced_assignment import assignment
    from scorw_p7.extract_rays import extract_scene
    from scorw_p7.integrate_pool import integrate_cloud
    from scorw_p7.signal_metrics import analyze_scene
    from scorw_p7.statistics import analyze
    from scorw_p7.decision import decide
    from scorw_p7.report import write_reports, archive_reports
    server_train(config["train_root"])
    smoke = args.mode == "smoke"
    output,logs = output_paths(config,smoke)
    history = historical_metadata(config)
    selection = server_selection(history,config)
    scope = selection["selected"] if not smoke else dict(generator_development=selection["selected"]["generator_development"][:1],independent_validation=[])
    names = {s for values in scope.values() for s in values}
    frames = [f for f in selection["frames"] if f["scene"] in names]
    inputs = input_fingerprint(config,frames)
    effective = dict(config)
    if smoke:
        effective["sampling"] = dict(config["sampling"],**{k:config["smoke"][k] for k in ("max_per_stratum","max_memberships_per_cloud")})
    request = dict(config=effective,config_file_sha256=digest(args.config),source_identity=sources["source_identity"],
        history_identity=history["identity"],selection_identity=identity(selection),inputs=inputs,
        mode=args.mode,runtime=runtime_identity())
    with lock(output):
        contract = begin(output,request,args.resume == "1")
        path = output/"VALIDATION_GATES.json"
        if path.exists():
            previous = read_json(path)
            if previous.get("contract") != contract:
                raise ValueError("Previous gate contract changed")
            for registered in previous.get("stages",{}).values():
                folder = Path(registered["root"])
                if output.resolve() not in folder.resolve().parents or digest(folder/"COMMIT.json") != registered["commit_sha256"]:
                    raise ValueError("Previously passed stage commit changed")
                verify_inventory(folder,registered["hashes"])
            verify_inventory(output,previous.get("published_hashes",{}))
        gates = dict(status="running",contract=contract,mode=args.mode,stages={})
        atomic_json(path,gates)
        stage = "P7-0"
        try:
            p0 = output/"stages/P7-0"
            def audit(code):
                atomic_json(p0/"P7_SOURCE_AUDIT.json",dict(sources,history={k:v for k,v in history.items() if k not in {"old","p6"}}))
                atomic_json(p0/"P7_SCENE_SELECTION.json",selection)
                from shared_weather_benchmark.provenance import read_json as old_json
                old_config = old_json(REPO/"shared_weather_benchmark/config/prototype.json")
                real_coordinate_audit(old_config,dict(frames=frames),Path(config["train_root"]),p0)
                if input_fingerprint(config,frames) != inputs:
                    raise ValueError("Original inputs changed during P7-0")
                domains = read_json(p0/"FIELD_DOMAINS.json")
                pools = {s:make_pool(s,effective,domains[s]) for s in sorted(names)}
                plans = {s:assignment(selection["inventories"][s]["cavs"],selection["inventories"][s]["ego"],20,config["negative_control_shift"]) for s in sorted(selection["inventories"])}
                from scorw_p7.core import seed_for
                all_seeds = {s:[seed_for(config["weather_root_seed"],"scorw-p7-pool-v1",s,j) for j in range(20)]
                             for values in selection["selected"].values() for s in values}
                gt_frames = read_json(p0/"CLEAN_GT.json")
                frozen = dict(schema=1,contract=contract,source_identity=sources["source_identity"],config=effective,
                    all_scene_selection=selection,active_scene_scope=scope,
                    active_frames=[{k:f[k] for k in ("scene","timestamp","partition","cavs","active_cavs","ego")} for f in gt_frames],
                    weather_seeds=all_seeds,assignments=plans,pool_identities={s:p["sha256"] for s,p in pools.items()},
                    sampling=effective["sampling"],primary_metrics=["within-target across-weather correlation","C mean/std/positive frequency","joint R/tau structure"],
                    tolerance=config["tolerance"],decision=config["decision"],
                    input_manifest_sha256=digest(p0/"INPUT_MANIFEST.json"),clean_gt_sha256=digest(p0/"CLEAN_GT.json"),
                    frozen_before_any_weather_integration=True,automatic_seed_expansion=False,
                    budget=dict(max_primary_ray_field_pairs=20*len(inputs)*effective["sampling"]["max_memberships_per_cloud"],
                                integration_batch=effective["integration"]["batch"],threads=1,weather_pcd_files=0),
                    approximation="clear-return-conditioned, sampled target attenuation only; uncalibrated project effect margins")
                atomic_json(p0/"P7_FIELD_POOL_MANIFEST.json",dict(pools=pools,assignments=plans,all_selected_scene_seeds=all_seeds))
                atomic_json(p0/"P7_PROTOCOL_FREEZE.json",frozen)
            def phase(name, folder, callback, status="passed"):
                c = transaction(folder,dict(parent=contract,stage=name),callback)
                gates["stages"][name] = dict(status=status,root=str(folder),commit_sha256=digest(folder/"COMMIT.json"),hashes=c["hashes"])
                atomic_json(path,gates)
                print(name+" "+status,flush=True)
            phase(stage,p0,audit)
            freeze = read_json(p0/"P7_PROTOCOL_FREEZE.json")
            manifest = read_json(p0/"P7_FIELD_POOL_MANIFEST.json")
            stage = "P7-1"
            p1 = output/"stages/P7-1"
            phase(stage,p1,lambda code:synthetic_checks(p1))
            def compute_partition(folder, partition):
                for scene in scope[partition]:
                    scene_root = folder/scene
                    def scene_run(code):
                        clouds = extract_scene(scene,p0,scene_root/"rays",effective,code)
                        rows = []
                        for cloud in clouds:
                            relative = cloud.relative_to(scene_root/"rays")
                            target = scene_root/"pool_cache"/relative
                            def integrate(cache_code):
                                integrate_cloud(cloud,target,manifest["pools"][scene],manifest["assignments"][scene],effective,cache_code)
                            transaction(target,dict(parent=code,rays_commit=digest(cloud/"COMMIT.json"),pool=manifest["pools"][scene]["sha256"]),integrate)
                            matrix = read_json(target/"MATRIX.json")
                            if identity(matrix["rows"]) != matrix["matrix_identity"]:
                                raise ValueError("Signal matrix rows changed")
                            rows.extend(matrix["rows"])
                        result = analyze_scene(rows,manifest["assignments"][scene],effective)
                        atomic_json(scene_root/"SCENE_RESULT.json",result)
                    transaction(scene_root,dict(parent=contract,stage=partition,scene=scene,
                        pool=manifest["pools"][scene]["sha256"],assignment=manifest["assignments"][scene]["sha256"]),scene_run)
            stage = "P7-2"
            p2 = output/"stages/P7-2"
            phase(stage,p2,lambda code:compute_partition(p2,"generator_development"))
            stage = "P7-3"
            p3 = output/"stages/P7-3"
            if smoke or not scope["independent_validation"]:
                status = "not_requested_smoke" if smoke else "not_executable"
                phase(stage,p3,lambda code:atomic_json(p3/"UNAVAILABLE.json",dict(status=status,
                    reason="SMOKE never evaluates validation" if smoke else "INCONCLUSIVE_VALIDATION_UNAVAILABLE",
                    eligible_candidates=selection["available_validation_candidates"],no_reserved_scene_substitution=True)),status)
            else:
                phase(stage,p3,lambda code:compute_partition(p3,"independent_validation"))
            stage = "P7-4"
            p4 = output/"stages/P7-4"
            scenes = [read_json(folder/scene/"SCENE_RESULT.json") for partition,folder in (("generator_development",p2),("independent_validation",p3)) for scene in scope[partition]]
            matrices = [read_json(p) for folder in (p2,p3) for p in sorted(folder.glob("*/pool_cache/*/*/MATRIX.json"))]
            def final(code):
                analysis = analyze(scenes,effective)
                verdict = decide(analysis,scenes,selection,effective,smoke)
                write_reports(p4,scenes,analysis,verdict,selection,freeze,matrices)
            phase(stage,p4,final)
            for name in ("P7_PROTOCOL_FREEZE.json","P7_SOURCE_AUDIT.json","P7_SCENE_SELECTION.json","P7_FIELD_POOL_MANIFEST.json"):
                source = p0/name
                target = output/name
                if target.exists() and digest(target) != digest(source):
                    raise ValueError("Published frozen report differs: "+name)
                atomic_bytes(target,source.read_bytes())
            for name in read_json(p4/"COMMIT.json")["hashes"]:
                if name == "CONTRACT.json":
                    continue
                source,target = p4/name,output/name
                if target.exists() and digest(target) != digest(source):
                    raise ValueError("Published result changed: "+name)
                atomic_bytes(target,source.read_bytes())
            if input_fingerprint(config,frames) != inputs or source_audit()["source_identity"] != sources["source_identity"]:
                raise ValueError("Sources or original inputs changed before final publication")
            gates["status"] = "complete_smoke_only" if smoke else "complete" if scope["independent_validation"] else "complete_development_only"
            gates["decision"] = read_json(output/"P7_DECISION.json")["decision"]
            published = ["CONTRACT.json","P7_PROTOCOL_FREEZE.json","P7_SOURCE_AUDIT.json","P7_SCENE_SELECTION.json","P7_FIELD_POOL_MANIFEST.json"]
            published += [n for n in read_json(p4/"COMMIT.json")["hashes"] if n != "CONTRACT.json"]
            allowed = set(published) | {"VALIDATION_GATES.json","P7_REPORT_MANIFEST.json","FAILURE.json","P7_SMALL_RESULTS.tar.gz",".p7.lock"}
            if any(p.is_file() and p.name not in allowed for p in output.iterdir()):
                raise ValueError("Unregistered file in published output root")
            gates["published_hashes"] = {name:digest(output/name) for name in sorted(published)}
            atomic_json(path,gates)
            # Repeated completion produces the same archive; the gate contains no clock timestamps.
            archive = archive_reports(output,[*published,"VALIDATION_GATES.json"])
            logs.mkdir(parents=True,exist_ok=True)
            atomic_json(logs/"LATEST.json",dict(status=gates["status"],decision=gates["decision"],output=str(output),archive=archive))
            print("P7 complete: "+str(output)+"; "+gates["decision"],flush=True)
        except BaseException as error:
            gates.update(status="stopped",failure=dict(stage=stage,type=type(error).__name__,message=str(error)))
            atomic_json(path,gates)
            atomic_json(output/"FAILURE.json",dict(stage=stage,traceback=traceback.format_exc()))
            raise


def interrupted(*_):
    raise InterruptedError("SIGTERM received; uncommitted calculations will be recomputed deterministically")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(HERE/"config/protocol.json"))
    parser.add_argument("--mode",choices=("checks","smoke","full"),default="checks")
    parser.add_argument("--resume",choices=("0","1"),default="0")
    parser.add_argument("--check-output")
    args = parser.parse_args()
    config = load_config(args.config)
    sources = source_audit()
    if args.mode == "checks":
        output = Path(args.check_output) if args.check_output else Path(config["logs_root"])/"checks" if platform.system()=="Linux" else HERE/"local_validation"
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/"P7_SOURCE_AUDIT.json",sources)
        synthetic_checks(output)
        atomic_json(output/"VALIDATION_GATES.json",dict(status="checks_passed",stages={
            "P7-0":dict(status="passed",scope="static source audit only; historical real-data audit deferred to server smoke/full"),
            "P7-1":dict(status="passed",scope="synthetic numerical tests only")}))
        return
    signal.signal(signal.SIGTERM,interrupted)
    run_experiment(args,config,sources)


if __name__ == "__main__":
    main()
