"""P6-A -> P6-B -> P6-C, with immutable prerequisites and CPU-only execution."""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import platform
import sys
import traceback
import unittest

from shared_weather_benchmark_p6.core import (
    HERE, REPO, LEGACY_CONTRACT, read_json, digest, identity, atomic_json, atomic_text,
    verify_code, begin, lock_output, verify_external)


def load_config(path):
    c = read_json(path)
    if c["schema"] != 1 or c["legacy_contract"] != LEGACY_CONTRACT:
        raise ValueError("Unknown P6 schema or historical contract")
    old = read_json(REPO/"shared_weather_benchmark/config/prototype.json")
    for key in ("field","integration"):
        if c[key] != old[key]:
            raise ValueError("P6 cannot change inherited physical parameters: " + key)
    if c["matching"] != {k:v for k,v in old["matching"].items() if k != "parameter_fit"}:
        raise ValueError("Legacy matching tolerances must remain frozen")
    if c["thresholds"] != old["observability"]["intensity_threshold_sensitivity"]:
        raise ValueError("All three frozen thresholds required")
    if (c["scene_counts"] != dict(generator_development=4,physics_validation=8)
            or c["frames_per_scene"] != 2 or c["repeat_indices"] != [0,1,2]
            or c["protocols"] != ["E1","E2","E3"] or c["smoke_repeat_indices"] != [0]
            or c["smoke_scenes_per_partition"] != 1):
        raise ValueError("P6 formal/smoke scope differs from preregistration")
    if c["detectors_enabled"] or c["physical_model_changes"]:
        raise ValueError("Model execution/physics changes forbidden in P6")
    if c["bootstrap"]["minimum_scenes"] < 5 or c["bootstrap"]["replicates"] < 100:
        raise ValueError("Invalid exploratory bootstrap")
    for key,edges in c["strata"].items():
        if len(edges) < 2 or any(b <= a for a,b in zip(edges,edges[1:])):
            raise ValueError("Invalid strata: " + key)
    for key in ("minimum_matching_fraction","minimum_direction_fraction"):
        if not 0 < c["decision"][key] <= 1:
            raise ValueError("Invalid declared decision heuristic")
    return c


def server_paths(c,smoke):
    from shared_weather_benchmark.dataset_io import server_only,approved_path
    server_only()
    source = approved_path(c["train_root"])
    legacy = Path(c["legacy_root"]).resolve(strict=True)
    base = Path("/data/cjm/datasets").resolve(strict=True)
    data = Path(c["data_root"]+"-smoke" if smoke else c["data_root"]).resolve()
    logs = Path(c["logs_root"]).resolve()
    protected = [source,legacy,REPO]
    if (data.parent != base or not data.name.startswith("shared-weather-benchmark-p6-")
            or logs.parent != base/"logs" or not logs.name.startswith("scorw-p6-")):
        raise ValueError("P6 outputs require dedicated sibling namespaces: /data/cjm/datasets/shared-weather-benchmark-p6-* and logs/scorw-p6-*")
    for target in (data,logs):
        if base not in target.parents or any(target == p or p in target.parents or target in p.parents for p in protected):
            raise ValueError("P6 output overlaps frozen/source tree: " + str(target))
        if any("v2x" in p.lower() or "opv2v-w" in p.lower() or p.lower() in {"train","test","validate"} for p in target.parts):
            raise ValueError("Refusing protected dataset path")
    if data == logs or data in logs.parents or logs in data.parents:
        raise ValueError("Data/log roots must be separate")
    return data,logs


def registered_files(root):
    return [p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
            and p.name not in {".scor-w.lock","COMMIT.json","VALIDATION_GATES.json"}
            and not p.name.endswith(".partial") and not any(part.startswith(".recheck-") or part.startswith(".comparison-") for part in p.parts)]


def products_for(root):
    return [dict(path=str(p.resolve()),sha256=digest(p),bytes=p.stat().st_size) for p in registered_files(root)]


def runtime_identity():
    from importlib.metadata import version
    return dict(python=platform.python_version(),platform=platform.system(),machine=platform.machine(),
        packages={name:version(name) for name in ("numpy","open3d","PyYAML")},
        threads={name:os.environ.get(name) for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})


def phase(output,request,run):
    if output.is_symlink():
        raise ValueError("Stage output may not alias a frozen directory")
    with lock_output(output):
        code = begin(output,request,resume=(output/"CONTRACT.json").exists())
        commit = output/"COMMIT.json"
        if commit.exists():
            stored = read_json(commit)
            if stored["contract"] != code or stored["status"] != "passed":
                raise ValueError("Stage commit contract mismatch")
            if {str(p.resolve()) for p in registered_files(output)} != {p["path"] for p in stored["products"]}:
                raise ValueError("Committed stage product inventory changed")
            verify_external(stored["products"])
            return stored
        run(code)
        stored = dict(status="passed",contract=code,products=products_for(output))
        atomic_json(commit,stored)
        return stored


def checks(output):
    from shared_weather_benchmark_p6.tests import test_p6
    for path in HERE.rglob("*"):
        if not path.is_file() or {"local_validation","__pycache__"}.intersection(path.relative_to(HERE).parts):
            continue
        if path.suffix in {".py",".sh",".json",".md"} and b"\r" in path.read_bytes():
            raise ValueError("CR/CRLF in delivery: " + str(path))
        if path.suffix == ".py":
            ast.parse(path.read_text("utf-8"),filename=str(path))
        if path.suffix == ".json":
            read_json(path)
    shell = (HERE/"run_server.sh").read_text("utf-8")
    if "pipefail" in shell or "[[" in shell:
        raise ValueError("Non-POSIX shell construct")
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_p6)
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"CHECKS.json",dict(status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),log=stream.getvalue(),
        scope="AST/JSON/hash/LF + Python/NumPy synthetic arrays and synthetic file fixtures only",
        shell_parser="server launcher runs sh -n; Windows parser availability not assumed",real_data_read=False))
    if not result.wasSuccessful():
        raise ValueError("Synthetic P6 checks failed; tolerances unchanged")


def execute(args):
    config = load_config(args.config)
    code_id = verify_code()
    if args.mode == "checks":
        checks(Path(args.check_output) if args.check_output else
               HERE/"local_validation" if os.name != "posix" else Path(config["logs_root"])/"checks")
        return
    data,logs = server_paths(config,args.smoke)
    state_root = logs/("state-smoke" if args.smoke else "state-full")
    if args.mode == "status":
        print((state_root/"VALIDATION_GATES.json").read_text("utf-8"))
        return
    from shared_weather_benchmark_p6.legacy import verify_legacy,audit_legacy
    from shared_weather_benchmark_p6.metrics import write_analysis
    from shared_weather_benchmark_p6.core import read_jsonl
    from shared_weather_benchmark_p6.repeat import run_repeats,verify_generated
    old_gate_path = Path(config["legacy_root"])/"VALIDATION_GATES.json"
    request = dict(config=config,config_file_sha256=digest(args.config),source_identity=code_id,smoke=args.smoke,
                   runtime=runtime_identity(),legacy_contract=config["legacy_contract"],
                   legacy_gate_sha256=digest(old_gate_path) if old_gate_path.is_file() else None)
    shared_request = {k:v for k,v in request.items() if k != "smoke"}
    with lock_output(state_root):
        contract = begin(state_root,request,args.resume)
        atomic_json(state_root/"RESOLVED_CONFIG.json",config)
        gates_path = state_root/"VALIDATION_GATES.json"
        state = read_json(gates_path) if gates_path.exists() else dict(contract=contract,stages={})
        if state["contract"] != contract:
            raise ValueError("P6 gate contract mismatch")
        state.pop("failure",None)
        stage = "P6-A"
        try:
            audit_root,metric_root = logs/"audit",logs/"metrics"
            # Rehash all old inputs before trusting any cached prerequisite, and
            # preserve a machine-readable stopped state on identity failures.
            evidence = verify_legacy(Path(config["legacy_root"]),config["legacy_contract"],config["train_root"],config["historical_scenes"])
            for previous,folder in (("P6-A",audit_root),("P6-B",metric_root),("P6-C",data)):
                committed = state["stages"].get(previous)
                if committed and committed["status"] == "passed":
                    if not (folder/"COMMIT.json").is_file() or digest(folder/"COMMIT.json") != committed["commit_sha256"]:
                        raise ValueError("Previously committed stage manifest changed: "+previous)
                    verify_external(committed["products"])
            if args.mode in {"metrics","repeat"} and not (audit_root/"COMMIT.json").exists():
                raise ValueError("P6-A must pass first; run MODE=audit or MODE=all")
            a = phase(audit_root,dict(shared_request,stage="P6-A"),lambda code:audit_legacy(config,audit_root,evidence))
            state["stages"][stage] = dict(status="passed",commit_sha256=digest(audit_root/"COMMIT.json"),products=a["products"])
            state["status"] = "P6A_passed"
            atomic_json(gates_path,state)
            if args.mode == "audit":
                return
            stage = "P6-B"
            if args.mode == "repeat" and not (metric_root/"COMMIT.json").exists():
                raise ValueError("P6-B must pass first; run MODE=metrics or MODE=all")
            def run_metrics(code):
                cache = read_json(audit_root/"cache/CACHE_MANIFEST.json")
                from shared_weather_benchmark_p6.core import verify_hashes
                verify_hashes(audit_root/"cache",cache["hashes"])
                rows = [r for name in cache["hashes"] for r in read_json(audit_root/"cache"/name)["rows"]]
                write_analysis(metric_root,rows,config,["E0","E1","E2","E3"])
                atomic_text(metric_root/"P6B_METRIC_DEFINITIONS.md",(HERE/"METRIC_DEFINITIONS.md").read_text("utf-8"))
            b = phase(metric_root,dict(shared_request,stage=stage,audit_commit_sha256=digest(audit_root/"COMMIT.json")),run_metrics)
            state["stages"][stage] = dict(status="passed",commit_sha256=digest(metric_root/"COMMIT.json"),products=b["products"])
            state["status"] = "P6B_passed"
            atomic_json(gates_path,state)
            if args.mode == "metrics":
                return
            stage = "P6-C"
            c = phase(data,dict(request,stage=stage,metrics_commit_sha256=digest(metric_root/"COMMIT.json")),
                lambda code:run_repeats(config,evidence,data,code,args.smoke,args.resume))
            # The nested dataset manifests bind metadata links and large clouds.
            for folder in sorted((data/"repeats").iterdir()):
                cfg = read_json(folder/"RESOLVED_CONFIG.json")
                verify_generated(folder,cfg,read_json(folder/"CONTRACT.json")["identity"])
            state["stages"][stage] = dict(status="passed",commit_sha256=digest(data/"COMMIT.json"),products=c["products"])
            state["status"] = "complete_smoke_only" if args.smoke else "complete"
            atomic_json(gates_path,state)
            atomic_json(data/"VALIDATION_GATES.json",state)
            print("P6 complete: "+str(data),flush=True)
        except BaseException as error:
            state.update(status="stopped",failure=dict(stage=stage,type=type(error).__name__,message=str(error)))
            atomic_json(gates_path,state)
            atomic_text(state_root/"FAILURE.txt",traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(HERE/"config/protocol.json"))
    parser.add_argument("--mode",choices=["audit","metrics","repeat","all","checks","status"],default="checks")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--check-output")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
