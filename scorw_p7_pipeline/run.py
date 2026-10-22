"""One command: checks -> SMOKE -> full P7 -> verified small result archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback

from scorw_p7.core import (REPO, read_json, atomic_json, atomic_text, digest, identity,
                            child, verify_inventory, lock, begin)
from scorw_p7.audit_sources import source_audit

HERE = Path(__file__).resolve().parent
STEPS = ("checks", "smoke", "full")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def delivery_identity(config_path):
    return dict(p7_source_identity=source_audit()["source_identity"],config_path=str(config_path),
                config_sha256=digest(config_path),
                pipeline_sources={n:digest(HERE/n) for n in ("run.py","run_all.sh")},
                python=str(Path(sys.executable).resolve()),python_version=platform.python_version(),steps=list(STEPS))


def validate_layout(config):
    if platform.system() != "Linux":
        raise RuntimeError("Execute real P7 stages only on the Linux server")
    if REPO != Path("/home/cjm/OpenCOOD-main/cjmnet").resolve():
        raise ValueError("Existing P7 entry requires /home/cjm/OpenCOOD-main/cjmnet")
    data,logs = Path(config["output_root"]),Path(config["logs_root"])
    for p,parent in ((data,Path("/data/cjm/datasets")),(logs,Path("/data/cjm/datasets/logs"))):
        if p.is_symlink() or not p.is_absolute() or p.resolve().parent != parent.resolve() or not p.name.startswith("scorw-p7-"):
            raise ValueError("Unexpected P7 output namespace: "+str(p))
    state = logs/"pipeline"
    if state.is_symlink():
        raise ValueError("Pipeline state may not alias another directory")
    return data,logs,state


def run_child(command, env, cwd):
    process = subprocess.Popen(command,env=env,cwd=cwd,start_new_session=True)
    try:
        return process.wait()
    except BaseException:
        # Keep the outer lock until its child has stopped writing results.
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        raise


def verify_gate(mode, config):
    logs = Path(config["logs_root"])
    if mode == "checks":
        root = logs/"checks"
        gates = read_json(root/"VALIDATION_GATES.json")
        checks = read_json(root/"CHECKS.json")
        if (gates.get("status") != "checks_passed" or checks.get("status") != "passed"
                or checks.get("tests",0) <= 0 or checks.get("failures") != 0 or checks.get("errors") != 0
                or any(gates.get("stages",{}).get(s,{}).get("status") != "passed" for s in ("P7-0","P7-1"))):
            raise ValueError("P7 checks did not pass all required gates")
        return dict(status="checks_passed",tests=checks["tests"],gate_sha256=digest(root/"VALIDATION_GATES.json"))
    smoke = mode == "smoke"
    root = Path(config["output_root"]+("-smoke" if smoke else ""))
    gates = read_json(root/"VALIDATION_GATES.json")
    status = gates.get("status")
    expected = {"complete_smoke_only"} if smoke else {"complete","complete_development_only"}
    if status not in expected:
        raise ValueError(mode+": required completion gate missing or failed")
    stages = gates.get("stages",{})
    if any(stages.get(s,{}).get("status") != "passed" for s in ("P7-0","P7-1","P7-2","P7-4")):
        raise ValueError(mode+": prerequisite or analysis stage did not pass")
    validation = "not_requested_smoke" if smoke else "passed" if status == "complete" else "not_executable"
    if stages.get("P7-3",{}).get("status") != validation:
        raise ValueError(mode+": independent validation status is inconsistent")
    verdict = read_json(root/"P7_DECISION.json")
    if verdict.get("decision") != gates.get("decision") or verdict.get("detector_started") is not False:
        raise ValueError(mode+": scientific decision or detector scope is inconsistent")
    if status == "complete_development_only" and verdict["decision"] != "INCONCLUSIVE_VALIDATION_UNAVAILABLE":
        raise ValueError("Unavailable independent validation must remain explicit")
    verify_inventory(root,gates["published_hashes"])
    manifest = read_json(root/"P7_REPORT_MANIFEST.json")
    if identity(manifest["files"]) != manifest["report_identity"]:
        raise ValueError("Report manifest identity changed")
    for name,row in manifest["files"].items():
        path = child(root,name)
        if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Report archive input changed: "+name)
    latest_path = Path(str(logs)+"-smoke" if smoke else str(logs))/"LATEST.json"
    latest = read_json(latest_path)
    archive = root/"P7_SMALL_RESULTS.tar.gz"
    if (latest.get("status") != status or latest.get("decision") != verdict["decision"]
            or Path(latest["output"]).resolve() != root.resolve()
            or Path(latest["archive"]["path"]).resolve() != archive.resolve()
            or digest(archive) != latest["archive"]["sha256"]):
        raise ValueError("Final small archive or completion record differs")
    return dict(status=status,decision=verdict["decision"],gate_sha256=digest(root/"VALIDATION_GATES.json"),
                validation_status=validation,output=str(root),archive=str(archive),archive_sha256=digest(archive))


def verify_same_plan(data):
    plans = [read_json(Path(str(data)+suffix)/"P7_PROTOCOL_FREEZE.json") for suffix in ("-smoke","")]
    for key in ("all_scene_selection","weather_seeds","assignments","source_identity"):
        if key not in plans[0] or plans[0][key] != plans[1].get(key):
            raise ValueError("SMOKE/formal frozen plan or source differs: "+key)
    for scene,sha in plans[0]["pool_identities"].items():
        if plans[1]["pool_identities"].get(scene) != sha:
            raise ValueError("SMOKE/formal shared scene weather identity differs")
    return identity({k:plans[0][k] for k in ("all_scene_selection","weather_seeds","assignments")})


def execute_steps(config_path, config, state_root, code, invoke=run_child):
    status_path = state_root/"PIPELINE_STATUS.json"
    previous = read_json(status_path) if status_path.exists() else {}
    state = dict(status="running",pid=os.getpid(),attempt=previous.get("attempt",0)+1,
                 started_utc=now(),active_stage=None,steps=[])
    atomic_json(status_path,state)
    try:
        for index,mode in enumerate(STEPS,start=1):
            state.update(active_stage=mode,updated_utc=now())
            atomic_json(status_path,state)
            print(f"[{now()}] START {index}/3 P7-{mode.upper()}",flush=True)
            if delivery_identity(config_path) != code:
                raise ValueError("Source/config changed during pipeline execution")
            env = dict(os.environ,PYTHON=sys.executable,CONFIG=str(config_path),MODE=mode,RESUME="1")
            command = ["sh",str(REPO/"scorw_p7/run_all.sh")]
            result = invoke(command,env,REPO)
            if result != 0:
                raise subprocess.CalledProcessError(result,command)
            if delivery_identity(config_path) != code:
                raise ValueError("Source/config changed while a stage was running")
            gate = verify_gate(mode,config)
            if mode == "full":
                gate["shared_plan_identity"] = verify_same_plan(Path(config["output_root"]))
            state["steps"].append(dict(mode=mode,finished_utc=now(),**gate))
            atomic_json(status_path,state)
            print(f"[{now()}] PASS {index}/3 P7-{mode.upper()}: {gate['status']}",flush=True)
        final = state["steps"][-1]
        state.update(status=final["status"],decision=final["decision"],active_stage=None,finished_utc=now(),
                     output=final["output"],archive=final["archive"],archive_sha256=final["archive_sha256"])
        atomic_json(status_path,state)
        atomic_json(state_root/"PIPELINE_RESULT.json",state)
        root = Path(config["output_root"])
        atomic_text(state_root/"RESULT_SUMMARY.txt",(root/"RESULTS.md").read_text("utf-8")+
                    "\n\n"+(root/"NEXT_STAGE_DECISION.md").read_text("utf-8"))
        print(f"[{now()}] ALL REQUESTED STAGES FINISHED: {state['status']}; {state['decision']}",flush=True)
        print("Results: "+str(root)+"\nSmall archive: "+state["archive"]+"\nSHA256: "+state["archive_sha256"],flush=True)
    except BaseException as error:
        state.update(status="stopped",finished_utc=now(),failure=dict(type=type(error).__name__,
                     message=str(error),traceback=traceback.format_exc()))
        atomic_json(status_path,state)
        print(f"[{now()}] STOP {state['active_stage']}: {error}",file=sys.stderr,flush=True)
        raise


def interrupted(*_):
    raise InterruptedError("SIGTERM received")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--resume",choices=("0","1"),default="1")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = read_json(config_path)
    data,logs,state = validate_layout(config)
    code = delivery_identity(config_path)
    with lock(state):
        if args.resume == "0":
            for root in (state,data,Path(str(data)+"-smoke")):
                if root.exists() and any(p.name != ".p7.lock" for p in root.iterdir()):
                    raise ValueError("Existing run found; use RESUME=1. Existing results were preserved")
        begin(state,code,args.resume == "1")
        previous = signal.signal(signal.SIGTERM,interrupted)
        try:
            execute_steps(config_path,config,state,code)
        finally:
            signal.signal(signal.SIGTERM,previous)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
