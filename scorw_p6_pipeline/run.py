"""CPU-only sequential launcher. Does not import or change research code."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STEPS = (
    ("checks", "checks", False, "passed", None),
    ("P6-A", "audit", False, "P6A_passed", "P6-A"),
    ("P6-B", "metrics", False, "P6B_passed", "P6-B"),
    ("P6-C-SMOKE", "repeat", True, "complete_smoke_only", "P6-C"),
    ("P6-C-FULL", "repeat", False, "complete", "P6-C"),
)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".pipeline-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def delivery_identity(repo, config_path):
    snapshot_path = repo / "shared_weather_benchmark_p6/source_snapshot.json"
    pinned = read_json(snapshot_path)["sources"]
    for name, expected in pinned.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe pinned source path: " + name)
        if sha256(repo / relative) != expected:
            raise ValueError("P6 pinned source changed; sync the complete delivery: " + name)
    return dict(config_path=str(config_path), config_sha256=sha256(config_path),
                p6_snapshot_sha256=sha256(snapshot_path),
                pipeline_sources={name: sha256(HERE / name) for name in ("run.py", "run_all.sh")},
                python=str(Path(sys.executable).resolve()), python_version=platform.python_version(),
                steps=[list(step) for step in STEPS])


def validate_layout(config, repo):
    if platform.system() != "Linux":
        raise RuntimeError("Run the experiment pipeline only on the Linux server")
    if repo != Path("/home/cjm/OpenCOOD-main/cjmnet").resolve():
        raise ValueError("The existing P6 launcher requires /home/cjm/OpenCOOD-main/cjmnet")
    data = Path(config["data_root"])
    logs = Path(config["logs_root"])
    for path, parent, prefix in (
        (data, Path("/data/cjm/datasets"), "shared-weather-benchmark-p6-"),
        (logs, Path("/data/cjm/datasets/logs"), "scorw-p6-"),
    ):
        if not path.is_absolute() or path.resolve().parent != parent.resolve() or not path.name.startswith(prefix):
            raise ValueError("Unexpected P6 output namespace: " + str(path))
    pipeline = logs / "pipeline"
    if pipeline.is_symlink():
        raise ValueError("Pipeline state directory must not be a symbolic link")
    return data, logs, pipeline


@contextmanager
def pipeline_lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".pipeline.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another P6 pipeline is running; no second copy started") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_child(command, env, cwd):
    child = subprocess.Popen(command, env=env, cwd=cwd, start_new_session=True)
    try:
        return child.wait()
    except BaseException:
        # Hold the pipeline lock until its child has stopped writing artifacts.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise


def verify_gate(step, logs):
    name, mode, smoke, status, stage = step
    path = logs / "checks/CHECKS.json" if mode == "checks" else (
        logs / ("state-smoke" if smoke else "state-full") / "VALIDATION_GATES.json")
    gate = read_json(path)
    if gate.get("status") != status:
        raise ValueError(f"{name}: expected gate {status}, got {gate.get('status')}: {path}")
    if stage and gate.get("stages", {}).get(stage, {}).get("status") != "passed":
        raise ValueError(f"{name}: required stage gate did not pass: {path}")
    if mode == "checks" and (gate.get("tests", 0) <= 0 or gate.get("failures") != 0 or gate.get("errors") != 0):
        raise ValueError("checks did not record successful tests")
    return dict(path=str(path), sha256=sha256(path), status=status)


def verify_same_plan(data):
    plans = [read_json(Path(str(data) + suffix) / "P6C_FROZEN_CONFIG.json")
             for suffix in ("-smoke", "")]
    keys = ("formal_plan_identity", "config", "weather_seeds")
    if not plans[0].get("formal_plan_identity") or any(plans[0].get(k) != plans[1].get(k) for k in keys):
        raise ValueError("SMOKE and formal scene/seed plans differ; do not combine results")
    if plans[0].get("smoke") is not True or plans[1].get("smoke") is not False:
        raise ValueError("SMOKE/formal scope flags differ from the scheduled stages")
    return plans[0]["formal_plan_identity"]


def interrupted(*_):
    raise InterruptedError("SIGTERM received")


def execute_steps(repo, config_path, logs, data, state_root, identity, invoke=run_child):
    status_path = state_root / "PIPELINE_STATUS.json"
    previous = read_json(status_path) if status_path.exists() else {}
    state = dict(status="running", pid=os.getpid(), started_utc=now(),
                 attempt=previous.get("attempt", 0) + 1, steps=[], active_stage=None,
                 config_path=str(config_path), formal_data_root=str(data))
    atomic_json(status_path, state)
    try:
        for index, step in enumerate(STEPS, start=1):
            name, mode, smoke, _, _ = step
            state.update(active_stage=name, updated_utc=now())
            atomic_json(status_path, state)
            print(f"[{now()}] START {index}/{len(STEPS)} {name}", flush=True)
            if delivery_identity(repo, config_path) != identity:
                raise ValueError("Source/config identity changed during pipeline execution")
            env = dict(os.environ, CONFIG=str(config_path), PYTHON=sys.executable,
                       MODE=mode, SMOKE="1" if smoke else "0", RESUME="1")
            # Each research entry revalidates its committed hashes; never skip
            # a stage merely because this wrapper's status says it passed.
            command = ["sh", str(repo / "shared_weather_benchmark_p6/run_server.sh")]
            result = invoke(command, env, repo)
            if result != 0:
                raise subprocess.CalledProcessError(result, command)
            if delivery_identity(repo, config_path) != identity:
                raise ValueError("Source/config identity changed while a stage was running")
            gate = verify_gate(step, logs)
            record = dict(stage=name, status="passed", finished_utc=now(), gate=gate)
            if name == "P6-C-FULL":
                record["formal_plan_identity"] = verify_same_plan(data)
            state["steps"].append(record)
            atomic_json(status_path, state)
            print(f"[{now()}] PASS {index}/{len(STEPS)} {name}", flush=True)
        state.update(status="complete", active_stage=None, finished_utc=now())
        atomic_json(status_path, state)
        print(f"[{now()}] ALL STAGES COMPLETE: {data}", flush=True)
    except BaseException as error:
        state.update(status="stopped", finished_utc=now(),
                     failure=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        atomic_json(status_path, state)
        print(f"[{now()}] STOP {state['active_stage']}: {error}", file=sys.stderr, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", choices=("0", "1"), default="1")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = read_json(config_path)
    data, logs, state_root = validate_layout(config, REPO)
    identity = delivery_identity(REPO, config_path)
    contract_path = state_root / "PIPELINE_CONTRACT.json"
    with pipeline_lock(state_root):
        if args.resume == "0":
            roots = [state_root, logs/"state-full", logs/"state-smoke", logs/"audit", logs/"metrics", data, Path(str(data)+"-smoke")]
            if any(p.exists() and any(c.name != ".pipeline.lock" for c in p.iterdir()) for p in roots):
                raise ValueError("Existing P6 results/state found; use RESUME=1. No data deleted")
        if contract_path.exists():
            if read_json(contract_path) != identity:
                raise ValueError("Pipeline identity changed; do not overwrite the contract or existing results")
        else:
            atomic_json(contract_path, identity)
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            execute_steps(REPO, config_path, logs, data, state_root, identity)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
