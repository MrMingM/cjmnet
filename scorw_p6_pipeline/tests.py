"""Launcher engineering tests only; no research imports, data or experiments."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scorw_p6_pipeline import run


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".launcher-test-", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root / "logs"
        self.state = self.logs / "pipeline"
        self.state.mkdir(parents=True)
        self.data = self.root / "data"
        self.config = self.root / "config.json"
        run.atomic_json(self.config, {})
        self.identity = {"delivery": "synthetic"}
        self.calls = []

    def child(self, command, env, cwd):
        self.calls.append((env["MODE"], env["SMOKE"], env["RESUME"]))
        mode, smoke = env["MODE"], env["SMOKE"] == "1"
        if mode == "checks":
            path = self.logs / "checks/CHECKS.json"
            gate = dict(status="passed", tests=1, failures=0, errors=0)
        else:
            path = self.logs / ("state-smoke" if smoke else "state-full") / "VALIDATION_GATES.json"
            stage, status = {"audit": ("P6-A", "P6A_passed"), "metrics": ("P6-B", "P6B_passed"),
                             "repeat": ("P6-C", "complete_smoke_only" if smoke else "complete")}[mode]
            gate = dict(status=status, stages={stage: dict(status="passed")})
        path.parent.mkdir(parents=True, exist_ok=True)
        run.atomic_json(path, gate)
        if mode == "repeat":
            data = Path(str(self.data) + "-smoke") if smoke else self.data
            data.mkdir(parents=True, exist_ok=True)
            run.atomic_json(data / "P6C_FROZEN_CONFIG.json", dict(
                formal_plan_identity="same-synthetic-plan", config={}, weather_seeds={"0": 1}, smoke=smoke))
            # Scientific negative outcomes must not stop a computationally valid run.
            run.atomic_json(data / "DECISION.json", dict(decision="STOP_OR_REDESIGN", matched=False))
        return 0

    def execute(self, invoke=None, identity_mock=None):
        with patch.object(run, "delivery_identity", **(identity_mock or {"return_value": self.identity})), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            run.execute_steps(self.root, self.config, self.logs, self.data,
                              self.state, self.identity, invoke or self.child)

    def test_exact_order_scopes_and_resume(self):
        self.execute()
        self.assertEqual(self.calls, [("checks", "0", "1"), ("audit", "0", "1"),
                                     ("metrics", "0", "1"), ("repeat", "1", "1"), ("repeat", "0", "1")])
        state = run.read_json(self.state / "PIPELINE_STATUS.json")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(state["steps"]), 5)

    def test_every_nonzero_stage_stops_successors(self):
        for failure_index in range(5):
            with self.subTest(stage=failure_index):
                self.calls.clear()
                def invoke(command, env, cwd):
                    code = self.child(command, env, cwd)
                    return 9 if len(self.calls) - 1 == failure_index else code
                with self.assertRaises(subprocess.CalledProcessError):
                    self.execute(invoke)
                self.assertEqual(len(self.calls), failure_index + 1)
                state = run.read_json(self.state / "PIPELINE_STATUS.json")
                self.assertEqual(state["status"], "stopped")
                self.assertEqual(state["active_stage"], run.STEPS[failure_index][0])

    def test_zero_exit_with_failed_gate_stops(self):
        def invoke(command, env, cwd):
            code = self.child(command, env, cwd)
            if env["MODE"] == "audit":
                run.atomic_json(self.logs / "state-full/VALIDATION_GATES.json", dict(status="stopped"))
            return code
        with self.assertRaises(ValueError):
            self.execute(invoke)
        self.assertEqual(len(self.calls), 2)

    def test_missing_gate_stops(self):
        with self.assertRaises(FileNotFoundError):
            self.execute(lambda *_: 0)
        self.assertEqual(run.read_json(self.state / "PIPELINE_STATUS.json")["active_stage"], "checks")

    def test_incorrect_smoke_scope_never_launches_formal(self):
        def invoke(command, env, cwd):
            code = self.child(command, env, cwd)
            if env["SMOKE"] == "1":
                run.atomic_json(self.logs / "state-smoke/VALIDATION_GATES.json", dict(
                    status="complete", stages={"P6-C": dict(status="passed")}))
            return code
        with self.assertRaises(ValueError):
            self.execute(invoke)
        self.assertEqual(len(self.calls), 4)

    def test_failed_checks_never_launch_audit(self):
        def invoke(command, env, cwd):
            code = self.child(command, env, cwd)
            run.atomic_json(self.logs / "checks/CHECKS.json", dict(status="passed", tests=1, failures=1, errors=0))
            return code
        with self.assertRaises(ValueError):
            self.execute(invoke)
        self.assertEqual(len(self.calls), 1)

    def test_changed_delivery_stops_before_next_child(self):
        with self.assertRaises(ValueError):
            self.execute(identity_mock={"side_effect": [self.identity, self.identity, {"changed": True}]})
        self.assertEqual(len(self.calls), 1)

    def test_resuming_revalidates_every_entry(self):
        self.execute()
        self.execute()
        self.assertEqual(len(self.calls), 10)
        self.assertEqual(run.read_json(self.state / "PIPELINE_STATUS.json")["attempt"], 2)

    def test_different_formal_plan_cannot_report_complete(self):
        def invoke(command, env, cwd):
            code = self.child(command, env, cwd)
            if env["MODE"] == "repeat" and env["SMOKE"] == "0":
                plan = self.data / "P6C_FROZEN_CONFIG.json"
                value = run.read_json(plan)
                value["formal_plan_identity"] = "different"
                run.atomic_json(plan, value)
            return code
        with self.assertRaises(ValueError):
            self.execute(invoke)
        self.assertEqual(run.read_json(self.state / "PIPELINE_STATUS.json")["status"], "stopped")

    def test_server_guard_rejects_windows(self):
        with patch.object(run.platform, "system", return_value="Windows"):
            with self.assertRaises(RuntimeError):
                run.validate_layout({}, self.root)

    def test_delivery_identity_survives_json_roundtrip(self):
        # Tiny fake source snapshot tests wrapper hashing; no research file reads.
        package = self.root / "shared_weather_benchmark_p6"
        package.mkdir()
        source = self.root / "synthetic.txt"
        source.write_text("synthetic", encoding="utf-8")
        run.atomic_json(package / "source_snapshot.json", {"sources": {source.name: run.sha256(source)}})
        identity = run.delivery_identity(self.root, self.config)
        run.atomic_json(self.root / "identity.json", identity)
        self.assertEqual(identity, run.read_json(self.root / "identity.json"))
        source.write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            run.delivery_identity(self.root, self.config)


if __name__ == "__main__":
    unittest.main()
