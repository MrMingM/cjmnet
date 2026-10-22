"""Engineering-only orchestration tests. Real P7 entry is never executed."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scorw_p7_v2 import pipeline as run


class PipelineTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1]/"local_validation"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix=".launcher-test-",dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root/"logs"
        self.data = self.root/"data"
        self.state = self.logs/"pipeline"
        self.state.mkdir(parents=True)
        self.config_path = self.root/"config.json"
        self.config = dict(output_root=str(self.data),logs_root=str(self.logs))
        run.atomic_json(self.config_path,self.config)
        self.code = dict(source="synthetic")
        self.calls = []
        self.formal_status = "complete"
        self.formal_decision = "INCONCLUSIVE"

    def child(self,command,env,cwd):
        mode = env["MODE"]
        self.calls.append((mode,env["RESUME"]))
        if mode == "checks":
            root = self.logs/"checks"
            root.mkdir(parents=True,exist_ok=True)
            run.atomic_json(root/"CHECKS.json",dict(status="passed",tests=1,failures=0,errors=0))
            run.atomic_json(root/"VALIDATION_GATES.json",dict(status="checks_passed",stages={
                s:dict(status="passed") for s in ("P7-0","P7-1")}))
            return 0
        smoke = mode == "smoke"
        root = Path(str(self.data)+"-smoke") if smoke else self.data
        root.mkdir(parents=True,exist_ok=True)
        status = "complete_smoke_only" if smoke else self.formal_status
        decision = "INCONCLUSIVE" if smoke else self.formal_decision
        validation = "not_requested_smoke" if smoke else "passed" if status == "complete" else "not_executable"
        run.atomic_json(root/"P7_DECISION.json",dict(decision=decision,detector_started=False))
        run.atomic_text(root/"RESULTS.md","Synthetic test result only\n")
        run.atomic_text(root/"NEXT_STAGE_DECISION.md",decision+"\n")
        run.atomic_json(root/"P7_PROTOCOL_FREEZE.json",dict(all_scene_selection=dict(selected=["synthetic"]),
            weather_seeds=list(range(20)),assignments=dict(ego=list(range(20))),source_identity="fixed-synthetic-source",
            pool_identities=dict(synthetic="same-field-pool"),sampling_budget=1 if smoke else 8))
        names = ("P7_DECISION.json","P7_PROTOCOL_FREEZE.json","RESULTS.md","NEXT_STAGE_DECISION.md")
        gates = dict(status=status,decision=decision,stages={s:dict(status="passed") for s in ("P7-0","P7-1","P7-2","P7-4")},
                     published_hashes={n:run.digest(root/n) for n in names})
        gates["stages"]["P7-3"] = dict(status=validation)
        run.atomic_json(root/"VALIDATION_GATES.json",gates)
        rows = {n:dict(sha256=run.digest(root/n),bytes=(root/n).stat().st_size) for n in (*names,"VALIDATION_GATES.json")}
        run.atomic_json(root/"P7_REPORT_MANIFEST.json",dict(files=rows,report_identity=run.identity(rows)))
        archive = root/"P7_SMALL_RESULTS.tar.gz"
        archive.write_bytes(b"synthetic archive fixture, not real P7 output")
        logroot = Path(str(self.logs)+"-smoke") if smoke else self.logs
        logroot.mkdir(parents=True,exist_ok=True)
        run.atomic_json(logroot/"LATEST.json",dict(status=status,decision=decision,output=str(root),
            archive=dict(path=str(archive),sha256=run.digest(archive))))
        return 0

    def execute(self,invoke=None,delivery=None):
        with patch.object(run,"delivery_identity",**(delivery or dict(return_value=self.code))), \
                redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            run.execute_steps(self.config_path,self.config,self.state,self.code,invoke or self.child)

    def test_checks_smoke_full_exact_order(self):
        self.execute()
        self.assertEqual(self.calls,[("checks","1"),("smoke","1"),("full","1")])
        status = run.read_json(self.state/"PIPELINE_STATUS.json")
        self.assertEqual(status["status"],"complete")
        self.assertTrue((self.state/"RESULT_SUMMARY.txt").is_file())

    def test_each_exit_failure_stops_successors(self):
        for failure in range(3):
            with self.subTest(failure=failure):
                self.calls.clear()
                def invoke(command,env,cwd):
                    code = self.child(command,env,cwd)
                    return 7 if len(self.calls)-1 == failure else code
                with self.assertRaises(subprocess.CalledProcessError):
                    self.execute(invoke)
                self.assertEqual(len(self.calls),failure+1)
                self.assertEqual(run.read_json(self.state/"PIPELINE_STATUS.json")["status"],"stopped")

    def test_zero_exit_without_gate_stops(self):
        with self.assertRaises(FileNotFoundError):
            self.execute(lambda *_:0)
        self.assertEqual(run.read_json(self.state/"PIPELINE_STATUS.json")["active_stage"],"checks")

    def test_checks_failure_blocks_smoke(self):
        def invoke(command,env,cwd):
            self.child(command,env,cwd)
            run.atomic_json(self.logs/"checks/CHECKS.json",dict(status="passed",tests=1,failures=1,errors=0))
            return 0
        with self.assertRaises(ValueError):
            self.execute(invoke)
        self.assertEqual(len(self.calls),1)

    def test_invalid_smoke_scope_blocks_formal(self):
        def invoke(command,env,cwd):
            self.child(command,env,cwd)
            if env["MODE"] == "smoke":
                path = Path(str(self.data)+"-smoke")/"VALIDATION_GATES.json"
                gates = run.read_json(path)
                gates["status"] = "complete"
                run.atomic_json(path,gates)
            return 0
        with self.assertRaises(ValueError):
            self.execute(invoke)
        self.assertEqual(len(self.calls),2)

    def test_report_archive_corruption_blocks_successors(self):
        def invoke(command,env,cwd):
            self.child(command,env,cwd)
            if env["MODE"] == "smoke":
                (Path(str(self.data)+"-smoke")/"P7_SMALL_RESULTS.tar.gz").write_bytes(b"changed")
            return 0
        with self.assertRaises(ValueError):
            self.execute(invoke)
        self.assertEqual(len(self.calls),2)

    def test_source_change_stops_before_next_child(self):
        with self.assertRaises(ValueError):
            self.execute(delivery=dict(side_effect=[self.code,self.code,dict(changed=True)]))
        self.assertEqual(len(self.calls),1)

    def test_resume_reenters_every_research_entry(self):
        self.execute()
        self.execute()
        self.assertEqual(len(self.calls),6)
        self.assertEqual(run.read_json(self.state/"PIPELINE_STATUS.json")["attempt"],2)

    def test_unavailable_validation_stays_explicit(self):
        self.formal_status = "complete_development_only"
        self.formal_decision = "INCONCLUSIVE_VALIDATION_UNAVAILABLE"
        self.execute()
        status = run.read_json(self.state/"PIPELINE_STATUS.json")
        self.assertEqual(status["status"],"complete_development_only")
        self.assertEqual(status["decision"],"INCONCLUSIVE_VALIDATION_UNAVAILABLE")
        self.assertEqual(status["steps"][-1]["validation_status"],"not_executable")

    def test_scientific_negative_result_is_preserved(self):
        self.formal_decision = "STOP_OR_REDESIGN"
        self.execute()
        self.assertEqual(run.read_json(self.state/"PIPELINE_RESULT.json")["decision"],"STOP_OR_REDESIGN")
        self.assertEqual(len(self.calls),3)

    def test_inconsistent_no_validation_decision_is_rejected(self):
        self.formal_status = "complete_development_only"
        self.formal_decision = "CONTINUE_TO_DETECTOR"
        with self.assertRaises(ValueError):
            self.execute()

    def test_changed_plan_rejected(self):
        self.execute()
        path = self.data/"P7_PROTOCOL_FREEZE.json"
        frozen = run.read_json(path)
        frozen["weather_seeds"] = [999]*20
        run.atomic_json(path,frozen)
        with self.assertRaises(ValueError):
            run.verify_same_plan(self.data)

    def test_server_guard_rejects_windows(self):
        with patch.object(run.platform,"system",return_value="Windows"):
            with self.assertRaises(RuntimeError):
                run.validate_layout(self.config)

    def test_duplicate_pipeline_lock_is_rejected(self):
        with run.lock(self.root/"locked"):
            with self.assertRaises(OSError):
                with run.lock(self.root/"locked"):
                    self.fail("Concurrent pipeline acquired the same lock")


if __name__ == "__main__":
    unittest.main()
