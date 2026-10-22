"""Synthetic metadata fixtures only. The delivered server checks run this suite."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scorw_p7.core import identity, read_json, transaction
from scorw_p7_v2.select_scenes import (frame_connectivity, qualify_selection, require_connectivity,
    verify_metadata_inputs, verify_audited_connectivity, server_selection)

HERE = Path(__file__).resolve().parents[1]


def config():
    return read_json(HERE/"config/protocol.json")


def fixture(original=("002", "003"), distances=(20, 20, 90, 90, 30)):
    stamps = [f"{i:03}" for i in range(len(distances))]
    cavs = ["ego", "neighbor"]
    scene = "unused-train-scene"
    plan = dict(selected=dict(generator_development=[], independent_validation=[scene]),
        validation_status="available_8_new_scenes", available_validation_candidates=10,
        excluded_future=["reserved-scene"], eligible_order=[scene, "unselected-alternative"],
        inventories={scene:dict(cavs=cavs, ego="ego", selected=list(original),
            timestamp_count=len(stamps), timestamps_sha256=identity(stamps))},
        frames=[dict(scene=scene, timestamp=s, cavs=cavs, ego="ego", partition="independent_validation") for s in original],
        metadata_inputs=[])
    calls = []
    def read_pose(s, stamp, cav):
        if s != scene or cav not in cavs:
            raise AssertionError("Unexpected scene/CAV identity read")
        calls.append((s, stamp, cav))
        return [0 if cav == "ego" else distances[int(stamp)], 0, 0, 0, 0, 0]
    return plan, stamps, read_pose, calls


class ConnectivityTests(unittest.TestCase):
    def test_inclusive_xy_radius_matches_frozen_audit(self):
        poses = dict(ego=[0, 0, 0, 0, 0, 0], b=[70, 0, 100, 0, 0, 0], c=[70.000001, 0, 0, 0, 0, 0])
        out = frame_connectivity("s", "001", list(poses), "ego", lambda s,t,c: poses[c], 70.)
        self.assertEqual(out["active_cavs"], ["ego", "b"])
        self.assertEqual(out["xy_distance_from_ego_m"]["b"], 70.)

    def test_diagonal_xy_distance_not_axiswise_threshold(self):
        poses = dict(ego=[0]*6, b=[50, 50, 0, 0, 0, 0])
        out = frame_connectivity("s", "001", list(poses), "ego", lambda s,t,c: poses[c], 70.)
        self.assertFalse(out["admissible"])

    def test_invalid_identity_or_range_stops(self):
        for cavs, ego, radius in ((["a", "a"], "a", 70), (["a", "b"], "c", 70), (["a", "b"], "a", 100)):
            with self.subTest(cavs=cavs, ego=ego, radius=radius), self.assertRaises(ValueError):
                frame_connectivity("s", "1", cavs, ego, lambda *a: [0]*6, radius)

    def test_invalid_pose_stops(self):
        for pose in ([0]*5, [float("nan")]*6, [float("inf")]*6):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                frame_connectivity("s", "1", ["a", "b"], "a", lambda *a: pose, 70.)

    def test_cav_identity_is_not_reassigned_by_input_order(self):
        poses = dict(ego=[0]*6, a=[20,0,0,0,0,0], b=[100,0,0,0,0,0])
        a = frame_connectivity("s", "1", ["ego","a","b"], "ego", lambda s,t,c: poses[c], 70.)
        b = frame_connectivity("s", "1", ["b","a","ego"], "ego", lambda s,t,c: poses[c], 70.)
        self.assertEqual(a["xy_distance_from_ego_m"], b["xy_distance_from_ego_m"])
        self.assertEqual(set(a["active_cavs"]), set(b["active_cavs"]))
        self.assertEqual(a["ego"], b["ego"])


class FrameSelectionTests(unittest.TestCase):
    def qualify(self, plan, stamps, reader):
        return qualify_selection(plan, config(), lambda scene: stamps, reader)

    def test_original_admissible_block_kept(self):
        plan, stamps, reader, calls = fixture(distances=(90,90,30,40,90))
        out = self.qualify(plan, stamps, reader)
        self.assertEqual(out["frames"], plan["frames"])
        self.assertFalse(out["connectivity_preflight"]["scene_reports"]["unused-train-scene"]["changed_from_v1"])
        self.assertEqual(len(calls), 4)

    def test_first_admissible_block_in_seed_anchored_cyclic_order(self):
        plan, stamps, reader, calls = fixture()
        out = self.qualify(plan, stamps, reader)
        info = out["connectivity_preflight"]["scene_reports"]["unused-train-scene"]
        self.assertEqual(info["selected_frames"], ["000", "001"])
        self.assertEqual([v["start_index"] for v in info["tested_blocks"]], [2,3,0])
        self.assertTrue(info["changed_from_v1"])
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(out["selected"], plan["selected"])

    def test_both_frames_must_be_qualified(self):
        plan, stamps, reader, calls = fixture(distances=(20,20,20,90,90))
        out = self.qualify(plan, stamps, reader)
        self.assertEqual(out["inventories"]["unused-train-scene"]["selected"], ["000", "001"])

    def test_no_disconnected_frame_substitution(self):
        plan, stamps, reader, calls = fixture(distances=(20,90,20,90,20))
        out = self.qualify(plan, stamps, reader)
        self.assertEqual(out["selected"]["independent_validation"], [])
        self.assertEqual(out["frames"], [])
        self.assertEqual(out["validation_status"], "INCONCLUSIVE_VALIDATION_UNAVAILABLE")
        self.assertIn("No admissible block", out["validation_unavailable_reason"])
        self.assertEqual(out["original_v1_scene_selection"], plan["selected"])
        self.assertEqual(len(calls), len(stamps)*2)

    def test_no_new_scene_or_future_access_when_block_unavailable(self):
        plan, stamps, reader, calls = fixture(distances=(90,90,90,90,90))
        out = self.qualify(plan, stamps, reader)
        self.assertEqual(out["eligible_order"], plan["eligible_order"])
        self.assertEqual(out["excluded_future"], plan["excluded_future"])
        self.assertEqual({s for s,t,c in calls}, {"unused-train-scene"})
        self.assertFalse(out["connectivity_preflight"]["scene_replacement"])

    def test_development_frames_frozen_and_fail_with_diagnostics(self):
        plan, stamps, reader, calls = fixture()
        scene = "unused-train-scene"
        plan["selected"] = dict(generator_development=[scene], independent_validation=[])
        for f in plan["frames"]:
            f["partition"] = "generator_development"
        out = self.qualify(plan, stamps, reader)
        with self.assertRaisesRegex(ValueError, "unused-train-scene.*002"):
            require_connectivity(out)
        self.assertEqual({t for s,t,c in calls}, {"002", "003"})
        self.assertEqual(len(out["connectivity_preflight"]["scene_reports"][scene]["tested_blocks"]), 1)

    def test_development_valid_frames_unchanged(self):
        plan, stamps, reader, calls = fixture(distances=(90,90,30,40,90))
        plan["selected"] = dict(generator_development=["unused-train-scene"], independent_validation=[])
        for f in plan["frames"]:
            f["partition"] = "generator_development"
        out = self.qualify(plan, stamps, reader)
        require_connectivity(out)
        self.assertEqual(out["frames"], plan["frames"])

    def test_repeated_selection_is_identical_and_input_is_unmodified(self):
        plan, stamps, reader, calls = fixture()
        frozen = deepcopy(plan)
        a = self.qualify(plan, stamps, reader)
        b = self.qualify(plan, stamps, reader)
        self.assertEqual(a, b)
        self.assertEqual(plan, frozen)

    def test_timestamp_or_cav_identity_change_stops(self):
        plan, stamps, reader, calls = fixture()
        with self.assertRaisesRegex(ValueError, "Timestamp inventory changed"):
            self.qualify(plan, stamps[:-1], reader)
        plan["frames"][0]["ego"] = "neighbor"
        with self.assertRaisesRegex(ValueError, "frame/CAV/ego identities"):
            self.qualify(plan, stamps, reader)

    def test_one_failed_scene_disables_the_entire_eight_scene_cohort(self):
        plan, stamps, reader, calls = fixture()
        original = deepcopy(plan)
        plan.update(inventories={}, frames=[])
        scenes = [f"candidate{i}" for i in range(8)]
        plan["selected"]["independent_validation"] = scenes
        for s in scenes:
            plan["inventories"][s] = deepcopy(original["inventories"]["unused-train-scene"])
            plan["frames"].extend(dict(f, scene=s) for f in original["frames"])
        def pose(scene, stamp, cav):
            return [0 if cav == "ego" else 90 if scene == "candidate7" else 20, 0, 0, 0, 0, 0]
        out = self.qualify(plan, stamps, pose)
        self.assertEqual(out["selected"]["independent_validation"], [])
        self.assertEqual(out["frames"], [])
        self.assertEqual(out["connectivity_preflight"]["attempted_validation_scenes"], scenes)
        self.assertEqual(out["connectivity_preflight"]["unavailable_validation_scenes"], ["candidate7"])
        self.assertEqual(out["available_validation_candidates"], 10)
        self.assertNotIn("Fewer than eight", out["validation_unavailable_reason"])


class DeliveryTests(unittest.TestCase):
    def test_preflight_active_cavs_are_checked_against_frozen_audit(self):
        plan, stamps, reader, calls = fixture(distances=(90,90,30,40,90))
        plan = qualify_selection(plan, config(), lambda s: stamps, reader)
        audited = [dict(f, active_cavs=f["cavs"]) for f in plan["frames"]]
        verify_audited_connectivity(plan, audited, plan["frames"])
        with self.assertRaisesRegex(ValueError, "frame inventory mismatch"):
            verify_audited_connectivity(plan, audited[:1], plan["frames"])
        audited[0]["active_cavs"] = ["ego"]
        with self.assertRaisesRegex(ValueError, "CAV identity mismatch"):
            verify_audited_connectivity(plan, audited, plan["frames"])

    def test_all_consulted_metadata_are_pinned_and_pose_reads_cached(self):
        plan, stamps, reader, calls = fixture()
        base = HERE/"local_validation"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root = Path(temp)
            scene = "unused-train-scene"
            for stamp in stamps:
                for cav in plan["inventories"][scene]["cavs"]:
                    path = root/scene/cav/(stamp+".yaml")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((stamp+":"+cav).encode())
            def metadata(path, train):
                return dict(lidar_pose=reader(path.parent.parent.name, path.stem, path.parent.name))
            with patch("scorw_p7.select_scenes.server_selection", return_value=plan), \
                 patch("scorw_p7.core.server_train", return_value=root), \
                 patch("shared_weather_benchmark.dataset_io.approved_path", side_effect=lambda p,r: p), \
                 patch("shared_weather_benchmark.dataset_io.timestamps", return_value=stamps), \
                 patch("shared_weather_benchmark.dataset_io.read_metadata", side_effect=metadata):
                out = server_selection(dict(identity="synthetic-history"), config())
            self.assertEqual(len(calls), len(set(calls)))
            self.assertEqual(len(out["metadata_inputs"]), len(calls))
            verify_metadata_inputs(out)
            rejected = root/scene/"neighbor/002.yaml"
            self.assertIn(str(rejected), {r["path"] for r in out["metadata_inputs"]})
            rejected.write_bytes(b"changed rejected candidate metadata")
            with self.assertRaisesRegex(ValueError, "Consulted selection metadata changed"):
                verify_metadata_inputs(out)

    def test_experimental_values_and_seeds_are_identical_to_v1(self):
        a = read_json(HERE.parent/"scorw_p7/config/protocol.json")
        b = config()
        skip = {"revision", "output_root", "logs_root", "frame_selection"}
        self.assertEqual({k:v for k,v in a.items() if k not in skip}, {k:v for k,v in b.items() if k not in skip})
        self.assertNotEqual(identity(a), identity(b))
        self.assertEqual(b["frame_selection"]["communication_range_m"], 70.)

    def test_old_contract_cannot_be_reused_after_protocol_revision(self):
        base = HERE/"local_validation"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root = Path(temp)/"run"
            transaction(root, dict(revision="scorw-p7-v1"), lambda code: None)
            with self.assertRaisesRegex(ValueError, "contract changed"):
                transaction(root, dict(revision="scorw-p7-v2"), lambda code: None)

    def test_delivered_config_is_accepted_and_old_config_rejected(self):
        from scorw_p7_v2.run import load_config
        self.assertEqual(load_config(HERE/"config/protocol.json"), config())
        with self.assertRaises(ValueError):
            load_config(HERE.parent/"scorw_p7/config/protocol.json")

    def test_unavailable_decision_keeps_its_actual_reason(self):
        from scorw_p7_v2.decision import decide
        reason = "Eight scene identities frozen; one has no admissible consecutive pair"
        selection = dict(validation_status="INCONCLUSIVE_VALIDATION_UNAVAILABLE", validation_unavailable_reason=reason)
        out = decide(dict(partitions=dict(independent_validation={})), [], selection, config())
        self.assertEqual(out["decision"], "INCONCLUSIVE_VALIDATION_UNAVAILABLE")
        self.assertEqual(out["reason"], reason)
        self.assertFalse(out["detector_started"])

    def test_server_preflight_dispatch_is_before_frozen_coordinate_audit(self):
        # Exercise the actual v2 controller with mocked data I/O; no Clean PCD is opened.
        from scorw_p7_v2 import run
        from argparse import Namespace
        cfg = config()
        plan, stamps, reader, calls = fixture(distances=(90,90,30,40,90))
        plan = qualify_selection(plan, cfg, lambda s: stamps, reader)
        plan["selected"]["generator_development"] = []
        plan["history_identity"] = "synthetic"
        base = HERE/"local_validation"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            output = Path(temp)/"result"
            args = Namespace(mode="full", resume="0", config=HERE/"config/protocol.json")
            with patch("scorw_p7_v2.select_scenes.server_selection", return_value=plan) as select, \
                 patch.object(run, "server_train"), patch.object(run, "historical_metadata", return_value=dict(identity="h")), \
                 patch.object(run, "input_fingerprint", return_value=[]), patch.object(run, "runtime_identity", return_value={}), \
                 patch.object(run, "output_paths", return_value=(output, Path(temp)/"logs")), \
                 patch("shared_weather_benchmark.audit.real_coordinate_audit", side_effect=ValueError("synthetic stop before field creation")) as audit:
                with self.assertRaisesRegex(ValueError, "synthetic stop before field creation"):
                    run.run_experiment(args, cfg, dict(source_identity="synthetic-source"))
            select.assert_called_once()
            audit.assert_called_once()
            self.assertEqual(read_json(output/"VALIDATION_GATES.json")["status"], "stopped")
            saved = read_json(output/"stages/P7-0/P7_SCENE_SELECTION.json")
            self.assertEqual(saved["connectivity_preflight"], plan["connectivity_preflight"])
            self.assertFalse((output/"stages/P7-0/P7_FIELD_POOL_MANIFEST.json").exists())
