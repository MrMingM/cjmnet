"""CPU-only tests for Stage-3 pure bookkeeping and interpretation logic."""
import unittest

from gspr_evidence.stage3_common import failure_stage, scene_failure_table, choose_frame_oracles


class Stage3CommonTest(unittest.TestCase):
    def test_failure_stage(self):
        self.assertEqual(
            failure_stage({
                "decoded": True, "score": False, "geometry": False,
                "nms_top1000": False, "nms": False, "range": False,
                "final_matched": False,
            }),
            "score_threshold",
        )
        self.assertEqual(
            failure_stage({
                "decoded": True, "score": True, "geometry": True,
                "nms_top1000": True, "nms": True, "range": True,
                "final_matched": False,
            }),
            "final_matching_competition",
        )
        self.assertEqual(
            failure_stage({
                "decoded": False, "score": False, "geometry": False,
                "nms_top1000": False, "nms": False, "range": False,
                "final_matched": False,
            }),
            "no_iou70_decoded",
        )

    def test_scene_rate_uses_source_valid_denominator(self):
        stage1 = {
            (1, 0): {"scene": 0, "distance": 10.0},
            (2, 0): {"scene": 0, "distance": 30.0},
            (3, 0): {"scene": 1, "distance": 30.0},
        }
        stage2 = [
            {"sample_index": 1, "target_index": 0, "any_peer_alone_detected": True,
             "full": {"matched": False}},
            {"sample_index": 2, "target_index": 0, "any_peer_alone_detected": True,
             "full": {"matched": True}},
            {"sample_index": 3, "target_index": 0, "any_peer_alone_detected": False,
             "full": {"matched": False}},
        ]
        result = scene_failure_table(stage1, stage2)
        self.assertEqual(result["overall"]["source_valid_target_occurrences"], 2)
        self.assertEqual(result["overall"]["failure_target_occurrences"], 1)
        self.assertAlmostEqual(result["overall"]["failure_rate_given_source_valid"], 0.5)
        self.assertEqual(result["by_scene"]["0"]["source_valid_target_occurrences"], 2)
        self.assertEqual(result["by_scene"]["0"]["failure_target_occurrences"], 1)
        self.assertNotIn("1", result["by_scene"])

    def test_target_oracle_not_merged_into_fake_frame(self):
        results = [
            {"subset": [1], "matched": {0, 2}, "fp": 1},
            {"subset": [2], "matched": {1, 2}, "fp": 1},
            {"subset": [1, 2], "matched": {2}, "fp": 0},
        ]
        oracle = choose_frame_oracles(
            results,
            candidate_targets={0, 1},
            full_matched={2},
            full_fp=0,
        )
        self.assertEqual(oracle["target_wise_recoverable_count"], 2)
        self.assertEqual(oracle["best_frame_subset"]["recovered_candidate_count"], 1)
        self.assertEqual(oracle["safe_best_frame_subset"]["recovered_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
