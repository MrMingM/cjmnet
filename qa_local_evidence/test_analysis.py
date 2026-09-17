import unittest

from qa_local_evidence.analysis import classify, summarize_rows


def row(failure, score_pass, q25=True, scene=0, sample=1):
    decoded = 0 if failure == "no_iou70_after_decode" else 1
    return {
        "sample_index": sample,
        "scene": scene,
        "proxy_strong": {"q20": q25, "q25": q25, "q30": q25, "q40": q25},
        "weather_path": {
            "matched": False,
            "failure_stage": failure,
            "stages": {"decoded": {"qualifying_count": decoded}},
        },
        "clean_matched_anchor_in_weather": {
            "score_passes_threshold": score_pass,
            "iou70": failure != "no_iou70_after_decode",
            "score_delta": -0.2,
            "iou_delta": -0.1,
            "logit_delta": -1.0,
        },
    }


class AnalysisTest(unittest.TestCase):
    def test_late_loss(self):
        self.assertEqual(classify(row("score_filtered", False)), "late_detection_path_loss")

    def test_regression_support(self):
        self.assertEqual(
            classify(row("no_iou70_after_decode", True)),
            "regression_or_localization_loss_with_cls_survival",
        )

    def test_unresolved(self):
        self.assertEqual(
            classify(row("no_iou70_after_decode", False)),
            "joint_head_or_upstream_unresolved",
        )

    def test_summary(self):
        rows = [
            row("score_filtered", False, sample=1),
            row("no_iou70_after_decode", True, sample=1),
            row("no_iou70_after_decode", False, sample=2),
        ]
        s = summarize_rows(rows)
        self.assertEqual(s["occurrences"], 3)
        self.assertEqual(s["unique_frames"], 2)
        self.assertEqual(s["q25"]["strong_occurrences"], 3)
        self.assertAlmostEqual(s["q25"]["a5b_support_fraction"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
