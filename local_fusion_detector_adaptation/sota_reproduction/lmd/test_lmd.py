import unittest

import numpy as np

from local_fusion_detector_adaptation.sota_reproduction.lmd.features import (
    build_frame_features, feature_names,
)


class LmdFeatureTest(unittest.TestCase):
    def _row(self, cid, x, score, quality):
        # hwl: decoded dims = h,w,l; rectangle length=4,width=2.
        return {
            "candidate_id": cid, "score": score, "max_gt_iou": quality,
            "fused_decoded_box": [x, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
            "fused_bev_corners": [
                [x-2, -1], [x-2, 1], [x+2, 1], [x+2, -1]],
            "within_range": True,
        }

    def test_schema_and_labels(self):
        rows = [self._row(0, 0.0, 0.8, 0.9),
                self._row(1, 0.5, 0.3, 0.4)]
        x, y, ids = build_frame_features(rows, "hwl", 0.2)
        self.assertEqual(x.shape, (2, len(feature_names())))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(y, [0.9, 0.4])
        np.testing.assert_array_equal(ids, [0, 1])

    def test_far_box_is_not_neighbor(self):
        rows = [self._row(0, 0.0, 0.8, 0.9),
                self._row(1, 100.0, 0.3, 0.0)]
        x, _, _ = build_frame_features(rows, "hwl", 0.2)
        neighbor_index = feature_names().index("neighbor_count")
        np.testing.assert_allclose(x[:, neighbor_index], [1, 1])

    def test_dimension_mapping(self):
        row = self._row(0, 0.0, 0.5, 0.5)
        x, _, _ = build_frame_features([row], "hwl", 0.2)
        names = feature_names()
        self.assertAlmostEqual(float(x[0, names.index("own_height")]), 1.5)
        self.assertAlmostEqual(float(x[0, names.index("own_width")]), 2.0)
        self.assertAlmostEqual(float(x[0, names.index("own_length")]), 4.0)


class LmdCachedEvaluationTest(LmdFeatureTest):
    def test_cached_ap_and_matching_reproduce_opencood(self):
        from local_fusion_detector_adaptation.candidate_hypothesis_replay import _stats
        from local_fusion_detector_adaptation.sota_reproduction.lmd.evaluate import (
            _nms, _prepare_rows, _assign,
        )
        from local_fusion_detector_adaptation.sota_reproduction.lmd.fast_geometry import (
            candidate_gt_iou, evaluate_frame_cached, assign_cached,
            assert_same_as_opencood,
        )

        raw = [self._row(0, 0.0, 0.9, 1.0),
               self._row(1, 0.25, 0.6, 0.85),
               self._row(2, 50.0, 0.35, 0.0)]
        rows = _prepare_rows(raw)
        gt = np.asarray([raw[0]["fused_bev_corners"]], dtype=np.float32)
        x, y, ids, matrix = build_frame_features(raw, "hwl", 0.2,
                                                  return_overlap=True)
        self.assertEqual(matrix.shape, (3, 3))
        scores = np.asarray([0.9, 0.6, 0.35], dtype=np.float32)
        slow = _nms(rows, scores, 0.01)
        fast = _nms(rows, scores, 0.01, overlap_matrix=matrix)
        self.assertEqual(slow, fast)
        gt_ious = candidate_gt_iou(rows, gt, "synthetic", 1)
        self.assertAlmostEqual(float(gt_ious[0, 0]), 1.0, places=5)
        for selected in (slow, [0, 1, 2], [], [2, 1]):
            stats = _stats()
            evaluate_frame_cached(stats, selected, scores, gt_ious, len(gt))
            assert_same_as_opencood(stats, rows, selected, scores, gt)
            self.assertEqual(
                assign_cached(rows, selected, scores, gt_ious, len(gt)),
                _assign(rows, selected, gt, scores))

    def test_empty_feature_pool_with_cached_matrix(self):
        x, y, ids, matrix = build_frame_features(
            [], "hwl", 0.2, return_overlap=True)
        self.assertEqual(x.shape, (0, len(feature_names())))
        self.assertEqual(y.shape, (0,))
        self.assertEqual(ids.shape, (0,))
        self.assertEqual(matrix.shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
