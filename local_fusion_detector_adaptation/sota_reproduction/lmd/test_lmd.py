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


if __name__ == "__main__":
    unittest.main()
