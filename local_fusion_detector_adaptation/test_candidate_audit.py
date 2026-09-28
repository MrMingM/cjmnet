import unittest
from types import SimpleNamespace

import numpy as np
import torch

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import box_utils

from .candidate_audit import (_fixed_candidate_features, _project_selected,
                              pool_gate, POOL_SPECS, WEATHERS, ARMS)


class CandidateAuditTests(unittest.TestCase):
    def test_feature_cache_uses_the_same_anchor_cell(self):
        feature = torch.tensor([[[[1., 2.]]]])
        detector = SimpleNamespace(
            backbone=SimpleNamespace(deblocks=[torch.nn.Identity()]),
            cls_head=torch.nn.Conv2d(1, 2, 1, bias=False),
            reg_head=torch.nn.Conv2d(1, 14, 1, bias=False))
        arm = SimpleNamespace(fusion=lambda levels: (levels, {}), detector=detector)
        prediction = {'psm': detector.cls_head(feature),
                      'rm': detector.reg_head(feature)}
        vectors = _fixed_candidate_features(
            arm, [feature], prediction, np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(vectors[:, 0], [1., 1., 2.])

    def test_selected_multisource_decode_preserves_anchor_identity(self):
        torch.manual_seed(9)
        anchors = torch.zeros(2, 2, 2, 7)
        anchors[..., 3:6] = 1
        anchors[..., 0] = torch.arange(8).reshape(2, 2, 2)
        ego = {'anchor_box': anchors, 'transformation_matrix': torch.eye(4)}
        pp = SimpleNamespace(params={'order': 'hwl'},
                             delta_to_boxes3d=VoxelPostprocessor.delta_to_boxes3d)
        prediction = {'rm': torch.randn(3, 14, 2, 2) * .05}
        ids = np.array([0, 3, 7], dtype=np.int64)
        multi = _project_selected(pp, prediction, ego, ids)
        self.assertEqual(multi.shape, (3, 3, 8, 3))
        for source in range(3):
            decoded = pp.delta_to_boxes3d(prediction['rm'][source:source+1], anchors)[0, ids]
            expected = box_utils.project_box3d(
                box_utils.boxes_to_corners_3d(decoded, order='hwl'), torch.eye(4))
            torch.testing.assert_close(torch.as_tensor(multi[source]), expected)

    def test_pool_gate_checks_every_arm_and_weather(self):
        conditions = {}
        for weather in WEATHERS:
            conditions[weather] = {'pools': {}}
            for arm in ARMS:
                conditions[weather]['pools'][arm] = {
                    name: {'candidates_per_frame_p90': 200,
                           'range_covered_70': 90 if name != 'all_geometry' else 100}
                    for name, _, _ in POOL_SPECS
                }
        self.assertIn('top_256', pool_gate(conditions)['passing_pools'])
        conditions['snow']['pools']['F+D']['top_256']['range_covered_70'] = 89
        self.assertNotIn('top_256', pool_gate(conditions)['passing_pools'])


if __name__ == '__main__':
    unittest.main()
