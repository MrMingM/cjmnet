import unittest

import torch
from torch import nn

from local_fusion_utility_v2.fusion import build_context
from local_fusion_v3.network import SpatialSourceFusion
from .core import (
    FEATURE_NAMES, candidate_features, choose_frames, outcome_class,
    patch_prediction, residual_with_gate_maps, selected_indices,
)


class DecisionPilotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.deblocks = nn.ModuleList(
                    [nn.Identity(), nn.Upsample(scale_factor=2), nn.Upsample(scale_factor=4)])
                self.cls_head = nn.Conv2d(56, 2, 1)
                self.reg_head = nn.Conv2d(56, 14, 1)

        self.base = Base().eval()
        self.levels = [torch.randn(3, 8, 8, 8),
                       torch.randn(3, 16, 4, 4),
                       torch.randn(3, 32, 2, 2)]
        self.module = SpatialSourceFusion([8, 16, 32], hidden=4).eval()

    def test_scene_sample_is_deterministic_and_scene_bounded(self):
        a = choose_frames([5, 12, 20, 28], 19, 2, 3)
        self.assertEqual(a, choose_frames([5, 12, 20, 28], 19, 2, 3))
        self.assertEqual(len(a[1]), 6)
        self.assertEqual(len(set(a[1])), 6)

    def test_keep_and_full_tile_patch(self):
        with torch.no_grad():
            context = build_context(self.base, self.levels, tile_size=8)
            fused, gates = residual_with_gate_maps(self.module, self.levels)
            descriptors, _ = candidate_features(context, fused, gates, self.base)
            self.assertEqual(descriptors.shape[1], len(FEATURE_NAMES))
            original = patch_prediction(self.base, context, fused, [], (0, 1))
            modified = patch_prediction(self.base, context, fused, [0], (0, 1))
            from local_fusion_utility_v2.fusion import predict_from_levels
            expected = predict_from_levels(self.base, fused)
            for key in ('psm', 'rm'):
                torch.testing.assert_close(original[key], context['baseline_prediction'][key])
                torch.testing.assert_close(modified[key], expected[key])

    def test_single_source_has_no_candidate_gate(self):
        with torch.no_grad():
            one = [value[:1] for value in self.levels]
            context = build_context(self.base, one, tile_size=8)
            fused, gates = residual_with_gate_maps(self.module, one)
            descriptors, prediction = candidate_features(context, fused, gates, self.base)
            self.assertFalse(gates)
            self.assertIsNone(prediction)
            self.assertEqual(descriptors.shape[0], 0)

    def test_matching_budget_and_outcome_classes(self):
        scores = torch.tensor([.4, .9, .1, .9])
        self.assertEqual(selected_indices(scores, 2), [1, 3])
        self.assertEqual(selected_indices(torch.tensor([-.2, .1]), 2, True), [1])
        targets = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 1., 0.]])
        self.assertEqual(outcome_class(targets).tolist(), [2, 1, 0])


if __name__ == '__main__':
    unittest.main()
