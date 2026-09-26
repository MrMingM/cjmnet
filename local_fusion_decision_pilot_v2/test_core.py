import unittest

import torch
from torch import nn

from local_fusion_decision_pilot.core import candidate_features, residual_with_gate_maps
from local_fusion_utility_v2.fusion import build_context
from local_fusion_v3.network import SpatialSourceFusion
from .core import (ACTION_FEATURE_NAMES, action_features, actions_for_tiles,
                   choose_distinct_rows, levels_for_choices, prediction_for_choices,
                   select_tiles, simple_scores)


class FollowUpTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.deblocks = nn.ModuleList(
                    [nn.Identity(),nn.Upsample(scale_factor=2),nn.Upsample(scale_factor=4)])
                self.cls_head = nn.Conv2d(56,2,1)
                self.reg_head = nn.Conv2d(56,14,1)

        self.base = Base().eval()
        self.levels = [torch.randn(3,8,16,8),torch.randn(3,16,8,4),
                       torch.randn(3,32,4,2)]
        self.module = SpatialSourceFusion([8,16,32],hidden=4).eval()
        self.settings = dict(tiles_per_frame=2,low_score_floor=.02,
                             high_score_floor=.2,max_peer_actions_per_tile=2)

    def test_actions_preserve_unselected_feature_cells(self):
        with torch.no_grad():
            context = build_context(self.base,self.levels,8)
            residual,gates = residual_with_gate_maps(self.module,self.levels)
            base_features,_ = candidate_features(context,residual,gates,self.base)
            tiles = select_tiles(context,base_features,self.settings)
            self.assertEqual(sorted(tiles),[0,1])
            actions = actions_for_tiles(context,tiles,self.settings)
            self.assertTrue(any(code == 1 for _,code in actions))
            self.assertTrue(any(code > 1 for _,code in actions))
            kept = levels_for_choices(context,residual,[])
            for old,new in zip(context['baseline_levels'],kept):
                torch.testing.assert_close(old,new)
            one = levels_for_choices(context,residual,[(0,1)])
            torch.testing.assert_close(one[0][:,:,8:],context['baseline_levels'][0][:,:,8:])
            torch.testing.assert_close(one[1][:,:,4:],context['baseline_levels'][1][:,:,4:])
            torch.testing.assert_close(one[2],context['baseline_levels'][2])
            peer_action = next(action for action in actions if action[1] > 1)
            prediction = prediction_for_choices(self.base,context,residual,[peer_action])
            self.assertEqual(prediction['psm'].shape[-2:],(16,8))
            features = action_features(context,base_features,peer_action,prediction)
            self.assertEqual(features.numel(),len(ACTION_FEATURE_NAMES))
            confidence,response = simple_scores(context,peer_action,prediction)
            self.assertTrue(torch.isfinite(torch.tensor([confidence,response])).all())

    def test_conflicting_tile_actions_are_rejected(self):
        with torch.no_grad():
            context = build_context(self.base,self.levels,8)
            residual,_ = residual_with_gate_maps(self.module,self.levels)
            with self.assertRaises(ValueError):
                levels_for_choices(context,residual,[(0,1),(0,2)])

    def test_equal_budget_uses_distinct_tiles(self):
        actions = [(0,1),(0,2),(1,1),(1,2)]
        self.assertEqual(choose_distinct_rows([10,9,8,7],actions,2),[0,2])
        self.assertEqual(choose_distinct_rows([-.1,.4,-.2,.3],actions,2,True),[1,3])


if __name__ == '__main__':
    unittest.main()
