import unittest

import torch
from torch import nn

from local_fusion_v3.network import SpatialSourceFusion
from .model import AdaptationArm
from .evaluate import decide


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.deblocks = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(8, 8, 1), nn.BatchNorm2d(8)),
                    nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(16, 8, 1), nn.BatchNorm2d(8)),
                    nn.Sequential(nn.Upsample(scale_factor=4), nn.Conv2d(32, 8, 1), nn.BatchNorm2d(8)),
                ])
                self.cls_head = nn.Conv2d(24, 2, 1)
                self.reg_head = nn.Conv2d(24, 14, 1)
        self.base = Base().eval().requires_grad_(False)
        self.fusion = SpatialSourceFusion([8, 16, 32], hidden=4)
        self.levels = [torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4),
                       torch.randn(2, 32, 2, 2)]

    def test_detector_copy_preserves_source_path_and_initial_outputs(self):
        frozen = AdaptationArm(self.fusion, self.base, False).eval()
        adapted = AdaptationArm(self.fusion, self.base, True).eval()
        for key in ('psm', 'rm'):
            torch.testing.assert_close(frozen.predict(self.base, self.levels)[0][key],
                                       adapted.predict(self.base, self.levels)[0][key],
                                       rtol=0, atol=0)
        before = self.base.cls_head.weight.detach().clone()
        adapted.detector.cls_head.weight.data.add_(1)
        torch.testing.assert_close(self.base.cls_head.weight, before, rtol=0, atol=0)
        self.assertFalse(torch.equal(adapted.detector.cls_head.weight, before))

    def test_gradient_boundary_and_fixed_running_statistics(self):
        for adapt in (False, True):
            arm = AdaptationArm(self.fusion, self.base, adapt).train()
            norms = [layer for layer in arm.detector.modules() if isinstance(layer, nn.BatchNorm2d)]
            original = [layer.running_mean.clone() for layer in norms]
            self.assertTrue(all(not layer.training for layer in norms))
            prediction, info = arm.predict(self.base, self.levels)
            (prediction['psm'].square().mean() + prediction['rm'].square().mean()
             + .01 * info['change']).backward()
            self.assertTrue(any(p.grad is not None for p in arm.fusion.parameters()))
            self.assertEqual(any(p.grad is not None for p in arm.detector.parameters()), adapt)
            self.assertTrue(all(p.grad is None for p in self.base.parameters()))
            for layer, value in zip(norms, original):
                torch.testing.assert_close(layer.running_mean, value, rtol=0, atol=0)

    def test_gate_requires_clean_and_weather_gain(self):
        settings = dict(minimum_weather_mean_ap70_gain=.005, minimum_positive_weathers=2,
                        clean_ap70_tolerance=.001, ap50_tolerance=.001,
                        max_lost_tp_fraction=.01, max_new_fp_per_reference_tp=.01)
        conditions = {}
        for weather in ('clean', 'fog', 'rain', 'snow'):
            conditions[weather] = dict(
                results={name: dict(ap30=.8, ap50=.8, ap70=.7)
                         for name in ('baseline', 'F', 'F+D')},
                diagnostics_vs_F={'F+D': dict(lost=0, new_fp=0)},
                diagnostics_vs_baseline={'F+D': dict(lost=0, new_fp=0)},
                F_tp=100, baseline_tp=100)
        for weather in ('fog', 'rain'):
            conditions[weather]['results']['F+D']['ap70'] = .708
        self.assertTrue(decide(conditions, settings)['expand'])
        conditions['clean']['results']['F+D']['ap70'] = .698
        self.assertFalse(decide(conditions, settings)['expand'])


if __name__ == '__main__':
    unittest.main()
