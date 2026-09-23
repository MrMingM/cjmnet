import unittest
import torch
from torch import nn
from .network import SpatialSourceFusion
from local_fusion_utility_v2.fusion import attention_fusion


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.levels = [torch.randn(3,8,8,8),torch.randn(3,16,4,4),torch.randn(3,32,2,2)]

    def test_peer_permutation_and_single_source(self):
        for variant in ('attention','residual'):
            m = SpatialSourceFusion([8,16,32],variant,hidden=4)
            result,_ = m(self.levels)
            shuffled,_ = m([x[[0,2,1]] for x in self.levels])
            for a,b in zip(result,shuffled):
                torch.testing.assert_close(a,b)
            alone,info = m([x[:1] for x in self.levels])
            for a,b in zip(alone,self.levels):
                torch.testing.assert_close(a,b[:1])
            self.assertEqual(float(info['change']),0.)

    def test_bounded_mix_and_keep(self):
        m = SpatialSourceFusion([8,16,32],hidden=4,max_gate=.5)
        result,info = m(self.levels)
        self.assertLessEqual(float(info['change']),1.)
        torch.testing.assert_close(result[2],attention_fusion(self.levels[2])[0])
        for gate in m.gates.values():
            nn.init.constant_(gate.bias,-100.)
        result,_ = m(self.levels)
        for x,y in zip(result,self.levels):
            torch.testing.assert_close(x,attention_fusion(y)[0])

    def test_gradient_through_frozen_detector(self):
        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.deblocks = nn.ModuleList([nn.Identity(),nn.Upsample(scale_factor=2),nn.Upsample(scale_factor=4)])
                self.cls_head = nn.Conv2d(56,2,1)
                self.reg_head = nn.Conv2d(56,14,1)
        base = Base().requires_grad_(False).eval()
        m = SpatialSourceFusion([8,16,32],hidden=4)
        prediction,info = m.predict(base,self.levels)
        loss = prediction['psm'].square().mean()+prediction['rm'].square().mean()+.01*info['change']
        loss.backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        for group in (m.encoders,m.routers,m.gates):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in group.parameters() if p.grad is not None),0.)

    def test_validation_keep_for_harm_and_ap50_drop(self):
        import copy
        from .evaluate import choose
        gates = dict(max_lost_tp_fraction=.01,max_new_fp_per_baseline_tp=.01,
                     ap50_tolerance=.001,clean_ap70_tolerance=.001,minimum_mean_ap70_gain=.0001)
        row = dict(results={'baseline':dict(ap50=.8,ap70=.7),'residual':dict(ap50=.8,ap70=.71)},
                   baseline_tp=1000,diagnostics_vs_baseline={'residual':dict(lost=0,new_fp=0)})
        data = {k:copy.deepcopy(row) for k in ('clean','fog','rain','snow')}
        self.assertTrue(choose(data,'residual',gates)['enabled'])
        data['snow']['results']['residual']['ap50'] = .79
        self.assertFalse(choose(data,'residual',gates)['enabled'])
        data['snow'] = copy.deepcopy(row)
        data['rain']['diagnostics_vs_baseline']['residual']['lost'] = 11
        self.assertFalse(choose(data,'residual',gates)['enabled'])


if __name__ == '__main__':
    unittest.main()
