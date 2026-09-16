import unittest
import torch
from .bev_location import correct_bev


class LocationTests(unittest.TestCase):
    def fixture(self):
        levels = [torch.ones(1, c, 8//s, 8//s) for c, s in ((2, 2), (3, 4), (4, 8))]
        processed = {'voxel_coords': torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 6, 6]])}
        delta = torch.tensor([[.1, .2], [-.15, .2], [.25, .25]], requires_grad=True)
        supported = torch.tensor([[True, True], [True, False], [False, False]])
        return levels, processed, delta, supported

    def test_placement_pooling_padding_and_gradient(self):
        levels, processed, delta, supported = self.fixture()
        outputs, maps = correct_bev(levels, processed, delta, supported)
        self.assertAlmostEqual(float(maps[0][0, 0, 0, 0]), .05, places=6)
        for before, after, m in zip(levels, outputs, maps):
            torch.testing.assert_close(after, before*(1+m))
            self.assertEqual(int((m != 0).sum()), 1)
            torch.testing.assert_close(before, torch.ones_like(before), atol=0, rtol=0)
        sum(v.sum() for v in outputs).backward()
        self.assertTrue((delta.grad[supported] != 0).all())
        self.assertTrue((delta.grad[~supported] == 0).all())

    def test_zero_and_no_support_are_identity(self):
        levels, processed, delta, supported = self.fixture()
        for d, mask in ((delta*0, supported), (delta, torch.zeros_like(supported))):
            outputs, maps = correct_bev(levels, processed, d, mask)
            for before, after, m in zip(levels, outputs, maps):
                torch.testing.assert_close(before, after, atol=0, rtol=0)
                self.assertEqual(float(m.abs().sum()), 0.)

    def test_sign_bound_and_no_neighbor_input(self):
        levels, processed, delta, supported = self.fixture()
        for sign in (-1, 1):
            outputs, maps = correct_bev(levels, processed, torch.full_like(delta, sign*.25), supported)
            for m in maps:
                self.assertTrue((m.abs() <= .25+1e-7).all())
                self.assertTrue((m*sign >= 0).all())

    def test_head_receives_gradient_at_bev_location(self):
        from .counterfactual import CounterfactualReviewer
        from .test_tensors import sample
        from .evidence import summarize
        torch.manual_seed(31)
        p, r = sample()
        ids = torch.tensor([0])
        evidence = summarize(p, r, ids, (1, 2), (-2, 2), 4)
        reviewer = CounterfactualReviewer(use_gain=False)
        proposed, supported = reviewer(p, r, ids, evidence, (1, 2), (-2, 2), 4)
        levels = [torch.ones(1, c, 8//s, 16//s) for c, s in ((2, 2), (3, 4), (4, 8))]
        updated, _ = correct_bev(levels, p, proposed-r['point_reliability'], supported)
        sum(x.square().mean() for x in updated).backward()
        self.assertGreater(float(reviewer.net.score[-1].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in reviewer.parameters()))


if __name__ == '__main__':
    unittest.main()
