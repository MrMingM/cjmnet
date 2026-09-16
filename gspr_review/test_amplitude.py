import unittest
import torch
from .counterfactual import CounterfactualReviewer
from .test_tensors import sample
from . import test_counterfactual as fixtures
from .evidence import summarize


class AmplitudeTests(unittest.TestCase):
    def test_identity_scaling_limits_mask_and_state(self):
        reviewer = CounterfactualReviewer(use_gain=False).eval()
        fixtures.CounterfactualTests().activate(reviewer.net)
        p, r = sample()
        ids = torch.tensor([0])
        evidence = summarize(p, r, ids, (1, 2), (-2, 2), 4)
        saved = {k: v.clone() for k, v in reviewer.state_dict().items()}
        def run(scale):
            reviewer.evaluation_scale = scale
            return reviewer(p, r, ids, evidence, (1, 2), (-2, 2), 4)
        zero, mask = run(0)
        torch.testing.assert_close(zero, r['point_reliability'], atol=0, rtol=0)
        normal, _ = run(1)
        twice, _ = run(2)
        torch.testing.assert_close(twice[mask]-zero[mask], 2*(normal[mask]-zero[mask]))
        large, mask2 = run(10000)
        self.assertTrue(torch.equal(mask, mask2))
        torch.testing.assert_close(large[~mask], zero[~mask], atol=0, rtol=0)
        self.assertTrue(((large-zero).abs() <= reviewer.maximum+1e-7).all())
        self.assertTrue(((large[mask] >= reviewer.floor) & (large[mask] <= 1)).all())
        reviewer.intervention = 'neutral'
        neutral, _ = run(100)
        torch.testing.assert_close(neutral, zero, atol=0, rtol=0)
        for k, v in reviewer.state_dict().items():
            torch.testing.assert_close(v, saved[k], atol=0, rtol=0)

    def test_invalid_scale_and_training_guard(self):
        reviewer = CounterfactualReviewer().eval()
        x = torch.zeros(2, 11)
        for scale in (-1, float('nan'), float('inf')):
            reviewer.evaluation_scale = scale
            with self.assertRaises(ValueError):
                reviewer.weight_delta(x)
        reviewer.evaluation_scale = 2
        reviewer.train()
        with self.assertRaises(ValueError):
            reviewer.weight_delta(x)

    def test_default_is_original_expression(self):
        reviewer = CounterfactualReviewer().eval()
        fixtures.CounterfactualTests().activate(reviewer.net)
        x = torch.rand(20, 11)
        expected = reviewer.maximum*reviewer.net(x).squeeze(-1).tanh()
        torch.testing.assert_close(reviewer.weight_delta(x), expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
