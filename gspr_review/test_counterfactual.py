import unittest
import torch
from .counterfactual import (ContrastGain, CounterfactualReviewer, neutral_reference,
                             two_source_interactions)
from .test_tensors import sample
from .evidence import summarize


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def inputs(self):
        x = torch.rand(24, 11)
        x[:, 7:10] = torch.tensor([.7, .1, .2])
        return x

    def activate(self, net):
        with torch.no_grad():
            for p in net.score.parameters():
                p.zero_()
            net.score[0].weight[0, 7] = 1
            net.score[0].weight[0, 8] = -1
            net.score[2].weight[0, 0] = 1
            net.score[-1].weight[0, 0] = 1

    def test_reference_preserves_mass_quality_and_local_inputs(self):
        x = self.inputs()
        r = neutral_reference(x)
        torch.testing.assert_close(r[:, :7], x[:, :7], atol=0, rtol=0)
        torch.testing.assert_close(r[:, 9:], x[:, 9:], atol=0, rtol=0)
        torch.testing.assert_close(r[:, 7]+r[:, 8], x[:, 7]+x[:, 8])
        torch.testing.assert_close(r[:, 7], r[:, 8], atol=0, rtol=0)

    def test_initial_identity_neutral_identity_and_content_response(self):
        net = ContrastGain(16)
        x = self.inputs()
        self.assertEqual(float(net(x).abs().sum()), 0.)
        self.activate(net)
        self.assertGreater(float(net(x).sum()), 0.)
        self.assertEqual(float(net(neutral_reference(x)).abs().sum()), 0.)
        reverse = x.clone()
        reverse[:, 7:9] = x[:, [8, 7]]
        self.assertLess(float(net(reverse).sum()), 0.)

    def test_only_local_dependence_cancels(self):
        net = ContrastGain(16)
        with torch.no_grad():
            net.score[-1].weight.normal_()
            net.score[0].weight[:, 7:].zero_()
        self.assertEqual(float(net(self.inputs()).abs().sum()), 0.)

    def test_score_then_gain_gradients_and_ablation(self):
        net = ContrastGain(16)
        opt = torch.optim.SGD(net.parameters(), lr=.1)
        x = self.inputs()
        loss = (net(x)-.1).square().mean()
        loss.backward()
        self.assertGreater(float(net.score[-1].weight.grad.abs().sum()), 0.)
        opt.step()
        opt.zero_grad()
        (net(x)-.1).square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in net.gain.parameters()), 0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
        ablation = ContrastGain(16, use_gain=False)
        ablation.score.load_state_dict(net.score.state_dict())
        contrast, gain = net.components(x)
        torch.testing.assert_close(ablation(x), contrast)
        self.assertTrue(((gain > 0) & (gain < 1)).all())

    def test_padding_abstention_bounds_and_intervention_keep_mask(self):
        p, r = sample()
        ids = torch.tensor([0])
        s = summarize(p, r, ids, (1, 2), (-2, 2), 4)
        reviewer = CounterfactualReviewer()
        before, mask = reviewer(p, r, ids, s, (1, 2), (-2, 2), 4)
        torch.testing.assert_close(before, r['point_reliability'], atol=0, rtol=0)
        self.activate(reviewer.net)
        changed, mask2 = reviewer(p, r, ids, s, (1, 2), (-2, 2), 4)
        self.assertTrue(torch.equal(mask, mask2))
        self.assertTrue((changed[mask] > before[mask]).all())
        self.assertTrue(((changed-before).abs() <= reviewer.maximum).all())
        torch.testing.assert_close(changed[~mask], before[~mask], atol=0, rtol=0)
        for intervention in ('neutral', 'permuted'):
            reviewer.intervention = intervention
            result, current_mask = reviewer(p, r, ids, s, (1, 2), (-2, 2), 4)
            self.assertTrue(torch.equal(mask, current_mask))
            if intervention == 'neutral':
                torch.testing.assert_close(result, before, atol=0, rtol=0)
        reviewer.intervention = 'normal'
        empty, empty_mask = reviewer(p, r, ids, torch.zeros_like(s), (1, 2), (-2, 2), 4)
        self.assertFalse(empty_mask.any())
        torch.testing.assert_close(empty, before, atol=0, rtol=0)

    def test_interaction_additive_product_and_completeness(self):
        # Additive sources have zero mixed effect; product has nonzero effect.
        local, peer, off = two_source_interactions(1., 3., 4., 6.)
        self.assertEqual(off, 0.)
        self.assertEqual(local+peer+2*off, 5.)
        local, peer, off = two_source_interactions(0., 0., 0., 6.)
        self.assertEqual(off, 3.)
        self.assertEqual(local+peer+2*off, 6.)


if __name__ == '__main__':
    unittest.main()
