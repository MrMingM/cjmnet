"""Run on the server: python -m unittest gspr_communication.test_tensors -v."""
import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed locally; run on server')
class TensorTests(unittest.TestCase):
    def test_residual_identity_bound_and_learning(self):
        from .residual import PriorityCorrection
        from .selection import hard_topk, soft_topk
        from .masked_attfuse import fuse
        torch.manual_seed(19)
        model = PriorityCorrection(hidden=16, max_adjustment=.1)
        stats = torch.rand(1, 6, 3, 4)
        confidence = torch.rand(1, 1, 3, 4)
        request = torch.rand_like(confidence)
        reference = request * confidence
        score = model(stats, confidence, request)
        torch.testing.assert_close(score, reference, atol=0, rtol=0)
        torch.testing.assert_close(hard_topk(score, 3), hard_topk(reference, 3), atol=0, rtol=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        features = torch.randn(2, 8, 6, 8)
        for _ in range(2):
            optimizer.zero_grad()
            score = model(stats, confidence, request)
            hard = hard_topk(score, 3)
            soft = soft_topk(score, hard, .15)
            out = fuse(features, torch.cat([torch.ones_like(hard), hard]),
                       torch.cat([torch.ones_like(soft), soft]))
            out.square().mean().backward()
            self.assertGreater(float(model.net[-1].weight.grad.abs().sum()), 0)
            optimizer.step()
        self.assertGreater(float(model.net[0].weight.grad.abs().sum()), 0)
        adjusted = model(stats, confidence, request)
        self.assertGreater(float((adjusted-reference).abs().max()), 0)
        self.assertLessEqual(float((adjusted-reference).abs().max()), .100001)
        # Save/load preserves scores; the amplitude belongs to config.
        restored = PriorityCorrection(hidden=16, max_adjustment=.1)
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(stats, confidence, request), adjusted)

    def test_attention_endpoints_and_no_unreceived_leakage(self):
        from .masked_attfuse import fuse
        from attfuse_code.self_attn import AttFusion
        x = torch.randn(3, 8, 4, 6)
        masks = torch.ones(3, 1, 2, 3)
        expected = AttFusion(8)(x, torch.tensor([3]))
        torch.testing.assert_close(fuse(x, masks), expected)
        masks[1:] = 0
        torch.testing.assert_close(fuse(x, masks), x[:1])
        changed = x.clone()
        changed[1:] = float('nan')
        torch.testing.assert_close(fuse(changed, masks), x[:1])

    def test_raw_probability_and_padding(self):
        from .observation_stats import block_statistics
        # Reliable probability .75, mapped weight .7625; padding must not count.
        rel = {'point_evidence': torch.tensor([[[2., 0.], [999., 0.]]]),
               'point_valid_mask': torch.tensor([[True, False]]),
               'point_uncertainty': torch.tensor([[.5, 0.]])}
        p = {'voxel_coords': torch.tensor([[0, 0, 0, 0]])}
        stats = block_statistics(rel, p, 2, (1, 1))
        self.assertAlmostEqual(float(stats[0, 0, 0, 0]), .75)
        self.assertAlmostEqual(float(stats[0, 2, 0, 0]), float(torch.log1p(torch.tensor(.75))))
        self.assertEqual(float(stats[1].sum()), 0)

    def test_a_b_receive_gradients_through_hard_selection(self):
        from .heads import RequestHead, ResponseHead
        from .selection import hard_topk, soft_topk
        from .masked_attfuse import fuse
        torch.manual_seed(7)
        a, b = RequestHead(8), ResponseHead(8)
        sem = torch.randn(2, 8, 2, 3)
        stats = torch.rand(2, 6, 2, 3)
        conf = torch.rand(2, 1, 2, 3)
        request = a(sem[:1], stats[:1], conf[:1])
        q = request + ((request*255).round()/255-request).detach()
        scores = b(sem[1:], stats[1:], conf[1:], q)
        hard = hard_topk(scores, 2)
        soft = soft_topk(scores, hard, .15)
        h = torch.cat([torch.ones_like(hard), hard])
        s = torch.cat([torch.ones_like(soft), soft])
        feat = torch.randn(2, 8, 4, 6)
        output = fuse(feat, h, s)
        torch.testing.assert_close(output, fuse(feat, h))
        output.square().mean().backward()
        for module in (a, b):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)


if __name__ == '__main__':
    unittest.main()
