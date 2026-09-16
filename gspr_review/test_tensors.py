import unittest
import torch
from .evidence import PointReviewer, summarize, combine


def sample():
    points = torch.tensor([[[.1, .2, 0., .5], [.2, .1, 0., .6], [0., 0., 0., 0.]],
                           [[8.1, .2, 0., .5], [0., 0., 0., 0.], [0., 0., 0., 0.]]])
    valid = torch.tensor([[True, True, False], [True, False, False]])
    e = torch.tensor([6., 1.]).expand(2, 3, 2).clone()
    rel = {'point_evidence': e, 'point_valid_mask': valid, 'point_reliability': valid.float()*.7}
    processed = {'voxel_features': points, 'voxel_coords': torch.tensor([[0, 0, 0, 0], [0, 0, 0, 8]]), 'voxel_num_points': valid.sum(1)}
    return processed, rel


class EvidenceTests(unittest.TestCase):
    def test_summary_uses_evidence_not_floor_and_no_padding(self):
        p, r = sample()
        s = summarize(p, r, torch.tensor([0]), (1, 2), (-2, 2), 4)
        torch.testing.assert_close(s[0, 0, 0, 2], torch.tensor([6/9, 1/9, 2/9, 2/6]))
        r['point_evidence'][~r['point_valid_mask']] = 10000
        torch.testing.assert_close(s, summarize(p, r, torch.tensor([0]), (1, 2), (-2, 2), 4))

    def test_identity_abstention_unrequested_and_gradients(self):
        p, r = sample()
        ids = torch.tensor([0])
        s = summarize(p, r, ids, (1, 2), (-2, 2), 4)
        net = PointReviewer()
        out, eligible = net(p, r, ids, s, (1, 2), (-2, 2), 4)
        torch.testing.assert_close(out, r['point_reliability'], atol=0, rtol=0)
        self.assertEqual(int(eligible.sum()), 2)
        out.sum().backward()
        self.assertGreater(float(net.net[-1].bias.grad.abs().sum()), 0)
        with torch.no_grad():
            net.net[-1].bias.fill_(1)
        changed, _ = net(p, r, ids, s, (1, 2), (-2, 2), 4)
        self.assertTrue((changed[eligible] > out[eligible]).all())
        torch.testing.assert_close(changed[~eligible], out[~eligible], atol=0, rtol=0)
        empty, _ = net(p, r, ids, torch.zeros_like(s), (1, 2), (-2, 2), 4)
        torch.testing.assert_close(empty, out, atol=0, rtol=0)

    def test_unrequested_sender_data_cannot_change_message(self):
        p, r = sample()
        ids = torch.tensor([0])
        before = summarize(p, r, ids, (1, 2), (-2, 2), 4)
        r['point_evidence'][1] = 300
        p['voxel_features'][1, :, 2] = 1.5
        after = summarize(p, r, ids, (1, 2), (-2, 2), 4)
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_evidence_not_independently_accumulated(self):
        p, r = sample()
        s = summarize(p, r, torch.tensor([0]), (1, 2), (-2, 2), 4)
        torch.testing.assert_close(combine(s, s), s, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
