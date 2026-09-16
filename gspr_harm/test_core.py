import unittest
from types import SimpleNamespace
import torch
from torch import nn
from gspr_communication import codec
from gspr_communication.masked_attfuse import fuse
from .core import Messages, capture, predict, drop_mask, groups, random_deletions, classify, loss_values


class HarmTests(unittest.TestCase):
    def test_legacy_fusion_config_defaults_and_rejections(self):
        from .config import fusion_args_for_probe
        base = {'core_method': 'IntermediateFusionDataset'}
        for extra in ({}, {'args': []}, {'args': {}}, {'args': {'proj_first': True}}):
            self.assertEqual(fusion_args_for_probe(dict(base, **extra)), {'proj_first': True})
        saved = dict(base, args={'proj_first': True, 'cur_ego_pose_flag': False})
        self.assertEqual(fusion_args_for_probe(saved), saved['args'])
        self.assertIsNot(fusion_args_for_probe(saved), saved['args'])
        for value in ({'proj_first': False}, {'proj_first': 'false'}, ['proj_first'], None):
            with self.assertRaises(ValueError):
                fusion_args_for_probe(dict(base, args=value))

    def test_absence_is_not_zero_and_ego_fallback(self):
        # A zero sender changes attention; actual absence returns the negative ego.
        features = torch.tensor([-2., 0.]).reshape(2, 1, 1, 1)
        mask = torch.ones(2, 1, 1, 1)
        baseline = fuse(features, mask)
        actual_drop = fuse(features, drop_mask(mask, [(1, 0)]))
        self.assertGreater(float(baseline), -2.)
        torch.testing.assert_close(actual_drop, features[:1], atol=0, rtol=0)
        features[1] = 1e6
        torch.testing.assert_close(fuse(features, drop_mask(mask, [(1, 0)])), actual_drop)
        with self.assertRaises(ValueError):
            drop_mask(mask, [(0, 0)])

    def test_positive_delta_means_harmful_not_beneficial(self):
        mask = torch.ones(2, 1, 1, 1)
        for sender, target, expected in [(3., 2., 'harmful'), (0., 1., 'beneficial')]:
            x = torch.tensor([2., sender]).reshape(2, 1, 1, 1)
            send = (fuse(x, mask)-target).square().sum()
            drop = (fuse(x, drop_mask(mask, [(1, 0)]))-target).square().sum()
            self.assertEqual(classify(float(send-drop), 1e-6), expected)
        self.assertEqual(classify(1e-6, 1e-6), 'neutral')

    def test_multiscale_capture_deletion_and_budget(self):
        shapes = [(2, 1, 8, 8), (2, 1, 4, 4), (2, 1, 2, 2)]
        levels = [torch.arange(torch.tensor(s).prod()).float().reshape(s)/100 for s in shapes]
        grid = (2, 2)
        cost = codec.block_bytes([1, 1, 1], [4, 2, 1])
        head = nn.Identity()
        model = SimpleNamespace(training=False, variant='a0b0', grid=grid, cost=cost,
            budget=codec.request_bytes(*grid)+codec.HEADER.size+2*cost, value_bytes=4,
            _request=lambda a,b,c: 1-c, _score=lambda a,b,c,r: c*r,
            base=SimpleNamespace(backbone=SimpleNamespace(deblocks=[nn.Identity(), nn.Upsample(scale_factor=2), nn.Upsample(scale_factor=4)]),
                                 cls_head=head, reg_head=head))
        conf = torch.tensor([.1, .2, .3, .4, .9, .8, .7, .6]).reshape(2, 1, 2, 2)
        msg = capture(model, (levels, conf, conf, conf))
        self.assertEqual(msg.total_bytes, model.budget)
        self.assertEqual(int(msg.mask[1].sum()), 2)
        original_mask = msg.mask.clone()
        selected = [pair for group in groups(msg.mask) for pair in group]
        mask = drop_mask(msg.mask, selected)
        output = predict(model, msg, mask)['psm']
        expected = torch.cat([d(x[:1]) for d,x in zip(model.base.backbone.deblocks, levels)], 1)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(msg.mask, original_mask, atol=0, rtol=0)
        for x in msg.levels:
            x[1] = -1000
        torch.testing.assert_close(predict(model, msg, mask)['psm'], expected, atol=0, rtol=0)

    def test_grouping_and_random_match_each_sender(self):
        mask = torch.ones(3, 1, 4, 4)
        mask[1, :, 0, 0] = 0
        grouped = groups(mask, 2)
        pairs = [p for g in grouped for p in g]
        self.assertEqual(len(set(pairs)), 31)
        harm = grouped[0]+grouped[-1]
        rand = random_deletions(mask, harm, 42)
        self.assertEqual(rand, random_deletions(mask, harm, 42))
        self.assertEqual(len(set(rand)), len(harm))
        for peer in (1, 2):
            self.assertEqual(sum(p == peer for p,b in harm), sum(p == peer for p,b in rand))
        with self.assertRaises(ValueError):
            drop_mask(mask, [(1, 0)])

    def test_detection_components_fixed_labels_and_nonfinite(self):
        from opencood.loss.point_pillar_loss import PointPillarLoss
        criterion = PointPillarLoss({'cls_weight': 1., 'reg': 2.})
        labels = {'pos_equal_one': torch.tensor([[[[1.], [0.]]]]), 'targets': torch.zeros(1, 1, 2, 7)}
        saved = {k: v.clone() for k,v in labels.items()}
        output = {'psm': torch.zeros(1, 1, 1, 2), 'rm': torch.zeros(1, 7, 1, 2)}
        first = loss_values(criterion, output, labels)
        self.assertAlmostEqual(first['total_loss'], first['reg_loss']+first['conf_loss'])
        self.assertEqual(first, loss_values(criterion, output, labels))
        for k in labels:
            torch.testing.assert_close(labels[k], saved[k], atol=0, rtol=0)
        output['psm'][0, 0, 0, 0] = float('nan')
        with self.assertRaises(RuntimeError):
            loss_values(criterion, output, labels)


if __name__ == '__main__':
    unittest.main()
