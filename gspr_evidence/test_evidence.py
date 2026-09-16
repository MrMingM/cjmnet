import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from . import protocol
from .geometry import describe, traversal_samples
from .heads import RiskHead, GainHead, pair_ranking_loss, risk_loss
from .model import EvidenceModel
from .supervision import dense_targets, candidate_targets


def toy_model():
    model = EvidenceModel.__new__(EvidenceModel)
    nn.Module.__init__(model)
    model.options = dict(queries=4, teacher_candidates=4)
    model.grid = (2, 2)
    model.disable_u = False
    engine = nn.Module()
    engine.cost = protocol.block_bytes([2, 2, 2], [4, 2, 1], 4)
    engine.value_bytes = 4
    engine.budget = protocol.request_bytes(4)+protocol.HEADER.size+2*engine.cost
    engine.base = nn.Module()
    engine.base.backbone = nn.Module()
    engine.base.backbone.deblocks = nn.ModuleList([nn.Identity(), nn.Upsample(scale_factor=2), nn.Upsample(scale_factor=4)])
    engine.base.cls_head = nn.Conv2d(6, 1, 1)
    engine.base.reg_head = nn.Conv2d(6, 7, 1)
    model.engine = engine
    model.heads = nn.ModuleDict({'risk': RiskHead(), 'gain': GainHead()})
    encoded = dict(levels=[torch.randn(2, 2, 8, 8), torch.randn(2, 2, 4, 4), torch.randn(2, 2, 2, 2)],
                   obs=torch.rand(2, 13, 2, 2), semantics=torch.randn(2, 64, 2, 2),
                   poses=torch.eye(4)[None].repeat(2, 1, 1))
    return model.eval(), encoded


class EvidenceTests(unittest.TestCase):
    def test_request_roundtrip_and_invalid_packets(self):
        rng = np.random.default_rng(2)
        profile = rng.random((17, 3, 4))
        packet = protocol.pack_request(profile, [0, 9], np.eye(4))
        values, ids, pose, grid = protocol.unpack_request(packet)
        self.assertEqual(len(packet), protocol.request_bytes(2))
        self.assertEqual(grid, (3, 4))
        np.testing.assert_allclose(values, profile.reshape(17, -1)[:, ids].T, atol=.5/255)
        for broken in (packet[:-1], packet+b'0', b'BAD!'+packet[4:]):
            with self.assertRaises(ValueError):
                protocol.unpack_request(broken)
        with self.assertRaises(ValueError):
            protocol.pack_request(profile, [0, 0], pose)

    def test_budget_edges(self):
        for budget in (0, 20, 300, 262144):
            for peers in range(5):
                active, quotas = protocol.allocate(budget, peers, 256, 7172)
                used = peers*(protocol.request_bytes(256)+16)+sum(quotas)*7172 if active else 0
                self.assertLessEqual(used, budget)
                self.assertTrue(all(0 <= x <= 256 for x in quotas))

    def test_geometry_uses_true_origin_and_keeps_unseen_unknown(self):
        transform = torch.eye(4)
        transform[0, 3] = 20
        cloud = torch.tensor([[30., 0, 0, 1]])
        query = torch.tensor([[25., 0, 0], [35., 0, 0], [25., 3, 0]])
        self.assertEqual(traversal_samples(cloud, transform, query).tolist(), [1., 0., 0.])
        translation = torch.tensor([5., -7, 2])
        cloud[:, :3] += translation
        transform[:3, 3] += translation
        self.assertEqual(traversal_samples(cloud, transform, query+translation).tolist(), [1., 0., 0.])
        self.assertEqual(float(traversal_samples(cloud[:0], transform, query).sum()), 0.)

    def test_raw_probability_padding_and_empty_regions(self):
        rel = dict(point_valid_mask=torch.tensor([[True, False]]), point_evidence=torch.tensor([[[8., 0.], [999., 0.]]]),
                   point_uncertainty=torch.tensor([[.2, .001]]), point_reliability=torch.ones(1, 2))
        data = dict(voxel_features=torch.tensor([[[0., 0., -.5, 1.], [0., 0, -.5, 1.]]]),
                    voxel_coords=torch.tensor([[0, 0, 0, 0]]))
        obs = describe(rel, data, torch.zeros(1, 1, 1, 2), torch.eye(4)[None], [torch.zeros(0, 4)], (1, 2), [-4,-4,-2,12,4,2])
        self.assertAlmostEqual(float(obs[0, 0, 0, 0]), .9, places=6)
        self.assertEqual(float(obs[0, 3, 0, 1]), 0)
        self.assertEqual(float(obs[0, 1, 0, 1]), 1)
        self.assertAlmostEqual(float(obs[0, 4:8].sum()), .9/64, places=6)

    def test_missed_gt_is_in_dense_risk_target(self):
        output = dict(psm=torch.full((1, 1, 4, 4), -20.), rm=torch.zeros(1, 7, 4, 4))
        pos = torch.zeros(1, 4, 4, 1)
        pos[0, 3, 3, 0] = 1
        labels = dict(pos_equal_one=pos, neg_equal_one=1-pos, targets=torch.zeros(1, 4, 4, 7))
        risk, fg = dense_targets(output, labels, (2, 2))
        self.assertGreater(float(risk[0, 0, 1, 1]), .99)
        self.assertEqual(float(fg.sum()), 1)

    def test_rank_ties_and_both_signs(self):
        pred = torch.tensor([0., 0., 0.], requires_grad=True)
        truth = torch.tensor([-1., 0., 1.])
        pair_ranking_loss(pred, truth).backward()
        self.assertGreater(float(pred.grad[0]), 0)
        self.assertLess(float(pred.grad[2]), 0)
        self.assertEqual(float(pair_ranking_loss(pred, torch.ones_like(truth))), 0)

    def test_matching_and_concat_have_equal_parameters(self):
        self.assertEqual(sum(x.numel() for x in GainHead().parameters()),
                         sum(x.numel() for x in GainHead(matching=False).parameters()))
        req, own, sem, pose = torch.rand(6, 17), torch.rand(6, 13), torch.rand(6, 64), torch.zeros(1, 5)
        head = GainHead()
        pred = head(req, own, sem, pose)
        pair_ranking_loss(pred, torch.arange(6).float()).backward()
        self.assertGreater(sum(float(x.grad.abs().sum()) for x in head.parameters() if x.grad is not None), 0)

    def test_wire_fusion_and_unreceived_feature_isolation(self):
        model, encoded = toy_model()
        mask = model.empty_masks(encoded)
        mask[1, 0, 0, 0] = 1
        wire, packets = model.detect(encoded, mask, True)
        replay, _ = model.detect(encoded, mask, False)
        for key in wire:
            torch.testing.assert_close(wire[key], replay[key])
        changed = dict(encoded, levels=[x.clone() for x in encoded['levels']])
        for level in changed['levels']:
            patch = level.shape[-1]//2
            level[1, :, patch:, :] = 1e5
        isolated, _ = model.detect(changed, mask, True)
        for key in wire:
            torch.testing.assert_close(wire[key], isolated[key])

    def test_same_request_equal_bytes_and_sender_isolation(self):
        model, encoded = toy_model()
        exchange = model.request(encoded)
        _, left = model.run(encoded, 'learned', exchange)
        _, right = model.run(encoded, 'protocol', exchange)
        self.assertEqual(left['total_bytes'], right['total_bytes'])
        self.assertLessEqual(left['total_bytes'], model.engine.budget)
        # Once request is sent, unsent ego fields cannot affect B's ranking.
        score = model.scores(encoded, 1, *exchange[1:], 'learned')
        changed = dict(encoded, semantics=encoded['semantics'].clone(), obs=encoded['obs'].clone())
        changed['semantics'][0] = 999
        changed['obs'][0] = 999
        torch.testing.assert_close(score, model.scores(changed, 1, *exchange[1:], 'learned'))

    def test_teacher_reuses_context_and_equal_bytes(self):
        model, encoded = toy_model()
        criterion = lambda output, labels: output['psm'].square().mean()
        groups = candidate_targets(model, encoded, {}, criterion, 19)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]['ids']), 3)
        self.assertEqual(len(groups[0]['context_ids'][0]), 1)
        self.assertTrue(torch.isfinite(groups[0]['gains']).all())
        repeat = candidate_targets(model, encoded, {}, criterion, 19)
        torch.testing.assert_close(groups[0]['gains'], repeat[0]['gains'])

    def test_no_u_removes_both_transmitted_and_local_u(self):
        model, encoded = toy_model()
        model.disable_u = True
        exchange = model.request(encoded)
        original = model.scores(encoded, 1, *exchange[1:], 'learned')
        encoded['obs'][:, 1] = 100
        changed = model.request(encoded)
        self.assertEqual(exchange[0], changed[0])
        torch.testing.assert_close(original, model.scores(encoded, 1, *changed[1:], 'learned'))

    def test_cpu_cached_training_and_resume(self):
        from . import runtime as rt, train
        options = dict(seed=13, learning_rate=.001, risk_epochs=1, gain_epochs=1, gain_scale=1000.)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract = rt.contract(options, __file__, 'synthetic-test')
            for split in ('train', 'validation'):
                (root/split).mkdir()
                rt.write_json(root/split/'manifest.json', dict(contract=contract, complete=True, split=split, indices=[0, 1]))
                for index in range(2):
                    row = dict(semantics=torch.randn(2, 64, 2, 2).half(), obs=torch.rand(2, 13, 2, 2).half(),
                               poses=torch.eye(4)[None].repeat(2, 1, 1), target=torch.rand(1, 4, 2, 2).half(),
                               foreground=torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.uint8),
                               groups=[dict(peer=1, ids=torch.tensor([0, 1, 3]), gains=torch.tensor([-.001, .003, 0.]))])
                    torch.save(dict(clean=row, weather=row), root/split/f'{index:08d}.pt')
            argv = ['train', '--train-cache', str(root/'train'), '--validation-cache', str(root/'validation'),
                    '--output-dir', str(root/'model')]
            def new_output(path):
                Path(path).mkdir()
                return Path(path)
            with patch.object(rt, 'device', return_value=torch.device('cpu')), patch.object(rt, 'new_output', side_effect=new_output):
                with patch.object(sys, 'argv', argv):
                    train.main()
                saved = torch.load(root/'model'/'gain_best.pth', weights_only=True)
                self.assertEqual(saved['phase'], 'gain')
                self.assertTrue(all(torch.isfinite(x).all() for x in saved['heads'].values()))
                with patch.object(sys, 'argv', argv+['--resume']):
                    train.main()


if __name__ == '__main__':
    unittest.main()
