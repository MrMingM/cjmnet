"""CPU unit tests; run before extraction/train/evaluation."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from .data import make_labels
from .model import D2DRescore
from .evaluate import _record_from_iou


def rectangle(x=0., y=0., width=2., length=2.):
    half_w, half_l = width / 2., length / 2.
    xy = np.asarray([
        [x - half_w, y - half_l],
        [x + half_w, y - half_l],
        [x + half_w, y + half_l],
        [x - half_w, y + half_l],
    ], dtype=np.float32)
    bottom = np.column_stack((xy, np.full(4, -.5, dtype=np.float32)))
    top = np.column_stack((xy, np.full(4, .5, dtype=np.float32)))
    return np.vstack((bottom, top))


class TestD2DRescore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = D2DRescore(width=32, layers=2, heads=4).eval()
        self.boxes = torch.tensor([
            [1., 2., 0., 4., 2., 1.5, .1],
            [1.2, 2.1, 0., 4., 2., 1.5, .1],
            [20., -2., 0., 4., 2., 1.5, .2],
        ], dtype=torch.float32)[None]
        self.scores = torch.tensor([[.8, .5, .2]], dtype=torch.float32)

    def test_zero_residual_preserves_original_score(self):
        with torch.no_grad():
            got = self.model.rescore(
                self.boxes, self.scores,
                torch.ones_like(self.scores, dtype=torch.bool))
        torch.testing.assert_close(
            got, self.scores, atol=1e-6, rtol=1e-6)

    def test_permutation_equivariance(self):
        with torch.no_grad():
            self.model.score_head[-1].weight.normal_(std=.02)
            mask = torch.ones_like(self.scores, dtype=torch.bool)
            reference = self.model.rescore(self.boxes, self.scores, mask)
            perm = torch.tensor([2, 0, 1])
            permuted = self.model.rescore(
                self.boxes[:, perm], self.scores[:, perm], mask[:, perm])
        torch.testing.assert_close(
            permuted, reference[:, perm], atol=2e-6, rtol=2e-6)

    def test_padding_is_ignored(self):
        padded_boxes = torch.cat(
            (self.boxes, torch.zeros(1, 2, 7)), dim=1)
        padded_scores = torch.cat(
            (self.scores, torch.zeros(1, 2)), dim=1)
        padded_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        with torch.no_grad():
            raw = self.model.rescore(
                self.boxes, self.scores,
                torch.ones_like(self.scores, dtype=torch.bool))
            padded = self.model.rescore(
                padded_boxes, padded_scores, padded_mask)
        torch.testing.assert_close(
            raw, padded[:, :3], atol=2e-6, rtol=2e-6)

    def test_gossip_variant_and_mask(self):
        gossip = D2DRescore(
            width=32, layers=2, heads=4, variant='gossip', radius=5.).eval()
        mask = torch.ones_like(self.scores, dtype=torch.bool)
        with torch.no_grad():
            identity = gossip.rescore(self.boxes, self.scores, mask)
            gossip.score_head[-1].weight.normal_(std=.02)
            changed = gossip.rescore(self.boxes, self.scores, mask)
            perm = torch.tensor([2, 0, 1])
            reordered = gossip.rescore(
                self.boxes[:, perm], self.scores[:, perm], mask[:, perm])
        torch.testing.assert_close(identity, self.scores, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            reordered, changed[:, perm], atol=2e-6, rtol=2e-6)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.boxes, self.scores, torch.zeros_like(
                self.scores, dtype=torch.bool))


class TestGreedyMatching(unittest.TestCase):
    def test_duplicate_competition(self):
        gt = rectangle()[None]
        boxes = np.stack((
            rectangle(), rectangle(), rectangle(x=20.)))
        labels = make_labels(
            boxes, gt, np.asarray([.9, .8, .7]), threshold=.7)
        np.testing.assert_array_equal(
            labels, np.asarray([1., 0., 0.]))

    def test_fixed_cached_iou_and_empty_gt(self):
        corners = np.stack((rectangle(), rectangle(x=20.)))
        gt = rectangle()[None]
        labels = make_labels(
            corners, gt, np.asarray([.6, .9]), .7,
            iou_matrix=np.asarray([[1.], [0.]], dtype=np.float32))
        np.testing.assert_array_equal(
            labels, np.asarray([1., 0.]))
        self.assertEqual(
            make_labels(corners, np.zeros((0, 8, 3), dtype=np.float32),
                        np.asarray([.6, .9])).sum(), 0.)


class TestCachedEvaluation(unittest.TestCase):
    def test_cached_iou_matches_open_cood_tp_fp(self):
        from ceif_audit.scoring import empty_stats
        from opencood.utils import eval_utils

        gt = np.stack((rectangle(), rectangle(x=20.)))
        boxes = np.stack((
            rectangle(),
            rectangle(),
            rectangle(x=20.),
            rectangle(x=40.),
        ))
        scores = np.asarray([.9, .8, .7, .6], dtype=np.float32)
        # Exact matrix for the synthetic non-overlapping rectangles above.
        ious = np.asarray([
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [0., 0.],
        ], dtype=np.float32)

        cached = empty_stats()
        _record_from_iou(cached, scores, ious)

        reference = empty_stats()
        tb = torch.as_tensor(boxes, dtype=torch.float32)
        ts = torch.as_tensor(scores, dtype=torch.float32)
        tg = torch.as_tensor(gt, dtype=torch.float32)
        for threshold in (.3, .5, .7):
            eval_utils.caluclate_tp_fp(
                tb, ts, tg, reference, threshold)

        self.assertEqual(cached, reference)

    def test_cached_iou_handles_empty_detections(self):
        from ceif_audit.scoring import empty_stats
        stats = empty_stats()
        _record_from_iou(
            stats,
            np.empty(0, dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
        for threshold in (.3, .5, .7):
            self.assertEqual(stats[threshold]['gt'], 2)
            self.assertEqual(stats[threshold]['tp'], [])
            self.assertEqual(stats[threshold]['fp'], [])
            self.assertEqual(stats[threshold]['score'], [])


if __name__ == '__main__':
    unittest.main()
