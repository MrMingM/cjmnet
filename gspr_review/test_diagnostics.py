import importlib.util
import unittest
import numpy as np
from .diagnostic_utils import near_boxes_xy, point_totals, pillar_totals, summarize_rows


class DiagnosticTests(unittest.TestCase):
    def test_rotated_boxes_and_order(self):
        box = np.array([[0, 0, 0, 2, 2, 4, np.pi/2]])  # hwl; long axis rotates to y
        points = np.array([[0, 1.9], [1.9, 0], [0, 3.1]])
        np.testing.assert_array_equal(near_boxes_xy(points, box, 'hwl', 0), [True, False, False])
        np.testing.assert_array_equal(near_boxes_xy(points, box, 'hwl', 1), [True, True, False])
        lwh = box[:, [0, 1, 2, 5, 4, 3, 6]]
        np.testing.assert_array_equal(near_boxes_xy(points, box, 'hwl', 0), near_boxes_xy(points, lwh, 'lwh', 0))
        self.assertFalse(near_boxes_xy(points, np.empty((0, 7))).any())

    def test_padding_excluded_and_feature_changes(self):
        old, new = np.array([[.5, 0.]]), np.array([[.6, 1.]])
        mask = np.array([[True, False]])
        stats = point_totals(old, new, new, mask, mask, np.ones_like(mask), 1e-6)
        self.assertEqual(stats['valid_points'], 1)
        self.assertEqual(stats['changed_points'], 1)
        self.assertAlmostEqual(stats['abs_weight_change_sum'], .1)
        self.assertEqual(stats['abs_change_vs_normal_sum'], 0)
        pillars = pillar_totals(np.ones((2, 3)), np.array([[1, 1, 2], [7, 7, 7]]), np.array([True, False]), np.ones(2, bool), 1e-6)
        self.assertEqual(pillars['changed_pillars'], 1)
        self.assertEqual(pillars['feature_abs_change_sum'], 1)

    def test_summary_uses_point_weighted_totals(self):
        rows = []
        for n, change in ((1, .1), (9, .3)):
            mask = np.ones(n, bool)
            stats = point_totals(np.zeros(n), np.full(n, change), np.zeros(n), mask, mask, mask, 1e-6)
            stats.update(pillar_totals(np.ones((n, 2)), np.ones((n, 2)), mask, mask, 1e-6))
            groups = {k: stats for k in ('all', 'target_near', 'outside_target_near')}
            groups['peer_input_abs_change_sum'] = 0.
            rows.append({'branch': 'test', 'baseline_repeat_max_error': 0., 'variants': {k: groups for k in ('normal', 'permuted', 'constant', 'zero')}})
        result = summarize_rows(rows)['test']['variants']['normal']['all']
        self.assertAlmostEqual(result['mean_abs_weight_change'], .28)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires server PyTorch')
class FixedMaskTests(unittest.TestCase):
    def test_cell_permutation_preserves_local_inputs_and_support(self):
        import torch
        from .diagnostic_utils import altered_inputs, fixed_mask_weights
        from .evidence import PointReviewer
        x = torch.zeros(1, 4, 11)
        x[..., -4:] = torch.tensor([[[.8, .1, .1, .5], [.8, .1, .1, .5], [.3, .2, .5, .8], [0., 0., 0., 0.]]])
        x[..., :7] = .4
        cell = torch.tensor([[2, 2, 5, 7]])
        eligible = torch.tensor([[True, True, True, False]])
        original = torch.full((1, 4), .5)
        reviewer = PointReviewer()
        with torch.no_grad():
            for parameter in reviewer.parameters():
                parameter.fill_(.02)
        for variant in ('normal', 'permuted', 'constant', 'zero'):
            changed = altered_inputs(x, cell, eligible, variant)
            torch.testing.assert_close(changed[..., :7], x[..., :7], atol=0, rtol=0)
            torch.testing.assert_close(changed[~eligible], x[~eligible], atol=0, rtol=0)
            weights = fixed_mask_weights(reviewer, changed, original, eligible)
            torch.testing.assert_close(weights[~eligible], original[~eligible], atol=0, rtol=0)
            if variant == 'permuted':
                torch.testing.assert_close(changed[0, 0, -4:], x[0, 2, -4:])
                torch.testing.assert_close(changed[0, 0], changed[0, 1])
        # Empty support cannot produce an update, even with nonzero reviewer bias.
        empty = torch.zeros_like(eligible)
        torch.testing.assert_close(fixed_mask_weights(reviewer, x, original, empty), original, atol=0, rtol=0)
