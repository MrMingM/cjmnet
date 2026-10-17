"""Server-only synthetic checks against the pinned original official AP code."""
import copy
import unittest

from support import server_only


def box(x=0., y=0.):
    # Four BEV corners followed by their top-face copies.
    return [[x-1, y-1, 0.], [x+1, y-1, 0.], [x+1, y+1, 0.], [x-1, y+1, 0.],
            [x-1, y-1, 2.], [x+1, y-1, 2.], [x+1, y+1, 2.], [x-1, y+1, 2.]]


class OfficialContract(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        from opencood.utils import eval_utils_opv2v
        self.eval = eval_utils_opv2v

    def test_none_contributes_gt_to_no_skip_denominator(self):
        from evaluate import statistics
        stat = statistics()
        gt = self.torch.tensor([box()])
        self.eval.caluclate_tp_fp(gt.clone(), self.torch.tensor([.9]), gt, stat, .7)
        before = self.eval.calculate_ap(copy.deepcopy(stat), .7, False)[0]
        self.eval.caluclate_tp_fp(None, None, gt, stat, .7)
        after = self.eval.calculate_ap(copy.deepcopy(stat), .7, False)[0]
        self.assertEqual(stat[.7]['gt'], 2)
        self.assertEqual((before, after), (1., .5))

    def test_empty_is_distinct_from_none_and_is_counted(self):
        from evaluate import statistics
        stat = statistics()
        self.eval.caluclate_tp_fp(self.torch.empty((0, 8, 3)), self.torch.empty(0), self.torch.tensor([box()]), stat, .3)
        self.assertEqual(stat[.3]['gt'], 1)
        self.assertEqual(stat[.3]['tp'], [])

    def test_one_prediction_cannot_match_removed_and_retained_twice(self):
        from evaluate import official_match_trace, statistics
        stat = statistics()
        pred, gt = self.torch.tensor([box()]), self.torch.tensor([box(), box(.2)])
        pairs, _ = official_match_trace(pred, self.torch.tensor([.9]), gt, ['removed', 'retained'], stat, .3)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['object_id'], 'removed')
        self.assertEqual(sum(stat[.3]['tp']), 1)

    def test_two_predictions_cannot_reuse_one_gt(self):
        from evaluate import official_match_trace, statistics
        stat = statistics()
        pairs, _ = official_match_trace(self.torch.tensor([box(), box()]), self.torch.tensor([.8, .9]),
            self.torch.tensor([box()]), ['g'], stat, .7)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['prediction_index'], 1)
        self.assertEqual(stat[.7]['tp'], [1, 0])
        self.assertEqual(stat[.7]['fp'], [0, 1])

    def test_observer_preserves_original_matching_flags(self):
        from evaluate import official_match_trace, statistics
        args = (self.torch.tensor([box(8), box(), box(.2)]), self.torch.tensor([.6, .9, .8]),
                self.torch.tensor([box(), box(12)]))
        plain, traced = statistics(), statistics()
        self.eval.caluclate_tp_fp(*args, plain, .5)
        official_match_trace(*args, ['a', 'b'], traced, .5)
        self.assertEqual(plain, traced)

    def test_scores_stay_in_original_order_before_official_sort(self):
        from evaluate import official_match_trace, statistics
        pairs, _ = official_match_trace(self.torch.tensor([box(), box(8)]), self.torch.tensor([.1, .9]),
            self.torch.tensor([box()]), ['g'], statistics(), .7)
        self.assertEqual(pairs[0]['prediction_index'], 0)

    def test_fixed_gt_only_adds_clean_selected_targets(self):
        from evaluate import statistics
        official, fixed = statistics(), statistics()
        pred, scores = self.torch.tensor([box()]), self.torch.tensor([.9])
        self.eval.caluclate_tp_fp(pred, scores, self.torch.tensor([box()]), official, .7)
        self.eval.caluclate_tp_fp(pred, scores, self.torch.tensor([box(), box(12)]), fixed, .7)
        self.assertEqual(self.eval.calculate_ap(copy.deepcopy(official), .7, False)[0], 1.)
        self.assertEqual(self.eval.calculate_ap(copy.deepcopy(fixed), .7, False)[0], .5)


def run_tests():
    server_only()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OfficialContract))
    if not result.wasSuccessful():
        raise ValueError('Synthetic checks of official AP/one-to-one trace failed')
