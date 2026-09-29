"""Small synthetic policy tests; no dataset, detector or experiment is run."""
import unittest

import numpy as np

from .candidate_ranker_rescue import (
    METHODS, PRIMARY_THRESHOLD, _eligible_edges, _gate,
    _pair_probability, _single_swap,
)


class CandidateRankerRescueTests(unittest.TestCase):
    def test_only_direct_selected_multi_source_score_crossing_is_eligible(self):
        rows = [
            {'score': .30, 'within_range': True, 'source_count': 2},
            {'score': .10, 'within_range': True, 'source_count': 2},
            {'score': .11, 'within_range': True, 'source_count': 1},
            {'score': .32, 'within_range': True, 'source_count': 2},
        ]
        fixed = {
            'selected': {'top256_fused': [0]},
            'suppressors': {1: (0, .2), 2: (0, .2), 3: (0, .2)},
        }
        self.assertEqual(_eligible_edges(rows, fixed), [(1, 0)])

    def test_policy_swaps_one_pair_only_above_fixed_threshold(self):
        order = np.asarray([0, 2, 1, 3])
        probabilities = [.1, .9, .2, .8]
        revised, edit, choices = _single_swap(
            order, [(1, 0), (3, 2)], probabilities, PRIMARY_THRESHOLD)
        np.testing.assert_array_equal(revised, [1, 2, 0, 3])
        np.testing.assert_array_equal(order, [0, 2, 1, 3])
        self.assertEqual(edit[:2], (1, 0))
        self.assertEqual(len(choices), 2)
        untouched, no_edit, _ = _single_swap(
            order, [(1, 0)], probabilities, .999)
        np.testing.assert_array_equal(untouched, order)
        self.assertIsNone(no_edit)
        self.assertGreater(_pair_probability(.9, .1), .9)

    def test_gate_requires_ap_controls_scene_robustness_and_net_gt(self):
        def metric(ap, net=1):
            return {
                'ap_frame_order': {'ap70': ap},
                'leave_one_scene_out_ap70_frame': {'0': ap, '1': ap},
                'new_matched_gt_vs_top256': max(0, net),
                'lost_matched_gt_vs_top256': max(0, -net),
            }
        report = {
            'protocol': {'seeds': [1, 2]},
            'baselines': {weather: {'scorepass': metric(.5)}
                          for weather in ('clean', 'fog', 'rain', 'snow')},
            'seeds': {},
        }
        for seed in (1, 2):
            report['seeds'][str(seed)] = {}
            for weather in ('clean', 'fog', 'rain', 'snow'):
                proposed = .5 if weather == 'clean' else .52
                control = .5 if weather == 'clean' else .51
                report['seeds'][str(seed)][weather] = {'methods': {
                    **{name: {'thresholds': {'0.90': metric(control)}}
                       for name in METHODS[:-1]},
                    'candidate_plus_distortion': {
                        'thresholds': {'0.90': metric(proposed)}},
                }}
        self.assertTrue(_gate(report)['rescue_gate_pass'])
        for weather in ('fog', 'rain'):
            row = report['seeds']['2'][weather]['methods'][
                'candidate_plus_distortion']['thresholds']['0.90']
            row['leave_one_scene_out_ap70_frame']['1'] = .49
        self.assertFalse(_gate(report)['rescue_gate_pass'])


if __name__ == '__main__':
    unittest.main()
