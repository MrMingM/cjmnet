import unittest

from .candidate_pair_distortion_audit import _group, _metrics, _summarize


def raw_row(score, quality):
    return {
        'score': score, 'max_gt_iou': quality, 'source_count': 2,
        'source_scores_same_anchor': [.12, .18],
        'source_box_iou_same_anchor': [.8, .7],
        'source_box_center_shift_to_fused': [.2, .4],
        'source_box_pairwise_iou_mean': .75,
        'score_source': 1, 'geometry_source': 0,
        'source_proxy_disagreement': True,
        'leave_one_source_out': {'1': {
            'logit_drop': .2, 'score_drop': .03, 'box_iou_to_full': .9,
            'center_shift': .1, 'size_shift_l1': .1, 'yaw_shift_abs': .1,
        }},
    }


class DirectPairAuditTests(unittest.TestCase):
    def test_metrics_do_not_use_gt_quality(self):
        row = raw_row(.1, .8)
        features = _metrics(row)
        row['max_gt_iou'] = .2
        self.assertEqual(features, _metrics(row))
        self.assertEqual(_group(raw_row(.1, .8)), 'low_good')
        self.assertEqual(_group(raw_row(.7, .2)), 'high_bad')

    def test_pair_summary_keeps_raw_orientation_and_scene_interval(self):
        good = {'sample_index': 1, 'group': 'low_good',
                'metrics': _metrics(raw_row(.1, .8))}
        bad = {'sample_index': 1, 'group': 'high_bad',
               'metrics': _metrics(raw_row(.7, .2))}
        summary = _summarize([good, bad], [(good, bad, .2)], {'1': 'scene_1'}, 5, 7)
        self.assertEqual(summary['overlapping_pairs'], 1)
        self.assertEqual(summary['metrics']['fused_score']
                         ['overlap_pair_good_higher_rate_ties_half'], 0)
        self.assertEqual(summary['metrics']['mean_source_pairwise_iou']
                         ['overlap_pair_good_higher_rate_ties_half'], .5)
        self.assertEqual(summary['metrics']['source_agreement']
                         ['overlap_pair_rate_scene_bootstrap_95pct'], [.5, .5])


if __name__ == '__main__':
    unittest.main()
