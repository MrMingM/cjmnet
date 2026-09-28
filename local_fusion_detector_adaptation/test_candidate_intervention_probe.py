import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .candidate_intervention_probe import (
    INTERVENTION_NAMES, WEATHERS, _candidate_features,
    _intervention_features, _label,
    load_rows, probe,
)


def example(score=.1, quality=.8, candidate_id=1):
    return {
        'weather': 'snow', 'arm': 'F', 'sample_index': 1,
        'candidate_id': candidate_id, 'source_count': 2,
        'score': score, 'fused_logit': float(np.log(score / (1 - score))),
        'fused_regression_deltas': [0.] * 7,
        'fused_decoded_box': [1., 2., 0., 4., 2., 1., 0.],
        'max_gt_iou': quality, 'within_range': True,
        'source_scores_same_anchor': [.12, .15],
        'source_box_iou_same_anchor': [.7, .8],
        'score_source': 1, 'geometry_source': 1,
        'source_proxy_disagreement': False,
        'fused_feature_cache': 'F_features_1.npz',
        'leave_one_source_out': {'1': {
            'logit_drop': .3, 'score_drop': .02,
            'box_iou_to_full': .9, 'center_shift': .1,
            'size_shift_l1': .2, 'yaw_shift_abs': .01,
            'gt_quality_change': 999.,
        }},
    }


class CandidateInterventionProbeTests(unittest.TestCase):
    def test_gt_does_not_enter_inference_features(self):
        first = example(quality=.8)
        second = example(quality=.2)
        second['leave_one_source_out']['1']['gt_quality_change'] = -999.
        np.testing.assert_array_equal(_candidate_features(first, [1., 2.]),
                                      _candidate_features(second, [1., 2.]))
        np.testing.assert_array_equal(_intervention_features(first),
                                      _intervention_features(second))
        self.assertEqual(len(_intervention_features(first)), len(INTERVENTION_NAMES))
        self.assertEqual(_label(first['score'], first['max_gt_iou']), 'low_good')
        self.assertEqual(_label(second['score'], second['max_gt_iou']), 'low_bad')

    def test_each_peer_is_required(self):
        row = example()
        row['source_count'] = 3
        with self.assertRaisesRegex(ValueError, 'every peer'):
            _intervention_features(row)

    def test_grouped_probe_keeps_frames_disjoint(self):
        rows = []
        for frame in range(10):
            for good in (False, True):
                raw = example(score=.1 + frame * .0001,
                              quality=.8 if good else .2)
                rows.append({
                    'sample_index': frame,
                    'group': 'low_good' if good else 'low_bad',
                    'candidate': _candidate_features(raw, [float(frame), float(good)]),
                    'proxy': np.asarray([float(good)]),
                    'intervention': _intervention_features(raw),
                })
        result = probe(rows, 'low', {}, 5, 5, 7)
        self.assertEqual(result['status'], 'available')
        self.assertEqual({group for fold in result['test_groups_by_fold'] for group in fold},
                         {str(frame) for frame in range(10)})

    def test_cache_and_anchor_ids_are_joined_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {'audit_pool': 'top_256', 'stage0_only': False,
                        'candidate_feature_cache': True, 'ablation_arm': 'F',
                        'conditions': {weather: {'ablation_frames': [1]}
                                       for weather in WEATHERS}}
            for weather in WEATHERS:
                folder = root / weather
                folder.mkdir()
                np.savez_compressed(folder / 'F_features_1.npz',
                                    candidate_id=np.asarray([2, 1]),
                                    feature=np.asarray([[9., 8.], [1., 2.]],
                                                       dtype=np.float16))
                row = example(candidate_id=1)
                row['weather'] = weather
                (folder / 'candidate_rows.jsonl').write_text(
                    json.dumps(row) + '\n', encoding='utf-8')
            rows, _ = load_rows(root, metadata, 'F')
            for weather in WEATHERS:
                np.testing.assert_array_equal(rows[weather][0]['candidate'][-2:],
                                              [1., 2.])


if __name__ == '__main__':
    unittest.main()
