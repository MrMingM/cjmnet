import unittest

import numpy as np

from .candidate_discriminability import (
    MODELS, _bev_iou, _features, _group, competition, paired_weather, probe,
    probe_or_insufficient,
)


def candidate(frame, candidate_id, score, quality, scores, ious, corners=None):
    raw = {
        'score': score, 'max_gt_iou': quality, 'source_count': len(scores),
        'source_scores_same_anchor': scores,
        'source_box_iou_same_anchor': ious,
        'score_source': max(range(len(scores)), key=scores.__getitem__),
        'geometry_source': max(range(len(ious)), key=ious.__getitem__),
    }
    raw['source_proxy_disagreement'] = raw['score_source'] != raw['geometry_source']
    features, _ = _features(raw)
    return {'key': ('snow', 'F', frame, candidate_id), 'sample_index': frame,
            'candidate_id': candidate_id, 'quality': quality,
            'group': _group(score, quality), 'features': features,
            'corners': corners}


class CandidateDiscriminabilityTests(unittest.TestCase):
    def test_labels_never_enter_inference_features(self):
        good = candidate(1, 1, .1, .8, [.05, .12], [.6, .9])
        bad = candidate(1, 2, .1, .2, [.05, .12], [.6, .9])
        self.assertEqual(good['features'], bad['features'])
        self.assertEqual(good['group'], 'low_good')
        self.assertEqual(bad['group'], 'low_bad')
        self.assertFalse(any('gt' in name or 'quality' in name
                             for columns in MODELS.values() for name in columns))

    def test_score_conditioning_and_frame_disjoint_folds(self):
        rows = []
        for frame in range(10):
            offset = frame / 1000
            rows.extend((
                candidate(frame, 1, .10 + offset, .8, [.08, .16], [.65, .90]),
                candidate(frame, 2, .11 + offset, .3, [.09, .10], [.50, .70]),
                candidate(frame, 3, .60 + offset, .8, [.52, .67], [.75, .92]),
                candidate(frame, 4, .61 + offset, .3, [.50, .65], [.49, .74]),
            ))
        low, _ = probe(rows, 'low', {}, 5, 5, 123)
        high, _ = probe(rows, 'high', {}, 5, 5, 123)
        self.assertEqual((low['good'], low['bad']), (10, 10))
        self.assertEqual((high['good'], high['bad']), (10, 10))
        fold_groups = low['test_groups_by_fold']
        self.assertEqual({item for fold in fold_groups for item in fold},
                         {str(frame) for frame in range(10)})
        self.assertEqual(sum(len(fold) for fold in fold_groups), 10)

    def test_nms_pair_uses_actual_fused_geometry(self):
        first = [[0, 0], [1, 0], [1, 1], [0, 1]]
        second = [[.2, 0], [1.2, 0], [1.2, 1], [.2, 1]]
        self.assertAlmostEqual(_bev_iou(np.asarray(first), np.asarray(second)), 2 / 3)
        self.assertAlmostEqual(_bev_iou(np.asarray(first), np.asarray(second[::-1])), 2 / 3)
        good = candidate(1, 1, .1, .8, [.08, .12], [.7, .9], first)
        bad = candidate(1, 2, .7, .2, [.4, .6], [.7, .8], second)
        oof = {
            good['key']: {name: .9 for name in MODELS},
            bad['key']: {name: .1 for name in MODELS},
        }
        result = competition([good, bad], oof, .15)
        self.assertEqual(result['pair_counts']['low_good_high_bad'], 1)
        self.assertEqual(result['correct_order_rate']['low_good_high_bad']['fused_score'], 0)
        self.assertEqual(result['correct_order_rate']['low_good_high_bad']
                         ['score_geometry_distortion_oof'], 1)
        self.assertEqual(competition([dict(good, corners=None)], None, .15)
                         ['status'], 'unavailable')

    def test_weather_pair_requires_same_anchor_and_source_count(self):
        clean = [candidate(1, 1, .1, .8, [.08, .12], [.7, .9]),
                 candidate(1, 2, .1, .8, [.08, .12], [.7, .9])]
        snow = [candidate(1, 1, .09, .8, [.08, .12], [.65, .85]),
                candidate(1, 2, .09, .8, [.08, .12], [.65, .85])]
        for row in clean:
            row['key'] = ('clean', 'F', row['sample_index'], row['candidate_id'])
        snow[1]['features']['source_count'] = 3
        result = paired_weather({'clean': {'F': clean}, 'snow': {'F': snow}},
                                'F', 'snow', {'1': 'scene_1'}, 5, 7)
        self.assertEqual(result['matched_candidates'], 1)
        self.assertEqual(result['different_source_count_excluded'], 1)
        self.assertAlmostEqual(result['weather_minus_clean']['fused_score']['median'], -.01)

    def test_sparse_band_is_reported_without_hiding_nms_pairs(self):
        good = candidate(1, 1, .1, .8, [.08, .12], [.7, .9],
                         [[0, 0], [1, 0], [1, 1], [0, 1]])
        bad = candidate(1, 2, .7, .2, [.4, .6], [.7, .8],
                        [[.2, 0], [1.2, 0], [1.2, 1], [.2, 1]])
        result, scores = probe_or_insufficient([good, bad], 'low', {}, 5, 5, 7)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIsNone(scores)
        pairs = competition([good, bad], scores, .15)
        self.assertEqual(pairs['pair_counts']['low_good_high_bad'], 1)


if __name__ == '__main__':
    unittest.main()
