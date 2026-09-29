"""Synthetic ordering tests; no detector, dataset, or research log is loaded."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .candidate_nms_scope import (
    _is_high_bad, _is_low_good, _load, _nms_ordered, _oracle_order,
    _suppression_components,
)


class CandidateNmsScopeTests(unittest.TestCase):
    def test_first_actual_suppressor_and_targeted_oracle_rank(self):
        # 0 is a high-score bad box, 2 is a low-score good box. Box 1 is
        # unrelated, and its occupied rank position must stay fixed.
        overlap_matrix = np.asarray([[1., 0., .7],
                                     [0., 1., 0.],
                                     [.7, 0., 1.]])
        overlap = lambda index, others: overlap_matrix[index, others]
        original = np.asarray([0, 1, 2])
        picked, suppressors = _nms_ordered(original, overlap, .15)
        self.assertEqual(picked, [0, 1])
        self.assertEqual(suppressors, {2: (0, .7)})
        self.assertEqual(_suppression_components(suppressors), {0: [0, 2]})
        oracle, groups, moved = _oracle_order(
            original, suppressors, [.2, .4, .8], [True] * 3, {0})
        np.testing.assert_array_equal(oracle, [2, 1, 0])
        self.assertEqual((groups, moved), (1, 2))
        new_picked, new_suppressed = _nms_ordered(oracle, overlap, .15)
        self.assertEqual(new_picked, [2, 1])
        self.assertEqual(new_suppressed[0][0], 2)

    def test_overlap_does_not_imply_bad_directly_suppresses_good(self):
        # The first picked good box removes both later boxes. A pairwise
        # high-bad/low-good overlap is not a bad-to-good suppression edge.
        matrix = np.asarray([[1., .8, .8],
                             [.8, 1., .8],
                             [.8, .8, 1.]])
        picked, suppressors = _nms_ordered(
            [0, 1, 2], lambda index, others: matrix[index, others], .15)
        self.assertEqual(picked, [0])
        self.assertEqual({loser: winner for loser, (winner, _) in suppressors.items()},
                         {1: 0, 2: 0})
        oracle, used, moved = _oracle_order(
            [0, 1, 2], suppressors, [.9, .2, .8], [True] * 3, set())
        np.testing.assert_array_equal(oracle, [0, 1, 2])
        self.assertEqual((used, moved), (0, 0))

    def test_range_and_gt_definitions_are_explicit(self):
        low_good = {'within_range': True, 'score': .2, 'quality': .7}
        high_bad = {'within_range': True, 'score': .21, 'quality': .49}
        self.assertTrue(_is_low_good(low_good))
        self.assertTrue(_is_high_bad(high_bad))
        self.assertFalse(_is_low_good(dict(low_good, within_range=False)))
        self.assertFalse(_is_high_bad(dict(high_bad, quality=.5)))

    def test_out_of_range_candidate_is_deprioritized_in_oracle(self):
        order, _, _ = _oracle_order(
            [0, 1], {1: (0, .5)}, [.99, .8], [False, True], {0})
        np.testing.assert_array_equal(order, [1, 0])

    def test_loader_rejects_incomplete_frame_without_running_experiment(self):
        corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meta = {
                'audit_pool': 'top_256', 'stage0_only': False,
                'validation_indices': [7], 'validation_scene_map': {'7': 0},
                'conditions': {weather: {
                    'original_score_threshold': .2, 'nms_iou_threshold': .15,
                    'pools': {'F': {
                        'top_256': {'total_candidates': 2},
                        'original': {'total_candidates': 1},
                    }},
                } for weather in ('clean', 'fog', 'rain', 'snow')},
            }
            (root / 'candidate_audit.json').write_text(json.dumps(meta), encoding='utf-8')
            for weather in ('clean', 'fog', 'rain', 'snow'):
                folder = root / weather
                folder.mkdir()
                target = {'weather': weather, 'sample_index': 7,
                          'gt_bev_corners': [corners]}
                (folder / 'frame_targets.jsonl').write_text(
                    json.dumps(target) + '\n', encoding='utf-8')
                rows = [dict(weather=weather, arm='F', sample_index=7,
                             candidate_id=index, fused_bev_corners=corners,
                             score=score, max_gt_iou=1., within_range=True)
                        for index, score in enumerate((.8, .1))]
                (folder / 'candidate_rows.jsonl').write_text(
                    ''.join(json.dumps(row) + '\n' for row in rows),
                    encoding='utf-8')
            _, conditions, hashes = _load(root, 'F')
            self.assertEqual(len(conditions['clean']['rows'][7]), 2)
            self.assertEqual(len(hashes), 9)
            (root / 'snow' / 'candidate_rows.jsonl').write_text('', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '1–256 candidates'):
                _load(root, 'F')


if __name__ == '__main__':
    unittest.main()
