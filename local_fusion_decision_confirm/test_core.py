import unittest

from .core import policy_rows


class ConfirmationTests(unittest.TestCase):
    def test_keep_comparator_matches_count_and_avoids_tile_conflicts(self):
        actions = [(0, 1), (0, 2), (1, 1), (2, 1)]
        scores = {'learned': [.8, .7, -.2, .4],
                  'confidence': [.1, .9, .8, .7]}
        rows = policy_rows(scores, actions, [1, 4])
        self.assertEqual(rows['learned_keep_top4'], [0, 3])
        self.assertEqual(rows['confidence_matched_keep'], [1, 2])
        self.assertEqual(len(rows['learned_keep_top4']),
                         len(rows['confidence_matched_keep']))
        self.assertEqual(rows['learned_top4'], [0, 3, 2])


if __name__ == '__main__':
    unittest.main()
