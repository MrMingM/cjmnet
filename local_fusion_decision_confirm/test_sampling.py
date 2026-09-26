import unittest

from .sampling import blocked_frames, holdout_frames


class SamplingTests(unittest.TestCase):
    def test_holdout_prefers_unseen_scenes_and_excludes_nearby_frames(self):
        args = ([10, 20, 30, 40], [0, 2], [2, 22], 4, 2, 3, 1)
        first = holdout_frames(*args)
        self.assertEqual(first, holdout_frames(*args))
        self.assertEqual(first[0], [1, 3])
        self.assertEqual(len(first[1]), 6)
        self.assertTrue(all(10 <= i < 20 or 30 <= i < 40 for i in first[1]))

    def test_all_scenes_used_still_keeps_new_frames(self):
        ends, pilot_scenes, pilot_indices = [10, 20], [0, 1], [2, 12]
        scenes, indices = holdout_frames(ends,pilot_scenes,pilot_indices,
                                         4, 2, 4, 1)
        self.assertEqual(scenes,[0,1])
        self.assertEqual(len(indices),8)
        self.assertFalse(set(indices) & blocked_frames(
            ends,pilot_scenes,pilot_indices,1))


if __name__ == '__main__':
    unittest.main()
