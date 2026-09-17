import unittest

import numpy as np

from qa_evidence_validity.offline import _group_neff_change, jaccard


class OfflineAuditTest(unittest.TestCase):
    def test_jaccard(self):
        self.assertAlmostEqual(jaccard({1, 2}, {2, 3}), 1 / 3)
        self.assertIsNone(jaccard(set(), set()))

    def test_neff_zero_denominator_is_explicit(self):
        pairs = [
            ({"ego": {"box_neff": 0.0}}, {"ego": {"box_neff": 2.0}}),
            ({"ego": {"box_neff": 4.0}}, {"ego": {"box_neff": 2.0}}),
        ]
        out = _group_neff_change(pairs)
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["clean_neff_zero_n"], 1)
        self.assertAlmostEqual(out["clean_neff_zero_fraction"], 0.5)
        # Relative loss is computed only on the positive-clean sample: (4-2)/4=0.5.
        self.assertAlmostEqual(out["relative_loss_clean_positive_mean"], 0.5)
        self.assertTrue(np.isfinite(out["bounded_symmetric_delta_mean"]))
        self.assertGreaterEqual(out["bounded_symmetric_delta_mean"], -1.0)
        self.assertLessEqual(out["bounded_symmetric_delta_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
