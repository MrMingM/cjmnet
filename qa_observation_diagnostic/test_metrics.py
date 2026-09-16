import unittest
import numpy as np
from qa_observation_diagnostic.metrics import pearson, spearman, partial_corr, binary_auc, rankdata


class MetricsTest(unittest.TestCase):
    def test_correlations(self):
        x = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(pearson(x, x), 1.0)
        self.assertAlmostEqual(spearman(x, x), 1.0)
        self.assertTrue(np.allclose(rankdata([1, 2, 2, 4]), [1, 2.5, 2.5, 4]))

    def test_auc(self):
        self.assertAlmostEqual(binary_auc([.1, .2, .8, .9], [0, 0, 1, 1]), 1.0)

    def test_partial(self):
        z = np.arange(20, dtype=float)
        x = 2*z + np.sin(z)
        y = -3*z + np.cos(z)
        value = partial_corr(x, y, z[:, None])
        self.assertTrue(np.isfinite(value))


if __name__ == '__main__':
    unittest.main()
