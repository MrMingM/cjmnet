import unittest
import torch
from .bev_location import correct_bev


class FixedBEVTests(unittest.TestCase):
    def test_constant_ignores_scores_preserves_support_all_scales(self):
        levels = [torch.ones(1, 2, 8//s, 8//s) for s in (2, 4, 8)]
        p = {'voxel_coords': torch.tensor([[0, 0, 0, 0], [0, 0, 6, 6]])}
        supported = torch.tensor([[True, False], [False, False]])
        for c in (0., .05, .1):
            a, maps = correct_bev(levels, p, torch.randn(2, 2), supported, c)
            b, _ = correct_bev(levels, p, torch.randn(2, 2)*100, supported, c)
            for before, x, y, m in zip(levels, a, b, maps):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
                self.assertAlmostEqual(float(m[0, 0, 0, 0]), c)
                torch.testing.assert_close(x[m.expand_as(x) == 0], before[m.expand_as(x) == 0], atol=0, rtol=0)
            empty, _ = correct_bev(levels, p, torch.randn(2, 2), supported*False, c)
            for x, y in zip(empty, levels):
                torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_invalid_and_default(self):
        levels = [torch.ones(1, 1, 4, 4)]
        p = {'voxel_coords': torch.tensor([[0, 0, 0, 0]])}
        d, mask = torch.tensor([[.15]]), torch.tensor([[True]])
        a, _ = correct_bev(levels, p, d, mask)
        b, _ = correct_bev(levels, p, d, mask, None)
        torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
        for c in (float('nan'), float('inf'), .3):
            with self.assertRaises(ValueError):
                correct_bev(levels, p, d, mask, c)


if __name__ == '__main__':
    unittest.main()
