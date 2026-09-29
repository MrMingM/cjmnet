import unittest

import numpy as np
import torch

from SAQC.adapter import (
    anchor_cells,
    decoded_center_cells,
    extract_patches,
)
from SAQC.core import (
    paper_fused_score,
    platt_calibrate,
    quality_ece,
    spearman,
)
from SAQC.model import (
    SpatialQualityHead,
    relative_coordinate_channels,
)


class TestSAQC(unittest.TestCase):
    def test_coordinates(self):
        value = relative_coordinate_channels(
            2,
            7,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        self.assertEqual(
            tuple(value.shape), (2, 2, 7, 7))
        self.assertEqual(
            float(value[0, 0, 0, 0]), -3.)
        self.assertEqual(
            float(value[0, 1, 3, 3]), 0.)
        self.assertEqual(
            float(value[0, 0, 6, 6]), 3.)

    def test_quality_head(self):
        head = SpatialQualityHead(
            4,
            hidden_channels=8,
            patch_size=7,
        )
        out = head(
            torch.randn(5, 4, 7, 7))
        self.assertEqual(
            tuple(out.shape), (5,))
        self.assertTrue(bool(
            ((out >= 0) & (out <= 1)).all()))

    def test_patch_zero_padding(self):
        feature = torch.arange(
            9,
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)
        patch = extract_patches(
            feature, [0], [0], 3)
        self.assertEqual(
            tuple(patch.shape),
            (1, 1, 3, 3),
        )
        self.assertEqual(
            float(patch[0, 0, 0, 0]), 0.)
        self.assertEqual(
            float(patch[0, 0, 1, 1]), 0.)
        self.assertEqual(
            float(patch[0, 0, 2, 2]), 4.)

    def test_anchor_and_decoded_mapping(self):
        psm = torch.zeros(1, 2, 2, 3)
        anchor = torch.zeros(
            1, 2, 3, 2, 7)
        for r in range(2):
            for c in range(3):
                anchor[
                    0, r, c, :, 0] = c * 2.
                anchor[
                    0, r, c, :, 1] = r * 3.

        decoded = torch.tensor(
            [[3.9, 2.9, 0, 1, 1, 1, 0]],
            dtype=torch.float32,
        )
        row, col, distance = (
            decoded_center_cells(
                decoded, anchor, psm))
        self.assertEqual(
            (int(row[0]), int(col[0])),
            (1, 2),
        )
        self.assertLess(
            float(distance[0]), .2)

        arow, acol = anchor_cells(
            [2 * (1 * 3 + 2)], psm)
        self.assertEqual(
            (int(arow[0]), int(acol[0])),
            (1, 2),
        )

    def test_score_and_calibration(self):
        score = np.array([.8, .2])
        quality = np.array([.25, 1.])
        fused = paper_fused_score(
            score, quality, 1.)
        np.testing.assert_allclose(
            fused, [.2, .2])

        calibrated = platt_calibrate(
            np.array([.2, .8]), 1., 0.)
        np.testing.assert_allclose(
            calibrated,
            [.2, .8],
            atol=1e-7,
        )
        self.assertAlmostEqual(
            spearman(
                [1, 2, 3],
                [10, 20, 30],
            ),
            1.,
        )
        self.assertAlmostEqual(
            quality_ece(
                [.2, .8],
                [.2, .8],
                bins=2,
            ),
            0.,
        )


if __name__ == '__main__':
    unittest.main()
