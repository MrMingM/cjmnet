"""Small deterministic contract tests; full detector/AP regression runs on server."""
import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from .model import mapped_iou, rectified_score


class ScoreTests(unittest.TestCase):
    def test_author_score_equation(self):
        raw = np.asarray([-1., 0., 1.], dtype=np.float32)
        np.testing.assert_allclose(mapped_iou(raw), [0., .5, 1.])
        np.testing.assert_allclose(rectified_score(np.asarray([.8, .8, .8]), raw),
                                   [0., .8/16., .8])

    def test_author_mapping_is_not_clipped(self):
        np.testing.assert_allclose(mapped_iou([-2., 2.]), [-.5, 1.5])
        np.testing.assert_allclose(rectified_score([1., 1.], [-2., 2.]),
                                   [.5**4, 1.5**4])


class TorchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('PyTorch tests run in the server environment')
        cls.torch = torch

    def test_patch_is_identical_to_full_map_convolution_at_border(self):
        from .extract import candidate_patches
        from .model import make_head, patch_logits
        torch = self.torch
        joined = torch.arange(1., 1.+2*3*4).reshape(1, 2, 3, 4)
        head = make_head(2, 2)
        with torch.no_grad():
            head.weight.copy_(torch.arange(head.weight.numel()).reshape_as(head.weight)/100.)
        ids = np.asarray([0, 1, 6, 15, 22, 23])
        patches = candidate_patches(joined, ids, 2)
        got = patch_logits(head, patches, torch.tensor(ids % 2))
        full = head(joined).permute(0, 2, 3, 1).reshape(-1)
        torch.testing.assert_close(got, full[ids])

    def test_smooth_l1_sigma_three(self):
        from .model import weighted_smooth_l1
        torch = self.torch
        pred = torch.tensor([0., 1./9., 1.])
        target = torch.zeros_like(pred)
        torch.testing.assert_close(weighted_smooth_l1(pred, target),
                                   torch.tensor([0., 1./18., 17./18.]))

    def test_project_gt_corners_casts_to_transform_dtype(self):
        import torch
        from .extract import project_gt_corners

        corners = torch.zeros((1, 8, 3), dtype=torch.float64)
        transform = torch.eye(4, dtype=torch.float32)
        projected = project_gt_corners(corners, transform)
        self.assertEqual(projected.dtype, torch.float32)
        self.assertEqual(tuple(projected.shape), (1, 8, 3))

    def test_aligned_3d_iou(self):
        from .extract import aligned_iou3d
        # Four footprint vertices, then the same four at the upper face.
        xy = np.asarray([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
        def corners(bottom, top):
            return np.concatenate((np.column_stack((xy, np.full(4, bottom))),
                                   np.column_stack((xy, np.full(4, top)))))
        base = corners(0., 2.)
        half_height = corners(1., 3.)
        np.testing.assert_allclose(aligned_iou3d(np.stack((base, base)),
                                                   np.stack((base, half_height))),
                                   [1., 1./3.])

    def test_offline_loader_excludes_negative_anchors(self):
        from .train import _frame_batches, load_positive_patches
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates, patches = root / 'candidates', root / 'patches'
            candidates.mkdir()
            patches.mkdir()
            audit = {'split': 'train', 'sample_indices': [4, 9],
                     'scene_map': {'4': 0, '9': 1},
                     'arm_sha256': {'F': 'frozen-f'}}
            payload = json.dumps(audit).encode()
            (candidates / 'candidate_audit.json').write_bytes(payload)
            (patches / 'manifest.json').write_text(json.dumps({
                'split': 'train', 'arm_sha256': 'frozen-f',
                'candidate_audit_sha256': hashlib.sha256(payload).hexdigest()}))
            for weather in ('clean', 'fog', 'rain', 'snow'):
                (patches / weather).mkdir()
                for sample in (4, 9):
                    np.savez_compressed(patches / weather / f'F_patches_{sample}.npz',
                                        candidate_id=np.asarray([10, 11]),
                                        patch=np.ones((2, 3, 3, 3), dtype=np.float32),
                                        positive=np.asarray([True, False]),
                                        target_iou3d=np.asarray([.75, np.nan]))
            x, y, a, scenes, frames, weights, _, _ = load_positive_patches(candidates, patches)
            self.assertEqual(x.shape, (8, 3, 3, 3))
            np.testing.assert_allclose(y, .75)
            np.testing.assert_array_equal(a, 0)
            self.assertEqual(set(scenes), {0, 1})
            self.assertEqual(len(set(frames)), 8)
            np.testing.assert_allclose(weights, 1.)
            batches = list(_frame_batches(frames, np.arange(len(frames)), 3))
            self.assertEqual(sum(count for _, count in batches), 8)
            self.assertEqual(sum(len(ids) for ids, _ in batches), 8)


if __name__ == '__main__':
    unittest.main()
