import math
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class RepairPureTests(unittest.TestCase):
    def _config(self):
        return {
            "candidate": {
                "max_sources": 2,
                "topk_per_source": 2,
                "max_candidates": 3,
                "score_floor": 0.01,
                "context_radius": 1,
                "positive_iou": 0.5,
                "negative_iou": 0.2,
            },
            "repair": {
                "center_scale_m": [8.0, 8.0, 2.0],
                "max_log_size": 0.7,
                "max_yaw_delta": math.pi / 2,
            },
        }

    def test_geometry_roundtrip_and_yaw_wrap(self):
        from .pipeline import geometry_residual, apply_geometry_residual
        cfg = self._config()["repair"]
        base = torch.tensor([[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 3.10]])
        target = torch.tensor([[1.5, 1.5, 0.2, 4.4, 1.8, 1.6, -3.10]])
        residual = geometry_residual(base, target, cfg)
        restored = apply_geometry_residual(base, residual)
        torch.testing.assert_close(restored[:, :6], target[:, :6], atol=1e-5, rtol=1e-5)
        yaw_error = torch.atan2(
            torch.sin(restored[:, 6] - target[:, 6]),
            torch.cos(restored[:, 6] - target[:, 6]),
        ).abs()
        self.assertLess(float(yaw_error.max()), 1e-5)
        self.assertTrue(bool((restored[:, 3:6] > 0).all()))

    def test_model_only_candidate_union_deduplicates_and_caps(self):
        from types import SimpleNamespace
        from .pipeline import generate_candidates

        class FakePost:
            def delta_to_boxes3d(self, rm, anchors):
                return rm.permute(0, 2, 3, 1).reshape(1, -1, 7)

        def pred(scores):
            logits = torch.logit(torch.tensor(scores).clamp(1e-4, 1 - 1e-4))
            psm = logits.reshape(1, 2, 2, 2).permute(0, 3, 1, 2).contiguous()
            rm = torch.zeros((1, 14, 2, 2))
            return {"psm": psm, "rm": rm}

        # Flat order is y,x,anchor. Peer proposes anchor 5 while full gives it low score.
        full = pred([0.9, 0.8, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
        peer = pred([0.9, 0.01, 0.01, 0.01, 0.01, 0.85, 0.01, 0.01])
        cfg = self._config()
        ds = SimpleNamespace(post_processor=FakePost())
        batch = {"ego": {"anchor_box": torch.zeros((1, 2, 2, 2, 7))}}
        result = generate_candidates(
            ds.post_processor, batch["ego"]["anchor_box"],
            full, [peer], cfg, [-10, -10, -3, 10, 10, 1]
        )
        ids = result["candidate_ids"].tolist()
        self.assertIn(5, ids)
        self.assertLessEqual(len(ids), cfg["candidate"]["max_candidates"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(result["local_region_count"], len(ids))


    def test_gt_alignment_handles_opencood_float64_centers(self):
        from types import SimpleNamespace
        from opencood.utils import box_utils
        from .pipeline import aligned_gt_centers

        centers = torch.tensor(
            [[[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]]],
            dtype=torch.float64,
        )
        mask = torch.tensor([[1.0]], dtype=torch.float64)
        transform = torch.eye(4, dtype=torch.float32)
        gt = box_utils.boxes_to_corners_3d(
            centers[0].float(), order="hwl"
        )
        ds = SimpleNamespace(
            post_processor=SimpleNamespace(params={"order": "hwl"})
        )
        batch = {"ego": {
            "object_bbx_center": centers,
            "object_bbx_mask": mask,
            "transformation_matrix": transform,
        }}
        reference = torch.zeros((1, 7), dtype=torch.float32)
        aligned = aligned_gt_centers(ds, batch, gt, reference=reference)
        self.assertEqual(aligned.dtype, torch.float32)
        torch.testing.assert_close(aligned, centers[0].float())

    def test_scene_split_is_disjoint_and_deterministic(self):
        from .runtime import scene_split
        a, b = scene_split(43, 0.8, 20260920)
        a2, b2 = scene_split(43, 0.8, 20260920)
        self.assertEqual((a, b), (a2, b2))
        self.assertFalse(set(a) & set(b))
        self.assertEqual(sorted(a + b), list(range(43)))
        self.assertGreater(len(a), len(b))

    def test_freeze_and_optimizer_scope(self):
        from .runtime import assert_frozen, optimizer_only
        frontend = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)
        )
        frontend.requires_grad_(False)
        frontend.eval()
        assert_frozen(frontend)
        repair = torch.nn.Linear(4, 1)
        optimizer = torch.optim.AdamW(repair.parameters())
        optimizer_only(optimizer, [repair])
        optimizer.param_groups[0]["params"].append(next(frontend.parameters()))
        with self.assertRaises(AssertionError):
            optimizer_only(optimizer, [repair])

    def test_selector_feature_dimension(self):
        from .pipeline import candidate_feature_dim, selector_feature_dim
        cfg = self._config()
        self.assertEqual(
            selector_feature_dim(cfg), candidate_feature_dim(cfg) + 12
        )


if __name__ == "__main__":
    unittest.main()
