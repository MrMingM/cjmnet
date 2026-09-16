import unittest
import numpy as np
import torch
from .core import Settings, Source, inside_box, box_patch, box_evidence, make_actions
from .scoring import greedy, choose_hindsight, empty_stats, merge
from .replay import received_levels, fused_levels, apply_action, detect


def box(x=5., y=0., z=0., length=2., width=2., height=2.):
    return np.array([[x+dx*length/2, y+dy*width/2, z+dz*height/2]
        for dz in (-1, 1) for dx, dy in ((-1,-1),(-1,1),(1,1),(1,-1))], np.float32)


def source(xyz, opinions=None, raw=None, pose=None, mask=None):
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    if opinions is None:
        opinions = np.tile([.9, .05, .05], (len(xyz), 1))
    return Source(xyz, np.asarray(opinions), xyz if raw is None else np.asarray(raw),
        np.eye(4) if pose is None else pose, np.ones((2, 4)) if mask is None else mask,
        (2, 4), [0,-5,-3,20,5,3], Settings())


class GeometryTests(unittest.TestCase):
    def test_unknown_does_not_refute(self):
        s = source([])
        self.assertEqual(s.free_at(np.array([[5.,0,0]])).tolist(), [0])

    def test_true_segment_not_entire_box_or_occluded_region(self):
        s = source([[10,0,0]])
        np.testing.assert_allclose(s.free_at(np.array([[5.,0,0],[11,0,0],[5,1,0],[9.5,0,0]])), [.9,0,0,0])

    def test_reliability_not_borrowed_and_raw_nearer_return_blocks(self):
        s = source([[10,0,0]], raw=[[4,0,0],[10,0,0]])
        self.assertEqual(len(s.ranges), 0)
        unmatched = source([[10,0,0]], raw=[[11,0,0]])
        self.assertEqual(len(unmatched.ranges), 0)
        noisy = source([[10,0,0]], opinions=[[.1,.8,.1]])
        self.assertEqual(len(noisy.ranges), 0)

    def test_receipt_masks_endpoint_and_query(self):
        mask = np.ones((2,4)); mask[1,2] = 0
        self.assertEqual(len(source([[10,0,0]], mask=mask).xyz), 0)
        mask = np.ones((2,4)); mask[1,1] = 0
        s = source([[10,0,0]], mask=mask)
        self.assertEqual(len(s.xyz), 1)
        self.assertEqual(s.free_at(np.array([[5.,0,0]])).tolist(), [0])

    def test_true_sensor_origin(self):
        pose = np.eye(4); pose[0,3] = 2
        s = source([[12,0,0]], pose=pose)
        np.testing.assert_allclose(s.free_at(np.array([[7.,0,0],[1,0,0]])), [.9,0])

    def test_directed_conflict_vs_missing(self):
        weak = source([[5,0,0],[5,.05,0]], opinions=[[.3,.6,.1],[.3,.6,.1]])
        free = source([[10,0,0]])
        e = box_evidence(box(), [weak, free], Settings())
        self.assertEqual(e['conflict_counts'], [0,2])
        e = box_evidence(box(), [weak, source([])], Settings())
        self.assertEqual(e['conflict_counts'], [0,0])

    def test_box_height_rotation_and_patch(self):
        corners = box()
        self.assertEqual(inside_box(np.array([[5,0,0],[5,0,3],[9,0,0]]), corners).tolist(), [True,False,False])
        self.assertTrue(box_patch(corners, (2,4), [0,-5,-3,20,5,3]).any())
        self.assertFalse(box_patch(box(50), (2,4), [0,-5,-3,20,5,3]).any())

    def test_completion_proposals_need_received_support(self):
        s = [source([]), source([[5,0,0],[5,.1,0],[5,-.1,0]])]
        actions, _ = make_actions(np.empty((0,8,3)), np.empty(0),
            [(1,box()[None],np.array([.9]))], s, np.ones((2,1,2,4)), [0,-5,-3,20,5,3], Settings())
        self.assertTrue(actions[0]['eligible'])
        self.assertEqual(actions[0]['kind'], 'complete')
        masks = np.ones((2,1,2,4)); masks[1] = 0
        actions, _ = make_actions(np.empty((0,8,3)), np.empty(0),
            [(1,box()[None],np.array([.9]))], s, masks, [0,-5,-3,20,5,3], Settings())
        self.assertFalse(actions)


class ReplayTests(unittest.TestCase):
    def test_baseline_replay_matches_wire_and_patch_only(self):
        from gspr_evidence.test_evidence import toy_model
        model, encoded = toy_model()
        masks = torch.ones_like(model.empty_masks(encoded)); masks[1,0,1,1] = 0
        received = received_levels(encoded, masks, 4)
        fused = fused_levels(received, masks)
        real, _ = model.detect(encoded, masks, True)
        replay = detect(model.engine.base, fused)
        for key in real:
            torch.testing.assert_close(real[key], replay[key])
        region = np.zeros((2,2), bool); region[0,0] = True
        before = [x.clone() for x in fused]
        changed = apply_action(fused, received, masks, dict(peer=1, region=region))
        for old, z, original, peer in zip(before, changed, fused, received):
            h,w = z.shape[-2:]
            torch.testing.assert_close(z[:,:,:h//2,:w//2], peer[1:2,:,:h//2,:w//2])
            torch.testing.assert_close(z[:,:,h//2:], old[:,:,h//2:])
            torch.testing.assert_close(original, old)
        region[1,1] = True
        with self.assertRaises(ValueError):
            apply_action(fused, received, masks, dict(peer=1, region=region))

    def test_unreceived_payload_cannot_affect_replay(self):
        encoded = dict(levels=[torch.randn(2,2,4,4)])
        masks = torch.ones(2,1,2,2); masks[1,0,1,1] = 0
        a = received_levels(encoded, masks, 2)
        encoded['levels'][0][1,:,2:,2:] = 99999
        b = received_levels(encoded, masks, 2)
        torch.testing.assert_close(a[0], b[0])


class OracleTests(unittest.TestCase):
    def test_greedy_duplicate_and_empty(self):
        iou = np.array([[.8,0],[.9,0],[0,.75]])
        np.testing.assert_equal(greedy(iou,np.array([.9,.8,.7])), [0,-1,1])
        self.assertEqual(greedy(np.zeros((2,0)),np.array([.8,.7])).tolist(), [-1,-1])

    def test_hindsight_preserves_identities_and_fp(self):
        rows = [dict(matched={1,2},fp=0,eligible=True),
                dict(matched={0,1},fp=2,eligible=True),
                dict(matched={0,1},fp=1,eligible=False),
                dict(matched={0},fp=0,eligible=True)]
        self.assertEqual(choose_hindsight({0},1,rows,True),3)
        self.assertEqual(choose_hindsight({0},1,rows,False),2)
        self.assertIsNone(choose_hindsight({0},1,rows[:2],True))

    def test_ap_records_not_aliased(self):
        a,b = empty_stats(), empty_stats(); b[.7]['tp'] = [1]
        merge(a,b); a[.7]['tp'][0] = 0
        self.assertEqual(b[.7]['tp'], [1])


if __name__ == '__main__':
    unittest.main()
