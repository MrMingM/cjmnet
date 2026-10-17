"""Server-only synthetic checks of scientific accounting; no real data or detector."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
from common import server_only
from report import compare_frame, geometry, summarize


def condition(kept, counts):
    candidates = ['a', 'b']
    boxes = [[10., 0., -1., 1.5, 2., 4., 0.], [20., 0., -1., 1.5, 2., 4., 0.]]
    return dict(context=dict(ego='1', participating_cavs=['1'], cavs=[dict(cav='1', lidar_file='condition.pcd')]),
        raw_yaml=[dict(cav='1', object_ids=candidates, count=2)],
        cavs=[dict(cav='1', ego=True, before_point_filter_ids=candidates,
            before_point_filter_boxes=boxes, official_points_in_box=counts,
            physical_post_fov_points_in_box=counts, official_empty_voxel_fallback=False)],
        final_gt=[dict(object_id=k, box=boxes[candidates.index(k)], corners=[[boxes[candidates.index(k)][0], 0., 0.]] * 8) for k in kept],
        final_gt_count=len(kept))


def frame(clean, fog, snow):
    return dict(index=0, frame_key=['scene', '000001'], conditions=dict(clean=clean, fog=fog, snow=snow))


class Reports(unittest.TestCase):
    def test_equal_count_id_swap_is_changed(self):
        f = frame(condition(['a'], [3, 0]), condition(['b'], [0, 2]), condition(['a'], [3, 0]))
        removed = compare_frame(f)
        summary = summarize([f], removed)
        result = summary['comparisons']['fog']
        self.assertEqual((result['removed'], result['added'], result['net_fewer']), (1, 1, 0))
        self.assertEqual(result['equal_count_but_changed_frames'], 1)
        self.assertEqual(result['point_filter_share_of_removed'], 1)
        self.assertFalse(summary['all_conditions_GT_identical'])

    def test_geometry_change_with_same_id_and_count(self):
        clean = condition(['a'], [3, 0])
        fog = copy.deepcopy(clean)
        # Candidate geometry is unchanged; final representative is altered for this unit case.
        fog['final_gt'][0]['box'] = list(fog['final_gt'][0]['box'])
        fog['final_gt'][0]['box'][0] += .01
        f = frame(clean, fog, copy.deepcopy(clean))
        summary = summarize([f], compare_frame(f))
        self.assertEqual(summary['comparisons']['fog']['geometry_changed_objects'], 1)
        self.assertEqual(summary['comparisons']['fog']['equal_count_but_changed_frames'], 1)

    def test_yaw_period_and_tolerance(self):
        a = dict(box=[0., 0., 0., 1., 1., 1., 0.])
        b = dict(box=[0., 0., 0., 1., 1., 1., 2 * math.pi])
        self.assertFalse(geometry(a, b)['changed'])
        b['box'][6] += .001
        self.assertTrue(geometry(a, b)['changed'])

    def test_empty_frames_are_included(self):
        empty = condition([], [0, 0])
        f = frame(empty, copy.deepcopy(empty), copy.deepcopy(empty))
        summary = summarize([f], compare_frame(f))
        self.assertEqual(summary['validation_frame_count'], 1)
        self.assertEqual(summary['Clean_GT_total'], 0)
        self.assertEqual(summary['comparisons']['fog']['GT_unchanged_frames'], 1)
        self.assertIsNone(summary['comparisons']['fog']['net_fewer_percent_clean'])

    def test_peer_only_evidence_and_fallback(self):
        clean, fog = condition(['a'], [0, 0]), condition([], [0, 0])
        peer = copy.deepcopy(clean['cavs'][0])
        peer.update(cav='2', ego=False, official_points_in_box=[8, 0], physical_post_fov_points_in_box=[8, 0])
        clean['cavs'].append(peer)
        target_peer = copy.deepcopy(peer)
        target_peer.update(official_points_in_box=[0, 0], physical_post_fov_points_in_box=[0, 0], official_empty_voxel_fallback=True)
        fog['cavs'].append(target_peer)
        f = frame(clean, fog, copy.deepcopy(clean))
        losses = compare_frame(f)
        self.assertEqual(losses[0]['clean_evidence_split'], 'peer_only')
        self.assertEqual(losses[0]['clean_evidence']['best_cav_points'], 8)
        self.assertTrue(losses[0]['fallback_involved'])

    def test_input_context_changes_fail(self):
        clean = condition(['a'], [3, 0])
        fog = copy.deepcopy(clean)
        fog['cavs'][0]['before_point_filter_ids'] = ['b', 'a']
        with self.assertRaises(ValueError):
            compare_frame(frame(clean, fog, copy.deepcopy(clean)))


if __name__ == '__main__':
    server_only()
    unittest.main()
