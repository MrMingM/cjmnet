import unittest

import torch

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor

from .candidate_rescore import (assert_scorepass_replay, extract_candidates,
                                rescore_postprocess, source_score_variants,
                                continuation_screen, WEATHERS)


class CandidateRescoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.pp = VoxelPostprocessor({
            'anchor_args': {'num': 2}, 'order': 'hwl',
            'target_args': {'score_threshold': .2}, 'nms_thresh': .15,
        }, train=False)
        anchors = torch.zeros(1, 1, 2, 7)
        anchors[..., 2] = -1
        anchors[..., 3:6] = 1
        anchors[0, 0, 1, 0] = 10
        self.ego = {'anchor_box': anchors, 'transformation_matrix': torch.eye(4)}

    def test_fused_scorepass_replays_voxel_postprocessor(self):
        prediction = {
            'psm': torch.tensor([[[[1.4]], [[-.1]]]]),
            'rm': torch.zeros(1, 14, 1, 1),
        }
        reference = self.pp.post_process({'ego': self.ego}, {'ego': prediction})
        trace, pools = extract_candidates(self.pp, self.ego, prediction)
        ids = pools['scorepass']
        self.assertEqual(ids.tolist(), [0, 1])
        self.assertTrue(set(ids).issubset(set(pools['top256'])))
        gt = torch.empty((0, 8, 3))
        replay = rescore_postprocess(trace, ids, trace['scores'][ids], self.pp, gt)
        assert_scorepass_replay((*reference, gt), replay)

    def test_source_agreement_penalizes_misaligned_high_source_score(self):
        prediction = {'psm': torch.tensor([[[[.4]], [[.4]]]]),
                      'rm': torch.zeros(1, 14, 1, 1)}
        trace, pools = extract_candidates(self.pp, self.ego, prediction)
        ids = pools['scorepass']
        local = {
            'psm': torch.tensor([[[[1.3862944]], [[0.]]],
                                 [[[2.1972246]], [[0.]]]]),
            'rm': torch.zeros(2, 14, 1, 1),
        }
        # Move the second source's first-anchor box away from the fused box.
        local['rm'][1, 0, 0, 0] = 1.0
        variants = source_score_variants(trace, ids, local, self.pp, self.ego)
        self.assertAlmostEqual(float(variants['max_source'][0]), .9, places=5)
        self.assertAlmostEqual(float(variants['ego'][0]), .8, places=5)
        self.assertAlmostEqual(float(variants['source_agreement'][0]), .8, places=5)
        self.assertLess(float(variants['source_agreement'][0]),
                        float(variants['max_source'][0]))

    def test_continuation_screen_requires_gain_and_clean_preservation(self):
        conditions = {}
        for weather in WEATHERS:
            conditions[weather] = {'results': {'F': {'ap': {
                'original': {'ap70': .7, 'ap50': .8},
                'top256_max_source': {'ap70': .701, 'ap50': .8},
                'top256_source_agreement': {'ap70': .709, 'ap50': .8},
            }}}}
        self.assertTrue(continuation_screen(conditions)['continue'])
        conditions['clean']['results']['F']['ap']['top256_source_agreement']['ap70'] = .698
        self.assertFalse(continuation_screen(conditions)['continue'])


if __name__ == '__main__':
    unittest.main()
