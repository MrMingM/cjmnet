import copy
import unittest
from .analyze import analyze_weather


class AnalyzeTests(unittest.TestCase):
    def data(self, lost):
        summary = {'frames': 1, 'region_classes': {'harmful': 1, 'beneficial': 0, 'neutral': 0},
            'controls': {'a0b0': {'ap70': .8}, 'gt_single_deletion_joint': {
                'lost_vs_a0b0': 1, 'recovered_vs_a0b0': 0, 'ap70': .7}}}
        region = {'sample_index': 2, 'class': 'harmful', 'epsilon': .001,
            'L_send': {'total_loss': 1.}, 'L_drop': {'total_loss': .9},
            'delta_send_minus_drop': {'total_loss': .1}, 'removed_native_blocks': 1,
            'lost_after_drop_iou70': lost, 'recovered_after_drop_iou70': 0,
            'fp_change_after_drop': 0, 'sender_id': 'car', 'blocks_yx': [[0, 0]]}
        frames = [{'sample_index': 2, 'controls': {'a0b0': {'loss': {'total_loss': 1.}},
            'gt_single_deletion_joint': {'loss': {'total_loss': .9}, 'removed_sender_blocks': [[1, 0]]}}}]
        return summary, [region], frames

    def test_single_conflict_is_not_attributed_only_to_interactions(self):
        result = analyze_weather(*self.data(1))
        self.assertEqual(result['single_loss_conflict_fraction'], 1.)
        self.assertIn('单独删除', result['interpretation'])
        self.assertAlmostEqual(result['per_frame_loss_interactions'][0]['joint_minus_sum_single'], 0.)

    def test_joint_only_target_loss(self):
        result = analyze_weather(*self.data(0))
        self.assertEqual(result['harmful_single_deletions_losing_targets'], 0)
        self.assertIn('联合删除才出现', result['interpretation'])

    def test_reject_inconsistent_records(self):
        data = self.data(0)
        for field, value in [('class', 'beneficial'), ('removed_native_blocks', 2)]:
            bad = copy.deepcopy(data)
            bad[1][0][field] = value
            with self.assertRaises(ValueError):
                analyze_weather(*bad)

    def test_new_target_ids_distinguish_joint_only_losses(self):
        summary, regions, frames = self.data(1)
        regions[0]['lost_gt_indices_iou70'] = [3]
        frames[0]['controls']['gt_single_deletion_joint']['lost_gt_indices_iou70'] = [3, 8]
        result = analyze_weather(summary, regions, frames)['per_frame_loss_interactions'][0]
        self.assertEqual(result['joint_lost_also_lost_in_single'], [3])
        self.assertEqual(result['joint_lost_not_lost_in_any_single'], [8])


if __name__ == '__main__':
    unittest.main()
