import unittest
import torch

from .pipeline import backward_if_trainable, choose_frames
from .preservation import preservation_loss, correct_anchor_mask


class PilotTests(unittest.TestCase):
    def test_ego_only_loss_skips_backward(self):
        self.assertFalse(backward_if_trainable(torch.tensor(1.), 1))
        with self.assertRaisesRegex(RuntimeError, 'Multi-source'):
            backward_if_trainable(torch.tensor(1.), 2)
        parameter = torch.tensor(2., requires_grad=True)
        self.assertTrue(backward_if_trainable(parameter.square(), 2))
        self.assertEqual(float(parameter.grad), 2.)

    def test_scene_sampling_is_deterministic_and_bounded(self):
        first = choose_frames([4, 9, 15], 13, 2, 2)
        self.assertEqual(first, choose_frames([4, 9, 15], 13, 2, 2))
        self.assertEqual(len(first[1]), 4)
        self.assertEqual(len(set(first[1])), 4)

    def test_preservation_loss_only_uses_selected_anchor(self):
        teacher = {'psm': torch.zeros(1, 2, 1, 1),
                   'rm': torch.zeros(1, 14, 1, 1)}
        student = {'psm': torch.tensor([[[[1.]], [[10.]]]], requires_grad=True),
                   'rm': torch.ones(1, 14, 1, 1, requires_grad=True)}
        mask = torch.tensor([[[[True, False]]]])
        loss = preservation_loss(student, teacher, mask)
        loss.backward()
        self.assertGreater(float(student['psm'].grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(student['psm'].grad[0, 1].abs().sum()), 0)
        self.assertEqual(float(student['rm'].grad[0, 7:].abs().sum()), 0)

    def test_empty_mask_has_zero_loss(self):
        student = {'psm': torch.randn(1, 2, 1, 1, requires_grad=True),
                   'rm': torch.randn(1, 14, 1, 1, requires_grad=True)}
        result = preservation_loss(student, student, torch.zeros(1, 1, 1, 2, dtype=torch.bool))
        self.assertEqual(float(result), 0.)

    def test_only_gt_matched_teacher_anchor_is_protected(self):
        from opencood.utils import box_utils

        center = torch.tensor([[0., 0., 0., 1., 1., 2., 0.]])
        corners = box_utils.boxes_to_corners_3d(center, order='hwl')

        class Postprocessor:
            params = {'order': 'hwl'}

            @staticmethod
            def delta_to_boxes3d(_regression, _anchors):
                return center[None]

        class Dataset:
            post_processor = Postprocessor()

            @staticmethod
            def post_process(_batch, _output):
                return corners, torch.tensor([.9]), corners

        batch = {'ego': {
            'label_dict': {'pos_equal_one': torch.ones(1, 1, 1, 1)},
            'anchor_box': center.reshape(1, 1, 1, 7),
        }}
        teacher = {'psm': torch.zeros(1, 1, 1, 1), 'rm': torch.zeros(1, 7, 1, 1)}
        mask, coverage = correct_anchor_mask(Dataset(), batch, teacher)
        self.assertTrue(bool(mask.item()))
        self.assertEqual(coverage, {'teacher_tp': 1, 'protected': 1})
        batch['ego']['label_dict']['pos_equal_one'].zero_()
        mask, coverage = correct_anchor_mask(Dataset(), batch, teacher)
        self.assertFalse(bool(mask.item()))
        self.assertEqual(coverage['protected'], 0)


if __name__ == '__main__':
    unittest.main()
