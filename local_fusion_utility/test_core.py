import unittest

try:
    import numpy as np
    import torch
    from torch import nn
    from local_fusion_utility.fusion import (
        DESCRIPTOR_NAMES, attention_fusion, build_context, fused_levels_for_actions,
        select_actions, expand_action_map)
    from local_fusion_utility.network import UtilityPredictor
    from local_fusion_utility.supervision import sample_action_pairs
    IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - supports source-only workstations
    IMPORT_ERROR = error


@unittest.skipIf(IMPORT_ERROR is not None, f'PyTorch stack unavailable: {IMPORT_ERROR}')
class CoreTests(unittest.TestCase):
    def test_query_switch_and_keep(self):
        torch.manual_seed(3)
        levels = [torch.randn(3, 4, 3, 5) for _ in range(3)]
        baseline = [attention_fusion(value, 0)[0] for value in levels]
        context = {'levels': levels, 'baseline_levels': baseline, 'grid': (2, 3),
                   'reference_hw': (3, 5), 'tile_size': 2}
        keep = fused_levels_for_actions(context, torch.zeros(2, 3, dtype=torch.long))
        for actual, expected in zip(keep, baseline):
            torch.testing.assert_close(actual, expected)
        all_peer = torch.full((2, 3), 2, dtype=torch.long)
        changed = fused_levels_for_actions(context, all_peer, (0, 1))
        torch.testing.assert_close(changed[0], attention_fusion(levels[0], 2)[0])
        torch.testing.assert_close(changed[1], attention_fusion(levels[1], 2)[0])
        torch.testing.assert_close(changed[2], baseline[2])
        local = torch.zeros(2, 3, dtype=torch.long)
        local[0, 0] = 1
        local_value = fused_levels_for_actions(context, local, (0,))[0]
        mask = expand_action_map(local, (3, 5), (3, 5), 2).bool()[None, None]
        expected = torch.where(mask, attention_fusion(levels[0], 1)[0], baseline[0])
        torch.testing.assert_close(local_value, expected)

    def test_ego_query_matches_frozen_full_fusion(self):
        from gspr_communication.masked_attfuse import fuse
        torch.manual_seed(31)
        features = torch.randn(4, 8, 7, 11)
        hard = torch.ones(4, 1, 2, 3)
        torch.testing.assert_close(attention_fusion(features, 0)[0], fuse(features, hard))

    def test_physical_tile_expansion(self):
        action = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expanded = expand_action_map(action, (3, 5), (3, 5), 2)
        expected = torch.tensor([[1, 1, 2, 2, 3], [1, 1, 2, 2, 3], [4, 4, 5, 5, 6]])
        self.assertTrue(torch.equal(expanded, expected))

    def test_descriptor_and_network_contract(self):
        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.deblocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
                self.cls_head = nn.Conv2d(12, 2, 1)
                self.reg_head = nn.Conv2d(12, 14, 1)
        torch.manual_seed(4)
        context = build_context(Base(), [torch.randn(3, 4, 3, 5) for _ in range(3)], 2)
        self.assertEqual(tuple(context['descriptors'].shape), (2, len(DESCRIPTOR_NAMES), 2, 3))
        self.assertTrue(torch.isfinite(context['descriptors']).all())
        network = UtilityPredictor(len(DESCRIPTOR_NAMES), 16)
        result = network(context['descriptors'])
        self.assertTrue((result['outcome'] >= 0).all())
        dense_first = result['outcome'][0, :, 0, 0]
        single_first = network(context['descriptors'][0:1, :, 0:1, 0:1])['outcome'][0, :, 0, 0]
        torch.testing.assert_close(dense_first, single_first)
        (result['outcome'].sum()+result['loss_gain'].sum()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in network.parameters()))

    def test_keep_threshold_and_no_keep(self):
        score = torch.tensor([[[.1, -.1]], [[.2, -.2]]])
        self.assertTrue(torch.equal(select_actions(score, .15), torch.tensor([[2, 0]])))
        self.assertTrue(torch.equal(select_actions(score, .15, False), torch.tensor([[2, 1]])))

    def test_sampling_is_deterministic_and_unique(self):
        activity = torch.tensor([[.9, .1, 0.], [.3, .01, .7]])
        first = sample_action_pairs(activity, 3, 10, 99)
        second = sample_action_pairs(activity, 3, 10, 99)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len({tuple(row) for row in first.tolist()}), len(first))
        self.assertTrue(((first[:, 0] >= 1) & (first[:, 0] <= 3)).all())


@unittest.skipIf(IMPORT_ERROR is not None, f'PyTorch stack unavailable: {IMPORT_ERROR}')
class OutcomeTests(unittest.TestCase):
    def test_new_fp_is_identity_based_not_net_count(self):
        try:
            from local_fusion_utility.outcomes import count_new_false_positives
        except Exception as error:
            self.skipTest(str(error))

        def box(cx):
            # OpenCOOD box corners; only BEV x/y are used by convert_format.
            return np.asarray([[cx-1, -1, 0], [cx-1, 1, 0], [cx+1, 1, 0], [cx+1, -1, 0],
                               [cx-1, -1, 1], [cx-1, 1, 1], [cx+1, 1, 1], [cx+1, -1, 1]],
                              dtype=np.float32)
        old = np.stack([box(0)])
        self.assertEqual(count_new_false_positives(np.stack([box(0)]), old, .7), 0)
        self.assertEqual(count_new_false_positives(np.stack([box(10)]), old, .7), 1)


if __name__ == '__main__':
    unittest.main()
