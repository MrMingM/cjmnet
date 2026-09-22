import unittest
import numpy as np
import torch
from torch import nn
from local_fusion_utility_v2.fusion import (
    DESCRIPTOR_NAMES, attention_fusion, build_context, fused_levels_for_actions,
    select_actions, expand_action_map, region_pool)
from local_fusion_utility_v2.network import UtilityPredictor
from local_fusion_utility_v2.supervision import sample_action_pairs


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

    def test_partial_tile_pool_matches_exact_action_support_on_all_scales(self):
        ids = torch.arange(13*44).reshape(13, 44)
        for height, width in ((100, 352), (50, 176), (25, 88)):
            x = torch.arange(height*width).float().reshape(1, 1, height, width)
            support = expand_action_map(ids, (height, width), (100, 352), 8)
            avg = region_pool(x, (100, 352), 8)[0, 0]
            maximum = region_pool(x, (100, 352), 8, True)[0, 0]
            for row in (0, 3, 6, 9, 12):
                for col in (0, 17, 43):
                    cells = x[0, 0][support == ids[row, col]]
                    torch.testing.assert_close(avg[row, col], cells.mean())
                    torch.testing.assert_close(maximum[row, col], cells.max())

    def test_sampling_labels_every_peer_in_each_stratum(self):
        activity = torch.tensor([[.9, .1, 0.], [.3, .01, .7]])
        for peers in (1, 2, 4):
            pairs = sample_action_pairs(activity, peers, 3, 51)
            self.assertEqual(len(pairs), 3*peers)
            tiles = pairs[:, 1].unique()
            self.assertEqual(len(tiles), 3)
            values = activity.flatten()[tiles]
            self.assertEqual(int((values >= .2).sum()), 1)
            self.assertEqual(int(((values >= .02) & (values < .2)).sum()), 1)
            self.assertEqual(int((values < .02).sum()), 1)
            for tile in tiles:
                self.assertEqual(set(pairs[pairs[:, 1] == tile, 0].tolist()), set(range(1, peers+1)))
        self.assertEqual(sample_action_pairs(activity, 0, 3, 1).shape, (0, 2))

    def test_validation_selection_enforces_clean_gate_and_keep_fallback(self):
        from local_fusion_utility_v2.calibrate import choose_threshold
        gates = dict(clean_ap70_tolerance=.001, max_lost_tp_fraction=.01,
                     max_new_fp_per_baseline_tp=.01)
        summaries = {}
        for condition in ('clean', 'fog', 'rain', 'snow'):
            summaries[condition] = dict(baseline_tp=1000,
                results={'baseline': {'ap70': .8}, 'utility@0': {'ap70': .8},
                         'utility@1': {'ap70': .79 if condition == 'clean' else .9}},
                diagnostics_vs_baseline={
                    'utility@0': dict(lost=0, new_fp=0, selected_tiles_per_frame=0),
                    'utility@1': dict(lost=0, new_fp=0, selected_tiles_per_frame=3)})
        result = choose_threshold(summaries, 'utility', [.02], gates)
        self.assertIsNone(result['threshold'])
        summaries['clean']['results']['utility@1']['ap70'] = .8
        self.assertEqual(choose_threshold(summaries, 'utility', [.02], gates)['threshold'], .02)
        summaries['rain']['diagnostics_vs_baseline']['utility@1']['new_fp'] = 11
        self.assertIsNone(choose_threshold(summaries, 'utility', [.02], gates)['threshold'])


class OutcomeTests(unittest.TestCase):
    def test_new_fp_is_identity_based_not_net_count(self):
        from local_fusion_utility_v2.outcomes import count_new_false_positives

        def box(cx):
            # OpenCOOD box corners; only BEV x/y are used by convert_format.
            return np.asarray([[cx-1, -1, 0], [cx-1, 1, 0], [cx+1, 1, 0], [cx+1, -1, 0],
                               [cx-1, -1, 1], [cx-1, 1, 1], [cx+1, 1, 1], [cx+1, -1, 1]],
                              dtype=np.float32)
        old = np.stack([box(0)])
        self.assertEqual(count_new_false_positives(np.stack([box(0)]), old, .7), 0)
        self.assertEqual(count_new_false_positives(np.stack([box(10)]), old, .7), 1)


class PipelineTests(unittest.TestCase):
    def run_mock_pipeline(self, fail_stage=None):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from local_fusion_utility_v2 import pipeline
        called = []
        with tempfile.TemporaryDirectory(prefix='pipeline_test_', dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)

            def fake_run(command, **kwargs):
                module = command[command.index('-m')+1]
                called.append(list(command))
                def argument(name):
                    return command[command.index(name)+1]
                if fail_stage == module:
                    return SimpleNamespace(returncode=1)
                if module.endswith('.prepare'):
                    dest = Path(argument('--output-dir'))
                    dest.mkdir()
                    (dest/'manifest.json').write_text(json.dumps({'contract': {'options': {
                        'weather_augmentation': {'physics_fog': {'lookup_dir': '/resolved/original'}}}}}))
                if module.endswith('.evaluate'):
                    dest = Path(argument('--output-dir'))
                    dest.mkdir()
                    (dest/'protocol.json').write_text(json.dumps({'conditions': {}}))
                return SimpleNamespace(returncode=0)

            argv = ['pipeline', '--frontend-config', 'front.yaml', '--frontend-checkpoint', 'front.pth',
                    '--run', str(root)]
            with patch('sys.argv', argv), patch('gspr_communication.runtime.new_output', return_value=root), \
                 patch.object(pipeline.subprocess, 'run', side_effect=fake_run):
                if fail_stage:
                    with self.assertRaises(RuntimeError):
                        pipeline.main()
                else:
                    pipeline.main()
                    self.assertTrue((root/'all_results.json').exists())
                    import yaml
                    resolved = yaml.safe_load((root/'resolved_experiment.yaml').read_text())
                    self.assertEqual(resolved['weather_augmentation']['physics_fog']['lookup_dir'], '/resolved/original')
        return called

    def test_pipeline_automatically_tests_after_development(self):
        calls = self.run_mock_pipeline()
        evaluations = [c for c in calls if 'local_fusion_utility_v2.evaluate' in c]
        self.assertEqual([c[c.index('--phase')+1] for c in evaluations], ['development', 'benchmark'])
        self.assertEqual(len(calls), 8)
        for call in evaluations:
            self.assertTrue(call[call.index('--config')+1].endswith('resolved_experiment.yaml'))

    def test_pipeline_failure_prevents_test_access(self):
        calls = self.run_mock_pipeline('local_fusion_utility_v2.calibrate')
        self.assertFalse(any('local_fusion_utility_v2.evaluate' in c for c in calls))


if __name__ == '__main__':
    unittest.main()
