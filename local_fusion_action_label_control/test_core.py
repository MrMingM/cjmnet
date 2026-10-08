"""Synthetic checks only. --static-only never imports Torch/OpenCOOD."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .common import KEEP, HERE, Manifest, atomic_json, digest, isolated_paths, read_json
from .inputs import Source, control_source, verify_scenes
from .labels import (category, detection_key, detection_pairs, detection_ranks,
                     focal_only_pair, keys, original_keys, original_pairs, original_ranks)
from .audit import Composition
from .probes import calibrate_common, fit_new, update_count
from .evaluate import DetectionMetrics
from .replay import check_ap, conflict_resolver
from .summarize import classify_execution
from local_fusion_action_utility_audit.features import feature_leakage_check, schema


def outcome(**changes):
    value = {'baseline_tp': 2, 'action_tp': 2, 'recovered': 0, 'lost': 0,
             'baseline_fp': 1, 'action_fp': 1, 'new_fp': 0,
             'focal_detected_before': True, 'focal_detected_after': True,
             'focal_score_before': .5, 'focal_score_after': .5,
             'focal_iou_before': .8, 'focal_iou_after': .8}
    value.update(changes)
    return value


class StaticTests(unittest.TestCase):
    def test_score_iou_only_original_prefers_new_equivalent(self):
        names = [KEEP, 'single:0', 'query:0']
        outcomes = [outcome(), outcome(focal_score_after=.9), outcome(focal_iou_after=.95)]
        self.assertGreater(original_keys(outcomes, names)[1], original_keys(outcomes, names)[0])
        self.assertEqual(detection_key(outcomes[0]), detection_key(outcomes[1]))
        self.assertEqual(detection_pairs(outcomes, names), [(0, 1), (0, 2)])
        self.assertEqual(detection_ranks(outcomes, names), [0., 0., 0.])
        self.assertTrue(focal_only_pair(outcomes[0], outcomes[1]))

    def test_priority_and_no_source_index_preference(self):
        names = [KEEP, 'single:0', 'query:0']
        rows = [outcome(), outcome(action_tp=3, recovered=1, action_fp=5), outcome(action_tp=3, recovered=1, action_fp=5)]
        self.assertIn((1, 0), detection_pairs(rows, names))
        self.assertNotIn((1, 2), detection_pairs(rows, names))
        self.assertNotIn((2, 1), detection_pairs(rows, names))
        self.assertGreater(detection_key(outcome(lost=0, new_fp=10)), detection_key(outcome(lost=1, new_fp=0)))
        self.assertGreater(detection_key(outcome(new_fp=0, action_fp=10)), detection_key(outcome(new_fp=1, action_fp=1)))

    def test_original_functions_are_direct_imports(self):
        from local_fusion_action_utility_audit import counterfactual as old
        self.assertIs(original_keys, old.action_keys)
        self.assertIs(original_pairs, old.pairwise_preferences)
        self.assertIs(original_ranks, old.outcome_ranks)

    def test_equal_detection_modify_regret_zero(self):
        group = {'outcomes': [outcome(), outcome(focal_score_after=.9)], 'names': [KEEP, 'single:0']}
        metric = DetectionMetrics()
        metric.add(group, np.array([0., 1.]))
        row = metric.result()
        self.assertEqual(row['mean_detection_outcome_regret'], 0.)
        self.assertEqual(row['same_detection_modify_ratio'], 1.)
        self.assertEqual(row['beneficial_modify_precision'], 0.)
        self.assertIsNone(row['non_tie_pairwise_accuracy'])

    def test_common_calibration_rejects_score_only_benefit(self):
        class Probe:
            def scores(self, x):
                return x
        spec = {'calibration_quantiles': [0., .5, 1.], 'calibration_min_actions': 1,
                'calibration_min_modify_precision': .8, 'calibration_max_lost_per_action': 0.,
                'calibration_max_new_fp_per_action': 0.}
        group = {'x': np.array([0., 1.]), 'outcomes': [outcome(), outcome(focal_score_after=.9)], 'names': [KEEP, 'single:0']}
        row = calibrate_common(Probe(), [group], spec)
        self.assertIsNone(row['threshold'])
        self.assertEqual(row['fallback'], 'KEEP_ALL')
        group['outcomes'][1] = outcome(action_tp=3, recovered=1)
        self.assertEqual(calibrate_common(Probe(), [group], spec)['threshold'], 0.)

    def test_mutually_exclusive_categories(self):
        values = [outcome(action_tp=3, recovered=2, lost=1, new_fp=1),
                  outcome(action_tp=1, lost=1), outcome(recovered=1, lost=1),
                  outcome(new_fp=1), outcome(focal_score_after=.8), outcome()]
        expected = ['tp_count_increase', 'tp_count_decrease', 'gt_identity_change_same_tp_count',
                    'fp_change_same_tp_and_gt_ids', 'score_or_localization_only', 'equivalent_saved_outcome']
        self.assertEqual([category(o, outcome()) for o in values], expected)
        self.assertEqual(sum(Composition().result()['exclusive_categories'].values()), 0)

    def test_unique_proposal_weight_and_simplicity_counts(self):
        names, rows = [KEEP, 'single:0', 'query:0'], [outcome(), outcome(focal_score_after=.9), outcome()]
        acc = Composition()
        acc.add(rows, names, .5)
        acc.add(rows, names, .5)
        c = acc.result()['counts']
        self.assertEqual(c['candidates'], 1.)
        self.assertEqual(c['non_keep_actions'], 2.)
        self.assertEqual(c['detection_keep_simplicity_pairs'], 2.)
        self.assertEqual(c['strict_detection_non_tie_pairs'], 0.)
        self.assertEqual(sum(acc.result()['exclusive_categories'].values()), 2.)

    def test_no_gt_feature_schema(self):
        for task in ('cls', 'reg'):
            self.assertTrue(feature_leakage_check(schema(task)))
            for forbidden in ('gt_iou', 'focal_score', 'recovered', 'target_index'):
                with self.assertRaises(ValueError):
                    feature_leakage_check(schema(task) + [forbidden])

    def test_source_run_isolation_including_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            source.mkdir()
            for output in (source, source / 'nested', source.parent):
                with self.assertRaises(ValueError):
                    isolated_paths(source, output, True)
            self.assertFalse((source / 'nested').exists())

    def test_readonly_source_hashes_and_missing_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity = {'source_hash': 'fake'}
            atomic_json(root / 'protocol.json', {'identity': identity, 'config': {}})
            atomic_json(root / 'manifest.json', {'identity': identity, 'entries': {}})
            source = Source(root)
            before = {p.name: digest(p) for p in root.iterdir()}
            self.assertFalse(source.complete('S0'))
            self.assertIn('stage S0', source.readiness())
            with self.assertRaises(RuntimeError):
                source.artifact('frame.pt', 'S0')
            self.assertEqual(before, {p.name: digest(p) for p in root.iterdir()})
            atomic_json(root / 'cache.json', {'x': 1})
            atomic_json(root / 'manifest.json', {'identity': identity, 'entries': {'S0/-/-/-': {
                **identity, 'status': 'complete', 'artifacts': {'cache.json': digest(root / 'cache.json')}}}})
            source.refresh()
            self.assertTrue(source.complete('S0'))
            atomic_json(root / 'cache.json', {'x': 2})
            with self.assertRaises(ValueError):
                source.complete('S0')

    def test_scene_protocol_mismatch_stops(self):
        p = {'train_indices': [0, 1], 'validation_indices': [0], 'train_scenes': [0, 1],
             'validation_scenes': [0], 'probe_fit_scenes': [0], 'probe_calibration_scenes': [0],
             'config': {'calibration_scene_fraction': .2, 'seed': 123}}
        with self.assertRaises(ValueError):
            verify_scenes(p)

    def test_source_cache_index_stays_pinned_when_manifest_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity = {'source_hash': 'fake'}
            name = 'cache/train/rain/00000000.pt'
            path = root / name
            atomic_json(path, {'x': 1})
            original_hash = digest(path)
            atomic_json(root / 'protocol.json', {'identity': identity, 'config': {}})
            entry = {**identity, 'status': 'complete', 'artifacts': {name: original_hash}}
            atomic_json(root / 'manifest.json', {'identity': identity, 'entries': {'S0-frame/train/rain/0': entry}})
            source = Source(root, {name: original_hash})
            self.assertEqual(source.artifact(name, 'S0-frame', 'train', 'rain', 0), path)
            atomic_json(path, {'x': 2})
            entry['artifacts'][name] = digest(path)
            atomic_json(root / 'manifest.json', {'identity': identity, 'entries': {'S0-frame/train/rain/0': entry}})
            source.refresh()
            with self.assertRaisesRegex(ValueError, 'changed after control preflight'):
                source.artifact(name, 'S0-frame', 'train', 'rain', 0)

    def test_control_cache_index_tamper_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            atomic_json(Path(folder) / 'source_cache_index.json', {'changed': 'hash'})
            with self.assertRaisesRegex(ValueError, 'cache index hash differs'):
                control_source(folder, {'identity': {'source_cache_index_hash': 'original'}})

    def test_original_replay_rules_reused_deterministically(self):
        from local_fusion_action_utility_audit.s3_replay import conflict_resolver as old
        self.assertIs(conflict_resolver, old)
        from .runtime import ReplayRuntime
        from local_fusion_action_utility_audit.common import Runtime
        self.assertIs(ReplayRuntime.loader, Runtime.loader)
        self.assertIs(ReplayRuntime.predict, Runtime.predict)
        actions = [{'position': i, 'anchor_id': i, 'shared_score': .8,
                    'cls_source': 'single:0', 'reg_source': KEEP, 'cls_margin': 1., 'reg_margin': 0.,
                    'mask': np.array([mask], bool)} for i, mask in ((1, [1, 1, 0]), (2, [0, 1, 1]))]
        first, counts = conflict_resolver(actions)
        second, repeated = conflict_resolver(list(reversed(actions)))
        self.assertEqual(counts, repeated)
        self.assertEqual(counts['modified_cells'], 3)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a['mask'], b['mask'])
        self.assertFalse((first[0]['mask'] & first[1]['mask']).any())

    def test_manifest_resume_checks_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'artifact.json'
            atomic_json(path, {'x': 1})
            manifest = Manifest(folder, {'control_source_hash': 'x'})
            manifest.mark('TRAIN', 'complete', artifacts=[path])
            self.assertTrue(Manifest(folder).complete('TRAIN'))
            atomic_json(path, {'x': 2})
            with self.assertRaises(ValueError):
                Manifest(folder).complete('TRAIN')

    def test_ap_tolerance_fails_immediately(self):
        row = {'ap30': .7, 'ap50': .6, 'ap70': .5}
        self.assertEqual(check_ap(row, row, 'synthetic'), {'ap30': 0., 'ap50': 0., 'ap70': 0.})
        with self.assertRaises(RuntimeError):
            check_ap({**row, 'ap70': .51}, row, 'synthetic')

    def test_do_not_call_keep_all_learning_success(self):
        old = {'proposals': 1000, 'changed_proposals': 500, 'recovered': 10, 'lost': 5, 'new_fp': 9}
        new = {'proposals': 1000, 'changed_proposals': 0, 'recovered': 0, 'lost': 0, 'new_fp': 0}
        self.assertIn('尚未证明', classify_execution(old, new, 1., .01))
        new.update(changed_proposals=300, recovered=10, lost=3, new_fp=5)
        self.assertIn('更好的执行结果', classify_execution(old, new, .2, .01))

    def test_pair_group_batch_rule(self):
        self.assertEqual(update_count([3, 3, 3], 5), 2)
        self.assertEqual(update_count([], 5), 0)

    def test_shell_lf_and_no_git_runtime_call(self):
        for name in ('launch.sh', 'run_all.sh'):
            value = (HERE / name).read_bytes()
            self.assertNotIn(b'\r', value)
            self.assertTrue(value.startswith(b'#!/bin/sh\nset -eu\n'))
        from . import common
        with patch.object(common, 'ROOT', Path('/nonexistent-code-copy')):
            self.assertIsNone(common.optional_commit())


class ServerTests(unittest.TestCase):
    def test_no_pairs_keep_all_zero_linear(self):
        import torch
        from local_fusion_action_utility_audit.linear_ranker import LinearProbe, selection
        with tempfile.TemporaryDirectory() as folder:
            state = {'feature_names': schema('cls'), 'mean': [0.] * len(schema('cls')), 'std': [1.] * len(schema('cls'))}
            spec = {'seed': 123, 'probe_learning_rate': .01, 'probe_weight_decay': .0001,
                    'probe_epochs': 2, 'pair_batch_size': 4}
            result = fit_new([], state, spec, folder, 'cls', {}, Manifest(folder))
            self.assertEqual(result['fallback'], 'KEEP_ALL_NO_TRAINABLE_PREFERENCES')
            probe = LinearProbe({**state, **result})
            scores = probe.scores(torch.randn(3, len(schema('cls'))))
            self.assertEqual(selection(scores, [KEEP, 'single:0', 'query:0'])[0], 0)

    def test_synthetic_linear_fit_and_resume(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            inherited = {'feature_names': schema('cls')}
            spec = {'seed': 123, 'probe_learning_rate': .01, 'probe_weight_decay': .0001,
                    'probe_epochs': 2, 'pair_batch_size': 1}
            x = torch.zeros(2, len(schema('cls')))
            x[1, 0] = 1.
            training = [(x, torch.tensor([[1, 0]]))]
            first = fit_new(training, inherited, spec, folder, 'cls', {'x': 1}, Manifest(folder))
            second = fit_new(training, inherited, spec, folder, 'cls', {'x': 1}, Manifest(folder))
            self.assertTrue(torch.equal(first['linear']['weight'], second['linear']['weight']))
            self.assertGreater(float(first['linear']['weight'][0, 0]), 0.)
            self.assertEqual(first['history'][0]['optimizer_updates'], 1)

    def test_shell_syntax(self):
        for name in ('run_all.sh', 'launch.sh'):
            subprocess.run(['sh', '-n', str(HERE / name)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-only', action='store_true')
    parser.add_argument('--require-server', action='store_true')
    args = parser.parse_args()
    if args.static_only and args.require_server:
        raise ValueError('Conflicting test modes')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StaticTests)
    if not args.static_only:
        if importlib.util.find_spec('torch') is None:
            raise RuntimeError('Torch missing in this environment; local static checks use --static-only')
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ServerTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
