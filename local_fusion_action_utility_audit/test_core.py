"""Synthetic protocol tests. Research data/experiments run only on server.

Local --static-only excludes Torch/OpenCOOD and shell tests. The server driver
requires all tests and fails on skips, missing dependencies or shell errors.
"""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from .common import (KEEP, Manifest, assert_development_paths, atomic_json,
                     digest, object_hash, read_json, scene_split, shell_check, source_identity,
                     verify_protocol_snapshot, write_source_snapshot)
from .counterfactual import (action_keys, outcome_ranks, pairwise_preferences,
                             path_independent_action_key)
from .features import (extract_features, feature_leakage_check, geometry,
                       schema, source_names)
from .linear_ranker import Metrics, calibrate, selection, verdict
from .s0_counterfactual import background_sample
from .s3_replay import conflict_resolver, s3_verdict
from .summarize import interpret, summarize as final_summarize


def outcome(**updates):
    row = {'baseline_tp': 2, 'action_tp': 2, 'recovered': 0, 'lost': 0,
           'baseline_fp': 1, 'action_fp': 1, 'new_fp': 0,
           'focal_detected_before': True, 'focal_detected_after': True,
           'focal_score_before': .5, 'focal_score_after': .5,
           'focal_iou_before': .8, 'focal_iou_after': .8}
    row.update(updates)
    return row


def boxes(count=4):
    template = np.array([[1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
                         [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]], float)
    return np.stack([template + np.array([3 * i, 0, 0]) for i in range(count)]).astype(np.float32)


def fake_pred_iou(left, right):
    # Test permutation/count behavior without importing OpenCOOD; real geometry
    # is tested below by the mandatory server suite.
    a = np.asarray(left).mean(1)
    b = np.asarray(right).mean(1)
    return np.exp(-np.linalg.norm(a[:, None] - b[None], axis=-1))


class StaticProtocolTests(unittest.TestCase):
    def test_feature_leakage(self):
        for task in ('cls', 'reg'):
            self.assertTrue(feature_leakage_check(schema(task)))
        for name in ('gt_iou', 'target_index', 'was_shared_missed', 'oracle_selected_source',
                     'recovered', 'lost', 'new_fp', 'focal_iou', 'post_nms_gt_matching', 'gt_box_x'):
            with self.assertRaises(ValueError):
                feature_leakage_check(schema('cls') + [name])

    def test_pairwise_priority_and_keep_tie(self):
        names = [KEEP, 'single:0', 'query:0']
        labels = [outcome(), outcome(action_tp=3, recovered=1, action_fp=4), outcome()]
        self.assertGreater(path_independent_action_key(labels[1], names[1]),
                           path_independent_action_key(labels[0], KEEP))
        self.assertIn((1, 0), pairwise_preferences(labels, names))
        self.assertIn((0, 2), pairwise_preferences(labels, names))
        self.assertEqual(outcome_ranks(labels, names), [0., 1., 0.])
        self.assertEqual(selection(np.array([0., 0., 0.]), names)[0], 0)

    def test_background_no_focal(self):
        row = outcome(**{key: None for key in outcome() if key.startswith('focal_')})
        self.assertEqual(path_independent_action_key(row, KEEP)[3], 0)
        group = {'outcomes': [row, row], 'names': [KEEP, 'single:0']}
        metrics = Metrics()
        metrics.add(group, np.array([1., 0.]))
        self.assertEqual(metrics.result()['mean_action_regret'], 0.)

    def test_stratified_background(self):
        first = background_sample(list(range(90)), 32, 10)
        self.assertEqual(first, background_sample(list(range(90)), 32, 10))
        self.assertEqual(len(first), 32)
        self.assertEqual([sum(lo <= i < lo + 30 for i in first) for lo in (0, 30, 60)], [11, 11, 10])

    def test_source_order_and_variable_count(self):
        spec = {'shared_top_k': 256, 'local_top_k': 5, 'nearby_distance_m': 5.}
        for cavs in (1, 2, 5):
            names = [KEEP] + [f'{family}:{i}' for family in ('single', 'query') for i in range(cavs)]
            proxies = {name: {'scores': np.array([.8, .7, .6, .5]) + i * .001,
                               'corners': boxes() + i * .01, 'anchor_num': 1, 'hw': (2, 2)}
                       for i, name in enumerate(names)}
            ids, scores, corners = np.arange(4), proxies[KEEP]['scores'], boxes()
            masks = [np.ones((2, 2), bool) for _ in ids]
            with patch('local_fusion_action_utility_audit.features.pred_iou', fake_pred_iou):
                ordered, a = extract_features(proxies, ids, scores, corners, masks, spec)
                reordered, b = extract_features(dict(reversed(list(proxies.items()))), ids, scores, corners, masks, spec)
            self.assertEqual(ordered, reordered)
            self.assertEqual(ordered, source_names(proxies))
            for task in ('cls', 'reg'):
                np.testing.assert_array_equal(a[task], b[task])
                self.assertEqual(a[task].shape[:2], (4, 1 + 2 * cavs))
                np.testing.assert_array_equal(a[task][:, :, schema(task).index('source_count')], 1 + 2 * cavs)
                self.assertTrue(np.isfinite(a[task]).all())
            centers, sizes, yaw = geometry(boxes())
            np.testing.assert_allclose(sizes, np.full((4, 3), 2.))
            np.testing.assert_allclose(yaw, 0.)

    def test_conflicts_deterministic_partial_first_wins(self):
        def row(anchor, margin, score, mask):
            return {'position': anchor, 'anchor_id': anchor, 'shared_score': score,
                    'cls_source': 'single:0', 'reg_source': KEEP,
                    'cls_margin': margin, 'reg_margin': 0., 'mask': np.array([mask], bool)}
        actions = [row(2, 1., .8, [1, 1, 0]), row(1, 1., .8, [0, 1, 1]), row(3, .5, .9, [0, 1, 0])]
        accepted, counts = conflict_resolver(actions)
        reverse, reverse_counts = conflict_resolver(list(reversed(actions)))
        self.assertEqual([r['anchor_id'] for r in accepted], [1, 2])
        self.assertEqual(counts, reverse_counts)
        for a, b in zip(accepted, reverse):
            np.testing.assert_array_equal(a['mask'], b['mask'])
        self.assertFalse((accepted[0]['mask'] & accepted[1]['mask']).any())
        self.assertEqual(counts['accepted_actions'], 2)
        self.assertEqual(counts['rejected_overlap_actions'], 2)
        self.assertEqual(counts['modified_cells'], 3)

    def test_manifest_resume_hash_and_child_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.json'
            atomic_json(path, {'frame': 1})
            identity = {'config_hash': object_hash({'x': 1}), 'source_commit': 'fake', 'checkpoint_hashes': {}}
            parent = Manifest(folder, identity)
            parent.mark('S0', 'in_progress')
            child = Manifest(folder, identity)
            child.mark('S0-frame', 'complete', 'train', 'fog', 1, [path])
            parent.mark('S0', 'failed', error='synthetic interruption')
            resumed = Manifest(folder, identity)
            self.assertTrue(resumed.complete('S0-frame', 'train', 'fog', 1))
            self.assertFalse(resumed.complete('S0'))
            with self.assertRaises(ValueError):
                Manifest(folder, {'config_hash': 'changed'})
            atomic_json(path, {'frame': 2})
            with self.assertRaises(ValueError):
                resumed.complete('S0-frame', 'train', 'fog', 1)

    def test_scene_and_test_guards(self):
        fit, calibration = scene_split([0, 1, 2, 3, 4], .2, 10)
        self.assertFalse(set(fit) & set(calibration))
        self.assertEqual(set(fit + calibration), set(range(5)))
        self.assertEqual((fit, calibration), scene_split([4, 3, 2, 1, 0], .2, 10))
        for path in ('/data/scd/datasets/opv2v_official_data_dumping/test', '/data/cjm/datasets/opv2v-w/snow/test'):
            with self.assertRaises(ValueError):
                assert_development_paths(path)

    def test_metrics_outcome_regret_not_source_id(self):
        group = {'outcomes': [outcome(), outcome(action_tp=3, recovered=1), outcome(action_tp=3, recovered=1)],
                 'names': [KEEP, 'single:0', 'query:0']}
        metric = Metrics()
        metric.add(group, np.array([0., .2, .3]))
        self.assertEqual(metric.result()['mean_action_regret'], 0.)
        self.assertEqual(metric.result()['selected_recovered'], 1)

    def test_interpretation_does_not_overclaim_interaction(self):
        self.assertEqual(interpret('S1_WEAK', 'S2_WEAK', 1., 0., .3)[0][0], 'A')
        self.assertEqual(interpret('S1_LEARNABLE', 'S2_LEARNABLE', 1., -.1, .3)[0][0], 'C')
        self.assertEqual(interpret('S1_LEARNABLE', 'S2_LEARNABLE', 1., .1, .3)[0][0], 'D')
        self.assertEqual(interpret('S1_LEARNABLE', 'S2_LEARNABLE', -.1, -.1, .3)[0][0], 'E')

    def test_calibration_feasible_and_keep_fallback(self):
        class FakeProbe:
            def scores(self, x):
                return x
        spec = {'calibration_quantiles': [0., .5, 1.], 'calibration_min_actions': 2,
                'calibration_min_modify_precision': .8, 'calibration_max_lost_per_action': 0.,
                'calibration_max_new_fp_per_action': 0.}
        good = {'x': np.array([0., 2.]), 'names': [KEEP, 'single:0'],
                'outcomes': [outcome(), outcome(action_tp=3, recovered=1)]}
        bad = {'x': np.array([0., .1]), 'names': [KEEP, 'single:0'],
               'outcomes': [outcome(), outcome(action_tp=1, lost=1)]}
        calibrated = calibrate(FakeProbe(), [bad, good, good], spec)
        self.assertIsNotNone(calibrated['threshold'])
        self.assertGreaterEqual(calibrated['threshold'], .1)
        self.assertEqual(selection(bad['x'], bad['names'], calibrated['threshold'])[0], 0)
        fallback = calibrate(FakeProbe(), [bad, bad], spec)
        self.assertIsNone(fallback['threshold'])
        self.assertEqual(fallback['fallback'], 'KEEP_ALL')

    def test_preregistered_gates(self):
        conditions = {w: {'linear': {'mean_action_regret': .5, 'pairwise_ranking_accuracy': .7},
                          'Always KEEP': {'mean_action_regret': 1.},
                          'simple': {'mean_action_regret': .8}}
                      for w in ('fog', 'rain', 'snow')}
        spec = {'s1_pairwise_accuracy_gate': .6, 's1_regret_reduction_gate': .2,
                's3_weather_mean_ap70_gain_pp': .3, 's3_min_positive_adverse_weathers': 2,
                's3_clean_max_drop_pp': .2}
        self.assertEqual(verdict(conditions, spec, 'cls')['verdict'], 'S1_LEARNABLE')
        conditions['fog']['simple']['mean_action_regret'] = 0.
        conditions['rain']['simple']['mean_action_regret'] = 0.
        self.assertEqual(verdict(conditions, spec, 'cls')['verdict'], 'S1_WEAK')
        replay = {w: {'Proposal-Task-Conservative': {
            'delta_ap_pp_vs_Shared': {'ap70': -.1 if w == 'clean' else .4},
            'counts': {'recovered': 2, 'lost': 1}}} for w in ('clean', 'fog', 'rain', 'snow')}
        self.assertEqual(s3_verdict(replay, spec)['verdict'], 'S3_PASS')
        replay['clean']['Proposal-Task-Conservative']['delta_ap_pp_vs_Shared']['ap70'] = -.21
        self.assertEqual(s3_verdict(replay, spec)['verdict'], 'S3_FAIL')

    def test_snapshot_rejects_source_and_config_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('a.py', 'v3.yaml', 'front.yaml', 'protocol.json', 'decision_results.json'):
                (root / name).write_text('{}', encoding='utf-8')
            inputs = {'config': str(root / 'audit.yaml'), 'v3_config': str(root / 'v3.yaml'),
                      'frontend_config': str(root / 'front.yaml'), 'b0_run': str(root)}
            source_hashes = {'a.py': digest(root / 'a.py')}
            identity = {'source_commit': 'fake', 'source_hash': object_hash(source_hashes),
                        'config_hash': object_hash({'spec': {}, 'inputs': inputs,
                            'v3_config_hash': digest(root / 'v3.yaml'),
                            'frontend_config_hash': digest(root / 'front.yaml'),
                            'b0_protocol_hash': digest(root / 'protocol.json'),
                            'b0_result_hash': digest(root / 'decision_results.json')})}
            run = root / 'run'
            Manifest(run, identity)
            atomic_json(run / 'protocol.json', {'identity': identity, 'inputs': inputs,
                                               'source_hashes': source_hashes})
            with patch('local_fusion_action_utility_audit.common.ROOT', root), \
                 patch('local_fusion_action_utility_audit.common.settings', return_value={}), \
                 patch('local_fusion_action_utility_audit.common.source_identity', return_value={'source_commit': 'fake'}):
                verify_protocol_snapshot(run)
                (root / 'a.py').write_text('changed', encoding='utf-8')
                with self.assertRaises(ValueError):
                    verify_protocol_snapshot(run)
                (root / 'a.py').write_text('{}', encoding='utf-8')
                (root / 'v3.yaml').write_text('changed', encoding='utf-8')
                with self.assertRaises(ValueError):
                    verify_protocol_snapshot(run)

    def test_outcome_source_alignment_required(self):
        with self.assertRaises(ValueError):
            action_keys([outcome()], [KEEP, 'single:0'])
        with self.assertRaises(ValueError):
            action_keys([outcome(), outcome()], ['single:0', KEEP])

    def test_exported_main_copy_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audit = root / 'local_fusion_action_utility_audit'
            audit.mkdir()
            (audit / 'common.py').write_text('# synthetic source\n', encoding='utf-8')
            (root / '.git').mkdir()
            source = {'source_commit': 'a' * 40, 'source_branch': 'main', 'provenance': 'git'}
            with patch('local_fusion_action_utility_audit.common.ROOT', root):
                with patch('local_fusion_action_utility_audit.common.source_identity', return_value=source):
                    write_source_snapshot()
                (root / '.git').rmdir()
                exported = source_identity()
                self.assertEqual(exported['provenance'], 'exported_main_snapshot')
                self.assertEqual(exported['source_commit'], 'a' * 40)
                (audit / 'common.py').write_text('# changed\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'differ'):
                    source_identity()

    def test_git_failures_are_not_silently_bypassed(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '.git').mkdir()
            with patch('local_fusion_action_utility_audit.common.ROOT', root):
                with patch('local_fusion_action_utility_audit.common.subprocess.run', return_value=
                           subprocess.CompletedProcess(['git'], 128, '', 'fatal: dubious ownership')):
                    with self.assertRaisesRegex(RuntimeError, 'dubious ownership'):
                        source_identity()
                with patch('local_fusion_action_utility_audit.common._git_output', return_value='other'):
                    with self.assertRaisesRegex(ValueError, 'current branch: other'):
                        source_identity()
                (root / '.git').rmdir()
                with self.assertRaisesRegex(RuntimeError, 'No .git'):
                    source_identity()

    def test_final_report_complete_and_same_policy_interpretation(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            protocol = {'config': {'assisted_meaningful_gain_pp': .3, 'task_source_disagreement_gate': .1}}
            s0 = {'conditions': {'validation': {w: {'strata': {'associated': {
                'task_source_disagreement_rate': .2}}} for w in ('fog', 'rain', 'snow')}}}
            effects = {w: {'pairwise_accuracy_difference': .02, 'regret_difference': -.01}
                       for w in ('clean', 'fog', 'rain', 'snow')}
            s1 = {'decision': {'verdict': 'S1_LEARNABLE'}, 'competition_feature_effect': effects}
            s2 = {'decision': {'verdict': 'S2_LEARNABLE'}, 'competition_feature_effect': effects}
            s3 = {'decision': {'verdict': 'S3_FAIL'}, 'conditions': {
                w: {method: {'ap70': .5 + gain / 100, 'delta_ap_pp_vs_Shared': {'ap70': gain},
                             'counts': {'recovered': 2, 'lost': 1, 'new_fp': 0, 'changed_proposals': 3}}
                    for method, gain in (('Shared', 0.), ('Assisted-Task', .5),
                                         ('Proposal-Task-Greedy', .2), ('Proposal-Task-Conservative', -.1))}
                for w in ('clean', 'fog', 'rain', 'snow')}}
            manifest = Manifest(run)
            manifest.mark('baseline', 'complete')
            for stage, value in (('S0', s0), ('S1', s1), ('S2', s2), ('S3', s3)):
                path = run / (stage + '_RESULTS.json')
                atomic_json(path, value)
                manifest.mark(stage, 'complete', artifacts=[path])
            with patch('local_fusion_action_utility_audit.summarize.verify_protocol_snapshot', return_value=protocol):
                final_summarize(run)
                report = read_json(run / 'final_results.json')
                self.assertEqual(report['interpretations'][0]['case'], 'D')
                self.assertFalse(report['test_data_used'])
                self.assertEqual(report['inference_gt_fields'], 0)
                md = (run / 'FINAL_RESULTS.md').read_text(encoding='utf-8')
                for heading in ('## 一句话结论', '## 目前能得出什么结论',
                                '## 还不能得出什么结论', '## 下一步应该验证什么'):
                    self.assertIn(heading, md)
                manifest.mark('S3', 'failed', error='synthetic')
                with self.assertRaises(RuntimeError):
                    final_summarize(run)


class ServerTests(unittest.TestCase):
    def test_same_shared_state_no_accumulation(self):
        import torch
        from .counterfactual import independent_prediction
        shared = {'psm': torch.zeros(1, 2, 2, 2), 'rm': torch.zeros(1, 14, 2, 2)}
        pool = {KEEP: shared, 'single:0': {k: torch.ones_like(v) for k, v in shared.items()},
                'query:0': {k: torch.full_like(v, 2.) for k, v in shared.items()}}
        mask_a = np.array([[1, 0], [0, 0]], bool)
        mask_b = np.array([[0, 0], [0, 1]], bool)
        first = independent_prediction(shared, pool, mask_a, 'cls', 'single:0')
        second = independent_prediction(shared, pool, mask_b, 'reg', 'query:0')
        self.assertTrue(torch.equal(second['psm'], shared['psm']))
        self.assertEqual(float(second['rm'][0, 0, 0, 0]), 0.)
        self.assertEqual(float(first['psm'][0, 0, 0, 0]), 1.)
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in shared.values()))

    def test_real_prediction_geometry_iou(self):
        from .features import pred_iou
        matrix = pred_iou(boxes(), boxes())
        np.testing.assert_allclose(np.diag(matrix), np.ones(4), atol=1e-6)
        self.assertEqual(matrix[0, 1], 0.)

    def test_linear_pairwise_gradient_and_gt_free_decisions(self):
        import torch
        from .linear_ranker import LinearProbe
        from .s3_replay import predict_actions
        spec = {'feature_names': schema('cls'), 'linear': torch.nn.Linear(len(schema('cls')), 1).state_dict(),
                'mean': [0.] * len(schema('cls')), 'std': [1.] * len(schema('cls')),
                'calibration': {'threshold': None}}
        probe = LinearProbe(spec)
        self.assertIsInstance(probe.model, torch.nn.Linear)
        reg_spec = {**spec, 'feature_names': schema('reg'),
                    'linear': torch.nn.Linear(len(schema('reg')), 1).state_dict(),
                    'mean': [0.] * len(schema('reg')), 'std': [1.] * len(schema('reg'))}
        features = {task: torch.zeros(1, 3, len(schema(task))) for task in ('cls', 'reg')}
        actions = predict_actions([0], [.5], [np.ones((2, 2), bool)], features,
                                   [KEEP, 'single:0', 'query:0'], probe, LinearProbe(reg_spec),
                                   'with_competition_features', True)
        self.assertEqual(actions[0]['cls_source'], KEEP)
        self.assertEqual(actions[0]['reg_source'], KEEP)
        model = torch.nn.Linear(2, 1)
        torch.nn.init.zeros_(model.weight)
        loss = torch.nn.functional.softplus(-(model(torch.tensor([[1., 0.]])) - model(torch.zeros(1, 2)))).mean()
        loss.backward()
        self.assertLess(float(model.weight.grad[0, 0]), 0.)

    def test_structured_background_global_outcome(self):
        import torch
        from .counterfactual import outcome_state, structured_outcome
        gt = torch.tensor(boxes(1))
        baseline_post = (None, None, gt)
        action_post = (gt.clone(), torch.tensor([.9]), gt)
        result = structured_outcome(outcome_state(baseline_post, .7), outcome_state(action_post, .7), None,
                                     {'target_iou': .7, 'fp_identity_iou': .7})
        self.assertEqual(result['recovered'], 1)
        self.assertTrue(all(result[k] is None for k in result if k.startswith('focal_')))

    def test_shell_syntax_and_lf(self):
        shell_check()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--static-only', action='store_true')
    cli.add_argument('--require-server', action='store_true')
    args = cli.parse_args()
    if args.static_only and args.require_server:
        raise ValueError('Cannot skip mandatory server tests')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StaticProtocolTests)
    if not args.static_only:
        if importlib.util.find_spec('torch') is None:
            raise RuntimeError('Server tests require Torch/OpenCOOD; local static checks use --static-only')
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ServerTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
