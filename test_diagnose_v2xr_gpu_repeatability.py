"""Local synthetic tests; no Torch, OpenCOOD, real checkpoints or GPU runs."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from diagnose_v2xr_gpu_repeatability import (array_difference, backend_mode, backend_state, choose_batch,
    compare_trials, compare_gradient_maps, compare_crop_trial, crop_arguments, verify_prior_probe)


class RepeatabilityTests(unittest.TestCase):
    def test_equal_and_small_gradient_difference(self):
        a = np.array([1., .2], np.float32)
        self.assertTrue(array_difference(a, a)['equal'])
        b = a.copy(); b[1] += np.float32(1e-5)
        report = array_difference(a, b)
        self.assertFalse(report['equal'])
        self.assertGreater(report['max_abs'], 0)
        self.assertGreater(report['relative_l2'], 0)
        self.assertEqual(report['elements'], 2)
        self.assertTrue(array_difference(None, None)['equal'])
        self.assertEqual(array_difference(np.array([], np.float32), np.array([], np.float32))['rms_delta'], 0)

    def test_missing_nonfinite_and_layout_are_rejected(self):
        a = np.ones(2, np.float32)
        for b in (None, np.ones(3, np.float32), np.ones(2, np.float64)):
            with self.assertRaises(ValueError):
                array_difference(a, b)
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(FloatingPointError):
                array_difference(np.array([value], np.float32), np.array([value], np.float32))

    def test_selection_preserves_committed_microbatch_order(self):
        index, chosen = choose_batch([9, 8, 7, 6], 2, lambda item: item == 6)
        self.assertEqual(index, 1)
        self.assertEqual(chosen, [7, 6])
        with self.assertRaisesRegex(ValueError, 'No supervised window'):
            choose_batch([9, 8], 2, lambda item: False)
        with self.assertRaisesRegex(ValueError, 'complete committed microbatches'):
            choose_batch([9, 8, 7], 2, lambda item: True)

    def backend(self):
        state = SimpleNamespace(enabled=False, warn_only=False)
        cudnn = SimpleNamespace(enabled=True, deterministic=False, benchmark=True)
        def set_mode(enabled, warn_only=False):
            state.enabled, state.warn_only = enabled, warn_only
        return SimpleNamespace(backends=SimpleNamespace(cudnn=cudnn), use_deterministic_algorithms=set_mode,
            are_deterministic_algorithms_enabled=lambda: state.enabled,
            is_deterministic_algorithms_warn_only_enabled=lambda: state.warn_only)

    def test_backend_preferences_restore_even_after_failure(self):
        substitute = self.backend()
        original = backend_state(substitute)
        with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            with backend_mode(substitute, 'prefer_deterministic') as active:
                self.assertTrue(active['enabled'])
                self.assertTrue(active['warn_only'])
                self.assertTrue(active['cudnn_deterministic'])
                self.assertFalse(active['cudnn_benchmark'])
                raise RuntimeError('synthetic interruption')
        self.assertEqual(backend_state(substitute), original)
        with backend_mode(substitute, 'current') as active:
            self.assertEqual(active, original)

    def trial(self):
        return dict(rng_before='same_before', rng_after='same_after', batch_digest='same_batch', capture=dict(
            detector_outputs='same_out', auxiliary_output='same_aux', fused_features='same_features',
            target_input='same_target_input', target_output='same_target_output',
            detection_loss='same_detection', generation_loss='same_generation', joint_loss='same_joint',
            target_output_gradients=[dict(phase='joint_backward', tensor='same_upstream')]))

    def test_backward_candidate_requires_same_forward_and_upstream_gradient(self):
        reference = self.trial()
        current = copy.deepcopy(reference)
        a = {'detector.target.weight': np.array([1., 2.], np.float32)}
        b = {'detector.target.weight': np.array([1., 2.0001], np.float32)}
        result = compare_trials(reference, current, a, b, 'detector.target.weight')
        self.assertEqual(result['observed'], 'backward_numeric_variation_observed')
        self.assertTrue(result['target_weight_backward_candidate'])
        current['capture']['target_output_gradients'][0]['tensor'] = 'different_upstream'
        result = compare_trials(reference, current, a, b, 'detector.target.weight')
        self.assertFalse(result['target_weight_backward_candidate'])

    def test_forward_variation_and_control_drift_are_distinct(self):
        reference = self.trial()
        current = copy.deepcopy(reference)
        gradients = {'detector.target.weight': np.array([1., 2.], np.float32)}
        self.assertEqual(compare_trials(reference, current, gradients, gradients, 'detector.target.weight')['observed'],
                         'no_numeric_variation_in_selected_probe')
        current['capture']['detector_outputs'] = 'different_forward'
        self.assertEqual(compare_trials(reference, current, gradients, gradients, 'detector.target.weight')['observed'],
                         'forward_bitwise_variation_observed')
        for field in ('rng_before', 'rng_after', 'batch_digest'):
            current = copy.deepcopy(reference)
            current[field] = 'changed'
            with self.assertRaisesRegex(ValueError, 'controls drifted'):
                compare_trials(reference, current, gradients, gradients, 'detector.target.weight')

    def test_no_optimizer_step_or_top_level_research_import(self):
        source = Path(__file__).with_name('diagnose_v2xr_gpu_repeatability.py')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'step'
                             for node in ast.walk(tree)))
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name in ('torch', 'opencood') for alias in node.names))
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('opencood', sys.modules)

    def test_missing_observer_and_unused_target_cannot_report_repeatability(self):
        reference = self.trial()
        current = copy.deepcopy(reference)
        gradients = {'detector.target.weight': np.array([1.], np.float32)}
        del current['capture']['target_input']
        with self.assertRaisesRegex(ValueError, 'observer did not execute'):
            compare_trials(reference, current, gradients, gradients, 'detector.target.weight')
        unused = {'detector.target.weight': None}
        with self.assertRaisesRegex(ValueError, 'did not receive a gradient'):
            compare_trials(reference, reference, unused, unused, 'detector.target.weight')

    def test_split_gradient_maps_allow_unused_other_path_and_collect_boundaries(self):
        reference = {'detector.head.weight': np.array([1.], np.float32), 'aux.weight': None,
                     'boundary.fused_features': np.array([2.], np.float32)}
        current = copy.deepcopy(reference)
        current['boundary.fused_features'][0] += np.float32(.001)
        result = compare_gradient_maps(reference, current)
        self.assertEqual(result['changed_gradient_tensors'], 1)
        self.assertEqual(result['top_changed_gradients'][0]['name'], 'boundary.fused_features')
        with self.assertRaisesRegex(ValueError, 'registration changed'):
            compare_gradient_maps(reference, {})
        current['aux.weight'] = np.ones(1, np.float32)
        with self.assertRaisesRegex(ValueError, 'Used/unused'):
            compare_gradient_maps(reference, current)

    def test_crop_arguments_include_all_windows_in_original_order(self):
        batch = dict(auxiliary=[[dict(center=[1., 2.], valid=False), dict(center=[3., 4.], valid=True)],
                                [], [dict(center=[5., 6.], valid=True)]])
        owners, centers = crop_arguments(batch)
        self.assertEqual(owners, [0, 0, 2])
        np.testing.assert_array_equal(centers, [[1., 2.], [3., 4.], [5., 6.]])
        with self.assertRaisesRegex(ValueError, 'No crop windows'):
            crop_arguments(dict(auxiliary=[[], []]))

    def test_crop_replay_requires_fixed_upstream_inputs_and_rng(self):
        reference = dict(input='input', owners='owners', centers='centers', upstream='upstream',
                         rng_before='before', rng_after='after', output='out')
        gradient = np.array([1., 2.], np.float32)
        current = copy.deepcopy(reference)
        changed = np.array([1., 2.001], np.float32)
        result = compare_crop_trial(reference, current, gradient, changed)
        self.assertTrue(result['forward_equal'])
        self.assertFalse(result['gradient']['equal'])
        for key in ('input', 'owners', 'centers', 'upstream', 'rng_before', 'rng_after'):
            current = copy.deepcopy(reference)
            current[key] = 'drifted'
            with self.assertRaisesRegex(ValueError, 'controls drifted: ' + key):
                compare_crop_trial(reference, current, gradient, gradient)

    def test_auxiliary_condition_forward_is_part_of_split_controls(self):
        reference = self.trial()
        reference['capture']['condition'] = 'condition_a'
        current = copy.deepcopy(reference)
        current['capture']['condition'] = 'condition_b'
        gradients = {'detector.target.weight': np.array([1.], np.float32)}
        result = compare_trials(reference, current, gradients, gradients, 'detector.target.weight')
        self.assertEqual(result['observed'], 'forward_bitwise_variation_observed')
        self.assertFalse(result['forward_fields_equal']['condition'])

    def test_prior_probe_binds_the_same_source_checkpoint_and_batch(self):
        prior = dict(status='complete', group='E2', execution={'physical_gpu': '2'}, sources={'hash': 'frozen'},
                     source_checkpoint_sha256='checkpoint', source_commit_sha256='commit',
                     selected_microbatch={'indices': [1, 2]})
        current = copy.deepcopy(prior)
        current['status'] = 'running'
        verify_prior_probe(prior, current)
        for key in ('group', 'execution', 'sources', 'source_checkpoint_sha256', 'source_commit_sha256', 'selected_microbatch'):
            changed = copy.deepcopy(current)
            changed[key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'identity differs: ' + key):
                verify_prior_probe(prior, changed)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            verify_prior_probe(dict(prior, status='partial'), current)


if __name__ == '__main__':
    unittest.main(verbosity=2)
