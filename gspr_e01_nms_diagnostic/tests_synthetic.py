"""Synthetic engineering tests only: no Torch/OpenCOOD, weights, point clouds or GPU."""
import ast
import copy
from contextlib import ExitStack
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from nms_common import HERE, PROJECT, atomic, epoch_audit, identity, sha, swap
from nms_numeric import array_hash, box_findings, footprint_area, json_numbers, map_indices, summary
from nms_layers import LayerTrace, batchnorm_observation, trace_summary
from nms_bn import compare_pair, current_batch_forward, instance_forward, install_batch_statistics
import nms_main


def compile_function(path, function, namespace, container=None):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    nodes = tree.body
    if container:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == container).body
        nodes = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == 'install').body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == function)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function]


def temporary():
    result = tempfile.TemporaryDirectory(prefix='.e01_nms_synthetic_', dir=PROJECT)
    target = Path(result.name).resolve()
    if target.parent != PROJECT.resolve() or not target.name.startswith('.e01_nms_synthetic_'):
        raise ValueError('Synthetic temporary directory is outside workspace')
    return result


class BNCounterfactual(unittest.TestCase):
    def test_forward_override_restores_class_method_without_shadow(self):
        class Native:
            def forward(self, value):
                return value
        module = Native()
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with instance_forward(module, lambda self, value: value + 1):
                self.assertEqual(module.forward(2), 3)
                raise RuntimeError('synthetic')
        self.assertNotIn('forward', module.__dict__)
        self.assertIs(module.forward.__func__, Native.forward)

    def test_forward_override_restores_existing_instance_method(self):
        module = SimpleNamespace(forward=Mock(return_value='original'))
        original = module.forward
        with instance_forward(module, lambda self, value: value):
            self.assertEqual(module.forward(3), 3)
        self.assertIs(module.forward, original)

    def fake_torch(self, grad=False):
        torch = ModuleType('torch')
        torch.is_grad_enabled = lambda: grad
        nn = ModuleType('torch.nn')
        nn.functional = SimpleNamespace(batch_norm=Mock(return_value=object()))
        return {'torch': torch, 'torch.nn': nn}, nn.functional.batch_norm

    def test_batch_stats_function_never_receives_buffers_and_returns_exact_result(self):
        modules, operation = self.fake_torch()
        module = SimpleNamespace(training=False, track_running_stats=True,
            weight=object(), bias=object(), eps=.001,
            running_mean=np.array([1.]), running_var=np.array([2.]), num_batches_tracked=8)
        before = copy.deepcopy((module.running_mean, module.running_var, module.num_batches_tracked))
        value = object()
        with patch.dict(sys.modules, modules):
            result = current_batch_forward(module, value)
        self.assertIs(result, operation.return_value)
        operation.assert_called_once_with(value, running_mean=None, running_var=None,
            weight=module.weight, bias=module.bias, training=True, momentum=0., eps=.001)
        np.testing.assert_array_equal(module.running_mean, before[0])
        np.testing.assert_array_equal(module.running_var, before[1])
        self.assertEqual(module.num_batches_tracked, before[2])

    def test_grad_or_training_mode_is_refused(self):
        for grad, training in ((True, False), (False, True)):
            modules, operation = self.fake_torch(grad)
            module = SimpleNamespace(training=training, track_running_stats=True)
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, 'no_grad'):
                    current_batch_forward(module, object())
            operation.assert_not_called()

    def test_install_only_bn_and_restore_forward(self):
        class FakeBN:
            track_running_stats = True
            running_mean, running_var = object(), object()
            def forward(self, value):
                return value
        bn = FakeBN()
        model = SimpleNamespace(named_modules=lambda: [('bn', bn), ('other', object())])
        module = ModuleType('torch.nn.modules.batchnorm')
        module._BatchNorm = FakeBN
        with patch.dict(sys.modules, {'torch.nn.modules.batchnorm': module}):
            with ExitStack() as stack:
                self.assertEqual(install_batch_statistics(model, stack), ['bn'])
                self.assertIs(bn.forward.__func__, current_batch_forward)
        self.assertNotIn('forward', bn.__dict__)

    def test_batch_formula_uses_population_stats_and_has_no_eval_claim(self):
        values = LayerObservations().values()
        mean, variance, _, _, gamma, beta, x, _, eps = values
        values[-2] = (x - mean[:, None]) / np.sqrt(variance[:, None] + eps) * gamma[:, None] + beta[:, None]
        result = batchnorm_observation(*values, use_batch_statistics=True)
        self.assertTrue(result['sampled_normalization_formula']['elementwise_close'])
        self.assertEqual(result['sampled_normalization_formula']['basis'], 'current_batch_population')
        self.assertIsNone(result['sampled_eval_formula'])

    def pair(self):
        request = {key: 'synthetic' for key in ('checkpoint_sha256', 'training_contract_sha256',
            'original_request_identity', 'original_request_sha256', 'model_root', 'train_root', 'original_run_root')}
        request.update(original_E1_gpu='2', original_threads={'OMP_NUM_THREADS': '1'}, sources={'one': 'source'},
                       bn_batch_stats=False)
        batch_request = dict(request, bn_batch_stats=True)
        arrays = {key: summary(np.array([1, 2], dtype=np.int64)) for key in
            ('input_voxel_features', 'input_voxel_coords', 'input_voxel_num_points',
             'input_record_len', 'input_transformation_matrix')}
        arrays.update(psm=summary(np.array([2e6])), rm=summary(np.array([1e7])))
        native = dict(no_training=True, eligible_for_AP_or_selection=False, protocol_changes=False,
            BN_current_batch_counterfactual=False, native_model_all_tensors_loaded_exactly=True,
            frame={'index': 0, 'frame_key': ['scene', '000060'], 'condition': 'clean'},
            postprocess={'order': 'hwl', 'score_threshold': .2, 'nms_threshold': .15},
            arrays=arrays, outcome='zero_union_reproduced', native_nms_called=True,
            NMS_geometry={'zero_area': 48, 'polygons': 30215})
        batch = copy.deepcopy(native)
        batch.update(protocol_changes=True, BN_current_batch_counterfactual=True,
                     BN_counterfactual_modules=['bn'], outcome='not_reproduced_first_frame',
                     native_inference_returned=True, NMS_geometry={'zero_area': 0, 'polygons': 2})
        batch['arrays'].update(psm=summary(np.array([2.])), rm=summary(np.array([10.])))
        trace = dict(summary=dict(all_model_state_exact_after_forward=True, BN_buffers_exact_after_forward=True,
            model_state_before_sha256='a' * 64, observation_errors=[], state_observation_errors=[],
            BN_sample_formula_failures=[], BN_observed_calls=1),
            rows=[dict(module='bn', BN={'sampled_normalization_formula': {'basis': 'stored_running'}})])
        batch_trace = copy.deepcopy(trace)
        batch_trace['rows'][0]['BN']['sampled_normalization_formula']['basis'] = 'current_batch_population'
        return [native, batch, request, batch_request, trace, batch_trace]

    def test_exact_pair_reports_scale_and_never_claims_evaluation(self):
        result = compare_pair(*self.pair())
        self.assertTrue(result['same_input_exact'])
        self.assertEqual(result['native_divided_by_current_batch_output_scale']['rm_abs_max'], 1e6)
        self.assertFalse(result['eligible_for_AP_or_selection'])
        self.assertFalse(result['scientific_evaluation_complete'])

    def test_different_gpu_or_checkpoint_is_refused(self):
        for key in ('original_E1_gpu', 'checkpoint_sha256'):
            values = self.pair()
            values[3][key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                compare_pair(*values)

    def test_changed_frame_or_postprocess_is_refused(self):
        for key in ('frame', 'postprocess'):
            values = self.pair()
            values[1][key]['changed'] = True
            with self.assertRaisesRegex(ValueError, 'frame/order/postprocess'):
                compare_pair(*values)

    def test_missing_exact_load_proof_is_refused(self):
        values = self.pair()
        values[1]['native_model_all_tensors_loaded_exactly'] = False
        with self.assertRaisesRegex(ValueError, 'strict loaded-model'):
            compare_pair(*values)

    def test_different_bn_call_count_is_refused(self):
        values = self.pair()
        values[5]['summary']['BN_observed_calls'] = 2
        with self.assertRaisesRegex(ValueError, 'BN call count differs'):
            compare_pair(*values)

    def test_input_dtype_shape_or_hash_difference_is_refused(self):
        for key, changed in (('dtype', 'float64'), ('shape', [2, 1]), ('sha256', 'changed')):
            values = self.pair()
            values[1]['arrays']['input_voxel_features'][key] = changed
            with self.assertRaisesRegex(ValueError, 'tensor differs'):
                compare_pair(*values)

    def test_missing_record_len_is_refused_even_when_both_are_missing(self):
        values = self.pair()
        for index in (0, 1):
            del values[index]['arrays']['input_record_len']
        with self.assertRaisesRegex(ValueError, 'Missing or different'):
            compare_pair(*values)

    def test_different_model_start_or_modified_buffers_is_refused(self):
        for key, changed in (('model_state_before_sha256', 'b' * 64),
                             ('all_model_state_exact_after_forward', False)):
            values = self.pair()
            values[5]['summary'][key] = changed
            with self.assertRaises(ValueError):
                compare_pair(*values)

    def test_unlabeled_intervention_or_scored_result_is_refused(self):
        for key, changed in (('protocol_changes', False), ('eligible_for_AP_or_selection', True)):
            values = self.pair()
            values[1][key] = changed
            with self.assertRaisesRegex(ValueError, 'unscored and labeled'):
                compare_pair(*values)

    def test_failed_observation_wrong_basis_or_missing_module_is_refused(self):
        for failure in ('observation', 'formula', 'module'):
            values = self.pair()
            if failure == 'observation':
                values[5]['summary']['observation_errors'] = [{'error': 'synthetic'}]
            elif failure == 'formula':
                values[5]['rows'][0]['BN']['sampled_normalization_formula']['basis'] = 'stored_running'
            else:
                values[1]['BN_counterfactual_modules'] = ['other']
            with self.assertRaises(ValueError):
                compare_pair(*values)

    def test_no_nms_geometry_is_not_reported_as_zero_zero_area_boxes(self):
        values = self.pair()
        del values[1]['NMS_geometry']
        values[1].update(native_nms_called=False, native_prediction_is_none=True)
        result = compare_pair(*values)
        self.assertIsNone(result['current_batch']['NMS_geometry'])
        self.assertFalse(result['current_batch']['native_nms_called'])

    def test_zero_counterfactual_scale_has_no_infinite_ratio(self):
        values = self.pair()
        values[1]['arrays']['rm']['finite_abs_max'] = 0.
        result = compare_pair(*values)
        self.assertIsNone(result['native_divided_by_current_batch_output_scale']['rm_abs_max'])

    def test_comparison_orchestration_uses_two_fresh_outputs_and_same_leases(self):
        pair = self.pair()
        calls = []
        def fake(root, request, e01, leases):
            mode = int(request['bn_batch_stats'])
            calls.append((root.name, tuple(leases)))
            atomic(root / 'report.json', pair[mode])
            atomic(root / 'request.json', request)
            atomic(root / 'layer_trace.json', pair[4 + mode])
        with temporary() as directory:
            root = Path(directory)
            with patch.object(nms_main, 'run_diagnostic', side_effect=fake):
                with redirect_stdout(io.StringIO()):
                    nms_main.run_comparison(root, pair[2], object(), (7, 8, 9))
            result = json.loads((root / 'bn_comparison.json').read_text())
            self.assertTrue(result['same_start_model_state_exact'])
        self.assertEqual(calls, [('native', (7, 8, 9)), ('current_batch', (7, 8, 9))])

    def test_second_process_failure_preserves_first_and_never_completes_pair(self):
        def fake(root, request, e01, leases):
            if request['bn_batch_stats']:
                raise RuntimeError('synthetic second process failure')
            atomic(root / 'report.json', {'synthetic': True})
        with temporary() as directory:
            root = Path(directory)
            with patch.object(nms_main, 'run_diagnostic', side_effect=fake):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'second process'):
                        nms_main.run_comparison(root, {}, object(), (7,))
            self.assertTrue((root / 'native/report.json').is_file())
            self.assertFalse((root / 'bn_comparison.json').exists())
            self.assertEqual(json.loads((root / 'comparison_status.json').read_text())['status'], 'failed')


class LayerObservations(unittest.TestCase):
    def values(self):
        mean, variance = np.array([2., 5.]), np.array([4., 9.])
        running_mean, running_var = np.array([1., -1.]), np.array([2., 3.])
        gamma, beta = np.array([2., -3.]), np.array([.5, -.1])
        x = np.array([[1., 2., 3.], [-2., 4., 5.]])
        y = (x - running_mean[:, None]) / np.sqrt(running_var[:, None] + .001) * gamma[:, None] + beta[:, None]
        return [mean, variance, running_mean, running_var, gamma, beta, x, y, .001]

    def test_eval_formula_and_negative_affine_weight_without_input_mutation(self):
        values = self.values()
        before = [value.copy() for value in values[:-1]]
        result = batchnorm_observation(*values)
        self.assertTrue(result['sampled_eval_formula']['elementwise_close'])
        self.assertEqual(result['sampled_eval_formula']['samples'], 6)
        self.assertEqual(result['sampled_eval_formula']['max_absolute_error'], 0)
        for actual, expected in zip(values[:-1], before):
            np.testing.assert_array_equal(actual, expected)

    def test_wrong_native_sample_is_reported_not_repaired(self):
        values = self.values()
        values[-2][0, 0] += 1
        result = batchnorm_observation(*values)
        self.assertFalse(result['sampled_eval_formula']['elementwise_close'])
        self.assertEqual(result['sampled_eval_formula']['max_absolute_error'], 1.)

    def test_large_distribution_drift_does_not_fail_correct_eval_arithmetic(self):
        values = self.values()
        values[1][:] = 1e12
        result = batchnorm_observation(*values)
        self.assertGreater(result['batch_to_running_std_ratio_max'], 100000)
        self.assertTrue(result['sampled_eval_formula']['elementwise_close'])

    def test_nonfinite_sample_is_never_marked_close(self):
        values = self.values()
        values[-2][0, 0] = np.inf
        result = batchnorm_observation(*values)
        self.assertFalse(result['sampled_eval_formula']['finite'])
        self.assertFalse(result['sampled_eval_formula']['elementwise_close'])
        json.dumps(json_numbers(result), allow_nan=False)

    def test_misaligned_channel_statistics_refused(self):
        values = self.values()
        values[0] = np.zeros(3)
        with self.assertRaisesRegex(ValueError, 'do not align'):
            batchnorm_observation(*values)

    def test_misaligned_samples_refused(self):
        values = self.values()
        values[-2] = np.zeros((2, 1))
        with self.assertRaisesRegex(ValueError, 'sampled input/output'):
            batchnorm_observation(*values)

    def test_invalid_variance_refused(self):
        values = self.values()
        values[3][0] = -1
        with self.assertRaisesRegex(ValueError, 'variance/epsilon'):
            batchnorm_observation(*values)

    def test_threshold_crossing_uses_observed_order_and_remains_diagnostic(self):
        rows = [dict(index=i, module='layer' + str(i), operator='synthetic',
                     outputs={'output': dict(finite_abs_max=value, all_finite=True)})
                for i, value in enumerate((2, 2000, 2e6, 500))]
        result = trace_summary(rows)
        self.assertEqual(result['first_observed_output_at_or_above']['100']['module'], 'layer1')
        self.assertEqual(result['first_observed_output_at_or_above']['1000000']['module'], 'layer2')
        self.assertEqual(result['largest_outputs'][0]['module'], 'layer2')
        self.assertFalse(result['evaluation_or_selection_gate'])

    def test_nonfinite_outputs_and_observation_errors_are_visible(self):
        rows = [dict(index=0, module='synthetic', operator='synthetic',
                     outputs={'output': dict(finite_abs_max=float('nan'), all_finite=False)},
                     observation_error='synthetic observation failure')]
        result = trace_summary(rows)
        self.assertEqual(result['nonfinite_outputs'], [dict(module='synthetic', slot='output')])
        self.assertEqual(len(result['observation_errors']), 1)
        self.assertIsNone(result['first_observed_output_at_or_above']['100'])
        json.dumps(json_numbers(result), allow_nan=False)

    def test_missing_forward_does_not_claim_state_preservation(self):
        report = LayerTrace().report()
        self.assertIsNone(report['summary']['BN_buffers_exact_after_forward'])
        self.assertFalse(report['eligible_for_AP_or_selection'])

    def test_layer_hook_always_returns_none_and_has_no_state_updates(self):
        tree = ast.parse((HERE / 'nms_layers.py').read_text(encoding='utf-8'))
        hook = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'hook')
        returns = [node for node in ast.walk(hook) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertIsNone(ast.literal_eval(returns[0].value))
        forbidden = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr in ('train', 'copy_', 'zero_', 'reset_running_stats', 'load_state_dict')]
        self.assertEqual(forbidden, [])


class IPCPaths(unittest.TestCase):
    directory = '/tmp/gnms-12345678'

    def environment(self):
        return {key: self.directory for key in ('TMPDIR', 'TEMP', 'TMP')}

    def listener(self, address=None):
        result = MagicMock()
        result.__enter__.return_value.address = address or self.directory + '/pymp-12345678/listener-12345678'
        return Mock(return_value=result)

    def test_original_long_output_path_rejected_and_short_path_fits(self):
        original = ('/data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/'
                    '20261022T093359Z_3360532/tmp/pymp-12345678/listener-12345678')
        with self.assertRaisesRegex(ValueError, 'AF_UNIX path too long'):
            nms_main.unix_address_bytes(original)
        self.assertLess(nms_main.unix_address_bytes(self.directory + '/pymp-12345678/listener-12345678'), 107)
        self.assertEqual(nms_main.unix_address_bytes('/' + 'x' * 106), 107)
        with self.assertRaises(ValueError):
            nms_main.unix_address_bytes('/' + 'x' * 107)

    def test_socket_length_counts_encoded_bytes_not_characters(self):
        address = '/tmp/' + '\u4e2d' * 40
        self.assertLess(len(address), 107)
        with self.assertRaisesRegex(ValueError, 'AF_UNIX path too long'):
            nms_main.unix_address_bytes(address)

    def test_all_temp_overrides_reset_without_changing_gpu_threads_or_input(self):
        original = dict(TMPDIR='/long/output', TEMP='/other', TMP='/third',
                        ROCR_VISIBLE_DEVICES='2', OMP_NUM_THREADS='1', PYTHONHASHSEED='20261016')
        before = original.copy()
        result = nms_main.ipc_environment(original, self.directory)
        self.assertEqual(original, before)
        self.assertEqual({key: result[key] for key in self.environment()}, self.environment())
        self.assertEqual({key: value for key, value in result.items() if key not in self.environment()},
                         {key: value for key, value in original.items() if key not in self.environment()})

    def test_real_ipc_directory_allocation_remains_server_only(self):
        with patch.object(nms_main.sys, 'platform', 'win32'):
            with patch.object(nms_main.tempfile, 'TemporaryDirectory') as factory:
                with self.assertRaisesRegex(RuntimeError, 'server-only'):
                    with nms_main.short_ipc_directory():
                        self.fail('Must refuse local allocation')
        factory.assert_not_called()

    def test_private_directory_uses_explicit_short_base_and_closes_on_error(self):
        manager = MagicMock()
        manager.__enter__.return_value = self.directory
        with patch.object(nms_main.sys, 'platform', 'linux'):
            with patch.object(nms_main.tempfile, 'TemporaryDirectory', return_value=manager) as factory:
                with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                    with nms_main.short_ipc_directory() as directory:
                        self.assertEqual(directory, self.directory)
                        manager.__exit__.assert_not_called()
                        raise RuntimeError('synthetic')
        factory.assert_called_once_with(prefix='gnms-', dir='/tmp')
        manager.__exit__.assert_called_once()

    def test_probe_binds_and_closes_before_any_model_import(self):
        factory = self.listener()
        with patch.dict(nms_main.os.environ, self.environment(), clear=True):
            with patch.object(nms_main.tempfile, 'gettempdir', return_value=self.directory):
                result = nms_main.probe_ipc(self.directory, factory)
        factory.assert_called_once_with(family='AF_UNIX')
        factory.return_value.__exit__.assert_called_once()
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['checked_before_model_import'])
        self.assertLess(result['socket_path_bytes'], 107)
        self.assertFalse(any(k == 'torch' or k.startswith('opencood') for k in sys.modules))

    def test_wrong_inherited_temp_environment_fails_before_binding(self):
        environment = self.environment()
        environment['TEMP'] = '/unexpected'
        factory = self.listener()
        with patch.dict(nms_main.os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, 'environment differs'):
                nms_main.probe_ipc(self.directory, factory)
        factory.assert_not_called()

    def test_cached_long_temp_directory_fails_before_binding(self):
        factory = self.listener()
        with patch.dict(nms_main.os.environ, self.environment(), clear=True):
            with patch.object(nms_main.tempfile, 'gettempdir', return_value='/old/output/tmp'):
                with self.assertRaisesRegex(ValueError, 'Cached Python temporary'):
                    nms_main.probe_ipc(self.directory, factory)
        factory.assert_not_called()

    def test_socket_outside_private_directory_rejected_and_closed(self):
        factory = self.listener('/tmp/gnms-12345678-other/pymp-12345678/listener-12345678')
        with patch.dict(nms_main.os.environ, self.environment(), clear=True):
            with patch.object(nms_main.tempfile, 'gettempdir', return_value=self.directory):
                with self.assertRaisesRegex(ValueError, 'outside the private'):
                    nms_main.probe_ipc(self.directory, factory)
        factory.return_value.__exit__.assert_called_once()

    def test_bind_error_reaches_caller_immediately(self):
        error = OSError('synthetic bind failure')
        factory = Mock(side_effect=error)
        with patch.dict(nms_main.os.environ, self.environment(), clear=True):
            with patch.object(nms_main.tempfile, 'gettempdir', return_value=self.directory):
                with self.assertRaises(OSError) as caught:
                    nms_main.probe_ipc(self.directory, factory)
        self.assertIs(caught.exception, error)


class FakeSignals:
    """No actual signals are installed or sent in local tests."""
    SIGTERM, SIGINT, SIGHUP, SIGKILL = 15, 2, 1, 9
    SIG_DFL, SIG_IGN = 0, 1

    def __init__(self, ignored=()):
        self.handlers = {number: self.SIG_IGN if number in ignored else self.SIG_DFL
                         for number in (self.SIGTERM, self.SIGINT, self.SIGHUP)}
        self.changes = []

    def getsignal(self, number):
        return self.handlers[number]

    def signal(self, number, handler):
        previous = self.handlers[number]
        self.handlers[number] = handler
        self.changes.append((number, handler))
        return previous


class SignalHandling(unittest.TestCase):
    def test_nohup_and_background_ignored_signals_stay_ignored(self):
        api = FakeSignals(ignored=(1, 2))
        original = api.handlers.copy()
        handler = Mock()
        with nms_main.termination_handlers(handler, api):
            self.assertEqual(api.handlers[api.SIGHUP], api.SIG_IGN)
            self.assertEqual(api.handlers[api.SIGINT], api.SIG_IGN)
            self.assertIs(api.handlers[api.SIGTERM], handler)
        self.assertEqual(api.handlers, original)
        self.assertEqual({number for number, _ in api.changes}, {api.SIGTERM})

    def test_original_handlers_restored_after_exception(self):
        api = FakeSignals()
        original_handler = Mock()
        api.handlers[api.SIGINT] = original_handler
        before = api.handlers.copy()
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with nms_main.termination_handlers(Mock(), api):
                raise RuntimeError('synthetic')
        self.assertEqual(api.handlers, before)

    def test_heartbeat_timeout_does_not_interrupt_worker(self):
        api = FakeSignals(ignored=(1, 2))
        process = Mock()
        process.wait.side_effect = [nms_main.subprocess.TimeoutExpired('synthetic', 20), 0]
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(nms_main.wait_for_worker(process, 'synthetic.log', api), 0)
        self.assertIn('RUNNING first_frame', output.getvalue())
        self.assertEqual(process.wait.call_count, 2)

    def test_sigterm_is_named_and_child_cleanup_cannot_be_reinterrupted(self):
        api = FakeSignals(ignored=(1, 2))
        before = api.handlers.copy()
        process = Mock()
        process.wait.side_effect = lambda **kwargs: api.handlers[api.SIGTERM](api.SIGTERM, None)
        def cleanup(actual):
            self.assertIs(actual, process)
            api.handlers[api.SIGTERM](api.SIGTERM, None)
        with patch.object(nms_main, 'stop_child', side_effect=cleanup) as stop:
            with self.assertRaises(nms_main.DiagnosticInterrupted) as caught:
                nms_main.wait_for_worker(process, 'synthetic.log', api)
        stop.assert_called_once_with(process)
        self.assertEqual(caught.exception.number, 15)
        self.assertEqual(caught.exception.signal_name, 'SIGTERM')
        self.assertEqual(api.handlers, before)

    def test_completed_worker_is_not_signalled(self):
        process = Mock()
        process.poll.return_value = 0
        with patch.object(nms_main.os, 'killpg', create=True) as kill:
            nms_main.stop_child(process)
        kill.assert_not_called()
        process.wait.assert_not_called()

    def test_active_worker_group_is_terminated_and_reaped(self):
        process = Mock(pid=123456)
        process.poll.return_value = None
        with patch.object(nms_main.os, 'killpg', create=True) as kill:
            nms_main.stop_child(process)
        kill.assert_called_once_with(process.pid, 15)
        process.wait.assert_called_once_with(timeout=20)

    def test_cleanup_timeout_kills_only_owned_group_and_waits(self):
        process = Mock(pid=123456)
        process.poll.return_value = None
        process.wait.side_effect = [nms_main.subprocess.TimeoutExpired('synthetic', 20), -9]
        with patch.object(nms_main, 'signal', FakeSignals()):
            with patch.object(nms_main.os, 'killpg', create=True) as kill:
                nms_main.stop_child(process)
        self.assertEqual([call.args for call in kill.call_args_list], [(123456, 15), (123456, 9)])
        self.assertEqual(process.wait.call_count, 2)
        self.assertEqual(process.wait.call_args.kwargs, {})

    def test_worker_exit_race_still_waits_before_returning(self):
        process = Mock(pid=123456)
        process.poll.return_value = None
        with patch.object(nms_main.os, 'killpg', create=True, side_effect=ProcessLookupError):
            nms_main.stop_child(process)
        process.wait.assert_called_once_with(timeout=20)


class NativeFailure(unittest.TestCase):
    def test_original_zero_union_error_reproduced_with_toy_geometry(self):
        class Geometry:
            def __init__(self, area):
                self.area = area
            def union(self, other):
                return Geometry(self.area + other.area)
            def intersection(self, other):
                return Geometry(min(self.area, other.area))
        native = compile_function(PROJECT / 'v2xr_official_repro/upstream/opencood/utils/common_utils.py', 'compute_iou', dict(np=np))
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ZeroDivisionError):
                native(Geometry(0.), [Geometry(0.)])
        np.testing.assert_array_equal(native(Geometry(1.), [Geometry(1.)]), np.array([.5], dtype=np.float32))

    def wrapper(self, original, recorder):
        fake = SimpleNamespace(context=dict(phase='nms'), capture_pair_failure=recorder, details={})
        return compile_function(HERE / 'nms_observe.py', 'iou', dict(self=fake, original_iou=original), 'Observer'), fake

    def test_observer_returns_identical_native_values(self):
        result = np.array([.2, .8], dtype=np.float32)
        wrapped, fake = self.wrapper(lambda *args: result, lambda *args: self.fail('Unexpected error callback'))
        self.assertIs(wrapped(object(), []), result)
        self.assertEqual(fake.details, {})

    def test_original_exception_object_is_preserved(self):
        error = ZeroDivisionError('synthetic native division')
        calls = []
        def fail(*args):
            raise error
        wrapped, _ = self.wrapper(fail, lambda *args: calls.append(args))
        with self.assertRaises(ZeroDivisionError) as caught:
            wrapped('picked', ['candidate'])
        self.assertIs(caught.exception, error)
        self.assertIs(calls[0][2], error)

    def test_observer_logging_failure_does_not_hide_original_exception(self):
        error = ZeroDivisionError('original')
        def fail(*args):
            raise error
        def recorder(*args):
            raise OSError('synthetic diagnostic storage failure')
        wrapped, fake = self.wrapper(fail, recorder)
        with self.assertRaises(ZeroDivisionError) as caught:
            wrapped(None, [])
        self.assertIs(caught.exception, error)
        self.assertIn('storage failure', fake.details['pair_observer_error'])


class MappingAndNumbers(unittest.TestCase):
    def arrays(self):
        decoded = np.arange(21, dtype=np.float32).reshape(1, 3, 7)
        projected = np.arange(48, dtype=np.float32).reshape(2, 8, 3)
        return dict(native_prob=np.array([.8, .1, .9], dtype=np.float32), decoded=decoded,
            selected=decoded[0, [0, 2]].copy(), projected=projected,
            large_mask=np.array([True, False]), z_mask=np.array([True, True]),
            nms_boxes=projected[:1].copy())

    def test_exact_mapping_and_original_arrays_unchanged(self):
        arrays = self.arrays()
        before = {k: array_hash(v) for k, v in arrays.items()}
        result = map_indices(arrays, .2)
        self.assertEqual(result['status'], 'exact')
        self.assertEqual(result['selected_to_anchor'].tolist(), [0, 2])
        self.assertEqual(result['nms_to_selected'].tolist(), [0])
        self.assertEqual(before, {k: array_hash(v) for k, v in arrays.items()})

    def test_missing_stage_rejects_attribution(self):
        arrays = self.arrays()
        del arrays['selected']
        self.assertEqual(map_indices(arrays, .2)['status'], 'unavailable')

    def test_different_score_selection_does_not_guess_anchor(self):
        arrays = self.arrays()
        arrays['selected'][0, 0] += 1
        self.assertEqual(map_indices(arrays, .2)['status'], 'unavailable')

    def test_different_existing_filter_does_not_guess_box_index(self):
        arrays = self.arrays()
        arrays['nms_boxes'][0, 0, 0] += 1
        self.assertEqual(map_indices(arrays, .2)['status'], 'unavailable')

    def test_multiple_cav_call_mapping_is_not_falsely_accepted(self):
        arrays = self.arrays()
        arrays['decoded_calls'] = np.array([0, 1])
        self.assertEqual(map_indices(arrays, .2)['status'], 'unavailable')

    def test_nonfinite_values_are_explicit_strict_json(self):
        value = np.array([0., -2., np.nan, np.inf, -np.inf], dtype=np.float32)
        record = summary(value)
        self.assertEqual((record['nan'], record['positive_infinity'], record['negative_infinity']), (1, 1, 1))
        self.assertEqual((record['finite_min'], record['finite_max']), (-2., 0.))
        result = json_numbers(value)
        self.assertEqual(result, [0., -2., 'NaN', '+Infinity', '-Infinity'])
        json.dumps(result, allow_nan=False)

    def test_zero_decoded_width_distinguished_from_corner_precision_loss(self):
        corners = np.full((8, 3), 100., dtype=np.float32)
        selected = np.array([100, 100, 0, 1, 0, 4, 0], dtype=np.float32)
        self.assertEqual(box_findings(selected, corners, corners), ['native_decoded_nonpositive_width_or_length'])
        selected[4:6] = 1e-9
        self.assertEqual(box_findings(selected, corners, corners), ['positive_width_length_but_native_corners_have_zero_BEV_area'])

    def test_projection_failure_distinguished_from_native_corners(self):
        corners = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]] * 2, dtype=np.float32)
        self.assertEqual(footprint_area(corners), 2.)
        selected = np.array([0, 0, 0, 1, 1, 2, 0], dtype=np.float32)
        self.assertEqual(box_findings(selected, corners, np.zeros_like(corners)), ['positive_native_corner_area_but_projection_has_zero_BEV_area'])

    def test_translation_safe_diagnostic_area(self):
        corners = np.array([[100000, 100000, 0], [100002, 100000, 0], [100002, 100001, 0], [100000, 100001, 0]] * 2, dtype=np.float32)
        self.assertEqual(footprint_area(corners), 2.)


class StateAndScope(unittest.TestCase):
    def make_commit(self, root, epoch, contract):
        paths = {}
        for field in ('checkpoint', 'resume_state', 'clean_summary', 'trace'):
            file = root / (field + str(epoch) + '.synthetic')
            file.write_bytes((field + str(epoch)).encode())
            paths[field], paths[field + '_sha256'] = str(file), sha(file)
        atomic(root / 'epochs' / f'{epoch:02d}' / 'complete.json',
               dict(status='complete', epoch=epoch, contract=contract['identity'], **paths))

    def test_pending_detector_weight_is_not_full_state_commit(self):
        with temporary() as directory:
            root = Path(directory)
            contract = dict(identity='synthetic')
            self.make_commit(root, 1, contract)
            (root / 'net_epoch2.pth').write_bytes(b'synthetic weights only')
            result = epoch_audit(root, contract)
            self.assertEqual(result['last_committed_epoch'], 1)
            self.assertFalse(result['epoch_02_committed'])
            self.assertFalse(result['epoch_02_full_state_exists'])

    def test_changed_committed_full_state_stops_diagnosis(self):
        with temporary() as directory:
            root = Path(directory)
            contract = dict(identity='synthetic')
            self.make_commit(root, 1, contract)
            (root / 'resume_state1.synthetic').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'committed bytes changed'):
                epoch_audit(root, contract)

    def test_noncontiguous_commit_rejected(self):
        with temporary() as directory:
            root = Path(directory)
            contract = dict(identity='synthetic')
            self.make_commit(root, 2, contract)
            with self.assertRaisesRegex(ValueError, 'Noncontiguous'):
                epoch_audit(root, contract)

    def test_original_and_frozen_identity_formats_remain_distinct(self):
        self.assertNotEqual(identity(dict(a=1)), identity(dict(a=1), compact=False))

    def test_staticmethod_descriptor_restored_even_after_exception(self):
        class Native:
            @staticmethod
            def decode(value):
                return value
        descriptor = Native.__dict__['decode']
        with self.assertRaises(RuntimeError):
            with swap(Native, 'decode', staticmethod(lambda x: x + 1)):
                self.assertEqual(Native().decode(1), 2)
                raise RuntimeError('synthetic exit')
        self.assertIs(Native.__dict__['decode'], descriptor)
        self.assertEqual(Native().decode(1), 1)

    def test_no_training_or_optimizer_calls(self):
        for path in HERE.glob('*.py'):
            if path.name == 'tests_synthetic.py':
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
            forbidden = [ast.unparse(n.func) for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ('backward', 'step', 'train', 'setup_optimizer', 'inference_mode')]
            self.assertEqual(forbidden, [], str(path))
        self.assertNotIn(b'\r', (HERE / 'run_server.sh').read_bytes())

    def test_inference_entry_is_clean_smoke_only(self):
        tree = ast.parse((HERE / 'nms_observe.py').read_text(encoding='utf-8'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'evaluation.evaluate']
        self.assertEqual(len(calls), 1)
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}, dict(clean_only=True, smoke=True))
        self.assertIn('raise FirstFrameObserved', (HERE / 'nms_observe.py').read_text(encoding='utf-8'))

    def test_no_research_runtime_imported_locally(self):
        self.assertFalse(any(k == 'torch' or k.startswith('torch.') or k == 'opencood' or k.startswith('opencood.') for k in sys.modules))


def run_tests():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    run_tests()
