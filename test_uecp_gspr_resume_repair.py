"""Synthetic state, AST and report fixtures only; no research runtime."""
import ast
import copy
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from uecp_gspr_resume_compare_v2 import KEYS, compare_states
from uecp_gspr_resume_repair import (amended_report, compile_replay_run,
                                    compile_report_run, compile_worker_command, worker_hash_seed)


def state():
    result = {k: None for k in KEYS}
    result.update(model={'weight': np.full((64, 10), .2, np.float32)},
        optimizer={'state': {60: {'exp_avg': np.full(8192, .04, np.float32), 'step': np.array(2., np.float32)}},
                   'param_groups': [{'lr': .002, 'params': [60]}]},
        scheduler={'last_epoch': 1, '_last_lr': [.002]}, criterion={}, rng={'cpu': np.array([1, 2], np.uint8)},
        global_step=2, position=16, order_identity='same order', microbatch=1, accumulation_position=0,
        contract_identity='same contract', variant='E0', epoch=1,
        totals={'frames': 16, 'loss': 30.}, actual=[{'input_digest': 'same input'}], amp_scaler=None)
    return result


def function_source(path, name, owner=None):
    text = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(text)
    body = tree.body if owner is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return ast.get_source_segment(text, next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name))


class RepairTests(unittest.TestCase):
    def test_new_policy_only_relaxes_adam_moment_metric(self):
        a = state(); a['optimizer']['state'][60]['exp_avg'][0] = 0.
        b = copy.deepcopy(a); b['optimizer']['state'][60]['exp_avg'][0] = .00004567951
        result = compare_states(a, b, True, True)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertTrue(any(row.get('optimizer_moment_norm_pass') for row in result['numeric']))
        b['model']['weight'].ravel()[:13] += .00011231
        b['rng']['cpu'][0] = 99
        result = compare_states(a, b, True, True)
        self.assertEqual(result['status'], 'FAILED')
        self.assertTrue(any(r['path'].startswith('model/') for r in result['failures']))
        self.assertTrue(any(r['path'].startswith('rng/') for r in result['failures']))

    def test_counters_lr_steps_inputs_and_common_checkpoint_cannot_be_relaxed(self):
        for change in ('lr', 'step', 'input', 'count'):
            with self.subTest(change=change):
                a, b = state(), state()
                if change == 'lr': b['optimizer']['param_groups'][0]['lr'] += 1e-10
                elif change == 'step': b['optimizer']['state'][60]['step'] += 1e-6
                elif change == 'input': b['actual'][0]['input_digest'] = 'wrong'
                else: b['totals']['frames'] += 1
                self.assertEqual(compare_states(a, b, True, True)['status'], 'FAILED')
        self.assertEqual(compare_states(state(), state(), False, True)['status'], 'FAILED')
        self.assertEqual(compare_states(state(), state(), True, False)['status'], 'FAILED')

    def test_missing_nonfinite_and_large_moments_fail(self):
        a, b = state(), state()
        del b['criterion']
        self.assertEqual(compare_states(a, b, True, True)['status'], 'FAILED')
        b = state(); b['optimizer']['state'][60]['exp_avg'][0] = np.nan
        self.assertEqual(compare_states(a, b, True, True)['status'], 'FAILED')
        b = state(); b['optimizer']['state'][60]['exp_avg'] += .01
        self.assertEqual(compare_states(a, b, True, True)['status'], 'FAILED')

    def test_actual_training_ast_changes_only_replay_directory(self):
        source = function_source('uecp_gspr_v2xr/train.py', 'run')
        namespace = {}
        fn = compile_replay_run(source, namespace)
        self.assertIs(fn.__globals__, namespace)
        before = ast.parse(source)
        for node in ast.walk(before):
            if isinstance(node, ast.Constant) and node.value == 'smoke_replay':
                node.value = 'smoke_replay_v2'
        expected = {}
        exec(compile(ast.fix_missing_locations(before), '<expected namespace-only edit>', 'exec'), expected)
        self.assertEqual(fn.__code__.co_code, expected['run'].__code__.co_code)
        self.assertEqual(fn.__code__.co_consts, expected['run'].__code__.co_consts)
        namespace['restore_rng'] = object()
        self.assertIs(fn.__globals__['restore_rng'], namespace['restore_rng'])

    def test_worker_command_contains_real_archive_worker_and_project(self):
        source = "def run(self, action, *extra):\n    cmd = [sys.executable, '-B', str(HERE / 'worker.py'), action, *extra]\n    return cmd\n"
        fn = compile_worker_command(source, {'sys': sys, 'HERE': Path('/original')}, PurePosixPath('/new/fix.zip'), PurePosixPath('/project'))
        actual = fn(None, 'train', '--output', '/output')
        self.assertEqual(actual, [sys.executable, '-B', '/new/fix.zip', 'worker', 'train', '--output', '/output', '--project', '/project'])
        real = function_source('uecp_gspr_v2xr/pipeline.py', 'run', owner='Pipeline')
        compile_worker_command(real, {}, Path('/fix.zip'), Path('/project'))
        with self.assertRaises(ValueError):
            compile_worker_command(source.replace('worker.py', 'unknown.py'), {}, Path('/fix.zip'), Path('/project'))

    def test_report_gate_rewrite_is_scoped_and_partial_never_creates_final(self):
        from uecp_gspr_v2xr import report
        source = function_source('uecp_gspr_v2xr/report.py', 'run')
        compile_report_run(source, {})
        def fixture_json(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding='utf-8')
        def fixture_bytes(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(value)
        # Prefix/immutability logic is under test; native atomic I/O is unchanged.
        with tempfile.TemporaryDirectory() as directory, patch.object(report, 'write', fixture_json), patch.object(report, 'atomic_bytes', fixture_bytes):
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'variants': ['E0'], 'epochs': 20, 'seed': 17}), encoding='utf-8')
            (root / 'protocol.json').write_text('{}', encoding='utf-8')
            marker = {'identity': 'repair', 'archive_sha256': 'archive'}
            result = amended_report(root, marker, partial=True)
            self.assertEqual(result['status'], 'FAILED')
            self.assertTrue((root / 'PARTIAL_RESULTS.json').exists())
            self.assertFalse((root / 'FINAL_RESULTS.json').exists())
            (root / 'FINAL_RESULTS.json').write_text(json.dumps({'status': 'COMPLETE', 'execution_source_repair': {'identity': 'repair'}}), encoding='utf-8')
            for name in ('FINAL_RESULTS.md', 'ABLATION_RESULTS.md', 'FINAL_metrics.csv'):
                (root / name).write_text('synthetic immutable report', encoding='utf-8')
            frozen = (root / 'FINAL_RESULTS.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'missing or invalid evidence'):
                amended_report(root, marker)
            self.assertEqual((root / 'FINAL_RESULTS.json').read_bytes(), frozen)

    def test_imports_keep_research_runtime_unloaded(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('opencood', sys.modules)

    def test_native_evaluation_hash_seed_is_preserved(self):
        for action in ('dev', 'final'):
            self.assertEqual(worker_hash_seed('worker', action, 20261017), '20261016')
        for action in ('train', 'resume-check', 'diagnose', 'select'):
            self.assertEqual(worker_hash_seed('worker', action, 20261017), '20261017')
        self.assertEqual(worker_hash_seed('run', None, 23), '23')


if __name__ == '__main__':
    unittest.main()
