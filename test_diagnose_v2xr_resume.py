"""Synthetic file/NumPy checks for the standalone server diagnostic; no research runtime."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from diagnose_v2xr_resume import ARTIFACTS, Comparison, KEYS, inspect, sha


class DiagnosticTests(unittest.TestCase):
    def test_local_diagnostic_import_keeps_research_runtime_unloaded(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('opencood', sys.modules)

    def test_full_report_does_not_stop_at_first_optimizer_failure(self):
        c = Comparison()
        c.compare({'state': {3: {'exp_avg': np.array([0.1], np.float32)}}},
                  {'state': {3: {'exp_avg': np.array([0.10006133], np.float32)}}}, 'optimizer')
        c.compare({'cpu': np.array([1, 2], np.uint8)}, {'cpu': np.array([1, 3], np.uint8)}, 'rng')
        report = c.summary()
        self.assertFalse(report['sections']['optimizer']['passed'])
        self.assertFalse(report['sections']['rng']['passed'])
        self.assertEqual(report['failed_floating_arrays'][0]['path'], 'optimizer/state/3/exp_avg')
        self.assertGreater(report['failed_floating_arrays'][0]['max_tolerance_ratio'], 1)

    def test_tensor_relative_tolerance_and_empty_arrays(self):
        c = Comparison()
        c.compare(np.array([1.], np.float32), np.array([1.00005], np.float32), 'model/w')
        c.compare(np.array([], np.float32), np.array([], np.float32), 'aux/w')
        self.assertTrue(c.summary()['registered_gate_passed'])

    def test_nonfinite_equal_nan_and_infinity_are_rejected(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            c = Comparison()
            c.compare(np.array([value], np.float32), np.array([value], np.float32), 'model/w')
            report = c.summary()
            self.assertFalse(report['registered_gate_passed'])
            json.dumps(report, allow_nan=False)

    def test_shape_dtype_and_dictionary_keys_are_exact(self):
        for a, b in [(np.ones(2, np.float32), np.ones(3, np.float32)),
                     (np.ones(2, np.float32), np.ones(2, np.float64)),
                     ({'a': 1}, {'b': 1}), ((1, 2), [1, 2])]:
            c = Comparison()
            c.compare(a, b, 'rng')
            self.assertFalse(c.summary()['registered_gate_passed'])

    def fixture(self):
        parent = Path(__file__).resolve().parent if sys.platform == 'win32' else Path(tempfile.gettempdir())
        temp = tempfile.TemporaryDirectory(prefix='.resume_diagnostic_fixture_', dir=parent)
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.assertIn(parent.resolve(), root.resolve().parents)
        (root / 'contract.json').write_text(json.dumps(dict(identity='synthetic contract')), encoding='utf-8')
        states = {}
        for branch in ('smoke', 'smoke_replay'):
            for epoch in (1, 2):
                folder = root / branch / 'E0/epochs' / f'{epoch:02d}'
                folder.mkdir(parents=True)
                for name in ARTIFACTS:
                    content = json.dumps(dict(epoch=epoch, synthetic=True)).encode() if name.endswith('.json') else b'synthetic checkpoint bytes'
                    (folder / name).write_bytes(content)
                (folder / 'train_complete.json').write_text(json.dumps(dict(status='complete', smoke_only=True,
                    group='E0', epoch=epoch, physical_gpu='2', gpu_name='synthetic GPU', contract_identity='synthetic contract',
                    output_hashes={name: sha(folder / name) for name in ARTIFACTS})), encoding='utf-8')
                state = {key: None for key in KEYS}
                state.update(model={'w': np.array([1.], np.float32)}, optimizer={'state': {3: {'exp_avg': np.array([.1], np.float32)}}},
                             rng={'cpu': np.array([1, 2], np.uint8)}, completed_training=True, group='E0', amp_scaler=None)
                states[folder / 'resume.pth'] = state
        return root, states

    def test_inspects_all_existing_states_without_writing_originals(self):
        root, states = self.fixture()
        snapshots = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
        repeated = states[root / 'smoke_replay/E0/epochs/02/resume.pth']
        repeated['optimizer']['state'][3]['exp_avg'] += np.float32(6.13e-5)
        report = inspect(root, 'E0', lambda path: states[path])
        self.assertEqual(report['conclusion'], 'numeric_tolerance_exceeded_cause_not_yet_verified')
        self.assertTrue(report['sections']['model']['passed'])
        self.assertTrue(report['sections']['rng']['passed'])
        self.assertTrue(report['epoch1_start_identical'])
        self.assertTrue(all(p.read_bytes() == contents for p, contents in snapshots.items()))
        self.assertEqual(len(snapshots), sum(p.is_file() for p in root.rglob('*')))

    def test_corrupt_committed_checkpoint_is_rejected_before_loading(self):
        root, states = self.fixture()
        path = root / 'smoke_replay/E0/epochs/02/resume.pth'
        path.write_bytes(b'changed bytes')
        with self.assertRaisesRegex(ValueError, 'artifact hash differs'):
            inspect(root, 'E0', lambda path: self.fail('must not load corrupted checkpoint'))

    def test_data_drift_is_reported_even_if_all_state_tensors_pass(self):
        root, states = self.fixture()
        folder = root / 'smoke_replay/E0/epochs/02'
        (folder / 'actual_samples.json').write_text(json.dumps(dict(synthetic='different input digests')), encoding='utf-8')
        commit = json.loads((folder / 'train_complete.json').read_text())
        commit['output_hashes']['actual_samples.json'] = sha(folder / 'actual_samples.json')
        (folder / 'train_complete.json').write_text(json.dumps(commit), encoding='utf-8')
        report = inspect(root, 'E0', lambda path: states[path])
        self.assertEqual(report['conclusion'], 'checkpoint_start_device_or_data_mismatch')
        self.assertFalse(report['epoch2_schedules_identical']['actual_samples.json'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
