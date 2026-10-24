"""Pure NumPy/file/observer fixtures; never import Torch or access research data."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from uecp_gspr_resume_probe import (ARMS, META_FIELDS, STATE_FIELDS, classify,
    exact_live_state, numpy_tree, overlay, run_locals, summary, write_new)
from uecp_gspr_resume_diagnose import compare, read, sha


def pairs():
    return [dict(left=a, right=b, exact_inputs_rng_metadata=True,
                 numerical_within_registered_tolerance=True)
            for i, a in enumerate(ARMS[:4]) for b in ARMS[i+1:4]]


class ProbeTests(unittest.TestCase):
    def test_no_research_runtime_imported(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('opencood', sys.modules)

    def test_repeatability_requires_independent_restart_difference(self):
        rows = pairs()
        self.assertEqual(classify(rows, True), 'CONTROLLED_COMPARISONS_WITHIN_ORIGINAL_TOLERANCE')
        rows[0]['numerical_within_registered_tolerance'] = False
        self.assertEqual(classify(rows, True), 'RESTART_ASSOCIATED_DIFFERENCE_REQUIRES_FURTHER_CHECK')
        rows[-1]['numerical_within_registered_tolerance'] = False
        self.assertEqual(classify(rows, True), 'REPEATED_CALCULATION_EXCEEDS_ORIGINAL_TOLERANCE')

    def test_exact_load_inputs_and_complete_coverage_take_precedence(self):
        rows = pairs()
        rows[-1]['numerical_within_registered_tolerance'] = False
        self.assertEqual(classify(rows, False), 'RESTORE_OR_INPUT_MISMATCH')
        rows[0]['exact_inputs_rng_metadata'] = False
        self.assertEqual(classify(rows, True), 'RESTORE_OR_INPUT_MISMATCH')
        self.assertEqual(classify([], True), 'INCOMPLETE_CONTROLLED_COMPARISONS')
        self.assertEqual(classify(rows[:-1], True), 'INCOMPLETE_CONTROLLED_COMPARISONS')

    def test_copied_metadata_preserves_original_and_refuses_arm_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'original'
            probe = root / 'diagnostics/probe'
            root.mkdir()
            write_new(root / 'config.json', {'seed': 17})
            write_new(root / 'preflight/train_manifest.json', {'fixture': True})
            expected = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
            target = overlay(root, probe, 'continuous')
            self.assertEqual(read(target / 'config.json'), {'seed': 17})
            self.assertEqual(read(target / 'preflight/train_manifest.json'), {'fixture': True})
            self.assertEqual(expected, {name: sha(name) for name in expected})
            with self.assertRaises(FileExistsError):
                overlay(root, probe, 'continuous')
            with self.assertRaises(FileExistsError):
                write_new(root / 'config.json', {'seed': 99})

    def test_observer_finds_original_frame_through_scheduler_style_wrapper(self):
        def observer():
            return run_locals(original_run)
        def scheduler_wrapper():
            return observer()
        def original_run():
            position = 8
            global_step = 1
            assert position == 8 and global_step == 1
            return scheduler_wrapper()
        result = original_run()
        self.assertEqual((result['position'], result['global_step']), (8, 1))
        with self.assertRaisesRegex(ValueError, 'original training function'):
            observer()

    def test_live_state_includes_all_serialized_resume_fields(self):
        class Stateful:
            def state_dict(self):
                return {'weight': np.array([1., 2.], np.float32)}
        local = dict(variant='E0', epoch=1, position=8, global_step=1, totals={'frames': 8}, elapsed=2.5,
            actual=['input'], gradients=[.5], distributions=[], model=Stateful(), opt=Stateful(),
            scheduler=Stateful(), crit=Stateful(), contract='source identity', micro=1, order=[0]*16)
        training = SimpleNamespace(rng_state=lambda: {'cpu': np.array([2, 3], np.uint8)}, identity=lambda order: str(order))
        state = exact_live_state(local, training)
        self.assertEqual(set(state), set(STATE_FIELDS))
        self.assertTrue(set(META_FIELDS) <= set(state))
        original = copy.deepcopy(state)
        self.assertTrue(compare(original, state, 'load', 0., 0.)['exact'])
        state['optimizer']['weight'][0] += .00001
        self.assertFalse(compare(original, state, 'load', 0., 0.)['exact'])

    def test_tensor_adapter_copies_buffers_and_preserves_container_types(self):
        class FakeTensor:
            def __init__(self):
                self.values = np.array([1, 2], np.uint8)
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.values
        tensor = FakeTensor()
        output = numpy_tree({'rng': (tensor,)}, SimpleNamespace(Tensor=FakeTensor))
        tensor.values[0] = 99
        self.assertIsInstance(output['rng'], tuple)
        np.testing.assert_array_equal(output['rng'][0], [1, 2])

    def test_original_tolerance_and_nonfinite_rejection_retained_in_summary(self):
        a = np.full((64, 10), .2, np.float32)
        b = a.copy()
        b.ravel()[:13] += np.float32(.00011231)
        row = summary(compare(a, b, 'model', 1e-5, 1e-4))
        self.assertEqual(row['failed_floating_arrays'], 1)
        self.assertEqual(row['largest_relative_changes'][0]['mismatched_elements'], 13)
        self.assertFalse(row['within_registered_tolerance'])
        invalid = summary(compare(np.array([np.nan]), np.array([np.nan]), 'model', 1e-5, 1e-4))
        self.assertFalse(invalid['within_registered_tolerance'])


if __name__ == '__main__':
    unittest.main()
