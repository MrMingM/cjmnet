"""Synthetic checks for the unapplied policy; no Torch or real data."""
import sys
import unittest

import numpy as np

from uecp_gspr_resume_policy_proposal import floating_array


class ProposalTests(unittest.TestCase):
    def decision(self, a, b, path, **options):
        return floating_array(a, b, path, same_checkpoint_verified=True, exact_load_verified=True, **options)

    def test_no_experiment_gate_is_passed_by_this_prototype(self):
        a = np.array([1.], np.float32)
        result = self.decision(a, a, 'model/weight')
        self.assertTrue(result['accepted'])
        self.assertFalse(result['policy_applied'])
        self.assertFalse(result['formal_recovery_gate_passed'])
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('opencood', sys.modules)

    def test_original_13_of_640_weight_failures_still_fail(self):
        a = np.full((64, 10), .2, np.float32)
        b = a.copy()
        b.ravel()[:13] += np.float32(.00011231)
        self.assertFalse(self.decision(a, b, 'model/pillar_vfe.weight')['accepted'])

    def test_small_moment_difference_near_zero_uses_per_tensor_norm(self):
        a = np.full(8192, .04, np.float32)
        a[0] = 0.
        b = a.copy()
        b[0] = np.float32(.00004567951)
        result = self.decision(a, b, 'optimizer/state/60/exp_avg')
        self.assertFalse(result['original_pointwise_pass'])
        self.assertTrue(result['optimizer_moment_norm_pass'])
        self.assertTrue(result['accepted'])
        self.assertFalse(self.decision(a, b, 'model/weight')['accepted'])

    def test_large_moment_errors_are_rejected(self):
        a = np.full(8192, .04, np.float32)
        b = a+.01
        self.assertFalse(self.decision(a, b, 'optimizer/state/60/exp_avg')['accepted'])

    def test_optimizer_steps_and_unknown_floating_state_remain_exact(self):
        a, b = np.array([2.], np.float32), np.array([2.0000002], np.float32)
        for path in ('optimizer/state/60/step', 'optimizer/state/60/unknown', 'rng/float_cache'):
            self.assertFalse(self.decision(a, b, path)['accepted'])

    def test_exact_loading_rejects_even_a_single_rounding_difference(self):
        a, b = np.array([1.], np.float32), np.array([1.0000001], np.float32)
        self.assertFalse(self.decision(a, b, 'model/weight', exact=True)['accepted'])
        self.assertFalse(self.decision(a, b, 'optimizer/state/1/exp_avg', exact=True)['accepted'])

    def test_dtype_nonfinite_and_missing_common_checkpoint_are_rejected(self):
        a = np.array([1.], np.float32)
        self.assertFalse(self.decision(a, a.astype(np.float64), 'model/weight')['accepted'])
        invalid = np.array([np.nan], np.float32)
        self.assertFalse(self.decision(invalid, invalid, 'model/weight')['accepted'])
        self.assertFalse(floating_array(a, a, 'model/weight')['accepted'])


if __name__ == '__main__':
    unittest.main()
