"""Pure synthetic diagnostic checks; no Torch or research data is loaded."""
import unittest

import numpy as np

from .shared_drift import choose_frame, numeric_difference, stats_difference, tensor_tree_hash


class DiagnosticHelpers(unittest.TestCase):
    def test_magnitude_shape_and_nonfinite(self):
        self.assertTrue(numeric_difference([1.], [1.])['exact'])
        small = numeric_difference([1.], [1.000001])
        self.assertFalse(small['exact'])
        self.assertTrue(small['allclose_2e5'])
        self.assertFalse(numeric_difference([1.], [2.])['allclose_2e5'])
        self.assertFalse(numeric_difference([1.], [1., 2.])['same_shape'])
        self.assertFalse(numeric_difference([np.nan], [np.nan])['finite'])

    def test_select_recorded_failure_and_reject_other_frames(self):
        protocol = {'validation_indices': [10, 20]}
        def row(frame, weather, updated):
            return {'stage': 'S0-frame', 'split': 'validation', 'status': 'failed',
                    'frame': frame, 'weather': weather, 'updated_utc_epoch': updated}
        manifest = {'entries': {'a': row(10, 'clean', 1), 'b': row(20, 'rain', 2)}}
        self.assertEqual(choose_frame(protocol, manifest), ('rain', 20))
        self.assertEqual(choose_frame(protocol, manifest, weather='snow'), ('snow', 10))
        with self.assertRaises(ValueError):
            choose_frame(protocol, manifest, frame=30)

    def test_input_hash_order_shape_and_content(self):
        first = {'a': np.array([1., 2.]), 'b': 1}
        self.assertEqual(tensor_tree_hash(first), tensor_tree_hash(dict(reversed(list(first.items())))))
        self.assertNotEqual(tensor_tree_hash(first), tensor_tree_hash({'a': np.array([[1., 2.]]), 'b': 1}))
        self.assertNotEqual(tensor_tree_hash(first), tensor_tree_hash({'a': np.array([1., 3.]), 'b': 1}))

    def test_detection_stats_changes_are_separate_from_score_noise(self):
        baseline = {t: {'gt': 1, 'tp': [1], 'fp': [0], 'score': [.8]} for t in (.3, .5, .7)}
        current = {t: {'gt': 1, 'tp': [1], 'fp': [0], 'score': [.800001]} for t in (.3, .5, .7)}
        result = stats_difference(baseline, current)
        self.assertTrue(result['0.7']['same_tp_sequence'])
        self.assertTrue(result['0.7']['scores']['allclose_2e5'])
        current[.7]['tp'] = [0]
        self.assertFalse(stats_difference(baseline, current)['0.7']['same_tp_sequence'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
