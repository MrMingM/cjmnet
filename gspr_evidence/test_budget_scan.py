import copy
import unittest
import torch
from .budget_scan import BUDGETS, choose_budget, passes
from .test_evidence import toy_model


class BudgetTests(unittest.TestCase):
    def summaries(self):
        full = dict(ap30=.9, ap50=.85, ap70=.8)
        return {w: dict(phase='development', comparison_contract={'seed': 1}, results={
            'full': full.copy(), '262144': dict(ap30=.89, ap50=.85, ap70=.8),
            '1048576': full.copy(), '2097152': full.copy()}) for w in ('clean', 'fog', 'rain', 'snow')}

    def test_all_twelve_metrics_and_no_rounding(self):
        rows = self.summaries()
        self.assertEqual(choose_budget(rows)['selected_budget_bytes'], 1048576)
        rows['snow']['results']['1048576']['ap70'] -= 1e-10
        self.assertEqual(choose_budget(rows)['selected_budget_bytes'], 2097152)
        rows['clean']['results']['2097152']['ap30'] -= .001
        self.assertEqual(choose_budget(rows)['selected_mode'], 'full')

    def test_incomplete_mismatched_and_test_results_rejected(self):
        rows = self.summaries()
        del rows['fog']
        with self.assertRaises(ValueError):
            choose_budget(rows)
        rows = self.summaries()
        rows['fog']['phase'] = 'benchmark'
        with self.assertRaises(ValueError):
            choose_budget(rows)
        rows = self.summaries()
        rows['rain']['comparison_contract']['seed'] = 2
        with self.assertRaises(ValueError):
            choose_budget(rows)

    def test_actual_packets_grow_and_saturate_at_full(self):
        model, encoded = toy_model()
        reference, full = model.run(encoded, 'full')
        previous = set()
        for budget in (0, 100, 350, 700, 100000):
            model.engine.budget = budget
            output, row = model.run(encoded, 'a0b0')
            self.assertLessEqual(row['total_bytes'], budget)
            selected = set(row['selected_ids'][0])
            self.assertTrue(previous <= selected)
            previous = selected
        self.assertEqual(row['total_bytes'], full['total_bytes'])
        for key in output:
            torch.testing.assert_close(output[key], reference[key])


if __name__ == '__main__':
    unittest.main()
