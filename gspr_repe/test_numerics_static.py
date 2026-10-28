"""Static/numeric-formula checks only. No Torch imports, models or real datasets."""
import ast
import math
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent


class NumericalRepairStaticTests(unittest.TestCase):
    def test_smoothed_spread_value_and_derivative(self):
        source=(ROOT/'modules.py').read_text(encoding='utf-8')
        self.assertIn('(spread.clamp_min(0) + 1e-6).sqrt() - 1e-3',source)
        self.assertNotIn('spread.clamp_min(0).sqrt()',source)
        for variance in (0.,1e-12,1e-6,.01,1.):
            value=math.sqrt(variance+1e-6)-1e-3
            derivative=.5/math.sqrt(variance+1e-6)
            self.assertTrue(math.isfinite(value) and value>=-1e-15)
            self.assertTrue(math.isfinite(derivative) and derivative<=500)
            self.assertTrue(math.isfinite(0*derivative))
        self.assertEqual(math.sqrt(1e-6)-1e-3,0.)

    def test_empty_trusted_sentinels_removed_before_norm(self):
        tree=ast.parse((ROOT/'modules.py').read_text(encoding='utf-8'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='statistics')
        text=ast.unparse(function)
        self.assertLess(text.index('low = torch.where'),text.index('extent ='))
        self.assertLess(text.index('high = torch.where'),text.index('extent ='))

    def test_full_model_checks_precede_update(self):
        tree=ast.parse((ROOT/'server_tests.py').read_text(encoding='utf-8'))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        text=ast.unparse(run)
        self.assertLess(text.index('finite_gradients(model, variant)'),text.index('optimizer.step()'))
        self.assertIn("finite_state(model, 'after optimizer', variant)",text)
        self.assertIn("phase='updated_model_eval'",text)
        self.assertIn('test_degenerate_statistics()',text)


if __name__=='__main__':
    unittest.main()
