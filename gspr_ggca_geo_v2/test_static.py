"""Local syntax/control-flow tests only. Do not import Torch or models."""
import ast
import dis
from pathlib import Path
import types
import unittest
from bridge import adapt_training

HERE = Path(__file__).resolve().parent


def tree(name):
    return ast.parse((HERE / name).read_text(encoding='utf-8'))


class StaticTests(unittest.TestCase):
    def test_pfn_forward_is_never_replaced(self):
        for node in ast.walk(tree('models.py')):
            if isinstance(node, ast.Assign):
                self.assertFalse(any(isinstance(t, ast.Attribute) and t.attr == 'forward' for t in node.targets))
        text = (HERE / 'models.py').read_text(encoding='utf-8')
        self.assertIn('super()._before_pfn(module, inputs)', text)
        self.assertIn('pfn.norm.register_forward_hook', text)
        self.assertNotIn('module.norm(', text)
        self.assertNotIn('module.linear(', text)

    def test_capture_preserves_autograd(self):
        cls = next(n for n in tree('models.py').body if isinstance(n, ast.ClassDef))
        hook = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_capture_normalized')
        self.assertNotIn('detach', ast.unparse(hook))
        self.assertNotIn('no_grad', ast.unparse(hook))

    def test_three_training_adaptations_compile_without_execution(self):
        module = types.ModuleType('ggca_training_fixture')
        adapt_training(module)
        names = {i.argval for i in dis.get_instructions(module.run) if isinstance(i.argval, str)}
        for name in ('initialize_model', 'restore_initial_rng', 'initialization_reference'):
            self.assertIn(name, names)

    def test_no_A0_training_stage(self):
        text = (HERE / 'coordinator.py').read_text(encoding='utf-8')
        self.assertNotIn("('A0', 'A1', 'A2', 'A3')", text)
        self.assertNotIn("command(args, 'train', folder, 'A0'", text)
        self.assertIn("for variant in ('A2','A3')", text)
        self.assertIn("command(args,'train',old_folder,'A1',old=True)", text)
        self.assertIn("if args.variant not in METHOD['trainable_variants']", (HERE / 'training.py').read_text())

    def test_records_survive_original_validation(self):
        old = ast.parse((HERE.parent / 'gspr_repe/training.py').read_text(encoding='utf-8'))
        run = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        initializers = [n for n in ast.walk(run) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'module_rows' for t in n.targets)]
        self.assertEqual(len(initializers), 1)

    def test_only_new_residual_uses_mask_and_no_global_scalar(self):
        cls = next(n for n in tree('models.py').body if isinstance(n, ast.ClassDef))
        before = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_before_vfe')
        self.assertNotIn('masked_fill', ast.unparse(before))
        module = (HERE / 'modules.py').read_text(encoding='utf-8')
        self.assertNotIn('nn.Parameter(', module)
        self.assertIn('nn.Linear(8, 16)', module)
        self.assertIn('nn.init.zeros_(self.project.weight)', module)

    def test_zero_neighbor_and_sparse_guards_present(self):
        text = (HERE / 'modules.py').read_text(encoding='utf-8')
        self.assertIn('torch.where(denominator > 1e-6, measured, prior.detach())', text)
        self.assertIn('((effective-2)/4).clamp(0, 1)', text)
        self.assertIn('use_reentrant=False', text)
        self.assertIn('if self.force_zero:', text)

    def test_inference_model_does_not_read_GT(self):
        text = (HERE / 'models.py').read_text(encoding='utf-8')
        for name in ('object_bbx', 'object_ids', 'final_gt', 'label_dict', 'clean_reference'):
            self.assertNotIn(name, text)
        diagnose = ast.unparse(tree('diagnose.py'))
        self.assertLess(diagnose.index('output = model(data)'), diagnose.index("['final_gt']"))

    def test_POSIX_shell(self):
        raw = (HERE / 'run_all.sh').read_bytes()
        self.assertNotIn(b'\r', raw)
        for token in (b'[[', b'pipefail', b'#!/bin/bash'):
            self.assertNotIn(token, raw)

    def test_preflight_creates_output_before_writing_config(self):
        run = next(n for n in tree('preflight.py').body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        text = ast.unparse(run)
        self.assertLess(text.index('root.mkdir('), text.index('reference = audit('))

    def test_evaluation_preserves_runtime_import_isolation(self):
        module = tree('evaluation.py')
        self.assertFalse(any(isinstance(n, ast.ImportFrom) and n.module == 'models' for n in ast.walk(module)))

    def test_A1_implementation_prefix_is_preserved_byte_for_byte(self):
        original = (HERE.parent / 'gspr_ggca/modules.py').read_bytes()
        new = (HERE / 'modules.py').read_bytes()
        self.assertTrue(new.startswith(original))
        self.assertEqual((HERE / 'models.py').read_bytes(), (HERE.parent / 'gspr_ggca/models.py').read_bytes())
        self.assertEqual((HERE / 'bridge.py').read_bytes(), (HERE.parent / 'gspr_ggca/bridge.py').read_bytes())

    def test_geometry_keeps_absolute_exchange_and_separate_trust(self):
        text = (HERE / 'geometry.py').read_text()
        self.assertIn("exchange = attention * pair['compatibility']", text)
        self.assertIn("(1-exchange_mass)[..., None]*f", text)
        self.assertIn('neighbor_trust', text)
        self.assertIn('mean_support', text)
        self.assertNotIn('nn.Parameter', text)

    def test_old_A1_is_referenced_not_copied_or_migrated(self):
        text = (HERE / 'coordinator.py').read_text()
        self.assertIn("folder=Path(migration['old_root'])/'A1'", text)
        self.assertIn('while not stop_complete(root)', text)
        self.assertNotIn('shutil.copy', text)
        self.assertIn("old_A2_A3_smoke_accepted_for_new_version=False", text)


if __name__ == '__main__':
    unittest.main()
