"""Control-flow and report fixtures only; no Torch, models or real data."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class TrainingRecordsTests(unittest.TestCase):
    def test_training_records_survive_validation(self):
        tree = ast.parse((HERE / 'training.py').read_text(encoding='utf-8'))
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        assignments = [n for n in ast.walk(run) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'module_rows' for t in n.targets)]
        self.assertEqual(len(assignments), 1, 'Training records must not be reset during validation')
        self.assertIsInstance(assignments[0].value, ast.List)
        # Compare syntax nodes: ast.unparse formats tuple targets differently in
        # Python 3.10 and 3.11, although they execute identically.
        train_loop = next(n for n in ast.walk(run) if isinstance(n, ast.For)
                          and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name)
                          and n.iter.func.id == 'enumerate' and n.iter.args
                          and isinstance(n.iter.args[0], ast.Name) and n.iter.args[0].id == 'train_loader')
        calls = [n for n in ast.walk(run) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)]
        update = next(n for n in calls if n.func.value.id == 'optimizer' and n.func.attr == 'step')
        append = next(n for n in calls if n.func.value.id == 'module_rows' and n.func.attr == 'append')
        validation = next(n for n in ast.walk(run) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'val_losses' for t in n.targets))
        self.assertLess(assignments[0].lineno, train_loop.lineno)
        self.assertLess(update.lineno, append.lineno)
        self.assertLess(append.lineno, validation.lineno)
        self.assertTrue(any(isinstance(n, ast.keyword) and n.arg == 'repe'
                            and isinstance(n.value, ast.Name) and n.value.id == 'module_rows'
                            for n in ast.walk(run)))

    def report_fixture(self, root, missing=None):
        def dump(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding='utf-8')

        def read(path):
            return json.loads(Path(path).read_text(encoding='utf-8'))

        def complete(path):
            value = read(path)
            if value['status'] != 'complete':
                raise ValueError('Incomplete fixture')
            return value

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        stub = types.ModuleType('auditlib')
        stub.atomic_json, stub.complete, stub.read_json, stub.sha256 = dump, complete, read, digest
        spec = importlib.util.spec_from_file_location('repe_report_fixture', HERE / 'results.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'auditlib': stub}):
            spec.loader.exec_module(module)
        dump(root / 'self_test/summary.json', {'status': 'complete'})
        for variant in module.VARIANTS:
            folder = root / 'smoke' / variant
            folder.mkdir(parents=True)
            weight = folder / 'fixture.bin'
            weight.write_bytes(b'report fixture; not a checkpoint')
            dump(folder / 'smoke_checkpoint.json', dict(status='complete', checkpoint=str(weight), checkpoint_sha256=digest(weight)))
            dump(folder / 'evaluation_smoke12/summary.json', dict(status='complete', frame_count=12, checkpoint_sha256=digest(weight)))
            dump(folder / 'history.json', dict(epochs=[dict(repe=[] if variant == missing else [dict(loss=1)],
                 parameter_counts=dict(total=1), duration_seconds=1, peak_memory_bytes=1)]))
        return module, read

    def test_report_accepts_preserved_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, read = self.report_fixture(root)
            module.stage1(root)
            self.assertTrue(read(root / 'stage1/acceptance.json')['stage2_allowed'])
            report = (root / 'stage1/STAGE1_REPORT.md').read_text(encoding='utf-8')
            for variant in module.VARIANTS:
                self.assertIn(variant + ' 最后一次模块统计', report)

    def test_report_still_rejects_lost_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module, _ = self.report_fixture(root, missing='B3')
            with self.assertRaisesRegex(ValueError, 'diagnostic records: B3'):
                module.stage1(root)
            self.assertFalse((root / 'stage1/acceptance.json').exists())


if __name__ == '__main__':
    unittest.main()
