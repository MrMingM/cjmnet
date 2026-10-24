"""Synthetic diagnostic/loader-extraction tests only; no Torch/OpenCOOD/data execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from uecp_gspr_resume_diagnose import (KNOWN_TRAINING_PATH, ORIGINAL_TRAINING_SHA256,
    REVIEWED_TRAINING_SHA256, committed_epoch, compare, diagnostic_sources,
    restore_block, review_helpers, run, sha)


class ResumeDiagnosticTests(unittest.TestCase):
    def test_research_runtime_not_imported_locally(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('opencood', sys.modules)

    def test_small_weight_difference_does_not_hide_rng_failure(self):
        x = np.full((64, 10), .2, dtype=np.float32)
        y = x.copy()
        y.ravel()[:13] += np.float32(.00011231)
        result = compare(dict(model=x, rng=np.array([1, 2], np.uint8)),
                         dict(model=y, rng=np.array([1, 3], np.uint8)), '', 1e-5, 1e-4)
        self.assertFalse(result['exact'])
        self.assertEqual(result['failed_floating_arrays'][0]['mismatched_elements'], 13)
        self.assertEqual(result['structural_or_exact_errors'][0]['path'], '/rng')

    def test_exact_load_comparison_allows_no_float_difference(self):
        result = compare(np.array([1.], np.float32), np.array([1.0000001], np.float32), 'load', 0, 0)
        self.assertFalse(result['exact'])
        self.assertEqual(len(result['failed_floating_arrays']), 1)

    def test_dtype_shape_and_nonfinite_cannot_pass(self):
        for a, b in ((np.zeros(1, np.float32), np.zeros(1, np.float64)),
                     (np.zeros(1), np.zeros(2)), (np.array([np.nan]), np.array([np.nan]))):
            with self.subTest(a=a, b=b):
                result = compare(a, b, 'value', 1e-5, 1e-4)
                self.assertFalse(result['within_registered_tolerance'])
                self.assertTrue(result['structural_or_exact_errors'])

    def test_saved_commit_checks_content_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'resume.pth'
            artifact.write_bytes(b'synthetic bytes, no checkpoint runtime')
            row = dict(status='COMPLETE', smoke_only=True, variant='E0', epoch=1,
                       output_hashes={'resume.pth': sha(artifact)})
            (root / 'train_complete.json').write_text(json.dumps(row), encoding='utf-8')
            self.assertEqual(committed_epoch(root, 1), row)
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact bytes changed'):
                committed_epoch(root, 1)

    def source_fixture(self, root, updated):
        project = Path(__file__).parent
        with zipfile.ZipFile(project / 'uecp_gspr_v2xr_sync_v2.zip') as archive:
            original = archive.read(KNOWN_TRAINING_PATH)
        target = root / KNOWN_TRAINING_PATH
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        self.assertEqual(sha(target), ORIGINAL_TRAINING_SHA256)
        pins = {KNOWN_TRAINING_PATH: sha(target)}
        dependency = root / 'synthetic_other_dependency.py'
        dependency.write_bytes(b'not imported, fixture only\n')
        pins[dependency.name] = sha(dependency)
        own = root / 'uecp_gspr_v2xr'
        own.mkdir()
        (own / 'PINS.json').write_text(json.dumps({'files': pins}), encoding='utf-8')
        (own / 'synthetic.py').write_bytes(b'not executed\n')
        files = {p.name: sha(p) for p in own.iterdir()}
        (own / 'source_snapshot.json').write_text(json.dumps({'files': files}), encoding='utf-8')
        if updated:
            target.write_bytes((project / KNOWN_TRAINING_PATH).read_bytes())
            self.assertEqual(sha(target), REVIEWED_TRAINING_SHA256)
        return dict(own=files, dependencies=pins)

    def test_original_sources_need_no_diagnostic_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorded = self.source_fixture(root, updated=False)
            result = diagnostic_sources(root, recorded)
            self.assertEqual(result['diagnosis_only_reviewed_differences'], [])

    def test_only_exact_known_legacy_update_is_reviewed_without_repinning(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorded = self.source_fixture(root, updated=True)
            files_before = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
            result = diagnostic_sources(root, recorded)
            self.assertEqual(len(result['diagnosis_only_reviewed_differences']), 1)
            self.assertFalse(result['original_pipeline_gate_modified'])
            self.assertEqual(files_before, {str(p): sha(p) for p in root.rglob('*') if p.is_file()})

    def test_unknown_legacy_change_and_other_dependency_change_are_refused(self):
        for changed in (KNOWN_TRAINING_PATH, 'synthetic_other_dependency.py'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                recorded = self.source_fixture(root, updated=True)
                target = root / changed
                target.write_bytes(target.read_bytes()+b'\n# unreviewed change\n')
                with self.assertRaisesRegex(ValueError, 'Unreviewed source differences'):
                    diagnostic_sources(root, recorded)

    def test_sealed_package_and_pin_changes_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorded = self.source_fixture(root, updated=True)
            (root / 'uecp_gspr_v2xr/synthetic.py').write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError, 'Sealed UECP package differs'):
                diagnostic_sources(root, recorded)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorded = self.source_fixture(root, updated=True)
            altered = copy.deepcopy(recorded)
            altered['dependencies'][KNOWN_TRAINING_PATH] = REVIEWED_TRAINING_SHA256
            with self.assertRaisesRegex(ValueError, 'dependency pins were changed'):
                diagnostic_sources(root, altered)

    def test_helper_review_compares_original_delivery_and_rejects_function_changes(self):
        with zipfile.ZipFile(Path(__file__).parent / 'uecp_gspr_v2xr_sync_v2.zip') as archive:
            source = archive.read(KNOWN_TRAINING_PATH).decode('utf-8')
        self.assertTrue(review_helpers(source)['all_helper_bodies_unchanged'])
        with self.assertRaisesRegex(ValueError, 'helper or module-level'):
            review_helpers(source.replace("torch.set_rng_state(rng['cpu'])", 'torch.manual_seed(0)'))
        with self.assertRaisesRegex(ValueError, 'helper or module-level'):
            review_helpers(source+'\nraise RuntimeError("not executed")\n')

    def test_exact_existing_loader_extracted_without_importing_train(self):
        path = Path(__file__).parent / 'uecp_gspr_v2xr/train.py'
        source = path.read_text(encoding='utf-8')
        code = restore_block(source)
        calls = []

        class Holder:
            def __init__(self, name):
                self.name = name

            def load_state_dict(self, value, **options):
                calls.append((self.name, copy.deepcopy(value), options))

        state = dict(model={'weight': 1}, optimizer={'step': 2}, scheduler={'epoch': 1},
                     criterion={}, global_step=2, epoch=1, position=16, totals={'updates': 2},
                     elapsed=3., actual=['input hash'], gradients=[.25], distributions=[], rng=['saved RNG'])
        env = dict(model=Holder('model'), opt=Holder('opt'), scheduler=Holder('scheduler'),
                   crit=Holder('crit'), restored=state, epoch=1,
                   restore_rng=lambda rng: calls.append(('rng', rng, {})))
        exec(code, env)
        self.assertEqual([r[0] for r in calls], ['model', 'opt', 'scheduler', 'crit', 'rng'])
        self.assertEqual(calls[0][2], {'strict': True})
        self.assertEqual(env['position'], 16)
        self.assertEqual(env['global_step'], 2)
        self.assertNotIn('restored', env)
        with self.assertRaisesRegex(ValueError, 'differs from reviewed loader'):
            restore_block(source.replace("restore_rng(restored['rng'])", 'restore_rng({})'))

    def test_server_execution_guard_runs_before_torch_import(self):
        if sys.platform == 'linux':
            self.skipTest('This guard is specifically for the local Windows environment')
        with self.assertRaisesRegex(RuntimeError, 'Linux server'):
            run('unused', 1)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
