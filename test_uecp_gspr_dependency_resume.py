"""Source/ZIP/AST and synthetic module fixtures only; never train or load a model."""
import ast
from contextlib import contextmanager
import copy
import importlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import uecp_gspr_dependency_resume as bridge
from build_uecp_dependency_resume import payloads

ROOT = Path(__file__).resolve().parent


def archive_write(path, members):
    values = dict(members)
    values['manifest.json'] = json.dumps(dict(files={n: bridge.digest(raw) for n, raw in members.items()})).encode()
    with zipfile.ZipFile(path, 'w') as archive:
        for name, raw in values.items():
            archive.writestr(name, raw)


@contextmanager
def package_namespace():
    prefix = 'v2xr_raygen_e0e3'
    saved = {k: v for k, v in list(sys.modules.items()) if k == prefix or k.startswith(prefix+'.')}
    for k in saved:
        del sys.modules[k]
    paths = list(sys.path)
    try:
        yield
    finally:
        for k in list(sys.modules):
            if k == prefix or k.startswith(prefix+'.'):
                del sys.modules[k]
        sys.modules.update(saved)
        sys.path[:] = paths


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = payloads()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='.uecp_source_synthetic_', dir=ROOT)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def zip(self):
        path = self.root / 'fixture.zip'
        archive_write(path, self.members)
        return path

    def test_archive_has_original_v2_bytes_and_exact_snapshot_hashes(self):
        archive = bridge.verify_zip(self.zip())
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(bridge.digest(z.read('legacy/uecp_gspr_resume_fix_v2.zip')), bridge.LEGACY_SHA)
            for name in bridge.MODULES:
                self.assertEqual(bridge.digest(z.read('snapshots/'+name+'.py')), bridge.SNAPSHOT[name])

    def test_manifest_tampering_rejected(self):
        archive = self.zip()
        with zipfile.ZipFile(archive) as z:
            members = {n: z.read(n) for n in z.namelist()}
        members['uecp_gspr_dependency_resume.py'] += b'\n# unexpected\n'
        with zipfile.ZipFile(archive, 'w') as z:
            for n, raw in members.items():
                z.writestr(n, raw)
        with self.assertRaises(ValueError):
            bridge.verify_zip(archive)

    def test_resealed_wrong_snapshot_still_rejected(self):
        members = dict(self.members)
        members['snapshots/runtime.py'] += b'\n# unknown snapshot\n'
        path = self.root / 'wrong.zip'
        archive_write(path, members)
        with self.assertRaises(ValueError):
            bridge.verify_zip(path)

    def test_resealed_wrong_v2_still_rejected(self):
        members = dict(self.members)
        members['legacy/uecp_gspr_resume_fix_v2.zip'] += b'unknown'
        path = self.root / 'wrong_v2.zip'
        archive_write(path, members)
        with self.assertRaises(ValueError):
            bridge.verify_zip(path)

    def test_prepare_files_is_byte_exact_and_idempotent(self):
        archive = self.zip()
        legacy, snapshots = bridge.prepare_files(archive, self.root)
        self.assertEqual(bridge.sha(legacy), bridge.LEGACY_SHA)
        bridge.prepare_files(archive, self.root)
        for n in bridge.MODULES:
            self.assertEqual(bridge.sha(snapshots / (n+'.py')), bridge.SNAPSHOT[n])

    def test_modified_private_source_cannot_be_overwritten(self):
        archive = self.zip()
        _, snapshots = bridge.prepare_files(archive, self.root)
        (snapshots / 'runtime.py').write_text('unexpected', encoding='utf-8')
        with self.assertRaises(ValueError):
            bridge.prepare_files(archive, self.root)
        self.assertEqual((snapshots / 'runtime.py').read_text(encoding='utf-8'), 'unexpected')

    def test_source_view_redirects_only_three_exact_paths(self):
        view = bridge.SnapshotProject('/actual', '/private')
        for n in bridge.MODULES:
            self.assertEqual(view / ('v2xr_raygen_e0e3/'+n+'.py'), Path('/private') / (n+'.py'))
        for n in ('uecp_gspr_v2xr', 'v2xr_raygen_e0e3/data.py', 'attfuse_gspr/reliability.py'):
            self.assertEqual(view / n, Path('/actual') / n)

    def test_original_source_checker_accepts_actual_executed_view(self):
        from uecp_gspr_resume_diagnose import diagnostic_sources
        _, snapshots = bridge.prepare_files(self.zip(), self.root)
        recorded = dict(own=bridge.read(ROOT / 'uecp_gspr_v2xr/source_snapshot.json')['files'],
            dependencies=bridge.read(ROOT / 'uecp_gspr_v2xr/PINS.json')['files'])
        result = diagnostic_sources(bridge.SnapshotProject(ROOT, snapshots), recorded)
        rows = result['diagnosis_only_reviewed_differences']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['path'], 'v2xr_raygen_e0e3/training.py')
        self.assertEqual(rows[0]['current_sha256'], bridge.SNAPSHOT['training'])
        self.assertTrue(rows[0]['all_helper_bodies_unchanged'])

    def test_physical_source_version_logged_and_unknown_refused(self):
        folder = self.root / 'project/v2xr_raygen_e0e3'
        folder.mkdir(parents=True)
        for n in bridge.MODULES:
            (folder / (n+'.py')).write_bytes((ROOT / 'v2xr_raygen_e0e3' / (n+'.py')).read_bytes())
        recorded = dict(dependencies={'v2xr_raygen_e0e3/'+n+'.py': bridge.ORIGINAL[n] for n in bridge.MODULES})
        result = bridge.physical_sources(folder.parent, recorded)
        self.assertEqual(result['v2xr_raygen_e0e3/runtime.py']['physical_sha256'], bridge.CURRENT['runtime'])
        self.assertEqual(result['v2xr_raygen_e0e3/runtime.py']['executed_snapshot_sha256'], bridge.ORIGINAL['runtime'])
        (folder / 'runtime.py').write_text('unknown computation', encoding='utf-8')
        with self.assertRaises(ValueError):
            bridge.physical_sources(folder.parent, recorded)

    def test_recorded_pins_cannot_be_replaced(self):
        recorded = dict(dependencies={'v2xr_raygen_e0e3/'+n+'.py': bridge.ORIGINAL[n] for n in bridge.MODULES})
        recorded['dependencies']['v2xr_raygen_e0e3/runtime.py'] = bridge.CURRENT['runtime']
        with self.assertRaises(ValueError):
            bridge.physical_sources(ROOT, recorded)

    def test_snapshot_bootstrap_has_no_new_compute_policy(self):
        source = self.members['snapshots/runtime.py'].decode()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bootstrap')
        self.assertNotIn('stable_crop', ast.unparse(fn))
        self.assertNotIn('configure(torch)', ast.unparse(fn))
        self.assertIn('official_bootstrap(args.runtime_root)', ast.unparse(fn))

    def test_all_snapshot_python_compiles_under_python310_grammar(self):
        for name, raw in self.members.items():
            if name.endswith('.py'):
                tree = ast.parse(raw.decode(), filename=name, feature_version=(3, 10))
                compile(tree, name, 'exec')

    def fake_modules(self):
        project, snapshots = self.root / 'fake_project', self.root / 'private'
        package = project / 'v2xr_raygen_e0e3'
        package.mkdir(parents=True)
        snapshots.mkdir()
        (package / '__init__.py').write_text('# synthetic package\n', encoding='utf-8')
        hashes = {}
        for n in bridge.MODULES:
            (package / (n+'.py')).write_text("raise RuntimeError('physical module must not execute')\n", encoding='utf-8')
            raw = ('VALUE = '+repr('snapshot-'+n)+'\n').encode()
            (snapshots / (n+'.py')).write_bytes(raw)
            hashes[n] = bridge.digest(raw)
        return project, snapshots, hashes

    def test_private_modules_own_import_namespace_without_running_physical_code(self):
        project, snapshots, hashes = self.fake_modules()
        with package_namespace(), patch.object(bridge, 'SNAPSHOT', hashes):
            bindings = bridge.load_snapshots(project, snapshots)
            bridge.verify_bindings(bindings)
            for n in bridge.MODULES:
                module = importlib.import_module('v2xr_raygen_e0e3.'+n)
                self.assertEqual(module.VALUE, 'snapshot-'+n)
                self.assertEqual(Path(module.__file__), snapshots / (n+'.py'))

    def test_early_import_is_rejected(self):
        project, snapshots, hashes = self.fake_modules()
        with package_namespace(), patch.object(bridge, 'SNAPSHOT', hashes):
            sys.path.insert(0, str(project))
            package = importlib.import_module('v2xr_raygen_e0e3')
            sys.modules['v2xr_raygen_e0e3.runtime'] = types.ModuleType('early')
            package.runtime = sys.modules['v2xr_raygen_e0e3.runtime']
            with self.assertRaises(ValueError):
                bridge.load_snapshots(project, snapshots)

    def test_loaded_binding_file_tamper_detected(self):
        project, snapshots, hashes = self.fake_modules()
        with package_namespace(), patch.object(bridge, 'SNAPSHOT', hashes):
            bindings = bridge.load_snapshots(project, snapshots)
            (snapshots / 'runtime.py').write_text('VALUE = 0\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                bridge.verify_bindings(bindings)

    def test_every_child_command_uses_snapshot_launcher(self):
        from uecp_gspr_resume_repair import compile_worker_command
        fake = types.SimpleNamespace(compile_worker_command=compile_worker_command,
            amended_report=lambda *a: {}, bind_commit=lambda *a: None)
        diagnose = types.SimpleNamespace(diagnostic_sources=lambda *a: {})
        probe = types.SimpleNamespace(diagnostic_sources=diagnose.diagnostic_sources)
        bridge.install_runner(fake, diagnose, probe, PurePosixPath('/new/snapshot.zip'), PurePosixPath('/private/original.zip'),
            Path('/output'), ROOT, Path('/snapshots'), {})
        source = "def run(self, action):\n    cmd = [sys.executable, '-B', str(HERE / 'worker.py'), action]\n    return cmd\n"
        fn = fake.compile_worker_command(source, dict(sys=sys, HERE=Path('/old')), Path('/private/original.zip'), ROOT)
        command = fn(None, 'dev')
        self.assertEqual(command[2:5], ['/new/snapshot.zip', 'worker', 'dev'])
        self.assertEqual(fake.verify_archive(), PurePosixPath('/private/original.zip'))

    def test_existing_private_enrollment_is_immutable(self):
        path = self.root / 'marker.json'
        record = dict(value=1)
        record['identity'] = bridge.identity(record)
        bridge.immutable_json(path, record)
        bridge.immutable_json(path, record)
        bad = copy.deepcopy(record)
        bad['value'] = 2
        with self.assertRaises(ValueError):
            bridge.immutable_json(path, bad)

    def fake_runner(self, report, bind):
        fake = types.SimpleNamespace(compile_worker_command=lambda *a: None, amended_report=report, bind_commit=bind)
        diagnose = types.SimpleNamespace(diagnostic_sources=lambda *a: {})
        probe = types.SimpleNamespace(diagnostic_sources=diagnose.diagnostic_sources)
        marker = self.root / bridge.MARKER
        bridge.immutable_json(marker, dict(identity='synthetic source marker'))
        archive = self.zip()
        bridge.install_runner(fake, diagnose, probe, archive, self.root / 'old.zip',
            self.root, ROOT, self.root / 'private', {})
        return fake, archive

    def test_new_stage_commit_records_snapshot_without_changing_metrics(self):
        from uecp_gspr_v2xr import common
        def bind(*a):
            common.commit = lambda path, outputs, **fields: dict(fields, original_v2_policy='unchanged')
        with patch.object(common, 'commit', common.commit):
            fake, archive = self.fake_runner(lambda *a: {}, bind)
            fake.bind_commit(self.root, dict(identity='old marker'))
            result = common.commit(self.root / 'synthetic_dev_complete.json', [], AP=dict(ap70=.42))
        self.assertEqual(result['AP']['ap70'], .42)
        self.assertEqual(result['original_v2_policy'], 'unchanged')
        self.assertEqual(result['execution_dependency_snapshot']['launcher_sha256'], bridge.sha(archive))

    def test_new_report_has_snapshot_provenance_and_completed_report_is_preserved(self):
        def old_report(root, marker, smoke=False, partial=False):
            path = root / 'FINAL_RESULTS.json'
            if path.exists():
                return bridge.read(path)
            result = dict(status='COMPLETE', AP=dict(ap70=.42), synthetic_fixture=True)
            bridge.immutable_json(path, result)
            path.with_suffix('.md').write_text('Synthetic report only\n', encoding='utf-8')
            return result
        fake, _ = self.fake_runner(old_report, lambda *a: None)
        result = fake.amended_report(self.root, {})
        self.assertEqual(result['AP']['ap70'], .42)
        self.assertIn('execution_dependency_snapshot', result)
        original = (self.root / 'FINAL_RESULTS.json').read_bytes()
        fake.amended_report(self.root, {})
        self.assertEqual(original, (self.root / 'FINAL_RESULTS.json').read_bytes())

    def test_argv0_restore_does_not_change_remaining_arguments(self):
        before = list(sys.argv)
        with bridge.argv0('private_legacy.zip'):
            self.assertEqual(sys.argv[0], 'private_legacy.zip')
            self.assertEqual(sys.argv[1:], before[1:])
        self.assertEqual(sys.argv, before)

    def test_no_research_runtime_imported(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('opencood', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
