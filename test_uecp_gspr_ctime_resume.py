"""Synthetic filesystem, AST and controller/worker wiring tests; no research runtime."""
import ast
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import uecp_gspr_ctime_resume as fix
from build_uecp_ctime_resume import build

ROOT = Path(__file__).resolve().parent


class CtimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='.ctime_synthetic_', dir=ROOT)
        self.root = Path(self.tmp.name)
        self.train = self.root / 'accepted'
        self.file = self.train / 'train/scene/cav/001.pcd'
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b'Synthetic point-cloud stand-in. No real point cloud.\n')
        self.live = fix.stamp(self.file)
        self.old = dict(self.live, ctime_ns=self.live['ctime_ns']-1)
        self.entry = dict(relative='scene/cav/001.pcd', size=self.live['size'], sha256=fix.sha(self.file))
        self.put('request.json', dict(paths=dict(TRAIN_ROOT=str(self.train))))
        self.put('contract.json', dict(synthetic=True))
        self.put('source_repair/uecp_dependency_snapshot_v1.json', dict(synthetic=True))
        self.put('preflight/train_manifest.json', dict(files=[self.entry]))
        self.put('preflight/input_stamps.json', [self.old])
        self.receipt = 'preflight/content/scene/cav/001.pcd.json'
        self.put(self.receipt, dict(status='complete', source=self.old, sha256=self.entry['sha256'], points=1))
        self.archive = self.root / 'synthetic_launcher.zip'
        self.archive.write_bytes(b'Synthetic archive identity only')

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def approve(self):
        return fix.changed_file(self.root, self.old, self.live, fix.train_files(self.root))

    def audit(self):
        return fix.audit_inputs(self.root, self.archive)

    def test_ctime_only_requires_matching_content_and_original_decode(self):
        result = self.approve()
        self.assertEqual(result['sha256'], self.entry['sha256'])
        self.assertEqual(result['original'], self.old)
        self.assertEqual(result['accepted'], self.live)

    def test_changed_bytes_with_same_size_are_rejected(self):
        self.file.write_bytes(b'X'*self.live['size'])
        live = fix.stamp(self.file)
        old = dict(live, ctime_ns=live['ctime_ns']-1)
        self.put(self.receipt, dict(status='complete', source=old, sha256=self.entry['sha256']))
        with self.assertRaisesRegex(ValueError, 'content SHA differs'):
            fix.changed_file(self.root, old, live, fix.train_files(self.root))

    def test_other_metadata_changes_are_rejected(self):
        for field in ('size', 'mtime_ns', 'path', 'resolved'):
            old = dict(self.old)
            old[field] = old[field]+1 if isinstance(old[field], int) else old[field]+'_changed'
            with self.subTest(field=field), self.assertRaises(ValueError):
                fix.changed_file(self.root, old, self.live, fix.train_files(self.root))

    def test_unknown_file_cannot_be_approved(self):
        with self.assertRaisesRegex(ValueError, 'lacks an original training SHA'):
            fix.changed_file(self.root, self.old, self.live, {})

    def test_original_decode_receipt_tampering_rejected(self):
        self.put(self.receipt, dict(status='complete', source=self.live, sha256=self.entry['sha256']))
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            self.approve()

    def test_mutation_while_hashing_rejected(self):
        def mutate(p):
            p.write_bytes(b'Mutation during synthetic hashing')
            return self.entry['sha256']
        with self.assertRaisesRegex(ValueError, 'during hashing'):
            fix.changed_file(self.root, self.old, self.live, fix.train_files(self.root), mutate)

    def test_audit_preserves_original_records_and_verifies_coverage(self):
        names = ['preflight/input_stamps.json', 'preflight/train_manifest.json', self.receipt]
        originals = {n: (self.root / n).read_bytes() for n in names}
        path = self.audit()
        rows = fix.verify_audit(self.root, self.archive, path, fix.sha(path))
        self.assertEqual(set(rows), {str(self.file)})
        for n in names:
            self.assertEqual((self.root / n).read_bytes(), originals[n])

    def test_unchanged_inputs_do_not_need_content_exception(self):
        self.put('preflight/input_stamps.json', [self.live])
        path = self.audit()
        self.assertEqual(fix.verify_audit(self.root, self.archive, path, fix.sha(path)), {})

    def test_failed_audit_records_failure_without_complete(self):
        self.file.write_bytes(b'Wrong content and length')
        with self.assertRaises(ValueError):
            self.audit()
        paths = list((self.root / fix.PREFIX / 'audits').glob('*.json'))
        self.assertEqual(len(paths), 1)
        self.assertEqual(fix.read(paths[0])['status'], 'FAILED')
        with self.assertRaises(ValueError):
            fix.verify_audit(self.root, self.archive, paths[0], fix.sha(paths[0]))

    def test_later_manifest_change_rejected(self):
        path = self.audit()
        self.put('preflight/train_manifest.json', dict(files=[]))
        with self.assertRaisesRegex(ValueError, 'lineage'):
            fix.verify_audit(self.root, self.archive, path, fix.sha(path))

    def test_tampered_receipt_rejected(self):
        path = self.audit()
        digest = fix.sha(path)
        path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'tampered'):
            fix.verify_audit(self.root, self.archive, path, digest)

    def test_duplicate_input_rows_rejected(self):
        self.put('preflight/input_stamps.json', [self.old, self.old])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.audit()

    def test_manifest_escape_rejected(self):
        row = dict(self.entry, relative='../../escaped.pcd')
        self.put('preflight/train_manifest.json', dict(files=[row]))
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            fix.train_files(self.root)

    def runtime(self):
        path = self.root / 'synthetic_runtime.py'
        path.write_text("def input_stamps(output):\n    for row in read(output / 'preflight/input_stamps.json'):\n        if stamp(row['path']) != row:\n            raise ValueError('Input changed')\n    upstream_source_check()\n", encoding='utf-8')
        spec = importlib.util.spec_from_file_location('_synthetic_ctime_runtime', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.read, module.stamp = fix.read, fix.stamp
        module.upstream_checks = []
        module.upstream_source_check = lambda: module.upstream_checks.append(True)
        return module

    def test_guard_preserves_upstream_check_and_rejects_later_changes(self):
        runtime = self.runtime()
        fix.adapt_guard(runtime, {str(self.file): self.approve()})
        runtime.input_stamps(self.root)
        self.assertEqual(runtime.upstream_checks, [True])
        self.file.write_bytes(b'Later modification')
        with self.assertRaisesRegex(ValueError, 'changed again'):
            runtime.input_stamps(self.root)

    def test_guard_rejects_unapproved_change(self):
        runtime = self.runtime()
        fix.adapt_guard(runtime, {})
        with self.assertRaisesRegex(ValueError, 'Input changed'):
            runtime.input_stamps(self.root)

    def test_existing_uecp_alias_is_updated(self):
        runtime = self.runtime()
        alias = types.SimpleNamespace(input_stamps=runtime.input_stamps)
        with patch.dict(sys.modules, {'uecp_gspr_v2xr.runtime': alias}):
            fix.adapt_guard(runtime, {str(self.file): self.approve()})
            self.assertIs(alias.input_stamps, runtime.input_stamps)

    def test_unknown_alias_binding_rejected(self):
        alias = types.SimpleNamespace(input_stamps=lambda root: None)
        with patch.dict(sys.modules, {'uecp_gspr_v2xr.runtime': alias}), self.assertRaisesRegex(ValueError, 'binding'):
            fix.adapt_guard(self.runtime(), {})

    def test_full_worker_wiring_uses_outer_launcher_and_records_provenance(self):
        path = self.audit()
        runtime = self.runtime()
        from uecp_gspr_resume_repair import compile_worker_command
        from uecp_gspr_v2xr import common
        lineage_calls = []
        repair = types.SimpleNamespace(compile_worker_command=compile_worker_command,
            lineage=lambda *a: lineage_calls.append(a), bind_commit=lambda *a: None,
            amended_report=lambda *a, **k: {})
        bridge = types.SimpleNamespace(install_runner=lambda *a: None)
        with patch.dict(os.environ, {fix.AUDIT_ENV: str(path), fix.HASH_ENV: fix.sha(path)}), \
                patch.dict(sys.modules, {'v2xr_raygen_e0e3.runtime': runtime}), \
                patch.object(common, 'commit', lambda p, outputs, **fields: fields):
            fix.install(bridge, self.archive, self.root, 'worker')
            bridge.install_runner(repair, None, None, 'prior', 'legacy', self.root, ROOT, 'snapshots', {})
            source = "def run(self, action):\n    cmd = [sys.executable, '-B', str(HERE / 'worker.py'), action]\n    return cmd\n"
            fn = repair.compile_worker_command(source, dict(sys=sys, HERE=ROOT), 'ignored', ROOT)
            self.assertEqual(fn(None, 'dev')[2:5], [str(self.archive), 'worker', 'dev'])
            self.assertEqual(lineage_calls, [(self.root, ROOT, 'legacy')])
            repair.bind_commit(self.root, {})
            result = common.commit(self.root / 'synthetic_commit.json', [], AP=dict(ap70=.4))
            self.assertEqual(result['AP']['ap70'], .4)
            self.assertEqual(result['input_ctime_audit']['audit_sha256'], fix.sha(path))

    def test_worker_missing_audit_env_fails_before_model(self):
        repair = types.SimpleNamespace(compile_worker_command=lambda *a: None, lineage=lambda *a: None)
        bridge = types.SimpleNamespace(install_runner=lambda *a: None)
        with patch.dict(os.environ, {}, clear=True):
            fix.install(bridge, self.archive, self.root, 'worker')
            with self.assertRaisesRegex(ValueError, 'requires its controller'):
                bridge.install_runner(repair, None, None, 'prior', 'legacy', self.root, ROOT, 'snapshots', {})

    def test_controller_verifies_epoch3_and_exports_audit_to_children(self):
        runtime = self.runtime()
        from uecp_gspr_v2xr import common
        verified = []
        repair = types.SimpleNamespace(compile_worker_command=lambda *a: None, lineage=lambda *a: None,
            bind_commit=lambda *a: None, amended_report=lambda *a: {})
        bridge = types.SimpleNamespace(install_runner=lambda *a: None, stopped=lambda root: nullcontext())
        with patch.dict(os.environ, {}, clear=True), \
                patch.dict(sys.modules, {'v2xr_raygen_e0e3.runtime': runtime}), \
                patch.object(common, 'complete', lambda path: verified.append(path)):
            fix.install(bridge, self.archive, self.root, 'run')
            bridge.install_runner(repair, None, None, 'prior', 'legacy', self.root, ROOT, 'snapshots', {})
            path = Path(os.environ[fix.AUDIT_ENV])
            self.assertEqual(os.environ[fix.HASH_ENV], fix.sha(path))
            self.assertEqual(verified, [self.root / 'E0/baseline/epochs/03/train_complete.json'])
            self.assertEqual(fix.read(path)['status'], 'COMPLETE')

    def test_completed_report_is_preserved_and_new_report_has_audit(self):
        path = self.audit()
        runtime = self.runtime()
        from uecp_gspr_v2xr import common
        def old_report(output, marker, smoke=False, partial=False):
            p = output / 'FINAL_RESULTS.json'
            if p.exists():
                return fix.read(p)
            value = dict(status='COMPLETE', AP=dict(ap70=.4))
            common.write(p, value)
            p.with_suffix('.md').write_text('Synthetic report\n', encoding='utf-8')
            return value
        repair = types.SimpleNamespace(compile_worker_command=lambda *a: None, lineage=lambda *a: None,
            bind_commit=lambda *a: None, amended_report=old_report)
        bridge = types.SimpleNamespace(install_runner=lambda *a: None)
        with patch.dict(os.environ, {fix.AUDIT_ENV: str(path), fix.HASH_ENV: fix.sha(path)}), \
                patch.dict(sys.modules, {'v2xr_raygen_e0e3.runtime': runtime}):
            fix.install(bridge, self.archive, self.root, 'worker')
            bridge.install_runner(repair, None, None, 'prior', 'legacy', self.root, ROOT, 'snapshots', {})
            result = repair.amended_report(self.root, {})
            self.assertEqual(result['AP']['ap70'], .4)
            self.assertEqual(result['input_ctime_audit']['audit_sha256'], fix.sha(path))
            original = (self.root / 'FINAL_RESULTS.json').read_bytes()
            repair.amended_report(self.root, {})
            self.assertEqual((self.root / 'FINAL_RESULTS.json').read_bytes(), original)

    def test_actual_frozen_input_guard_and_upstream_validation_are_preserved(self):
        archive = ROOT / 'project_backup_20261023/files/uecp_gspr_dependency_snapshot_v1.zip'
        with zipfile.ZipFile(archive) as z:
            raw = z.read('snapshots/runtime.py').decode()
        tree = ast.parse(raw)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'input_stamps')
        from pathlib import Path as PathType
        namespace = dict(Path=PathType, read=fix.read, stamp=fix.stamp, sha=fix.sha,
            evidence=lambda root: dict(a=dict(upstream_source_hashes={'frozen_source.py': 'wrong'})),
            arguments=lambda root: types.SimpleNamespace(runtime_root=str(self.root)))
        source_path = self.root / 'actual_frozen_guard_fixture.py'
        source_path.write_text(raw, encoding='utf-8')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
        runtime = types.SimpleNamespace(input_stamps=namespace['input_stamps'], stamp=fix.stamp)
        (self.root / 'frozen_source.py').write_text('Synthetic upstream source\n', encoding='utf-8')
        fix.adapt_guard(runtime, {str(self.file): self.approve()})
        with self.assertRaisesRegex(ValueError, 'Runtime source changed'):
            runtime.input_stamps(self.root)

    def test_bundle_contains_exact_original_zip(self):
        path = self.root / 'bundle.zip'
        build(path)
        fix.verify_zip(path)
        with zipfile.ZipFile(path) as z:
            self.assertEqual(__import__('hashlib').sha256(z.read('prior/dependency_snapshot_v1.zip')).hexdigest(), fix.PRIOR_SHA)
            for name in z.namelist():
                if name.endswith('.py'):
                    ast.parse(z.read(name).decode(), feature_version=(3, 10))

    def test_resealed_wrong_original_zip_is_rejected(self):
        path = self.root / 'bundle.zip'
        build(path)
        with zipfile.ZipFile(path) as z:
            files = {n: z.read(n) for n in z.namelist() if n != 'manifest.json'}
        files['prior/dependency_snapshot_v1.zip'] += b'wrong'
        hashes = {n: __import__('hashlib').sha256(raw).hexdigest() for n, raw in files.items()}
        files['manifest.json'] = json.dumps(dict(files=hashes)).encode()
        with zipfile.ZipFile(path, 'w') as z:
            for n, raw in files.items():
                z.writestr(n, raw)
        with self.assertRaisesRegex(ValueError, 'snapshot ZIP differs'):
            fix.verify_zip(path)

    def test_no_research_runtime_imported(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('opencood', sys.modules)


if __name__ == '__main__':
    unittest.main()
