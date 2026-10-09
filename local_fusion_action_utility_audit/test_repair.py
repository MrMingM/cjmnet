"""Synthetic metadata migration checks; no models or research data are loaded."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import WEATHERS, Manifest, atomic_json, digest, object_hash, read_json
from .repair import REPAIR_ID, finish_repair_transaction, migrate_if_authorized


def fixture(root):
    audit = root / 'local_fusion_action_utility_audit'
    audit.mkdir()
    source = 'local_fusion_action_utility_audit/common.py'
    (root / source).write_text('# repaired synthetic source', encoding='utf-8')
    (root / 'dependency.py').write_text('# frozen dependency', encoding='utf-8')
    legacy = {source: 'original-source-sha256'}
    atomic_json(audit / 'repair_compatibility.json', {'repair_id': REPAIR_ID,
                'legacy_audit_sources': legacy, 'legacy_snapshot_sha256': 'original-snapshot'})
    for name in ('frontend.pth', 'v3.pth', 'Shared.pth'):
        (root / name).write_bytes(b'synthetic checkpoint')
    inputs = {'frontend_checkpoint': str(root / 'frontend.pth'),
              'v3_checkpoint': str(root / 'v3.pth'), 'b0_run': str(root)}
    checkpoints = {'frontend_checkpoint': digest(root / 'frontend.pth'),
                   'v3_checkpoint': digest(root / 'v3.pth'), 'b0_shared': digest(root / 'Shared.pth')}
    old_sources = {**legacy, 'dependency.py': digest(root / 'dependency.py')}
    old = {'inputs': inputs, 'config': {'unchanged': True}, 'source_hashes': old_sources,
           'identity': {'source_hash': object_hash(old_sources), 'source_commit': 'legacy-metadata',
                        'config_hash': 'same-config', 'checkpoint_hashes': checkpoints}}
    for key in ('train_indices', 'validation_indices', 'train_scenes', 'validation_scenes',
                'probe_fit_scenes', 'probe_calibration_scenes'):
        old[key] = [0]
    old.update(b0_saved_results={}, frozen_sources={'frozen': 'unchanged'})
    new = copy.deepcopy(old)
    new['source_hashes'] = {name: digest(root / name) for name in old_sources}
    new['identity'].update(source_hash=object_hash(new['source_hashes']), source_commit=None)
    run = root / 'run'
    atomic_json(run / 'protocol.json', old)
    artifact = run / 'retained.pt'
    artifact.write_bytes(b'unchanged synthetic training cache')
    manifest = Manifest(run, old['identity'])
    manifest.mark('baseline', 'complete', artifacts=[artifact])
    manifest.mark('test_core', 'complete')
    for weather in WEATHERS:
        manifest.mark('S0-frame', 'complete', 'train', weather, 0, [artifact])
        manifest.mark('S0-weather', 'complete', 'train', weather, artifacts=[artifact])
        manifest.mark('baseline-frame', 'complete', 'validation', weather, 0, [artifact])
    summary = {key: True for key in ('same_input_in_all_cases', 'raw_hash_varies_between_repeats',
               'repeat_raw_differences_within_existing_2e5_tolerance',
               'saved_baseline_tp_fp_sequences_unchanged', 'all_inputs_and_buffers_unchanged')}
    summary['first_variable_intermediate_on_same_batch'] = 'spectral'
    atomic_json(run / 'diagnostics' / 'shared_drift_20261008.json', {'frame': 0, 'summary': summary,
                'cases': [{'stats_vs_saved_baseline': {'0.7': {'scores': {'allclose_2e5': True}}}}]})
    return run, old, new, artifact


class RepairTests(unittest.TestCase):
    def test_preserves_cache_bytes_and_producer_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, old, new, artifact = fixture(root)
            before = (digest(artifact), artifact.stat().st_mtime_ns)
            with patch('local_fusion_action_utility_audit.repair.ROOT', root):
                migrate_if_authorized(run, new, enabled=True)
                finish_repair_transaction(run)
                repeated = migrate_if_authorized(run, copy.deepcopy(new), enabled=True)
            self.assertEqual(repeated, read_json(run / 'protocol.json'))
            self.assertEqual(before, (digest(artifact), artifact.stat().st_mtime_ns))
            manifest = Manifest(run)
            self.assertEqual(manifest.value['identity'], new['identity'])
            row = manifest.value['entries'][manifest.key('S0-frame', 'train', 'clean', 0)]
            self.assertEqual(row['source_hash'], old['identity']['source_hash'])
            self.assertEqual(row['reuse_approval'], REPAIR_ID)
            self.assertFalse(manifest.complete('test_core'))
            self.assertEqual(read_json(run / 'repair_shared_drift/original_protocol.json'), old)

    def test_unsafe_migrations_are_rejected_before_metadata_changes(self):
        for change in ('disabled', 'config', 'version', 'dependency', 'cache', 'validation', 'evidence'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                run, old, new, artifact = fixture(root)
                if change == 'config':
                    new['config'] = {'changed': True}
                elif change == 'version':
                    old['source_hashes']['local_fusion_action_utility_audit/common.py'] = 'unknown'
                    old['identity']['source_hash'] = object_hash(old['source_hashes'])
                    atomic_json(run / 'protocol.json', old)
                    manifest = Manifest(run)
                    manifest.value['identity'] = old['identity']
                    manifest.save()
                elif change == 'dependency':
                    (root / 'dependency.py').write_text('changed', encoding='utf-8')
                elif change == 'cache':
                    artifact.write_bytes(b'corrupted')
                elif change == 'validation':
                    Manifest(run).mark('S0-frame', 'complete', 'validation', 'clean', 0, [artifact])
                elif change == 'evidence':
                    path = run / 'diagnostics/shared_drift_20261008.json'
                    evidence = read_json(path)
                    evidence['summary']['all_inputs_and_buffers_unchanged'] = False
                    atomic_json(path, evidence)
                before = digest(run / 'protocol.json')
                with patch('local_fusion_action_utility_audit.repair.ROOT', root):
                    with self.assertRaises(ValueError):
                        migrate_if_authorized(run, new, enabled=change != 'disabled')
                self.assertEqual(digest(run / 'protocol.json'), before)
                self.assertFalse((run / 'repair_shared_drift/transaction.json').exists())

    def test_interrupted_metadata_transaction_finishes_on_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, old, new, artifact = fixture(root)
            def interrupted(path, value):
                if Path(path) == run / 'manifest.json':
                    raise OSError('synthetic interruption after protocol replacement')
                atomic_json(path, value)
            with patch('local_fusion_action_utility_audit.repair.ROOT', root):
                with patch('local_fusion_action_utility_audit.repair.atomic_json', side_effect=interrupted):
                    with self.assertRaises(OSError):
                        migrate_if_authorized(run, new, enabled=True)
                self.assertEqual(read_json(run / 'protocol.json')['identity'], new['identity'])
                self.assertEqual(Manifest(run).value['identity'], old['identity'])
                finish_repair_transaction(run)
                finish_repair_transaction(run)
            self.assertEqual(Manifest(run).value['identity'], new['identity'])
            self.assertEqual(artifact.read_bytes(), b'unchanged synthetic training cache')
            self.assertEqual(read_json(run / 'repair_shared_drift/transaction.json')['status'], 'complete')

    def test_interrupted_transaction_rejects_changed_proposal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run, _, new, _ = fixture(root)
            with patch('local_fusion_action_utility_audit.repair.ROOT', root):
                with patch('local_fusion_action_utility_audit.repair.finish_repair_transaction', side_effect=OSError('stop')):
                    with self.assertRaises(OSError):
                        migrate_if_authorized(run, new, enabled=True)
                atomic_json(run / 'repair_shared_drift/proposed_manifest.json', {})
                with self.assertRaisesRegex(RuntimeError, 'corrupted'):
                    finish_repair_transaction(run)


if __name__ == '__main__':
    unittest.main(verbosity=2)
