"""Fake-artifact contract tests. No Torch import, model execution or real data."""
import copy
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import baseline
from auditlib import CONFIG, ROOT, SELECTION, SETTINGS, atomic_json, identity, read_json, sha256


class BaselineTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        scratch = Path(__file__).resolve().parent.parent / 'tmp' / 'ggca_fixtures'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary, ExitStack() as stack:
            base = Path(temporary); root = base / 'B0'; output = base / 'new/preflight'
            root.mkdir(); output.mkdir(parents=True)
            train = base / 'train'; train.mkdir()
            atomic_json(train / 'manifest.json', {'identity': 'train', 'train_data_acceptance': 'accepted'})
            config = {'train_params': {'batch_size': 8, 'epoches': 20}, 'fixture': True}
            atomic_json(root / 'config.yaml', config)
            contract = dict(variant='B0', smoke_only=False, epochs=20, settings=CONFIG, policy=SELECTION,
                gspr_frozen_settings=SETTINGS,
                loader=dict(batch_size=8, num_workers=16, shuffle=True, drop_last=True, pin_memory=True, prefetch_factor=4),
                environment=dict(backend='rocm', versions={'torch': 'fixture'}, hip='fixture'),
                scheduler='official end-of-epoch scheduler.step(zero_based_epoch)',
                sources=dict(files={f: sha256(ROOT / 'gspr_repe' / f) for f in
                    ('models.py', 'training.py', 'data.py', 'checkpoint.py', 'settings.json')},
                    dependencies={f: sha256(ROOT / f) for f in ('attfuse_gspr/reliability.py', 'v2xr_gspr_clean/integration.py')}),
                config_sha256=sha256(root / 'config.yaml'), train_manifest_sha256=sha256(train / 'manifest.json'),
                train_manifest_identity='train', train_data_acceptance='accepted', pair_identity='pairs',
                input_identity='inputs', frozen_gt='gt')
            contract['identity'] = identity(contract)
            atomic_json(root / 'contract.json', contract)
            rows = []
            for epoch in range(1, 21):
                folder = root / 'epochs' / f'{epoch:02d}'; folder.mkdir(parents=True)
                (folder / 'fixture.bin').write_bytes(b'report fixture only')
                weight = folder / 'fixture.bin'
                clean = dict(status='complete', conditions=['clean'], frame_count=829, full_validation=True,
                             checkpoint_sha256=sha256(weight), results={'clean': {'AP': {'O': {'ap70': epoch/100}}}})
                atomic_json(folder / 'clean.json', clean)
                row = dict(status='complete', epoch=epoch, contract=contract['identity'], smoke_only=False,
                    train_updates=1010, checkpoint=str(weight), checkpoint_sha256=sha256(weight), AP70=epoch/100,
                    clean_summary=str(folder / 'clean.json'), clean_summary_sha256=sha256(folder / 'clean.json'),
                    resume_state=str(weight), resume_state_sha256=sha256(weight), trace=str(weight), trace_sha256=sha256(weight))
                atomic_json(folder / 'complete.json', row); rows.append(row)
            chosen = dict(status='complete', epoch=20, checkpoint=rows[-1]['checkpoint'],
                checkpoint_sha256=rows[-1]['checkpoint_sha256'], clean_O_AP70=.20, policy=SELECTION,
                weather_used=False, candidate_count=20, training_contract_sha256=sha256(root / 'contract.json'))
            chosen['identity'] = identity(chosen)
            atomic_json(root / 'selected.json', chosen)
            atomic_json(root / 'training_summary.json', dict(status='complete', epochs=20, selected=chosen))
            atomic_json(root / 'history.json', dict(epochs=rows))
            atomic_json(root / 'selection_policy.json', SELECTION)
            initial = root / 'resume_states/00.pth'; initial.parent.mkdir()
            initial.write_bytes(b'fake epoch0 marker; no tensors')
            atomic_json(root / 'resume_states/00.json', dict(sha256=sha256(initial)))
            fake_state = dict(epoch=0, contract=contract['identity'], optimizer={'state': {}}, model={'fixture': 1},
                              scheduler={}, criterion={}, accumulators={}, rng={})
            fake_torch = types.ModuleType('torch')
            fake_torch.version = types.SimpleNamespace(hip='fixture')
            fake_torch.load = lambda *a, **kw: fake_state
            fake_training = types.ModuleType('training')
            def generate_config(args, folder):
                atomic_json(folder / 'config.yaml', config)
                return config
            fake_training.config = generate_config
            fake_data = types.ModuleType('data'); fake_data.paired_inventory = lambda *a: {'identity': 'pairs'}
            stack.enter_context(patch.dict(sys.modules, {'torch': fake_torch, 'training': fake_training, 'data': fake_data}))
            stack.enter_context(patch.object(baseline, 'frozen_c', lambda args: {'gt': {'identity': 'gt'}}))
            args = types.SimpleNamespace(a0_root=str(root), output_root=str(output), train_root=str(train), experiment_root=str(base / 'new'))
            evidence = {'input_manifest': {'identity': 'inputs'}, 'a': {'versions': {'torch': 'fixture'}}}
            yield args, read_json(train / 'manifest.json'), evidence, fake_state

    def test_accepts_A0_without_writing_old_files(self):
        with self.fixture() as (args, manifest, evidence, state):
            root = Path(args.a0_root)
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            result = baseline.audit(args, manifest, evidence)
            after = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertTrue(result['training_forbidden'])
            self.assertFalse(result['evaluation_reused'])
            self.assertIn('supplemental_evaluation', result['evaluation_root'])

    def test_missing_selected_checkpoint_stops(self):
        with self.fixture() as (args, manifest, evidence, state):
            selected = read_json(Path(args.a0_root) / 'selected.json')
            Path(selected['checkpoint']).unlink()
            with self.assertRaises((FileNotFoundError, ValueError)):
                baseline.audit(args, manifest, evidence)

    def test_incomplete_training_stops_without_fallback(self):
        with self.fixture() as (args, manifest, evidence, state):
            (Path(args.a0_root) / 'epochs/20/complete.json').unlink()
            with self.assertRaisesRegex(ValueError, 'not completed'):
                baseline.audit(args, manifest, evidence)

    def test_wrong_seed_stops(self):
        with self.fixture() as (args, manifest, evidence, state):
            path = Path(args.a0_root) / 'contract.json'
            contract = read_json(path); contract['settings']['seed'] = 9
            contract['identity'] = identity({k: v for k, v in contract.items() if k != 'identity'})
            atomic_json(path, contract)
            with self.assertRaisesRegex(ValueError, 'recipe differs: seed'):
                baseline.audit(args, manifest, evidence)

    def test_trained_initial_state_stops(self):
        with self.fixture() as (args, manifest, evidence, state):
            state['epoch'] = 1
            with self.assertRaisesRegex(ValueError, 'pristine'):
                baseline.audit(args, manifest, evidence)

    def test_complete_but_wrong_weather_protocol_stops(self):
        with self.fixture() as (args, manifest, evidence, state):
            root = Path(args.a0_root)
            selected = read_json(root / 'selected.json')
            folder = root / 'final_evaluation'
            atomic_json(folder / 'summary.json', dict(status='complete', full_validation=True,
                conditions=['clean', 'fog', 'snow'], frame_count=829, checkpoint_sha256=selected['checkpoint_sha256'],
                frozen_d1c_gt_identity='different'))
            atomic_json(folder / 'protocol.json', {'protocols': {}})
            with self.assertRaisesRegex(ValueError, 'cannot be matched'):
                baseline.audit(args, manifest, evidence)

    def test_changed_initial_hash_stops(self):
        with self.fixture() as (args, manifest, evidence, state):
            (Path(args.a0_root) / 'resume_states/00.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'initial training state'):
                baseline.audit(args, manifest, evidence)

    def test_complete_matching_evaluation_is_reused_read_only(self):
        from support import DEFINITIONS
        with self.fixture() as (args, manifest, evidence, state):
            root = Path(args.a0_root)
            folder = root / 'final_evaluation'; folder.mkdir()
            selected = read_json(root / 'selected.json')
            atomic_json(folder / 'summary.json', dict(status='complete', full_validation=True,
                conditions=['clean', 'fog', 'snow'], frame_count=829,
                checkpoint_sha256=selected['checkpoint_sha256'], frozen_d1c_gt_identity='gt',
                associated_d1d_training={'training_contract_sha256': sha256(root / 'contract.json')},
                results={w: {'AP': {'O': {}, 'O*': {}, 'F': {}}} for w in ('clean', 'fog', 'snow')}))
            atomic_json(folder / 'protocol.json', dict(protocols=DEFINITIONS, input_manifest_identity='inputs'))
            atomic_json(folder / 'gt_manifest.json', dict(fixture=True))
            atomic_json(folder / 'prediction_manifest.json', dict(files=[]))
            stub = types.ModuleType('evaluation')
            calls = []
            stub.inherited = types.SimpleNamespace(assert_frozen_gt=lambda *args: calls.append(args))
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch.dict(sys.modules, {'evaluation': stub}):
                result = baseline.audit(args, manifest, evidence)
            self.assertTrue(result['evaluation_reused'])
            self.assertEqual(result['evaluation_root'], str(folder))
            self.assertEqual(len(calls), 1)
            self.assertEqual(before, {str(p): sha256(p) for p in root.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
