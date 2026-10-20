"""Synthetic NumPy/filesystem tests only; no Torch, model, real data or checkpoint loading."""
import copy
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .common import HERE, identity, read, sha, write
from .stable_crop import POLICY_ID, neighbors, configure, check_backend
from .successor import import_caches, transfer, initialize


def sample_numpy(image, grid):
    out = np.zeros(grid.shape[:-1], dtype=image.dtype)
    for x, y, weight in neighbors(grid, image.shape[1], image.shape[0], np.floor):
        valid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        out += image[np.clip(y, 0, image.shape[0]-1).astype(int), np.clip(x, 0, image.shape[1]-1).astype(int)] * weight * valid
    return out


class StableMathTests(unittest.TestCase):
    def test_bilinear_values_and_half_pixel_zero_padding(self):
        image = np.array([[1., 3.], [5., 7.]])
        grid = np.array([[[-.25, 0.], [-1., -1.], [1., 1.], [0., 0.], [3., 3.]]])
        np.testing.assert_allclose(sample_numpy(image, grid), [[3.5, .25, 1.75, 4., 0.]], rtol=0, atol=0)

    def test_feature_adjoint_matches_finite_difference_and_overlap_sum(self):
        image = np.arange(12, dtype=np.float64).reshape(3, 4)
        grid = np.array([[[0., 0.], [.21, -.31], [0., 0.], [-1., 1.]]])
        upstream = np.array([[1., -.2, 2., .4]])
        analytical = np.zeros_like(image)
        for x, y, weight in neighbors(grid, 4, 3, np.floor):
            for at in np.ndindex(weight.shape):
                xx, yy = int(x[at]), int(y[at])
                if 0 <= xx < 4 and 0 <= yy < 3:
                    analytical[yy, xx] += weight[at] * upstream[at]
        for at in np.ndindex(image.shape):
            a, b = image.copy(), image.copy()
            a[at] += 1e-6; b[at] -= 1e-6
            numerical = ((sample_numpy(a, grid)-sample_numpy(b, grid))*upstream).sum() / 2e-6
            self.assertAlmostEqual(analytical[at], numerical, places=8)

    def test_float32_plan_partition_and_no_negative_weights(self):
        grid = np.array([[[0., 0.], [.2, .7]]], np.float32)
        plan = neighbors(grid, 352, 100, np.floor)
        np.testing.assert_allclose(sum(row[2] for row in plan), np.ones(grid.shape[:-1]), rtol=0, atol=1e-7)
        self.assertTrue(all(row[2].dtype == np.float32 and (row[2] >= 0).all() for row in plan))

    def test_backend_rejects_warn_only_or_benchmark_fallback(self):
        state = {'enabled': False, 'warn_only': True}
        backend = SimpleNamespace(backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, deterministic=False)),
            are_deterministic_algorithms_enabled=lambda: state['enabled'],
            is_deterministic_algorithms_warn_only_enabled=lambda: state['warn_only'])
        backend.use_deterministic_algorithms = lambda enabled, warn_only: state.update(enabled=enabled, warn_only=warn_only)
        configure(backend)
        self.assertTrue(check_backend(backend)['deterministic_algorithms'])
        state['warn_only'] = True
        with self.assertRaises(RuntimeError):
            check_backend(backend)

    def test_v3_acceptance_and_data_algorithms_unchanged(self):
        from .resume_compare import POLICY_V3_ID
        self.assertEqual(POLICY_V3_ID, '6ba0541284c4815d3ee022f38114c5dbeb9b95f353f00f2ac32d289b70a4f7d7')
        parent = read(HERE / 'STABLE_PARENT.json')
        for name in parent['cache_code_unchanged'] + ['resume_compare.py']:
            self.assertEqual(sha(HERE / name), parent['parent_sources']['files']['v2xr_raygen_e0e3/' + name])


class CacheImportTests(unittest.TestCase):
    def setUp(self):
        fixture_parent = HERE.parent if sys.platform == 'win32' else Path(tempfile.gettempdir())
        self.temp = tempfile.TemporaryDirectory(prefix='.stable_cache_fixture_', dir=fixture_parent)
        if fixture_parent.resolve() not in Path(self.temp.name).resolve().parents:
            raise AssertionError('Fixture cleanup must remain below its intended temporary directory')
        self.addCleanup(self.temp.cleanup)
        self.source, self.target = Path(self.temp.name) / 'old', Path(self.temp.name) / 'new'
        self.config, self.geometry = dict(epochs=2), dict(geometry='same')
        self.old = dict(inputs={'train': 'same'}, sources={'files': 'old'})
        self.new = dict(inputs={'train': 'same'}, sources={'files': 'new'})
        old_id = identity(dict(input=self.old['inputs'], config=self.config, sources=self.old['sources'], geometry=self.geometry))
        write(self.source / 'labels/contract.json', dict(identity=old_id))
        self.uid = ['scene', '000000', '1']
        write(self.source / 'preflight/train_manifest.json', dict(scenes=[dict(scene='scene', ego='1', timestamps=['000000'])]))
        sample = self.source / 'labels/samples/000000.json'
        sample.parent.mkdir(parents=True)
        sample.with_suffix('.npz').write_bytes(b'synthetic array bytes; never a real data file')
        write(sample, dict(status='complete', uid=self.uid, cache_identity=old_id, sha256=sha(sample.with_suffix('.npz'))))
        win = self.source / 'labels/windows/01/000000.json'
        win.parent.mkdir(parents=True)
        win.with_suffix('.npz').write_bytes(b'synthetic packed label bytes')
        write(win, dict(status='complete', sha256=sha(win.with_suffix('.npz')),
                       identity=identity(dict(base=sha(sample), config=self.config, uid=self.uid, epoch=1))))

    def run_import(self, config=None, geometry=None):
        return import_caches(self.source, self.target, self.old, self.new,
                             config or self.config, geometry or self.geometry)

    def test_import_rebases_metadata_only_and_is_restartable(self):
        before = {str(p): p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        metadata = {p: (Path(p).stat().st_mtime_ns, Path(p).stat().st_ctime_ns) for p in before}
        inventory = self.run_import()
        self.assertEqual(len(inventory), 2)
        self.assertEqual(self.run_import(), inventory)
        for p, data in before.items():
            self.assertEqual(Path(p).read_bytes(), data)
            self.assertEqual((Path(p).stat().st_mtime_ns, Path(p).stat().st_ctime_ns), metadata[p])
        new_base = self.target / 'labels/samples/000000.json'
        new_window = read(self.target / 'labels/windows/01/000000.json')
        self.assertEqual(new_window['identity'], identity(dict(base=sha(new_base), config=self.config, uid=self.uid, epoch=1)))
        self.assertNotEqual(sha(new_base), sha(self.source / 'labels/samples/000000.json'))
        self.assertEqual(sha(new_base.with_suffix('.npz')), sha(self.source / 'labels/samples/000000.npz'))

    def test_rejects_changed_config_or_geometry(self):
        with self.assertRaises(ValueError):
            self.run_import(config={'epochs': 3})
        with self.assertRaises(ValueError):
            self.run_import(geometry={'geometry': 'changed'})

    def test_rejects_corrupt_source_payload(self):
        (self.source / 'labels/samples/000000.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.run_import()

    def test_rejects_window_from_another_base_identity(self):
        p = self.source / 'labels/windows/01/000000.json'
        write(p, dict(read(p), identity='wrong'))
        with self.assertRaisesRegex(ValueError, 'window identity'):
            self.run_import()

    def test_rejects_existing_different_destination(self):
        self.target.mkdir()
        dest = self.target / 'payload.npz'
        dest.write_bytes(b'other')
        src = self.source / 'labels/samples/000000.npz'
        with self.assertRaisesRegex(ValueError, 'Successor artifact differs'):
            transfer(src, dest, sha(src))

    def test_stable_failure_precedes_any_capacity_or_training(self):
        from .controller import Controller
        control = Controller.__new__(Controller)
        control.root = self.target
        calls = []
        def run(name, *args, **kwargs):
            calls.append(name)
            if name == 'stable_calculation_checks':
                raise ValueError('synthetic stable failure')
        control.run = run
        with self.assertRaisesRegex(ValueError, 'synthetic stable failure'):
            control.smoke()
        self.assertEqual(calls, ['smoke_prepare', 'stable_calculation_checks'])

    def prepare_initializer(self):
        config = read(HERE / 'config.json')
        geometry = read(HERE / 'protocol.json')['geometry']
        files = {'v2xr_raygen_e0e3/' + name: sha(HERE / name)
                 for name in read(HERE / 'STABLE_PARENT.json')['cache_code_unchanged']}
        original_sources = dict(files=dict(files, model='old'), dependencies={})
        own = dict(files=dict(files, model='stable'), dependencies={})
        contract = dict(config=config, sources=original_sources, inputs={'train': 'same'},
                        environment=dict(python=sys.version, versions={}, physical_gpu='0'))
        contract['identity'] = identity(contract)
        write(self.source / 'contract.json', contract)
        write(self.source / 'protocol.json', read(HERE / 'protocol.json'))
        write(self.source / 'request.json', dict(family='v2xr_raygen_e0e3', config=config, sources=original_sources,
                                                paths={'RUNTIME_ROOT': '/accepted/runtime'}))
        write(self.source / 'stage_status.json', {stage: dict(status='complete') for stage in ('preflight', 'smoke_prepare')})
        write(self.source / 'preflight/complete.json', dict(contract_identity=contract['identity']))
        for name in ('evidence.json', 'input_stamps.json', 'smoke_indices.json'):
            write(self.source / 'preflight' / name, {})
        write(self.source / 'data_check.json', dict(status='complete'))
        old_id = identity(dict(input=contract['inputs'], config=config, sources=original_sources, geometry=geometry))
        write(self.source / 'labels/contract.json', dict(identity=old_id))
        base = self.source / 'labels/samples/000000.json'
        write(base, dict(read(base), cache_identity=old_id))
        win = self.source / 'labels/windows/01/000000.json'
        write(win, dict(read(win), identity=identity(dict(base=sha(base), config=config, uid=self.uid, epoch=1))))
        write(self.source / 'smoke/E0/old_training_evidence.json', dict(old_training=True))
        return own

    def test_successor_initialization_preserves_old_run_and_new_stage_progress(self):
        own = self.prepare_initializer()
        before = {str(p): p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        with patch('v2xr_raygen_e0e3.successor.verify_sources', return_value=own), patch(
                'v2xr_raygen_e0e3.resume_compat.verify_active_sources'), patch(
                'v2xr_raygen_e0e3.resume_compat.effective_execution', return_value=dict(physical_gpu='2', microbatch=8)), patch(
                'v2xr_raygen_e0e3.runtime.input_stamps'), patch.dict('os.environ', {'PYTHONHASHSEED': '20261019', 'RUNTIME_ROOT': '/accepted/runtime'}):
            initialize(self.source, self.target, '2')
            record = read(self.target / 'data_reuse.json')
            self.assertFalse(record['training_states_imported'])
            self.assertFalse((self.target / 'smoke').exists())
            self.assertFalse((self.target / 'execution.json').exists())
            stages = read(self.target / 'stage_status.json')
            stages['smoke_E0_train_1'] = dict(status='complete', hash='new-checkpoint')
            write(self.target / 'stage_status.json', stages)
            initialize(self.source, self.target, '2')
            self.assertEqual(read(self.target / 'stage_status.json'), stages)
        from .controller import inherit_successor_paths
        with patch.dict('os.environ', {}, clear=True):
            inherit_successor_paths(self.target)
            import os
            self.assertEqual(os.environ['RUNTIME_ROOT'], '/accepted/runtime')
        with patch.dict('os.environ', {'RUNTIME_ROOT': '/changed/runtime'}), self.assertRaisesRegex(ValueError, 'environment differs'):
            inherit_successor_paths(self.target)
        for p, value in before.items():
            self.assertEqual(Path(p).read_bytes(), value)

    def test_successor_refuses_changed_cache_algorithm(self):
        own = self.prepare_initializer()
        own['files']['v2xr_raygen_e0e3/cache.py'] = 'changed'
        with patch('v2xr_raygen_e0e3.successor.verify_sources', return_value=own), patch(
                'v2xr_raygen_e0e3.resume_compat.verify_active_sources'), self.assertRaisesRegex(ValueError, 'cache algorithm changed'):
            initialize(self.source, self.target, '2')

    def test_successor_smoke_rechecks_same_capacity_and_checks_replay_before_evaluation(self):
        from .controller import Controller
        self.target.mkdir()
        write(self.target / 'data_reuse.json', dict(provenance=dict(execution=dict(microbatch=4))))
        write(self.target / 'contract.json', dict(identity='new-contract'))
        write(self.target / 'config.json', read(HERE / 'config.json'))
        controller = Controller.__new__(Controller)
        controller.root, controller.status = self.target, {}
        calls = []
        def stage(name, action, *extra, **kwargs):
            calls.append(name)
            if action == 'probe':
                self.assertEqual(extra, ('--group', 'E3', '--epoch', '1', '--micro', 4))
                write(self.target / 'probe/micro4/E3/epochs/01/train_complete.json', dict(gpu_name='synthetic GPU'))
            if action == 'train':
                group = extra[extra.index('--group')+1]
                epoch = int(extra[extra.index('--epoch')+1])
                branch = 'smoke_replay' if '--replay' in extra else 'smoke'
                ep = self.target / branch / group / 'epochs' / f'{epoch:02d}'
                write(ep / 'train_complete.json', dict(frames_per_second=1.))
                (ep / 'detector.pth').write_bytes(b'synthetic checkpoint bytes only')
            return 0
        controller.run = stage
        with patch.dict('os.environ', {'GPU': '2'}), patch('v2xr_raygen_e0e3.controller.os.link', side_effect=shutil.copyfile):
            controller.smoke()
        self.assertEqual([x for x in calls if x.startswith('capacity_micro')], ['capacity_micro4'])
        for group in ('E0', 'E1', 'E2', 'E3'):
            self.assertLess(calls.index(f'smoke_{group}_resume_check'), calls.index(f'smoke_{group}_eval_1'))


def run():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (StableMathTests, CacheImportTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if 'torch' in sys.modules or 'opencood' in sys.modules:
        raise AssertionError('Synthetic tests imported the research runtime')
    if not result.wasSuccessful():
        raise SystemExit(1)
    return result.testsRun


if __name__ == '__main__':
    run()
