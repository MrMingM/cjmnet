"""Pure NumPy/stdlib synthetic checks only; no real data, Torch or OpenCOOD imports."""
import copy
import ast
from collections import OrderedDict
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .common import HERE, read, stream_seed
from .geometry import candidate_indices, labels, loss_weights, nearest_returns, transform, windows
from .report import choose


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.c = read(HERE / 'config.json')
        self.center = np.array([0., 0., -1.])

    def make(self, points, origins=None, source=None, nearest=None):
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        o = np.zeros_like(p) if origins is None else np.asarray(origins, dtype=float)
        s = np.zeros(len(p), dtype=int) if source is None else np.asarray(source)
        n = np.ones(len(p), dtype=bool) if nearest is None else np.asarray(nearest, dtype=bool)
        return labels(p, o, s, n, self.center, self.c)

    def idx(self, p):
        return tuple(np.floor((np.array(p) - (self.center - [6.4, 6.4, 2])) / [.4, .4, .5]).astype(int)[::-1])

    def test_endpoint_free_behind_unknown(self):
        lab = self.make([[3., .1, -.8]], [[-4., .1, -.8]])
        self.assertTrue(lab['O'][self.idx([3., .1, -.8])])
        self.assertTrue(lab['E'][self.idx([0., .1, -.8])])
        self.assertTrue(lab['U'][self.idx([5., .1, -.8])])

    def test_distinct_sensor_origins(self):
        a = self.make([[3., 2., -.8]], [[-4., 2., -.8]])
        b = self.make([[3., 2., -.8]], [[0., 0., 0.]])
        self.assertTrue(a['E'][self.idx([-2., 2., -.8])])
        self.assertFalse(b['E'][self.idx([-2., 2., -.8])])

    def test_conflict_unknown_and_shared_x0(self):
        x = self.make([[1., .1, -.8], [4., .1, -.8]], [[-4., .1, -.8]] * 2, [0, 1])
        q = self.idx([1., .1, -.8])
        self.assertTrue(x['O_raw'][q] and x['E_raw'][q])
        self.assertTrue(x['U'][q]); self.assertEqual(x['x0'][q], 1)
        self.assertEqual(loss_weights('E2', x)[q], 0)

    def test_duplicate_invariance(self):
        a = self.make([[3., .1, -.8]], [[-4., .1, -.8]])
        b = self.make([[3., .1, -.8]] * 4, [[-4., .1, -.8]] * 4)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_nearest_return_no_free_behind(self):
        points = np.array([[2., 0., 0.], [5., 0., 0.], [2., 0., 0.]])
        near = nearest_returns(points, np.zeros(3), 1e-5)
        self.assertEqual(near.sum(), 1)
        x = self.make(points, nearest=near)
        self.assertFalse(x['E'][self.idx([3.5, 0., 0.])])

    def test_angular_wrap_nearest_return(self):
        p = np.array([[-2., 0., 0.], [-5., -0., 0.], [2., 1e-9, 0.], [5., -1e-9, 0.]])
        np.testing.assert_array_equal(nearest_returns(p, np.zeros(3), 1e-5), [True, False, True, False])

    def test_boundary_ownership(self):
        x = self.make([[-6.4, 0., -1.], [6.4, 0., -1.], [0., 0., 1.]], nearest=[False] * 3)
        self.assertEqual(x['O_raw'].sum(), 1)
        self.assertTrue(x['O_raw'][4, 16, 0])

    def test_sources_partition(self):
        x = self.make([[1., 0., -1.], [1., 0., -1.], [3., 1., -1.]],
                      source=[0, 1, 2], nearest=[False] * 3)
        self.assertFalse((x['ego'] & x['peer']).any())
        np.testing.assert_array_equal(x['O'], x['ego'] | x['peer'])
        self.assertEqual(x['ego'].sum(), 1); self.assertEqual(x['peer'].sum(), 1)

    def test_synchronized_affine_and_window(self):
        p, o = np.array([[3., .1, -.8]]), np.array([[-4., .1, -.8]])
        a = self.make(p, o)
        # Grid-preserving 180-degree rotation and translation, including origin/window.
        matrix = np.diag([-1., -1., 1., 1.]); matrix[:3, 3] = [.2, .2, 0.]
        b = labels(transform(p, matrix), transform(o, matrix), [0], [True],
                   transform(self.center[None], matrix)[0], self.c)
        self.assertEqual(a['O_raw'].sum(), b['O_raw'].sum())
        self.assertEqual(a['E_raw'].sum(), b['E_raw'].sum())
        # Inverse transformation recovers the entire labelling exactly.
        d = self.make(transform(transform(p, matrix), np.linalg.inv(matrix)),
                      transform(transform(o, matrix), np.linalg.inv(matrix)))
        for k in a:
            np.testing.assert_array_equal(a[k], d[k])

    def test_candidate_query_keeps_entering_rays(self):
        p = np.array([[10., 0., -1.], [12., 15., -1.]])
        o = np.array([[-10., 0., -1.], [10., 15., -1.]])
        pick = candidate_indices(p, o, self.center, [12.8, 12.8, 4], 1.6)
        np.testing.assert_array_equal(pick, [0])

    def test_empty_all_unknown(self):
        x = self.make([])
        self.assertTrue(x['U'].all())
        self.assertEqual(loss_weights('E2', x).sum(), 0)
        self.assertEqual(loss_weights('E3', x).sum(), 0)
        self.assertAlmostEqual(float(loss_weights('E1', x).sum()), 1, places=6)

    def test_loss_denominators(self):
        x = self.make([[1., 0., -1.], [2., 1., -1.], [3., 1., -1.]], source=[0, 1, 1], nearest=[False] * 3)
        x['E'][0, 0, :4] = True
        w = loss_weights('E3', x)
        self.assertAlmostEqual(float(w[x['ego']].sum()), .25)
        self.assertAlmostEqual(float(w[x['peer']].sum()), .25)
        self.assertAlmostEqual(float(w[x['E']].sum()), .5)
        w2 = loss_weights('E2', x)
        self.assertAlmostEqual(float(w2[x['O']].sum()), .5)
        self.assertAlmostEqual(float(w2[x['E']].sum()), .5)
        x['E'][:] = False
        self.assertAlmostEqual(float(loss_weights('E3', x)[x['ego']].sum()), .5)
        x['ego'][:] = False; x['O'] = x['peer'].copy()
        self.assertAlmostEqual(float(loss_weights('E3', x).sum()), 1.)

    def test_fixed_windows_independent_rng(self):
        boxes = np.zeros((20, 7)); boxes[:, 0] = np.arange(20)
        a = windows(boxes, 20261019, 1, ['scene', 'stamp', 'ego'], self.c)
        np.random.seed(123); np.random.randn(1000)
        b = windows(boxes, 20261019, 1, ['scene', 'stamp', 'ego'], self.c)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(len(a[0]), 8)
        self.assertTrue(np.all(np.abs(a[1][:, 0] - boxes[a[0], 0]) <= 1.6))
        self.assertNotEqual(stream_seed(1, 1, 'uid', 'base'), stream_seed(1, 1, 'uid', 'diffusion'))

    def test_packed_masks_roundtrip(self):
        x = self.make([[2., 0., -1.]])
        for k in ('O_raw', 'E_raw', 'ego', 'peer'):
            packed = np.packbits(x[k].reshape(-1))
            self.assertEqual(packed.size, 1024)
            np.testing.assert_array_equal(np.unpackbits(packed).reshape(8, 32, 32).astype(bool), x[k])


class SelectionTests(unittest.TestCase):
    def row(self, epoch, clean, fog, snow):
        return dict(epoch=epoch, results={k: dict(AP={'O': dict(ap70=v)})
                                         for k, v in [('clean', clean), ('fog', fog), ('snow', snow)]})

    def test_tolerance_and_earliest_tie(self):
        rows = [self.row(1, .8, .4, .4), self.row(2, .797, .6, .6),
                self.row(3, .796, .6, .6), self.row(4, .79, .9, .9)]
        b, m, ids = choose(rows)
        self.assertEqual(b['epoch'], 2); self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(m, .8)


class SmokeFixtureRegressionTests(unittest.TestCase):
    def test_actual_all_unknown_fixture_supplies_loss_counter_fields(self):
        # Execute the exact NumPy-only fixture statements used by the server checks.
        # Parsing the loss consumer also catches future additions to its required counters.
        smoke_path = HERE / 'smoke.py'
        module = ast.parse(smoke_path.read_text(encoding='utf-8'))
        checks = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'checks')

        def assignment_index(name):
            return next(i for i, n in enumerate(checks.body) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))

        statements = (checks.body[assignment_index('lab'):assignment_index('prediction')]
                      + checks.body[assignment_index('all_unknown'):assignment_index('unknown_batch')])
        scope = dict(np=np, copy=copy)
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(smoke_path), 'exec'), scope)
        unknown = scope['all_unknown']
        losses = ast.parse((HERE / 'losses.py').read_text(encoding='utf-8'))
        consumer = next(n for n in losses.body if isinstance(n, ast.FunctionDef) and n.name == 'auxiliary_loss')
        counter_loop = next(n for n in ast.walk(consumer) if isinstance(n, ast.For)
                            and isinstance(n.target, ast.Name) and n.target.id == 'key'
                            and isinstance(n.iter, ast.Tuple))
        keys = ast.literal_eval(counter_loop.iter)
        counts = {key: int(unknown[key].sum()) for key in keys}
        self.assertEqual(counts, dict(O=0, E=0, U=8192, ego=0, peer=0))
        for key in keys:
            self.assertEqual(unknown[key].shape, unknown['x0'].shape)
            self.assertEqual(unknown[key].dtype, np.dtype(bool))
        np.testing.assert_array_equal(unknown['U'], ~(unknown['O'] | unknown['E']))
        self.assertTrue((unknown['x0'] == -1).all())
        for group in ('E2', 'E3'):
            self.assertEqual(float(loss_weights(group, unknown).sum()), 0.)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('opencood', sys.modules)


class RepairResumeTests(unittest.TestCase):
    def setUp(self):
        from .common import identity, write, sha
        from .resume_compat import ADDED, CHANGED
        fixture_parent = HERE.parent if sys.platform == 'win32' else Path(tempfile.gettempdir())
        self.temp = tempfile.TemporaryDirectory(prefix='.raygen_resume_fixture_', dir=fixture_parent)
        if fixture_parent.resolve() not in Path(self.temp.name).resolve().parents:
            raise AssertionError('Synthetic fixture cleanup must stay inside its designated temporary directory')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity, self.write, self.sha = identity, write, sha
        self.original = dict(files=read(HERE / 'U_FIX_COMPAT.json')['original_files'],
                             dependencies=read(HERE / 'PINS.json')['files'])
        self.active = copy.deepcopy(self.original)
        for name in CHANGED | ADDED:
            self.active['files'][name] = 'f' * 64
        config = read(HERE / 'config.json')
        self.old_request = dict(family='v2xr_raygen_e0e3', config=config, sources=self.original, paths={'PYTHON': '/same/python'})
        self.new_request = dict(self.old_request, sources=self.active)
        contract = dict(config=config, sources=self.original, inputs={'train': 'synthetic-identity'},
                        initialization='synthetic metadata only')
        contract['identity'] = identity(contract)
        self.contract = contract
        write(self.root / 'request.json', self.old_request)
        write(self.root / 'contract.json', contract)
        write(self.root / 'config.json', config)
        write(self.root / 'protocol.json', read(HERE / 'protocol.json'))
        self.status = {name: dict(status='complete') for name in
                       ('preflight', 'synthetic_checks', 'smoke_prepare', 'capacity_micro4')}
        self.status['server_checks'] = dict(status='failed')
        write(self.root / 'stage_status.json', self.status)
        write(self.root / 'failure.json', dict(stage='server_checks', returncode=1,
              traceback="zero_unknown, _ = auxiliary_loss(...)\nKeyError: 'U'"))
        execution = dict(microbatch=4, accumulation=2, effective_batch=8, physical_gpu='0', gpu_name='synthetic GPU A')
        write(self.root / 'execution.json', execution)
        write(self.root / 'execution_commit.json', dict(sha256=sha(self.root / 'execution.json'), parent_contract=contract['identity']))
        ep = self.root / 'probe/micro4/E3/epochs/01'
        ep.mkdir(parents=True)
        hashes = {}
        for name in ('detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json'):
            (ep / name).write_bytes(b'synthetic commit bytes; not a real model')
            hashes[name] = sha(ep / name)
        write(ep / 'train_complete.json', dict(status='complete', smoke_only=True, group='E3', epoch=1,
              contract_identity=contract['identity'], physical_gpu='0', gpu_name=execution['gpu_name'], output_hashes=hashes))
        labels_root = self.root / 'labels'
        cache_id = identity(dict(input=contract['inputs'], config=config, sources=self.original,
                                 geometry=read(HERE / 'protocol.json')['geometry']))
        write(labels_root / 'contract.json', dict(identity=cache_id))
        write(self.root / 'preflight/smoke_indices.json', dict(indices=[0]))
        sample = labels_root / 'samples/000000.json'
        sample.parent.mkdir(parents=True)
        sample.with_suffix('.npz').write_bytes(b'synthetic cache bytes; no real points')
        uid = ['synthetic_scene', '000000', '1']
        write(sample, dict(status='complete', cache_identity=cache_id, uid=uid,
                           sha256=sha(sample.with_suffix('.npz'))))
        for epoch in (1, 2):
            path = labels_root / 'windows' / f'{epoch:02d}/000000.json'
            path.parent.mkdir(parents=True)
            path.with_suffix('.npz').write_bytes(b'synthetic packed-window bytes')
            write(path, dict(status='complete', sha256=sha(path.with_suffix('.npz')),
                             identity=identity(dict(base=sha(sample), config=config, uid=uid, epoch=epoch))))

    def enroll(self, gpu='0'):
        from .resume_compat import accept_resume
        with patch.dict('os.environ', {'GPU': gpu}), patch('v2xr_raygen_e0e3.common.lock',
                new=lambda path: __import__('contextlib').nullcontext()), patch(
                'v2xr_raygen_e0e3.resume_compat.lock', new=lambda path: __import__('contextlib').nullcontext()), patch(
                'v2xr_raygen_e0e3.runtime.input_stamps'):
            return accept_resume(self.root, self.old_request, self.new_request)

    def prepare_gpu_probe(self, gpu='2', micro=4):
        from .resume_compat import probe_root
        ep = probe_root(self.root, micro, gpu) / 'E3/epochs/01'
        ep.mkdir(parents=True, exist_ok=True)
        prior = self.root / 'probe/micro4/E3/epochs/01'
        measured = read(prior / 'train_complete.json')
        for name in measured['output_hashes']:
            (ep / name).write_bytes((prior / name).read_bytes())
        self.write(ep / 'sampling.json', dict(microbatch=micro, accumulation=8 // micro))
        measured.update(physical_gpu=gpu, gpu_name='synthetic GPU B',
                        output_hashes={name: self.sha(ep / name) for name in measured['output_hashes']})
        self.write(ep / 'train_complete.json', measured)
        return ep

    def test_preserves_original_contract_and_cache_and_records_actual_sources(self):
        from .resume_compat import verify_active_sources
        original_files = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        record = self.enroll()
        for name, content in original_files.items():
            self.assertEqual((self.root / name).read_bytes(), content)
        self.assertEqual(record['active_sources'], self.active)
        self.assertEqual(record['preserved_smoke_samples'], 1)
        self.assertEqual(record['repeated_stages'], ['u_fix_regression', 'server_checks'])
        self.assertEqual(verify_active_sources(self.root, self.active), record)

    def test_rejects_scientific_code_changes(self):
        self.active['files']['v2xr_raygen_e0e3/losses.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Scientific source'):
            self.enroll()

    def test_rejects_changed_config(self):
        self.new_request['config'] = dict(self.new_request['config'], seed=1)
        with self.assertRaisesRegex(ValueError, 'identical configuration'):
            self.enroll()

    def test_rejects_damaged_prepared_cache(self):
        (self.root / 'labels/windows/01/000000.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'cache missing/changed'):
            self.enroll()

    def test_rejects_damaged_capacity_probe(self):
        (self.root / 'probe/micro4/E3/epochs/01/resume.pth').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'probe artifact changed'):
            self.enroll()

    def test_rejects_later_training_boundary(self):
        self.write(self.root / 'stage_status.json', dict(self.status, E0_train_01=dict(status='running')))
        with self.assertRaisesRegex(ValueError, 'pre-training failure boundary'):
            self.enroll()

    def test_rejects_unrelated_failure(self):
        self.write(self.root / 'failure.json', dict(stage='server_checks', returncode=1, traceback='OOM'))
        with self.assertRaisesRegex(ValueError, 'exact supported'):
            self.enroll()

    def test_repeated_resume_requires_same_enrolled_sources(self):
        from .resume_compat import verify_active_sources
        first = self.enroll()
        self.write(self.root / 'stage_status.json', dict(self.status, E0_train_01=dict(status='complete')))
        self.assertEqual(self.enroll()['identity'], first['identity'])
        changed = copy.deepcopy(self.active)
        changed['files']['v2xr_raygen_e0e3/smoke.py'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'lineage/source/config'):
            verify_active_sources(self.root, changed)

    def test_gpu2_enrollment_preserves_all_original_files_and_requires_new_probe(self):
        from .resume_compat import effective_execution
        originals = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        record = self.enroll('2')
        self.assertEqual(record['requested_physical_gpu'], '2')
        self.assertEqual(record['repeated_stages'], ['u_fix_regression', 'capacity_gpu2', 'server_checks'])
        self.assertTrue(all(p.read_bytes() == content for p, content in originals.items()))
        with self.assertRaisesRegex(ValueError, 'completed capacity recheck'):
            effective_execution(self.root)

    def test_gpu2_probe_can_only_use_requested_device_and_original_microbatch(self):
        from .resume_compat import effective_execution
        self.enroll('2')
        with patch.dict('os.environ', {'GPU': '2'}):
            selected = effective_execution(self.root, capacity_gpu='2', micro=4)
            self.assertEqual(selected['physical_gpu'], '2')
            self.assertIsNone(selected['gpu_name'])
            for device, micro in [('1', 4), ('2', 2), ('../2', 4)]:
                with self.assertRaisesRegex(ValueError, 'enrolled target'):
                    effective_execution(self.root, capacity_gpu=device, micro=micro)
        with patch.dict('os.environ', {'GPU': '0'}), self.assertRaisesRegex(ValueError, 'enrolled target'):
            effective_execution(self.root, capacity_gpu='2', micro=4)

    def test_gpu2_committed_capacity_keeps_recipe_and_original_execution(self):
        from .resume_compat import commit_gpu_capacity, effective_execution, verify_active_sources
        old = (self.root / 'execution.json').read_bytes()
        self.enroll('2')
        self.prepare_gpu_probe()
        with patch.dict('os.environ', {'GPU': '2'}):
            selected = commit_gpu_capacity(self.root)
            self.assertEqual(commit_gpu_capacity(self.root), selected)
        self.assertEqual(selected, dict(read(self.root / 'execution.json'), physical_gpu='2', gpu_name='synthetic GPU B'))
        self.assertEqual(effective_execution(self.root), selected)
        self.assertEqual((self.root / 'execution.json').read_bytes(), old)
        self.assertIsNotNone(verify_active_sources(self.root, self.active))
        self.assertEqual(self.enroll('2')['requested_physical_gpu'], '2')
        with self.assertRaisesRegex(ValueError, 'keep the GPU selected'):
            self.enroll('1')

    def test_gpu2_damaged_probe_cannot_be_committed(self):
        from .resume_compat import GPU_CAPACITY, commit_gpu_capacity
        self.enroll('2')
        ep = self.prepare_gpu_probe()
        (ep / 'resume.pth').write_bytes(b'damaged')
        with patch.dict('os.environ', {'GPU': '2'}), self.assertRaisesRegex(ValueError, 'probe artifact changed'):
            commit_gpu_capacity(self.root)
        self.assertFalse((self.root / GPU_CAPACITY).exists())

    def test_gpu2_probe_from_another_device_is_rejected(self):
        from .resume_compat import commit_gpu_capacity
        self.enroll('2')
        ep = self.prepare_gpu_probe()
        self.write(ep / 'train_complete.json', dict(read(ep / 'train_complete.json'), physical_gpu='1'))
        with patch.dict('os.environ', {'GPU': '2'}), self.assertRaisesRegex(ValueError, 'probe evidence differs'):
            commit_gpu_capacity(self.root)

    def test_gpu2_sampling_cannot_silently_reduce_microbatch(self):
        from .resume_compat import commit_gpu_capacity
        self.enroll('2')
        ep = self.prepare_gpu_probe()
        self.write(ep / 'sampling.json', dict(microbatch=2, accumulation=4))
        measured = read(ep / 'train_complete.json')
        measured['output_hashes']['sampling.json'] = self.sha(ep / 'sampling.json')
        self.write(ep / 'train_complete.json', measured)
        with patch.dict('os.environ', {'GPU': '2'}), self.assertRaisesRegex(ValueError, 'changed the selected batch'):
            commit_gpu_capacity(self.root)

    def test_gpu2_commit_changes_are_detected(self):
        from .resume_compat import commit_gpu_capacity, effective_execution
        self.enroll('2')
        ep = self.prepare_gpu_probe()
        with patch.dict('os.environ', {'GPU': '2'}):
            commit_gpu_capacity(self.root)
        self.write(ep / 'train_complete.json', dict(read(ep / 'train_complete.json'), gpu_name='different hardware'))
        with self.assertRaisesRegex(ValueError, 'recheck lineage'):
            effective_execution(self.root)

    def test_controller_rechecks_gpu2_before_server_checks_and_reuses_completed_probe(self):
        from .controller import Controller
        self.enroll('2')
        controller = Controller.__new__(Controller)
        controller.root = self.root
        calls = []
        class AtServerChecks(Exception):
            pass
        def stage(name, action, *extra):
            calls.append((name, action, extra))
            if name == 'capacity_gpu2':
                self.assertEqual(extra, ('--group', 'E3', '--epoch', '1', '--micro', 4, '--capacity-gpu', '2'))
                self.prepare_gpu_probe()
            if name == 'server_checks':
                raise AtServerChecks()
        controller.run = stage
        with patch.dict('os.environ', {'GPU': '2'}):
            with self.assertRaises(AtServerChecks):
                controller.smoke()
            self.assertEqual([r[0] for r in calls], ['smoke_prepare', 'stable_calculation_checks', 'capacity_gpu2', 'server_checks'])
            calls.clear()
            with self.assertRaises(AtServerChecks):
                controller.smoke()
            self.assertEqual([r[0] for r in calls], ['smoke_prepare', 'stable_calculation_checks', 'server_checks'])

    def test_controller_capacity_failure_stops_before_server_checks(self):
        from .controller import Controller
        from .resume_compat import GPU_CAPACITY
        self.enroll('2')
        controller = Controller.__new__(Controller)
        controller.root = self.root
        calls = []
        def stage(name, action, *extra):
            calls.append(name)
            if name == 'capacity_gpu2':
                raise RuntimeError('simulated server capacity failure')
        controller.run = stage
        with patch.dict('os.environ', {'GPU': '2'}), self.assertRaisesRegex(RuntimeError, 'simulated server'):
            controller.smoke()
        self.assertEqual(calls, ['smoke_prepare', 'stable_calculation_checks', 'capacity_gpu2'])
        self.assertFalse((self.root / GPU_CAPACITY).exists())


class ResumeComparisonTests(unittest.TestCase):
    def test_float_logs_use_tolerance_but_loss_count_is_exact(self):
        from .resume_compare import compare_states
        a = dict(criterion_accumulators=dict(conf_loss=1., loss_cnt=4))
        b = dict(criterion_accumulators=dict(conf_loss=1.00001, loss_cnt=4))
        self.assertEqual(compare_states(a, b)['status'], 'complete')
        b['criterion_accumulators']['loss_cnt'] = 5
        self.assertEqual(compare_states(a, b)['status'], 'failed')

    def test_moment_norm_does_not_relax_model_parameter_tolerance(self):
        from .resume_compare import compare_states
        a = np.array([10., .04], np.float32)
        b = np.array([10., .04006], np.float32)
        optimizer_a = dict(optimizer=dict(state={3: dict(exp_avg=a)}))
        optimizer_b = dict(optimizer=dict(state={3: dict(exp_avg=b)}))
        report = compare_states(optimizer_a, optimizer_b)
        self.assertEqual(report['status'], 'complete')
        self.assertFalse(report['numeric'][0]['original_elementwise_pass'])
        self.assertTrue(report['numeric'][0]['optimizer_norm_pass'])
        self.assertEqual(compare_states(dict(model=dict(w=a)), dict(model=dict(w=b)))['status'], 'failed')

    def test_large_moment_drift_is_rejected(self):
        from .resume_compare import compare_states
        a = dict(optimizer=dict(state={3: dict(exp_avg=np.array([.1, .1], np.float32))}))
        b = dict(optimizer=dict(state={3: dict(exp_avg=np.array([.1, .2], np.float32))}))
        self.assertEqual(compare_states(a, b)['status'], 'failed')

    def test_learning_rate_step_and_rng_remain_exact(self):
        from .resume_compare import compare_states
        for a, b in [(dict(optimizer=dict(param_groups=[dict(lr=.002)])), dict(optimizer=dict(param_groups=[dict(lr=.00200000001)]))),
                     (dict(optimizer=dict(state={3: dict(step=np.array(2., np.float32))})),
                      dict(optimizer=dict(state={3: dict(step=np.array(2.000001, np.float32))}))),
                     (dict(rng=dict(cpu=np.array([1, 2], np.uint8))), dict(rng=dict(cpu=np.array([1, 3], np.uint8))))]:
            self.assertEqual(compare_states(a, b)['status'], 'failed')

    def test_exact_postload_rejects_tiny_weight_change(self):
        from .resume_compare import compare_states
        a = dict(model=dict(w=np.array([1.], np.float32)))
        b = dict(model=dict(w=np.array([1.000001], np.float32)))
        self.assertEqual(compare_states(a, b)['status'], 'complete')
        self.assertEqual(compare_states(a, b, exact=True)['status'], 'failed')

    def test_nonfinite_arrays_and_scalars_never_pass(self):
        from .resume_compare import compare_states
        for value in (float('nan'), float('inf'), float('-inf')):
            a = dict(optimizer=dict(state={3: dict(exp_avg=np.array([value], np.float32))}))
            self.assertEqual(compare_states(a, a)['status'], 'failed')
            a = dict(criterion_accumulators=dict(total_loss=value))
            self.assertEqual(compare_states(a, a)['status'], 'failed')

    def test_actual_load_function_detects_a_loader_that_ignored_saved_optimizer(self):
        # Execute the actual function using NumPy state holders; no Torch/OpenCOOD import.
        from .resume_compare import POLICY_ID
        source = ast.parse((HERE / 'training.py').read_text(encoding='utf-8'))
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'load_resume_state')
        function.body = [n for n in function.body if not (isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names))]
        class Holder:
            def __init__(self, state, broken=False):
                self.state, self.broken = copy.deepcopy(state), broken
            def state_dict(self):
                return self.state
            def load_state_dict(self, state, **kw):
                if not self.broken:
                    self.state = copy.deepcopy(state)
        rng = {}
        namespace = dict(__package__='v2xr_raygen_e0e3', np=np,
            torch=SimpleNamespace(Tensor=type('UnusedTensor', (), {})),
            restore_rng=lambda state: rng.update(copy.deepcopy(state)), rng_state=lambda: copy.deepcopy(rng))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-resume-loader-numpy-stubs>', 'exec'), namespace)
        state = dict(model={'w': np.array([1.], np.float32)}, aux=None,
                     optimizer={'state': {3: {'exp_avg': np.array([.1], np.float32)}}},
                     scheduler={'last_epoch': 1}, criterion={}, criterion_accumulators={'loss_cnt': 4},
                     rng={'cpu': np.array([1, 2], np.uint8)}, global_step=2)
        args = [Holder({}), None, Holder({}), Holder({}), Holder({}), state]
        step, report = namespace['load_resume_state'](*args)
        self.assertEqual(step, 2)
        self.assertEqual(report['policy_identity'], POLICY_ID)
        self.assertEqual(report['status'], 'complete')
        args[3] = Holder({}, broken=True)
        with self.assertRaisesRegex(ValueError, 'not exact'):
            namespace['load_resume_state'](*args)


class NumericRepairTests(unittest.TestCase):
    def setUp(self):
        from .resume_compat import NUMERIC_ADDED, commit_gpu_capacity
        self.base = RepairResumeTests()
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root = self.base.root
        self.write, self.sha = self.base.write, self.base.sha
        compat = read(HERE / 'RESUME_CHECK_COMPAT.json')
        self.previous = dict(files=compat['previous_files'], dependencies=self.base.original['dependencies'])
        self.base.active = self.previous
        self.base.new_request = dict(self.base.old_request, sources=self.previous)
        self.base.enroll('2')
        self.base.prepare_gpu_probe()
        with patch.dict('os.environ', {'GPU': '2'}):
            commit_gpu_capacity(self.root)
        self.active = copy.deepcopy(self.previous)
        for name in ('training.py', 'smoke.py', 'controller.py', 'worker.py', 'report.py', 'resume_compat.py',
                     'tests_local.py', 'README.md', 'LOCAL_VALIDATION.md'):
            self.active['files']['v2xr_raygen_e0e3/' + name] = 'b' * 64
        for name in NUMERIC_ADDED:
            self.active['files'][name] = 'b' * 64
        self.base.new_request = dict(self.base.old_request, sources=self.active)
        status = dict(self.base.status, server_checks=dict(status='complete'))
        for name in ('smoke_E0_train_1', 'smoke_E0_eval_1', 'smoke_E0_train_2', 'smoke_E0_eval_2', 'smoke_E0_resume_replay'):
            status[name] = dict(status='complete')
        status['smoke_E0_resume_check'] = dict(status='failed')
        self.write(self.root / 'stage_status.json', status)
        self.write(self.root / 'failure.json', dict(stage='smoke_E0_resume_check', group='E0', returncode=1,
              traceback='HIP resume exceeds registered tolerance: optimizer/state/3/exp_avg delta=6e-5'))
        commits = []
        for epoch in (1, 2):
            for branch in ('smoke', 'smoke_replay'):
                folder = self.root / branch / 'E0/epochs' / f'{epoch:02d}'
                folder.mkdir(parents=True, exist_ok=True)
                for name in ('detector.pth', 'resume.pth'):
                    (folder / name).write_bytes(b'synthetic checkpoint only')
                for name in ('sampling.json', 'actual_samples.json'):
                    self.write(folder / name, dict(epoch=epoch, synthetic=True))
                commit = dict(status='complete', smoke_only=True, group='E0', epoch=epoch,
                    physical_gpu='2', gpu_name='synthetic GPU B', contract_identity=self.base.contract['identity'],
                    output_hashes={name: self.sha(folder / name) for name in
                                   ('detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json')})
                self.write(folder / 'train_complete.json', commit)
                commits.append(commit)
        self.diagnostic_path = self.root / 'diagnostics/resume_E0_1.json'
        self.diagnostic = dict(diagnostic_source_sha256=compat['diagnostic_source_sha256'], checkpoint_commits=commits,
            artifact_hashes_verified=True, epoch1_start_identical=True, same_device=True,
            epoch2_schedules_identical={'sampling.json': True, 'actual_samples.json': True},
            sections={key: dict(passed=True) for key in ('model', 'aux', 'rng', 'scheduler')},
            exact_errors=[dict(path='criterion_accumulators/conf_loss')],
            failed_floating_arrays=[dict(path='optimizer/state/3/exp_avg', relative_l2=6e-6, nonfinite_left=0, nonfinite_right=0)])
        self.write(self.diagnostic_path, self.diagnostic)

    def enroll(self):
        return self.base.enroll('2')

    def test_upgrade_preserves_every_existing_file_and_old_lineage(self):
        from .resume_compat import NUMERIC_MARKER, verify_active_sources
        original = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        self.enroll()
        self.assertTrue(all(path.read_bytes() == content for path, content in original.items()))
        self.assertTrue((self.root / NUMERIC_MARKER).exists())
        self.assertIsNotNone(verify_active_sources(self.root, self.active))
        self.enroll()

    def test_upgrade_rejects_nonmatching_rng_diagnostic(self):
        self.diagnostic['sections']['rng']['passed'] = False
        self.write(self.diagnostic_path, self.diagnostic)
        with self.assertRaisesRegex(ValueError, 'CPU diagnostic first'):
            self.enroll()

    def test_upgrade_rejects_unrelated_scientific_source_changes(self):
        self.active['files']['v2xr_raygen_e0e3/losses.py'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'Unrelated scientific source'):
            self.enroll()

    def test_upgrade_rejects_large_optimizer_difference(self):
        self.diagnostic['failed_floating_arrays'][0]['relative_l2'] = .01
        self.write(self.diagnostic_path, self.diagnostic)
        with self.assertRaisesRegex(ValueError, 'CPU diagnostic first'):
            self.enroll()

    def test_upgrade_rejects_corrupted_committed_artifact(self):
        (self.root / 'smoke/E0/epochs/02/resume.pth').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.enroll()

    def test_upgrade_rejects_already_started_next_group(self):
        status = read(self.root / 'stage_status.json')
        status['smoke_E1_train_1'] = dict(status='running')
        self.write(self.root / 'stage_status.json', status)
        with self.assertRaisesRegex(ValueError, 'E0-only'):
            self.enroll()

    def test_upgraded_worker_rejects_previous_or_changed_source_identity(self):
        from .resume_compat import verify_active_sources
        self.enroll()
        with self.assertRaisesRegex(ValueError, 'v2 source/policy/lineage'):
            verify_active_sources(self.root, self.previous)
        (self.root / 'smoke/E0/epochs/02/resume.pth').write_bytes(b'changed after enrollment')
        with self.assertRaisesRegex(ValueError, 'artifact metadata changed'):
            verify_active_sources(self.root, self.active)

    def test_controller_reuses_e0_train_eval_and_capacity_then_only_checks_resume(self):
        from .controller import Controller
        self.enroll()
        controller = Controller.__new__(Controller)
        controller.root = self.root
        controller.status = read(self.root / 'stage_status.json')
        started = []
        class NextGroup(Exception):
            pass
        def stage(name, action, *extra, **options):
            if controller.status.get(name, {}).get('status') == 'complete':
                return 0
            started.append(name)
            if name == 'smoke_E1_train_1':
                raise NextGroup()
            return 0
        controller.run = stage
        with patch.dict('os.environ', {'GPU': '2'}), self.assertRaises(NextGroup):
            controller.smoke()
        self.assertEqual(started, ['stable_calculation_checks', 'smoke_E0_restore_check_v2', 'smoke_E0_resume_check', 'smoke_E1_train_1'])

    def test_upgrade_rejects_loss_count_drift_or_nonfinite_moments(self):
        for row in [dict(path='criterion_accumulators/loss_cnt'), dict(path='optimizer/state/3/step')]:
            self.diagnostic['exact_errors'] = [row]
            self.write(self.diagnostic_path, self.diagnostic)
            with self.assertRaisesRegex(ValueError, 'CPU diagnostic first'):
                self.enroll()
        self.diagnostic['exact_errors'] = []
        self.diagnostic['failed_floating_arrays'][0]['nonfinite_left'] = 1
        self.write(self.diagnostic_path, self.diagnostic)
        with self.assertRaisesRegex(ValueError, 'CPU diagnostic first'):
            self.enroll()


class RMSPolicyTests(unittest.TestCase):
    def test_e1_sized_tensor_uses_absolute_budget_per_element(self):
        from .resume_compare import POLICY_V3, compare_states
        b = np.full(128, 1.0707758045136373 / np.sqrt(128), np.float32)
        a = b.copy()
        a[0] += np.float32(.00017879605)
        left = dict(optimizer=dict(state={17: dict(exp_avg=a)}))
        right = dict(optimizer=dict(state={17: dict(exp_avg=b)}))
        self.assertEqual(compare_states(left, right)['status'], 'failed')
        report = compare_states(left, right, policy=POLICY_V3)
        self.assertEqual(report['status'], 'complete')
        self.assertFalse(report['numeric'][0]['original_elementwise_pass'])
        self.assertLess(report['numeric'][0]['rms_delta'], 1e-5 + 1e-4 * report['numeric'][0]['rms_scale'])
        self.assertEqual(compare_states(dict(model={'w': a}), dict(model={'w': b}), policy=POLICY_V3)['status'], 'failed')

    def test_duplicate_coefficients_do_not_change_rms_decision(self):
        from .resume_compare import POLICY_V3, compare_states
        for count in (1, 100, 1000):
            a = np.tile(np.array([1., .010113], np.float32), count)
            b = np.tile(np.array([1., .01], np.float32), count)
            left = dict(optimizer=dict(state={17: dict(exp_avg=a)}))
            right = dict(optimizer=dict(state={17: dict(exp_avg=b)}))
            result = compare_states(left, right, policy=POLICY_V3)
            self.assertEqual(result['status'], 'complete')
            self.assertAlmostEqual(result['numeric'][0]['rms_delta'], .000113 / np.sqrt(2), places=8)

    def test_rms_does_not_allow_large_drift_or_nan(self):
        from .resume_compare import POLICY_V3, compare_states
        for value in (.01, float('nan'), float('inf')):
            a = np.zeros(128, np.float32)
            a[0] = value
            left = dict(optimizer=dict(state={17: dict(exp_avg=a)}))
            right = dict(optimizer=dict(state={17: dict(exp_avg=np.zeros(128, np.float32))}))
            self.assertEqual(compare_states(left, right, policy=POLICY_V3)['status'], 'failed')
        a = dict(optimizer=dict(param_groups=[dict(lr=.002)]))
        b = dict(optimizer=dict(param_groups=[dict(lr=.002000001)]))
        self.assertEqual(compare_states(a, b, policy=POLICY_V3)['status'], 'failed')


class RMSRepairTests(unittest.TestCase):
    def setUp(self):
        from .resume_compare import POLICY, POLICY_ID
        from .resume_compat import RMS_ADDED
        self.numeric = NumericRepairTests()
        self.numeric.setUp()
        self.addCleanup(self.numeric.doCleanups)
        self.root = self.numeric.root
        self.write, self.sha = self.numeric.write, self.numeric.sha
        previous = read(HERE / 'RESUME_RMS_COMPAT.json')['previous_files']
        self.previous = dict(files=previous, dependencies=self.numeric.previous['dependencies'])
        self.numeric.active = self.previous
        self.numeric.base.new_request = dict(self.numeric.base.old_request, sources=self.previous)
        self.numeric.enroll()
        self.active = copy.deepcopy(self.previous)
        for name in ('resume_compare.py', 'smoke.py', 'resume_compat.py', 'controller.py', 'report.py',
                     'tests_local.py', 'README.md', 'LOCAL_VALIDATION.md'):
            self.active['files']['v2xr_raygen_e0e3/' + name] = 'c' * 64
        for name in RMS_ADDED:
            self.active['files'][name] = 'c' * 64
        self.numeric.base.new_request = dict(self.numeric.base.old_request, sources=self.active)
        status = read(self.root / 'stage_status.json')
        status['smoke_E0_resume_check'] = dict(status='complete')
        status['smoke_E0_restore_check_v2'] = dict(status='complete')
        for action in ('train', 'eval'):
            for epoch in (1, 2):
                status[f'smoke_E1_{action}_{epoch}'] = dict(status='complete')
        for action in ('resume_replay', 'restore_check_v2'):
            status['smoke_E1_' + action] = dict(status='complete')
        status['smoke_E1_resume_check'] = dict(status='failed')
        self.write(self.root / 'stage_status.json', status)
        self.write(self.root / 'failure.json', dict(stage='smoke_E1_resume_check', group='E1', returncode=1,
                                                  traceback='Resume exceeds v2 state validation policy'))
        self.folders = [self.root / branch / 'E1/epochs' / f'{epoch:02d}'
                        for epoch in (1, 2) for branch in ('smoke', 'smoke_replay')]
        for folder in self.folders:
            folder.mkdir(parents=True)
            for name in ('detector.pth', 'resume.pth'):
                (folder / name).write_bytes(b'synthetic E1 checkpoint bytes only')
            for name in ('sampling.json', 'actual_samples.json'):
                self.write(folder / name, dict(epoch=int(folder.name), synthetic=True))
            self.write(folder / 'train_complete.json', dict(status='complete', smoke_only=True, group='E1',
                epoch=int(folder.name), physical_gpu='2', gpu_name='synthetic GPU B',
                contract_identity=self.numeric.base.contract['identity'], output_hashes={name: self.sha(folder / name) for name in
                    ('detector.pth', 'resume.pth', 'sampling.json', 'actual_samples.json')}))
        self.restored_path = self.root / 'smoke/E1/restore_check_v2.json'
        self.write(self.restored_path, dict(status='complete', exact=True, policy_identity=POLICY_ID,
            group='E1', physical_gpu='2', gpu_name='synthetic GPU B',
            source_checkpoint_sha256=self.sha(self.folders[0] / 'resume.pth'),
            source_commit_sha256=self.sha(self.folders[0] / 'train_complete.json'),
            comparison=dict(status='complete', exact=True, arrays_checked=564, failures=[])))
        self.diagnostic_path = self.root / 'smoke/E1/resume_check_v2_diagnostic.json'
        self.diagnostic = dict(status='failed', policy=POLICY, policy_identity=POLICY_ID,
            same_start_exact=True, data_schedule_exact=True,
            original_checkpoint_sha256=self.sha(self.folders[2] / 'resume.pth'),
            replay_checkpoint_sha256=self.sha(self.folders[3] / 'resume.pth'),
            exact_restore_report_sha256=self.sha(self.restored_path), comparison=dict(
                failures=[dict(path='optimizer/state/17/exp_avg')], numeric=[dict(path='optimizer/state/17/exp_avg',
                passed=False, l2_delta=.00017879605, l2_scale=1.0707758, maximum_absolute_delta=.00017854571, relative_l2=.000166978)]))
        self.write(self.diagnostic_path, self.diagnostic)

    def enroll(self):
        return self.numeric.base.enroll('2')

    def test_upgrade_preserves_all_results_and_both_previous_lineages(self):
        from .resume_compat import RMS_MARKER, verify_active_sources
        snapshots = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.enroll()
        self.assertTrue((self.root / RMS_MARKER).exists())
        self.assertTrue(all(p.read_bytes() == content for p, content in snapshots.items()))
        self.assertIsNotNone(verify_active_sources(self.root, self.active))
        self.enroll()
        with self.assertRaisesRegex(ValueError, 'RMS v3 source/policy'):
            verify_active_sources(self.root, self.previous)

    def test_upgrade_rejects_changes_to_training_or_bad_restore_proof(self):
        self.active['files']['v2xr_raygen_e0e3/training.py'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'unrelated source changed'):
            self.enroll()
        self.active['files']['v2xr_raygen_e0e3/training.py'] = self.previous['files']['v2xr_raygen_e0e3/training.py']
        proof = read(self.restored_path)
        proof['comparison']['status'] = 'failed'
        self.write(self.restored_path, proof)
        self.diagnostic['exact_restore_report_sha256'] = self.sha(self.restored_path)
        self.write(self.diagnostic_path, self.diagnostic)
        with self.assertRaisesRegex(ValueError, 'matching exact-load/data evidence'):
            self.enroll()

    def test_upgrade_rejects_corrupt_checkpoint_and_next_group(self):
        original = (self.folders[2] / 'resume.pth').read_bytes()
        (self.folders[2] / 'resume.pth').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'E1 smoke artifact changed'):
            self.enroll()
        (self.folders[2] / 'resume.pth').write_bytes(original)
        status = read(self.root / 'stage_status.json')
        status['smoke_E2_train_1'] = dict(status='running')
        self.write(self.root / 'stage_status.json', status)
        with self.assertRaisesRegex(ValueError, 'E1-only'):
            self.enroll()

    def test_controller_reuses_e0_e1_training_and_exact_restore(self):
        from .controller import Controller
        self.enroll()
        controller = Controller.__new__(Controller)
        controller.root = self.root
        controller.status = read(self.root / 'stage_status.json')
        started = []
        class NextGroup(Exception):
            pass
        def stage(name, action, *extra, **options):
            if controller.status.get(name, {}).get('status') == 'complete':
                return 0
            started.append(name)
            if name == 'smoke_E2_train_1':
                raise NextGroup()
            return 0
        controller.run = stage
        with patch.dict('os.environ', {'GPU': '2'}), self.assertRaises(NextGroup):
            controller.smoke()
        self.assertEqual(started, ['stable_calculation_checks', 'smoke_E1_resume_check', 'smoke_E2_train_1'])


class NativeAdapterASTTests(unittest.TestCase):
    def test_real_native_zero_GT_empty_CAV_ast_with_numpy_stubs(self):
        from .data import adapt_native
        native_file = HERE.parent / 'v2xr_official_repro/upstream/opencood/data_utils/datasets/opv2v/intermediate_fusion_dataset.py'
        tree = ast.parse(native_file.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                    ('__getitem__', 'get_item_single_car', 'merge_features_to_dict')]
        namespace = dict(np=np, math=math, random=__import__('random'), OrderedDict=OrderedDict,
            opencood=SimpleNamespace(data_utils=SimpleNamespace(datasets=SimpleNamespace(COM_RANGE=70))),
            points_in_boxes_cpu=lambda points, boxes: np.zeros((len(boxes), len(points)), dtype=int),
            box_utils=SimpleNamespace(project_points_by_matrix_torch=lambda points, matrix: points),
            shuffle_points=lambda points: points.copy(), mask_points_by_range=lambda points, limit: points,
            mask_ego_points=lambda points: points)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(native_file), 'exec'), namespace)
        d = namespace['IntermediateFusionDataset']()
        d.proj_first = True; d.use_Lidar = True; d.use_Radar = False; d.use_DeLidar = False
        d.fov = True; d.training = True; d.cur_ego_pose_flag = True; d.visualize = False; d.max_cav = 7
        d.params = dict(postprocess=dict(max_num=100), preprocess=dict(cav_lidar_range=[-140.8, -40, -3, 140.8, 40, 1],
                        args=dict(max_points_per_voxel=32)))
        d.data_augmentor = SimpleNamespace(forward=lambda x: x)
        d.pre_processor = SimpleNamespace(preprocess=lambda p: dict(voxel_features=np.empty((0, 32, 4)),
                        voxel_coords=np.empty((0, 3), dtype=int), voxel_num_points=np.empty(0, dtype=int)))
        d.post_processor = SimpleNamespace(
            generate_object_center=lambda *a: (np.zeros((100, 7)), np.zeros(100), []),
            generate_anchor_box=lambda: np.zeros((2, 3, 2, 7)),
            generate_label=lambda **k: (_ for _ in ()).throw(AssertionError('Native empty-GT label should be adapted')))
        base = OrderedDict({'1': dict(id='1', ego=True, time_delay=0,
            lidar_np=np.empty((0, 4)), radar_np=np.empty((0, 4)),
            params=dict(lidar_pose=[0.] * 6, transformation_matrix=np.eye(4), spatial_correction_matrix=np.eye(4), ego_speed=0))})
        calls = []
        def retrieve(idx, **kw):
            calls.append(idx)
            return copy.deepcopy(base), 0, '000000'
        d.retrieve_base_data = retrieve
        d.get_pairwise_transformation = lambda *a: np.tile(np.eye(4), (7, 7, 1, 1))
        d.mask_ego_fov_flag = lambda row, points, ego: points
        item = adapt_native(d)(0)
        self.assertEqual(calls, [0])
        self.assertEqual(item['ego']['object_bbx_mask'].sum(), 0)
        self.assertEqual(len(item['ego']['projected_lidar'][0]), 0)
        self.assertEqual(item['ego']['processed_lidar']['voxel_features'][0].shape, (1, 32, 4))
        self.assertEqual(item['ego']['label_dict']['pos_equal_one'].sum(), 0)
        self.assertTrue(item['ego']['label_dict']['neg_equal_one'].all())


def run():
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in (GeometryTests, SelectionTests, SmokeFixtureRegressionTests,
                                         RepairResumeTests, ResumeComparisonTests, NumericRepairTests,
                                         RMSPolicyTests, RMSRepairTests, NativeAdapterASTTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if 'torch' in sys.modules or 'opencood' in sys.modules:
        raise AssertionError('Local checks imported prohibited research runtime')
    if not result.wasSuccessful():
        raise SystemExit(1)
    return result.testsRun


if __name__ == '__main__':
    run()
