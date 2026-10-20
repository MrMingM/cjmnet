"""Pure Python/NumPy synthetic checks. No Torch, native runtime, GPU, PCD or experiments."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import sys
import zipfile

import numpy as np

from .common import HERE, ROOT, commit, identity, read, sha, write, verify_dependencies
from .uncertainty_targets import density, pairwise, split_scenes, pool_numpy
from .diagnostic_statistics import contrasts, scene_interval, shuffled_features
from .state_logic import choose, valid_stage, verify_resume
from .report import difference, number, run as report
from .upstream import Imports, PREFIX
from .point_origin import resolve_neighbors, aggregate_labels
from .dependency_scope import direct_files, validate_scope
from .install_update import apply_update, digest


def update_fixture(root):
    project = root / 'project'; project.mkdir()
    output = root / 'output'; output.mkdir()
    (output / 'launcher.log').write_bytes(b'original failure log')
    package = project / 'uecp_gspr_v2xr'; package.mkdir()
    history = project / 'attfuse_gspr'; history.mkdir()
    (history / 'README.md').write_bytes(b'server-specific documentation')
    (history / 'reliability.py').write_bytes(b'pinned runtime code')
    pins = json.dumps(dict(files={'attfuse_gspr/reliability.py': sha(history / 'reliability.py')})).encode()
    original = {'common.py': b'original helper\n', 'PINS.json': pins}
    for name, data in original.items():
        (package / name).write_bytes(data)
    old_snapshot = json.dumps(dict(files={n: digest(v) for n, v in original.items()})).encode()
    (package / 'source_snapshot.json').write_bytes(old_snapshot)
    updated = dict(original, **{'common.py': b'updated helper\n'})
    snapshot = json.dumps(dict(files={n: digest(v) for n, v in updated.items()})).encode()
    base = {'uecp_gspr_v2xr/'+n: digest(v) for n, v in original.items()}
    base['uecp_gspr_v2xr/source_snapshot.json'] = digest(old_snapshot)
    metadata = dict(kind='uecp_gspr_pre_request_source_scope_fix', base_files=base,
        target_snapshot_sha256=digest(snapshot))
    archive = root / 'fix.zip'
    with zipfile.ZipFile(archive, 'x') as bundle:
        for name, data in updated.items():
            bundle.writestr('uecp_gspr_v2xr/'+name, data)
        bundle.writestr('uecp_gspr_v2xr/source_snapshot.json', snapshot)
        bundle.writestr('UPDATE.json', json.dumps(metadata))
        bundle.writestr('__main__.py', '# synthetic installer fixture\n')
    return archive, project, output


class Checks(unittest.TestCase):
    def test_direct_runtime_dependency_scope(self):
        required = direct_files(ROOT)
        self.assertEqual(len(required), 14)
        self.assertIn('attfuse_gspr/reliability.py', required)
        self.assertNotIn('attfuse_gspr/README.md', required)
        self.assertNotIn('attfuse_gspr/loss.py', required)
        self.assertNotIn('related_code/UECP-main/README.md', required)
        validate_scope(ROOT, read(HERE / 'PINS.json')['files'])

    def test_document_changes_ignored_code_mismatches_aggregated(self):
        with tempfile.TemporaryDirectory(prefix='uecp_source_check_', dir=ROOT) as folder:
            root = Path(folder)
            for name in ('README.md', 'first.py', 'second.py'):
                (root / name).write_bytes(b'original')
            pins = {name: sha(root / name) for name in ('first.py', 'second.py')}
            (root / 'README.md').write_bytes(b'different server documentation')
            verify_dependencies(pins, root)
            for name in pins:
                (root / name).write_bytes(b'changed code')
            with self.assertRaises(ValueError) as failure:
                verify_dependencies(pins, root)
            self.assertIn('2 files', str(failure.exception))
            self.assertIn('first.py', str(failure.exception)); self.assertIn('second.py', str(failure.exception))

    def test_source_fix_preserves_history_logs_and_is_idempotent(self):
        with tempfile.TemporaryDirectory(prefix='uecp_update_check_', dir=ROOT) as folder:
            archive, project, output = update_fixture(Path(folder))
            result = apply_update(archive, project, output)
            self.assertEqual(result['status'], 'updated')
            self.assertTrue(Path(result['backup']).is_file())
            self.assertEqual((project / 'attfuse_gspr/README.md').read_bytes(), b'server-specific documentation')
            self.assertEqual((output / 'launcher.log').read_bytes(), b'original failure log')
            self.assertFalse((output / 'request.json').exists())
            self.assertEqual(apply_update(archive, project, output)['status'], 'already_applied')

    def test_source_fix_refuses_started_run_and_unrecognized_edits(self):
        for started in (True, False):
            with self.subTest(started=started), tempfile.TemporaryDirectory(prefix='uecp_update_check_', dir=ROOT) as folder:
                archive, project, output = update_fixture(Path(folder))
                path = project / 'uecp_gspr_v2xr/common.py'
                if started:
                    (output / 'request.json').write_bytes(b'{}')
                else:
                    path.write_bytes(b'user edits')
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    apply_update(archive, project, output)
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(list(project.glob('uecp_gspr_before_source_fix_*.zip')))

    def test_density_empty_and_current_weather(self):
        bounds = [0, 0, 0, 4, 4, 4]
        c, u = density(np.empty((0, 4)), bounds, 1)
        self.assertEqual(c.shape, (4, 4)); self.assertTrue((u == 1).all())
        p = np.array([[1.1, 2.1, 1, .5]]*5)
        c, u = density(p, bounds, 1)
        self.assertEqual(c[2, 1], 5); self.assertEqual(u[2, 1], .5)
        _, weather = density(p[:1], bounds, 1)
        self.assertNotEqual(weather[2, 1], u[2, 1])

    def test_density_half_open_and_finite(self):
        p = np.array([[0, 0, 0, 1], [4, 1, 1, 1], [1, 4, 1, 1], [1, 1, 4, 1], [np.nan, 1, 1, 1]])
        c, u = density(p, [0, 0, 0, 4, 4, 4], 1)
        self.assertEqual(c.sum(), 1); self.assertEqual(c[0, 0], 1)

    def test_density_clip(self):
        _, u = density(np.array([[1, 2, 1, 1]]*40), [0, 0, 0, 4, 4, 4], 1)
        self.assertEqual(u[2, 1], 0)

    def test_multicar_transforms_and_cycle(self):
        a, b, c = np.eye(4), np.eye(4), np.eye(4)
        b[:2, 3] = [4, 2]; c[:2, :2] = [[0, -1], [1, 0]]
        matrices = pairwise([a, b, c])
        self.assertEqual(matrices.shape, (7, 7, 4, 4))
        point = np.array([1, 2, 0, 1])
        np.testing.assert_allclose(b@matrices[0, 1]@point, a@point)
        np.testing.assert_allclose(matrices[1, 2]@matrices[0, 1], matrices[0, 2])
        np.testing.assert_allclose(matrices[0, 1][:2, 3], [-4, -2])

    def test_scene_split_keeps_all_weather_uid(self):
        m = dict(scenes=[dict(scene=str(i), timestamps=['0', '1'], ego='a') for i in range(32)])
        s = split_scenes(m, 20261017, 6)
        self.assertEqual(len(s['train_dev_scenes']), 6)
        self.assertFalse(set(s['train_core_scenes']) & set(s['train_dev_scenes']))
        self.assertEqual(sorted(s['train_core']+s['train_dev']), list(range(64)))
        self.assertEqual(s, split_scenes(m, 20261017, 6))

    def test_downsample_xy_orientation(self):
        x = np.arange(32).reshape(4, 8)
        np.testing.assert_equal(pool_numpy(x, 2, 'sum'), [[18, 26, 34, 42], [82, 90, 98, 106]])
        np.testing.assert_allclose(pool_numpy(x, 2), pool_numpy(x, 2, 'sum')/4)

    def test_masked_normalization_reference(self):
        # NumPy reference for the mask invariant; actual Torch implementation is server-only.
        precision = np.array([[0, .2, .7], [0, .8, .9]])
        mask = np.array([[False, True, False], [False, True, True]])
        value = precision*mask
        value /= np.maximum(value.sum(0), 1e-12)
        # Simulate smoothing leaking mass, then the mandated re-mask and renormalization.
        value = value+.1
        value *= mask
        value /= np.maximum(value.sum(0), 1e-12)
        self.assertEqual(value[~mask].sum(), 0)
        np.testing.assert_allclose(value.sum(0), [0, 1, 1])

    def test_point_index_padding_and_ambiguous_labels(self):
        labels = np.array([1, 0, -1])
        distance = np.array([[0, np.inf], [0, .0001], [np.inf, np.inf], [0, np.inf]])
        index = np.array([[0, 3], [0, 1], [3, 3], [1, 3]])
        values = resolve_neighbors(labels, distance, index)
        np.testing.assert_equal(values, [1, -1, -1, 0])
        valid = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)
        coords = np.array([[0, 0, 2, 4], [0, 0, 3, 5]])
        count, reliable = aggregate_labels(values, coords, valid, height=4, width=8)
        self.assertEqual(count[1, 2], 2); self.assertEqual(reliable[1, 2], 1)
        self.assertEqual(count.sum(), 2)

    def test_fov_index_retains_source_coordinates(self):
        raw = np.array([[1, 0, 0, .3], [2, 0, 0, .5], [3, 0, 0, .7]])
        projected = raw.copy(); projected[:, 0] += 10
        indexed = np.c_[projected, np.arange(len(raw))]
        visible = indexed[indexed[:, 0] >= 12]
        np.testing.assert_equal(raw[visible[:, 4].astype(int), 0], [2, 3])

    def test_quality_shuffle_density_strata(self):
        q = np.arange(40).reshape(20, 2); counts = np.array([1]*10+[8]*10); w = np.zeros(20)
        out = shuffled_features(q, counts, w, 8)
        np.testing.assert_equal(np.sort(out[:10], 0), np.sort(q[:10], 0))
        self.assertFalse(np.array_equal(out, q))
        np.testing.assert_equal(out, shuffled_features(q, counts, w, 8))

    def test_no_labels_inconclusive(self):
        empty = dict(x=np.empty((0, 6)), y=np.empty(0), scene=np.empty(0), weather=np.empty(0))
        cfg = read(HERE / 'settings.json')['diagnosis']
        result = contrasts(empty, empty, cfg, 1)
        self.assertEqual(result['conclusion'], 'INCONCLUSIVE')
        self.assertIsNone(result['A'])

    def test_predefined_quality_extra_information(self):
        rng = np.random.RandomState(21)
        def sample(n):
            r = rng.uniform(.1, .9, n)
            return dict(x=np.c_[np.ones(n)*np.log(9), np.ones(n)*.5, r, 1-r, 1-r, r*10],
                y=r, scene=np.repeat(np.arange(n//50), 50), weather=np.arange(n)%3)
        fit, diag = sample(500), sample(500)
        cfg = read(HERE / 'settings.json')['diagnosis']; cfg['bootstrap_repetitions'] = 100
        result = contrasts(fit, diag, cfg, 21)
        self.assertEqual(result['conclusion'], 'SUPPORTED')
        self.assertGreater(result['A']['brier'], result['C']['brier'])

    def test_scene_bootstrap_not_point_bootstrap(self):
        ci = scene_interval(np.r_[np.zeros(100), np.ones(2)], np.r_[np.zeros(100), np.ones(2)], 1000, 1)
        self.assertEqual(ci['mean'], .5)
        self.assertIsNone(scene_interval(np.ones(5), np.zeros(5), 100, 1))

    def test_selection_unrounded_and_earliest(self):
        rows = [dict(epoch=e, split='train_dev', condition='clean', protocol='O', AP=dict(ap70=v))
                for e, v in ((1, .20000001), (2, .20000002), (3, .20000002))]
        self.assertEqual(choose(rows)['epoch'], 2)
        rows[0]['split'] = 'validation'
        with self.assertRaises(ValueError):
            choose(rows)

    def test_atomic_commit_no_directory_skip(self):
        with tempfile.TemporaryDirectory(prefix='uecp_local_check_', dir=ROOT) as folder:
            p = Path(folder); data = p / 'weights'; data.write_bytes(b'abc')
            self.assertFalse(valid_stage(dict(status='RUNNING', artifacts={})))
            self.assertFalse(valid_stage(dict(status='COMPLETE', artifacts={})))
            commit(p / 'train_complete.json', [data], epoch=1)
            row = dict(status='COMPLETE', artifacts={str(p / 'train_complete.json'): sha(p / 'train_complete.json')})
            self.assertTrue(valid_stage(row))
            data.write_bytes(b'damaged')
            with self.assertRaises(ValueError):
                valid_stage(row)

    def test_resume_identity_and_boundary(self):
        order = [1, 2]*8
        state = dict(contract_identity='a', variant='E2', epoch=3, microbatch=1,
                     order_identity=identity(order), accumulation_position=0, position=8)
        verify_resume(state, 'a', 'E2', 3, 1, order)
        for key, bad in [('contract_identity', 'b'), ('position', 9), ('accumulation_position', 1), ('microbatch', 2)]:
            s = dict(state); s[key] = bad
            with self.assertRaises(ValueError):
                verify_resume(s, 'a', 'E2', 3, 1, order)

    def test_private_uecp_import_mapping(self):
        tree = Imports().visit(ast.parse('from opencood.utils.common_utils import limit_period\nimport opencood.utils.common_utils as cu'))
        result = ast.unparse(ast.fix_missing_locations(tree))
        self.assertIn(PREFIX+'.utils.common_utils', result)
        self.assertNotIn('from opencood', result)

    def test_report_delta_and_missing(self):
        self.assertAlmostEqual(difference(.23, .20), 3.)
        self.assertIsNone(difference(None, .2)); self.assertEqual(number(None), 'N/A')
        with tempfile.TemporaryDirectory(prefix='uecp_local_check_', dir=ROOT) as folder:
            root = Path(folder); write(root / 'config.json', read(HERE / 'settings.json'))
            write(root / 'protocol.json', dict(test_only=True))
            result = report(root, partial=True)
            self.assertEqual(result['status'], 'FAILED')
            self.assertGreaterEqual(len(result['failures']), 10)
            self.assertTrue(all(r['ap70'] is None for r in result['rows']))
            self.assertIn('N/A', (root / 'PARTIAL_RESULTS.md').read_text(encoding='utf-8'))


def run():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if 'torch' in sys.modules or any(n == 'opencood' or n.startswith('opencood.') for n in sys.modules):
        raise AssertionError('Local checks imported prohibited research runtime')


if __name__ == '__main__':
    run()
