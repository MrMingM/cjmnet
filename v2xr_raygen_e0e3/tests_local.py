"""Pure NumPy/stdlib synthetic checks only; no real data, Torch or OpenCOOD imports."""
import copy
import ast
from collections import OrderedDict
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

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
                               for c in (GeometryTests, SelectionTests, NativeAdapterASTTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if 'torch' in sys.modules or 'opencood' in sys.modules:
        raise AssertionError('Local checks imported prohibited research runtime')
    if not result.wasSuccessful():
        raise SystemExit(1)
    return result.testsRun


if __name__ == '__main__':
    run()
