"""Pure NumPy/static checks locally; mandatory synthetic Torch/OpenCOOD checks on server."""
import argparse
import ast
import inspect
from pathlib import Path
import subprocess
import unittest
import numpy as np
from . import coordinates, features
from .common import KEEP, PACKAGE, object_hash, safe_path
from .extract import removal_plan
from .r0 import compare


def box(yaw=0.):
    # Same corner ordering as boxes_to_corners_3d, length along corner 3 -> 0.
    corners = np.array([[2, -1, -1], [2, 1, -1], [-2, 1, -1], [-2, -1, -1],
                        [2, -1, 1], [2, 1, 1], [-2, 1, 1], [-2, -1, 1]], float)
    c, s = np.cos(yaw), np.sin(yaw)
    return corners @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


class PureChecks(unittest.TestCase):
    def test_01_output_schema_exact(self):
        from local_fusion_action_utility_audit.features import schema
        for task in ('cls', 'reg'):
            self.assertEqual(features.schema(task, 'output_only'), schema(task))
            self.assertTrue(any(x.startswith('competition_') for x in features.schema(task, 'output_only')))

    def test_02_gt_fields_rejected(self):
        for name in ('gt_iou', 'target_score', 'oracle_rank', 'recovered', 'new_fp', 'matched'):
            with self.assertRaises(ValueError):
                features.check_names([name], 'reg')

    def test_03_pred_prediction_iou_allowed(self):
        features.check_names(['source_shared_pred_iou', 'loo_shared_before_after_pred_iou'], 'reg')

    def test_04_predicted_box_roi(self):
        self.assertEqual(coordinates.inside(np.array([[0, 1.5, 0], [1.5, 0, 0]]), box(np.pi/2), 1.).tolist(), [True, False])
        with self.assertRaises(TypeError):
            coordinates.inside(np.zeros((1, 3)), box(), 1., gt_box=box())

    def test_05_coordinate_round_trip(self):
        angle = .71
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        source = np.array([[1., 2., 3.], [-2., 1., .5]])
        translation = np.array([9., -4., 1.])
        aligned = source @ rotation.T+translation
        np.testing.assert_allclose((aligned-translation) @ rotation, source, atol=1e-12)
        local, size = coordinates.local_xyz(box(angle), box(angle))
        np.testing.assert_allclose(local, box(), atol=1e-12)
        np.testing.assert_allclose(size, [4, 2, 2])

    def test_06_variable_sources(self):
        for count in (1, 2, 4, 7):
            for removed in range(count):
                retained, query = removal_plan(count, removed)
                self.assertEqual(len(retained), count-1)
                self.assertEqual(query, None if removed == 0 else 0)

    def test_07_permutation_equivariance(self):
        rng = np.random.default_rng(7)
        levels = rng.normal(size=(4, 3, 5, 6))
        mask = np.ones((5, 6), bool)
        a = features.semantic(levels[1], levels.mean(0), levels[0], levels, mask, 1)
        permutation = [0, 3, 1, 2]
        reordered = levels[permutation]
        b = features.semantic(reordered[2], reordered.mean(0), reordered[0], reordered, mask, 2)
        np.testing.assert_allclose(list(a.values()), list(b.values()), atol=1e-12)

    def test_08_absolute_cav_id_forbidden(self):
        for task in ('cls', 'reg'):
            for group in features.GROUP_BLOCKS:
                names = features.schema(task, group)
                self.assertFalse(any('cav_index' in x or 'source_id' in x or 'underlying_source' in x for x in names))
        with self.assertRaises(ValueError):
            features.check_names(['source_id'], 'cls')

    def test_09_empty_roi_validity(self):
        output = features.geometric(np.empty((0, 3)), np.empty((0, 3)), box(), 1.5, 8)
        vector = features.vector('geometry', output)
        self.assertEqual(output['valid_point_count'], 0)
        self.assertEqual(vector[len(features.GEO)+features.GEO.index('variance_x')], 0)
        self.assertEqual(vector[len(features.GEO)+features.GEO.index('valid_point_count')], 1)
        self.assertEqual(features.semantic(np.zeros((2, 3, 3)), np.zeros((2, 3, 3)), np.zeros((2, 3, 3)), np.zeros((1, 2, 3, 3)), np.zeros((3, 3), bool), 0), {})

    def test_10_gspr_unknown_not_low_quality(self):
        empty = features.distribution([])
        self.assertEqual(empty, {'valid_count': 0})
        zero = features.distribution([0., 0.])
        self.assertEqual(zero['mean'], 0.)
        self.assertEqual(zero['valid_count'], 2)
        x = features.vector('gspr', {'point_reliability_valid_count': 0, 'observation_valid': 0})
        offset = len(features.GSPR)
        self.assertEqual(x[offset+features.GSPR.index('point_reliability_mean')], 0)
        self.assertEqual(x[offset+features.GSPR.index('observation_valid')], 1)

    def test_11_loo_missing_value_mask(self):
        x = features.vector('loo', {'shared_applicable': 0, 'query_self_applicable': 0})
        self.assertEqual(x[len(features.LOO)+features.LOO.index('shared_delta_logit')], 0)
        self.assertEqual(x[len(features.LOO)+features.LOO.index('shared_applicable')], 1)

    def test_12_source_removal_count(self):
        for n in (2, 5):
            retained, query = removal_plan(n, n-1, 1 if n > 2 else 0)
            self.assertEqual(len(retained), n-1)
            self.assertNotIn(n-1, retained)
            self.assertIsNotNone(query)

    def test_13_query_self_undefined(self):
        for i in range(4):
            self.assertIsNone(removal_plan(4, i, i)[1])

    def test_14_scene_split_contract(self):
        from local_fusion_action_utility_audit.common import scene_split
        fit, cal = scene_split([0, 1, 2, 3, 4], .2, 20261007)
        self.assertFalse(set(fit) & set(cal))
        self.assertEqual(set(fit) | set(cal), set(range(5)))
        text = (PACKAGE/'ranker.py').read_text(encoding='utf-8')
        self.assertIn("base_protocol['probe_fit_scenes']", text)
        self.assertIn("base_protocol['probe_calibration_scenes']", text)

    def test_15_labels_hash_and_readonly(self):
        from local_fusion_action_utility_audit.reproducibility import tensor_tree_hash
        labels = [{'outcomes': {'cls': [{'action_tp': 4, 'lost': 0, 'new_fp': 1}]}}]
        before = tensor_tree_hash(labels)
        features.vector('geometry', {'point_count': 4})
        self.assertEqual(before, tensor_tree_hash(labels))
        changed = [{'outcomes': {'cls': [{'action_tp': 5, 'lost': 0, 'new_fp': 1}]}}]
        self.assertNotEqual(before, tensor_tree_hash(changed))
        from . import common
        self.assertNotIn('atomic_', inspect.getsource(common.base_frame))

    def test_16_r0_stops_on_difference(self):
        compare({'pairwise': .5, 'top1': None}, {'pairwise': .5000005, 'top1': None})
        with self.assertRaises(ValueError):
            compare({'pairwise': .5}, {'pairwise': .500002})
        with self.assertRaises(ValueError):
            compare({'pairwise': None}, {'pairwise': 0})

    def test_17_replay_decision_gt_free(self):
        from .replay import predict_actions
        allowed = ('ids', 'scores', 'masks', 'features', 'names', 'cls_probe', 'reg_probe', 'conservative')
        self.assertEqual(tuple(inspect.signature(predict_actions).parameters), allowed)
        source = inspect.getsource(predict_actions)
        tree = ast.parse(source)
        self.assertFalse({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} & {'gt', 'labels', 'outcomes', 'association', 'target'})

    def test_18_posix_shell(self):
        for name in ('launch.sh', 'run_all.sh'):
            source = (PACKAGE/name).read_text(encoding='utf-8')
            self.assertTrue(source.startswith('#!/bin/sh\nset -eu\n'))
            for bad in ('[[', 'pipefail', 'BASH_SOURCE', 'source ', 'declare ', 'function '):
                self.assertNotIn(bad, source)

    def test_19_shell_lf(self):
        for name in ('launch.sh', 'run_all.sh'):
            self.assertNotIn(b'\r', (PACKAGE/name).read_bytes())

    def test_20_formal_test_guard(self):
        with self.assertRaises(ValueError):
            safe_path(Path('/data/dataset/test/scene/a.bin'))

    def test_21_fixed_groups_and_ablations(self):
        for task in ('cls', 'reg'):
            full = set(features.schema(task, 'all_evidence'))
            self.assertTrue(set(features.schema(task, 'all_evidence_without_loo')) < full)
            self.assertFalse(any(x.startswith('loo_') for x in features.schema(task, 'all_evidence_without_loo')))
            self.assertFalse(any(x.startswith('gspr_') for x in features.schema(task, 'all_evidence_without_gspr')))

    def test_22_geometry_semantic_loo_separated(self):
        geo = features.schema('reg', 'output_plus_geometry')
        sem = features.schema('cls', 'output_plus_semantic')
        self.assertNotIn('loo_shared_delta_logit', geo)
        self.assertNotIn('loo_shared_delta_x', sem)
        self.assertIn('loo_shared_delta_x', geo)
        self.assertIn('loo_shared_delta_logit', sem)

    def test_23_unknown_evidence_rejected(self):
        with self.assertRaises(ValueError):
            features.vector('geometry', {'gt_iou': .9})

    def test_24_complete_numpy_extraction_boundary(self):
        from unittest.mock import patch
        from .extract import source_blocks
        class ArrayTensor:
            """Only the read-only tensor interface; no Torch import or execution."""
            def __init__(self, value):
                self.value = np.asarray(value)
                self.shape = self.value.shape
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.value
            def __len__(self):
                return len(self.value)
        def prediction(value=0.):
            return {'psm': ArrayTensor(np.full((1, 2, 4, 4), value, np.float32)),
                    'rm': ArrayTensor(np.zeros((1, 14, 4, 4), np.float32))}
        names = [KEEP]+[f'single:{i}' for i in range(3)]+[f'query:{i}' for i in range(3)]
        pool = {name: prediction(i*.1) for i, name in enumerate(names)}
        proxy = {'scores': np.full(32, .5), 'corners': np.repeat(box()[None], 32, axis=0), 'anchor_num': 2}
        proxies = {name: proxy for name in names}
        processed = {'voxel_features': ArrayTensor([[[.1, .2, 0., 1.], [.2, .3, .1, 1.], [0., 0., 0., 0.]],
                                                   [[.1, .2, 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]]]),
                     'voxel_num_points': ArrayTensor([2, 1]), 'voxel_coords': ArrayTensor([[0, 0, 4, 4], [1, 0, 4, 4]]),
                     '_voxel_size': [1., 1., 2.]}
        reliability = {k: ArrayTensor(np.ones((2, 3), np.float32)*.7) for k in ('point_reliability', 'point_uncertainty')}
        reliability.update(point_valid_mask=ArrayTensor([[True, True, False], [True, False, False]]),
                           point_evidence=ArrayTensor(np.ones((2, 3, 2))),
                           pillar_reliability=ArrayTensor([.7, .7]), pillar_uncertainty=ArrayTensor([.2, .2]))
        levels = [ArrayTensor(np.arange(3*2*4*4, dtype=np.float32).reshape(3, 2, 4, 4)+1) for _ in range(2)]
        shared = [ArrayTensor(level.value.mean(0, keepdims=True)) for level in levels]
        removed = {0: None, 1: {'shared': prediction(-.1), 'attfuse': prediction(-.2)},
                   2: {'shared': prediction(-.1), 'attfuse': prediction(-.2)}}
        removed_proxies = {0: None, 1: {'shared': proxy, 'attfuse': proxy}, 2: {'shared': proxy, 'attfuse': proxy}}
        with patch('local_fusion_action_utility_audit.features.pred_iou', side_effect=lambda a, b: np.ones((len(a), len(b)))):
            blocks = source_blocks(box()[None], names, processed, reliability, levels, shared,
                                   np.repeat(np.eye(4)[None], 3, axis=0), [-4., -4., -1., 4., 4., 1.], 1.5,
                                   pool, proxies, removed, removed_proxies, prediction(), proxy,
                                   np.array([0]), [np.ones((4, 4), bool)])
        for block, value in blocks.items():
            self.assertEqual(value.shape, (1, 7, len(features.block_names(block))))
            self.assertTrue(np.isfinite(value).all())
        empty = names.index('single:2')
        self.assertEqual(blocks['geometry'][0, empty, features.GEO.index('point_count')], 0)
        self.assertEqual(blocks['gspr'][0, empty, len(features.GSPR)+features.GSPR.index('point_reliability_mean')], 0)
        for task in ('cls', 'reg'):
            old = np.zeros((1, 7, len(features.schema(task, 'output_only'))), np.float32)
            for group in features.GROUP_BLOCKS:
                self.assertEqual(features.combine(old, blocks, group).shape, (1, 7, len(features.schema(task, group))))

    def test_25_cases_do_not_overclaim(self):
        from .summarize import case_decision
        for expected, args in [(1, (True, False, 0, .3, False)), (2, (False, True, 0, .3, False)),
                               (3, (True, True, .3, .3, False)), (4, (True, True, 0, .3, True)),
                               (5, (False, False, 0, .3, False)), (6, (True, True, 0, .3, False)),
                               (0, (True, True, .1, .3, False))]:
            self.assertEqual(case_decision(*args)[0], expected)

    def test_26_schema_survives_resume_json(self):
        import json
        document = features.schema_document()
        self.assertEqual(json.loads(json.dumps(document)), document)

    def rank_fixture(self):
        from local_fusion_action_utility_audit.features import schema
        names = [KEEP]+[f'{family}:{i}' for family in ('single', 'query') for i in range(4)]
        columns = schema('cls')
        def build(scores):
            value = np.zeros((1, len(names), len(columns)), np.float32)
            scores = np.asarray(scores, np.float32)
            value[0, :, columns.index('roi_max')] = scores
            value[0, :, columns.index('source_relative_rank')] = (scores[None, 1:] > scores[:, None]).sum(1)/8
            return value
        scores = [.7, .1, .2, .3, .4, .5, .6, .6000005, .8]
        expected = build(scores)
        scores[6], scores[7] = scores[7], scores[6]
        return expected, build(scores), names, columns

    def test_27_near_tie_rank_step_allowed_without_mutation(self):
        from .reproduction import output_reproduction
        expected, actual, names, columns = self.rank_fixture()
        before = expected.copy()
        check = output_reproduction(actual, expected, 'cls', names)
        self.assertTrue(check['accepted'])
        self.assertFalse(check['numeric']['allclose_2e5'])
        self.assertEqual(check['numeric']['max_absolute_difference'], .125)
        self.assertEqual(check['rank_ties']['flips_outside_score_bounds'], 0)
        np.testing.assert_array_equal(expected, before)

    def test_28_validation_rank_still_exact(self):
        from .reproduction import output_reproduction
        expected, actual, names, _ = self.rank_fixture()
        self.assertFalse(output_reproduction(actual, expected, 'cls', names, True)['accepted'])
        self.assertTrue(output_reproduction(expected, expected, 'cls', names, True)['accepted'])

    def test_29_rank_exception_cannot_hide_other_feature_drift(self):
        from .reproduction import output_reproduction
        expected, actual, names, columns = self.rank_fixture()
        for field, difference in [('roi_mean', .01), ('competition_higher_score_overlap_count', 1.)]:
            broken = actual.copy()
            broken[0, 1, columns.index(field)] += difference
            check = output_reproduction(broken, expected, 'cls', names)
            self.assertFalse(check['accepted'])
            self.assertIn(field, check['reason'])
            self.assertIn('worst', check['columns'][field])

    def test_30_unexplained_rank_change_rejected(self):
        from .reproduction import output_reproduction
        expected, actual, names, columns = self.rank_fixture()
        actual[0, 1, columns.index('source_relative_rank')] -= .125
        check = output_reproduction(actual, expected, 'cls', names)
        self.assertFalse(check['accepted'])
        self.assertIn('reconstructed', check['reason'])

    def test_31_large_score_change_and_nonfinite_rejected(self):
        from .reproduction import output_reproduction
        expected, actual, names, columns = self.rank_fixture()
        actual[0, 6, columns.index('roi_max')] += .001
        self.assertFalse(output_reproduction(actual, expected, 'cls', names)['accepted'])
        actual[0, 6, 0] = np.nan
        self.assertFalse(output_reproduction(actual, expected, 'cls', names)['accepted'])
        self.assertFalse(output_reproduction(actual[:, :2], expected, 'cls', names)['accepted'])

    def test_32_regression_has_no_rank_exception(self):
        from .reproduction import output_reproduction
        from local_fusion_action_utility_audit.features import schema
        _, _, names, _ = self.rank_fixture()
        value = np.zeros((1, len(names), len(schema('reg'))), np.float32)
        actual = value.copy()
        actual[0, 1, schema('reg').index('source_shared_pred_iou')] = .125
        self.assertFalse(output_reproduction(actual, value, 'reg', names)['accepted'])

    def product_fixture(self):
        from .reproduction import PRODUCT
        from local_fusion_action_utility_audit.features import schema
        cls_e, cls_a, names, _ = self.rank_fixture()
        cls_e, cls_a = (np.repeat(x, 32, axis=0) for x in (cls_e, cls_a))
        reg_e = np.zeros((32, len(names), len(schema('reg'))), np.float32)
        reg_a = reg_e.copy()
        for value, distance in ((reg_e, 1e-6), (reg_a, 3e-6)):
            value[..., schema('reg').index('center_distance')] = distance
            value[..., schema('reg').index('delta_x')] = distance
            value[..., schema('reg').index('competition_iou_over_0p1_count')] = 30
            value[..., schema('reg').index(PRODUCT)] = distance*30
        for cls, reg in ((cls_e, reg_e), (cls_a, reg_a)):
            for name in (PRODUCT, 'competition_iou_over_0p1_count'):
                cls[..., schema('cls').index(name)] = reg[..., schema('reg').index(name)]
        return cls_e, cls_a, reg_e, reg_a, names

    def test_33_distance_product_uses_original_primitive_tolerance(self):
        from .reproduction import output_reproduction
        _, _, expected, actual, names = self.product_fixture()
        check = output_reproduction(actual, expected, 'reg', names)
        self.assertFalse(check['numeric']['allclose_2e5'])
        self.assertTrue(check['accepted'])
        self.assertTrue(check['geometry_product']['distance']['allclose_2e5'])
        self.assertEqual(check['geometry_product']['outside_propagated_bound'], 0)

    def test_34_product_and_rank_both_verified_without_input_mutation(self):
        from .reproduction import output_reproduction
        cls_e, cls_a, reg_e, reg_a, names = self.product_fixture()
        before = cls_e.copy()
        check = output_reproduction(cls_a, cls_e, 'cls', names,
                                    regression_actual=reg_a, regression_expected=reg_e)
        self.assertTrue(check['accepted'])
        self.assertTrue(check['geometry_product']['accepted'])
        self.assertGreater(check['rank_ties']['flipped_comparisons'], 0)
        np.testing.assert_array_equal(cls_e, before)

    def test_35_fabricated_product_rejected_even_inside_propagated_bound(self):
        from .reproduction import PRODUCT, output_reproduction
        from local_fusion_action_utility_audit.features import schema
        _, _, expected, actual, names = self.product_fixture()
        actual[0, 1, schema('reg').index(PRODUCT)] += .0004
        check = output_reproduction(actual, expected, 'reg', names)
        self.assertFalse(check['accepted'])
        self.assertIn('reconstructed', check['reason'])

    def test_36_changed_overlap_count_rejected(self):
        from .reproduction import output_reproduction
        from local_fusion_action_utility_audit.features import schema
        _, _, expected, actual, names = self.product_fixture()
        actual[0, :, schema('reg').index('competition_iou_over_0p1_count')] = 29
        self.assertFalse(output_reproduction(actual, expected, 'reg', names)['accepted'])

    def test_37_geometry_primitive_drift_not_hidden_by_product(self):
        from .reproduction import output_reproduction
        from local_fusion_action_utility_audit.features import schema
        _, _, expected, actual, names = self.product_fixture()
        for name in ('center_distance', 'delta_x', 'delta_y', 'delta_z'):
            broken = actual.copy()
            broken[0, 1, schema('reg').index(name)] += .001
            self.assertFalse(output_reproduction(broken, expected, 'reg', names)['accepted'])

    def test_38_cls_product_requires_aligned_regression_primitives(self):
        from .reproduction import PRODUCT, output_reproduction
        from local_fusion_action_utility_audit.features import schema
        cls_e, cls_a, reg_e, reg_a, names = self.product_fixture()
        self.assertFalse(output_reproduction(cls_a, cls_e, 'cls', names)['accepted'])
        cls_a[0, 1, schema('cls').index(PRODUCT)] += .0001
        check = output_reproduction(cls_a, cls_e, 'cls', names,
                                    regression_actual=reg_a, regression_expected=reg_e)
        self.assertFalse(check['accepted'])
        self.assertIn('align', check['reason'])

    def test_39_zero_overlap_has_zero_product(self):
        from .reproduction import PRODUCT, output_reproduction
        from local_fusion_action_utility_audit.features import schema
        _, _, expected, actual, names = self.product_fixture()
        for value in (expected, actual):
            value[..., schema('reg').index('competition_iou_over_0p1_count')] = 0
            value[..., schema('reg').index(PRODUCT)] = 0
        actual[0, 1, schema('reg').index(PRODUCT)] = .00005
        self.assertFalse(output_reproduction(actual, expected, 'reg', names)['accepted'])

    def test_40_validation_product_remains_exact(self):
        from .reproduction import output_reproduction
        _, _, expected, actual, names = self.product_fixture()
        self.assertFalse(output_reproduction(actual, expected, 'reg', names, True)['accepted'])

    def test_41_diagnostics_show_worst_actual_failure(self):
        from .reproduction import output_reproduction
        expected, actual, names, columns = self.rank_fixture()
        j = columns.index('roi_mean')
        expected[0, 1, j], actual[0, 1, j] = 100, 100.001
        actual[0, 2, j] = .00005
        check = output_reproduction(actual, expected, 'cls', names)
        self.assertFalse(check['accepted'])
        self.assertEqual(check['columns']['roi_mean']['worst']['source'], names[1])
        self.assertEqual(check['columns']['roi_mean']['worst_outside_tolerance']['source'], names[2])


def server_checks():
    # Missing server dependencies are a hard failure. No skip/try-except fallback.
    import torch
    from torch import nn
    from local_fusion_utility_v2.fusion import attention_fusion
    from local_fusion_task_source_oracle.oracle import build_candidate_pool, _proxy_cache
    from attfuse_gspr.reliability import GeometryFirstPillarReliability
    from .extract import leave_one_out, sensitivity, source_blocks
    from .replay import predict_actions
    from local_fusion_action_utility_audit.s3_replay import conflict_resolver, execute
    from types import SimpleNamespace
    torch.set_num_threads(1)
    torch.manual_seed(17)
    for n in (1, 2, 4, 7):
        level = torch.randn(n, 4, 4, 4)
        fused, weights = attention_fusion(level)
        assert fused.shape == (1, 4, 4, 4)
        assert torch.allclose(weights.sum(0), torch.ones_like(weights[0]))
        class Arm:
            def predict(self, base, levels):
                from local_fusion_utility_v2.fusion import predict_from_levels
                return predict_from_levels(base, [attention_fusion(x)[0] for x in levels]), {}
        base = SimpleNamespace(backbone=SimpleNamespace(deblocks=[nn.Identity()]),
                               cls_head=nn.Conv2d(4, 2, 1), reg_head=nn.Conv2d(4, 14, 1))
        runtime = SimpleNamespace(model=SimpleNamespace(engine=SimpleNamespace(base=base)), shared_arm=Arm())
        full, removed = leave_one_out(runtime, [level])
        assert removed[0] is None
        assert len(removed) == n
        pool = build_candidate_pool(base, [level], full, ['single', 'query'])
        assert len(pool) == 1+2*n
        for i in range(1, n):
            assert removed[i]['shared']['psm'].shape == full['psm'].shape
            assert not torch.equal(removed[i]['shared']['psm'], full['psm'])
    # Real frozen-module interface on a tiny synthetic grid, no data/checkpoint.
    reliability = GeometryFirstPillarReliability({'grid_size': [8, 8, 1], 'hidden_dim': 8, 'context_dim': 4},
                                                [1., 1., 2.], [-4., -4., -1., 4., 4., 1.])
    reliability.eval()
    voxels = torch.tensor([[[.1, .2, 0., .5], [.2, .3, .1, .4], [0., 0., 0., 0.]]])
    with torch.no_grad():
        rel = reliability(voxels, torch.tensor([2]), torch.tensor([[0, 0, 4, 4]]))
    assert rel['point_valid_mask'].tolist() == [[True, True, False]]
    assert rel['point_reliability'][0, 2] == 0
    assert torch.isfinite(rel['point_evidence']).all()
    # Actual OpenCOOD decoder and pred-pred IoU must be available.
    from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
    from opencood.utils import box_utils
    from local_fusion_action_utility_audit.features import pred_iou
    np.testing.assert_allclose(pred_iou(box()[None], box()[None]), [[1.]])
    # Exercise the full S4 extractor/decoder boundary, including a trailing CAV
    # with no retained voxel. These tiny synthetic checks run only on server.
    from local_fusion_action_utility_audit.features import source_names, schema as old_schema
    levels = [torch.randn(3, 2, 4, 4) for _ in range(2)]
    base = SimpleNamespace(backbone=SimpleNamespace(deblocks=[nn.Identity(), nn.Identity()]),
                           cls_head=nn.Conv2d(4, 2, 1), reg_head=nn.Conv2d(4, 14, 1))
    nn.init.zeros_(base.reg_head.weight)
    nn.init.zeros_(base.reg_head.bias)
    runtime = SimpleNamespace(model=SimpleNamespace(engine=SimpleNamespace(base=base, spatial_shape=(8, 8))),
                              shared_arm=Arm(), lidar_range=[-4., -4., -1., 4., 4., 1.],
                              hypes={'fusion': {'args': {'proj_first': True}}, 'preprocess': {'args': {'voxel_size': [1., 1., 2.]}}})
    full, _ = runtime.shared_arm.predict(base, levels)
    pool = build_candidate_pool(base, levels, full, ['single', 'query'])
    anchors = torch.zeros(4, 4, 2, 7)
    for y in range(4):
        for x in range(4):
            anchors[y, x, :, :3] = torch.tensor([-3+2*x, -3+2*y, 0.])
    anchors[..., 3:6] = torch.tensor([2., 2., 4.])
    batch = {'ego': {'anchor_box': anchors, 'transformation_matrix': torch.eye(4),
                     'record_len': torch.tensor([3]), 'communication_transforms': torch.eye(4)[None].repeat(3, 1, 1)}}
    post = SimpleNamespace(params={'order': 'hwl'}, delta_to_boxes3d=VoxelPostprocessor.delta_to_boxes3d)
    dataset = SimpleNamespace(post_processor=post)
    proxies = {name: _proxy_cache(dataset, batch, value) for name, value in pool.items()}
    att_full, removed = leave_one_out(runtime, levels)
    removed_proxies = {i: None if values is None else {ref: _proxy_cache(dataset, batch, value) for ref, value in values.items()} for i, values in removed.items()}
    processed = {'voxel_features': voxels.repeat(2, 1, 1), 'voxel_num_points': torch.tensor([2, 1]),
                 'voxel_coords': torch.tensor([[0, 0, 4, 4], [1, 0, 4, 4]])}
    with torch.no_grad():
        rel = reliability(processed['voxel_features'], processed['voxel_num_points'], processed['voxel_coords'])
    poses = coordinates.validate(runtime, batch, processed, levels)
    processed['_voxel_size'] = [1., 1., 2.]
    ids = np.array([20])
    corners = proxies[KEEP]['corners'][ids]
    mask = np.zeros((4, 4), bool)
    mask[2, 2] = True
    names = source_names(pool)
    blocks = source_blocks(corners, names, processed, rel, levels, [x.mean(0, keepdim=True) for x in levels],
                           poses, runtime.lidar_range, 1.5, pool, proxies, removed, removed_proxies,
                           att_full, _proxy_cache(dataset, batch, att_full), ids, [mask])
    for block, value in blocks.items():
        assert value.shape == (1, 7, len(features.block_names(block)))
        assert np.isfinite(value).all()
    empty = names.index('single:2')
    mean = features.GSPR.index('point_reliability_mean')
    assert blocks['gspr'][0, empty, len(features.GSPR)+mean] == 0
    query = names.index('query:1')
    applicability = features.LOO.index('query_self_applicable')
    assert blocks['loo'][0, query, applicability] == 0
    assert blocks['loo'][0, query, len(features.LOO)+applicability] == 1
    for task in ('cls', 'reg'):
        old = np.zeros((1, 7, len(old_schema(task))), np.float32)
        for group in features.GROUP_BLOCKS:
            combined = features.combine(old, blocks, group)
            assert combined.shape[-1] == len(features.schema(task, group))
    class DummyProbe:
        state = {'calibration': {'threshold': None}}
        def scores(self, x):
            return np.arange(len(x), dtype=float)
    matrices = {t: torch.zeros(1, 7, 1) for t in ('cls', 'reg')}
    actions = predict_actions(ids, [1.], [mask], matrices, names, DummyProbe(), DummyProbe())
    accepted, counts = conflict_resolver(actions)
    assert counts['accepted_actions'] == 1
    executed = execute(full, pool, accepted)
    assert torch.equal(executed['psm'][..., ~torch.from_numpy(mask)], full['psm'][..., ~torch.from_numpy(mask)])
    conservative = predict_actions(ids, [1.], [mask], matrices, names, DummyProbe(), DummyProbe(), True)
    assert conservative[0]['cls_source'] == KEEP and conservative[0]['reg_source'] == KEEP
    for name in ('launch.sh', 'run_all.sh'):
        subprocess.run(['sh', '-n', str(PACKAGE/name)], check=True)
    print('mandatory Torch/GSPR/AttFuse/OpenCOOD/shell synthetic checks complete', flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--require-server', action='store_true')
    args = cli.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PureChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.require_server:
        server_checks()
    else:
        print('Pure/static checks only; mandatory Torch/OpenCOOD checks run in the server driver.')


if __name__ == '__main__':
    main()
