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


class ArrayTensor:
    """Only the CPU array boundary; never imports Torch or executes a model."""
    def __init__(self, value):
        self.value = np.asarray(value)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.value


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

    def proposal_fixture(self):
        scores = np.asarray([.6, .2, .60000006, .5, .1], np.float32)
        corners = np.stack([box()+np.array([i*10., 0., 0.]) for i in range(5)]).astype(np.float32)
        ids = [2, 0, 3]
        reference = {'anchor_ids': ids, 'scores': scores[ids].tolist(),
                     'corners': corners[ids].copy(), 'ranks': [1, 2, 3]}
        names = [KEEP, 'single:0', 'query:0']
        return reference, scores, corners, names

    def test_42_reference_anchor_order_survives_repeated_score_ties(self):
        from .reproduction import proposal_reproduction
        reference, scores, corners, names = self.proposal_fixture()
        before = list(reference['anchor_ids'])
        scores[0], scores[2] = scores[2], scores[0]
        check = proposal_reproduction(reference, names, names, scores, corners, 3)
        self.assertTrue(check['accepted'])
        self.assertEqual(check['reordered_positions_in_repeated_scores'], 2)
        self.assertEqual(check['reference_anchor_ids'], before)
        self.assertEqual(reference['anchor_ids'], before)

    def test_43_reference_pool_is_not_reselected_at_topk_boundary(self):
        from .reproduction import proposal_reproduction
        reference, scores, corners, names = self.proposal_fixture()
        scores[4] = .50000006
        check = proposal_reproduction(reference, names, names, scores, corners, 3)
        self.assertTrue(check['accepted'])
        self.assertEqual(check['reference_anchor_ids'], [2, 0, 3])
        self.assertNotIn(4, check['reference_anchor_ids'])

    def test_44_source_pool_drift_remains_fatal(self):
        from .reproduction import proposal_reproduction
        reference, scores, corners, names = self.proposal_fixture()
        for changed in (names+['single:1'], names[:2], [names[0], names[2], names[1]]):
            self.assertFalse(proposal_reproduction(reference, names, changed, scores, corners, 3)['accepted'])

    def test_45_invalid_reference_proposals_rejected(self):
        from .reproduction import proposal_reproduction
        reference, scores, corners, names = self.proposal_fixture()
        for field, value in [('anchor_ids', [2, 0, 2]), ('anchor_ids', [2, 0, 9]),
                             ('anchor_ids', [2., 0., 3.]), ('ranks', [1, 3, 2]),
                             ('scores', [.1, .9, .5])]:
            broken = dict(reference, **{field: value})
            self.assertFalse(proposal_reproduction(broken, names, names, scores, corners, 3)['accepted'])

    def test_46_reference_score_or_geometry_drift_rejected(self):
        from .reproduction import proposal_reproduction
        reference, scores, corners, names = self.proposal_fixture()
        broken = scores.copy()
        broken[2] += .001
        self.assertFalse(proposal_reproduction(reference, names, names, broken, corners, 3)['accepted'])
        broken = corners.copy()
        broken[2, 0, 0] += .1
        self.assertFalse(proposal_reproduction(reference, names, names, scores, broken, 3)['accepted'])

    def test_47_validation_proposal_snapshot_remains_exact(self):
        from .reproduction import proposal_reproduction
        reference, scores, corners, names = self.proposal_fixture()
        self.assertTrue(proposal_reproduction(reference, names, names, scores, corners, 3, True)['accepted'])
        scores[0] += 1e-7
        self.assertFalse(proposal_reproduction(reference, names, names, scores, corners, 3, True)['accepted'])

    def test_48_extraction_uses_original_roi_and_ids_without_reselection(self):
        from .extract import fixed_proposals
        from unittest.mock import patch
        from types import SimpleNamespace
        reference, scores, corners, names = self.proposal_fixture()
        class ArrayTensor:
            def __init__(self, value):
                self.value = value
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.value
        cached = {'proposals': dict(reference, corners=ArrayTensor(reference['corners'])),
                  'source_names': names, 'labels': {'gt': 'must never enter feature construction'}}
        from local_fusion_action_utility_audit.features import schema
        expected = np.zeros((3, len(names), len(schema('cls'))), np.float32)
        expected[..., schema('cls').index('anchor_probability')] = np.asarray(reference['scores'], np.float32)[:, None]
        cached['features'] = {'cls': ArrayTensor(expected)}
        scores[0], scores[2] = scores[2], scores[0]
        corners = corners.copy()
        corners[2, :, 0] += 1e-6
        proxies = {name: {'scores': scores, 'corners': corners} for name in names}
        runtime = SimpleNamespace(spec={'shared_top_k': 3, 'roi_expansion': 1.5}, lidar_range=[0]*6)
        with patch('qa_local_intervention.operators.masks_for_frame', return_value=(np.ones((2, 2), bool), [], {})) as masks, \
             patch('local_fusion_action_utility_audit.features.extract_features', return_value=(names, {})) as extract:
            result, check = fixed_proposals({'ego': {'anchor_box': 'anchors'}}, proxies, {'levels': []}, runtime, cached)
        self.assertTrue(check['accepted'])
        self.assertEqual(result[0].tolist(), reference['anchor_ids'])
        for call, expected_roi in zip(masks.call_args_list, reference['corners']):
            np.testing.assert_array_equal(call.args[1], expected_roi)
        self.assertEqual(extract.call_args.args[1].tolist(), reference['anchor_ids'])
        np.testing.assert_array_equal(extract.call_args.args[2], np.asarray(reference['scores'], np.float64))
        self.assertEqual(extract.call_args.args[2].dtype, np.float64)
        np.testing.assert_array_equal(extract.call_args.args[3], reference['corners'])
        self.assertEqual(cached['labels'], {'gt': 'must never enter feature construction'})

    def full_output_fixture(self):
        from local_fusion_action_utility_audit.features import extract_features
        names = [KEEP, 'single:0', 'query:0']
        scores = np.array([.60000006, .6, .5], np.float32)
        corners = np.stack([box()+[i*.2, 0, 0] for i in range(3)]).astype(np.float32)
        proxies = {name: {'scores': scores.copy()+i*.03, 'corners': corners.copy(), 'anchor_num': 1}
                   for i, name in enumerate(names)}
        spec = {'shared_top_k': 3, 'roi_expansion': 1.5, 'local_top_k': 2, 'nearby_distance_m': 10.}
        ids, masks = np.arange(3), [np.ones((1, 3), bool) for _ in range(3)]
        _, old = extract_features(proxies, ids, scores.astype(np.float64), corners, masks, spec)
        reference = {'anchor_ids': ids.tolist(), 'scores': scores.tolist(),
                     'corners': ArrayTensor(corners.copy()), 'ranks': [1, 2, 3]}
        cached = {'proposals': reference, 'source_names': names,
                  'features': {k: ArrayTensor(v) for k, v in old.items()}, 'labels': {'do_not_read': 7}}
        return cached, proxies, spec, masks

    def test_49_full_extractor_tie_counts_and_products_use_original_candidates(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .extract import fixed_proposals, check_outputs
        from .reproduction import feature_roles
        from local_fusion_action_utility_audit.features import extract_features, schema
        with patch('local_fusion_action_utility_audit.features.pred_iou', side_effect=lambda a, b: np.full((len(a), len(b)), .7)):
            cached, proxies, spec, masks = self.full_output_fixture()
            proxies[KEEP]['scores'][[0, 1]] = proxies[KEEP]['scores'][[1, 0]]
            proxies[KEEP]['corners'][0, :, 0] += 1e-6
            before = {k: v.copy() for k, v in proxies[KEEP].items() if isinstance(v, np.ndarray)}
            _, broken = extract_features(proxies, np.arange(3), proxies[KEEP]['scores'],
                                         proxies[KEEP]['corners'], masks, spec)
            j = schema('cls').index('competition_higher_score_overlap_count')
            self.assertFalse(np.array_equal(broken['cls'][..., j], cached['features']['cls'].numpy()[..., j]))
            with patch('qa_local_intervention.operators.masks_for_frame', return_value=(masks[0], [], {})):
                result, check = fixed_proposals({'ego': {'anchor_box': None}}, proxies, {'levels': []},
                    SimpleNamespace(spec=spec, lidar_range=[0]*6), cached)
            self.assertTrue(check['accepted'])
            self.assertEqual(check_outputs(result, cached, 'train', {}), [])
            for task in ('cls', 'reg'):
                for field, role in feature_roles(task).items():
                    if role == 'exact_discrete' or field in ('competition_roi_delta_x_higher_count',):
                        j = schema(task).index(field)
                        np.testing.assert_array_equal(result[-1][task][..., j], cached['features'][task].numpy()[..., j])
            for key in before:
                np.testing.assert_array_equal(proxies[KEEP][key], before[key])
            self.assertEqual(cached['labels'], {'do_not_read': 7})

    def test_50_all_discrete_branches_stay_exact(self):
        from .reproduction import output_reproduction, feature_roles
        from local_fusion_action_utility_audit.features import schema
        names = [KEEP, 'single:0', 'query:0']
        for task in ('cls', 'reg'):
            columns = schema(task)
            self.assertEqual(set(feature_roles(task)), set(columns))
            for field, role in feature_roles(task).items():
                if role != 'exact_discrete':
                    continue
                expected = np.zeros((2, 3, len(columns)), np.float32)
                actual = expected.copy()
                actual[0, 0, columns.index(field)] = 1e-7
                self.assertFalse(output_reproduction(actual, expected, task, names)['accepted'], field)

    def score_product_fixture(self, field):
        from .reproduction import SCORE_PRODUCTS
        from local_fusion_action_utility_audit.features import schema
        names = [KEEP, 'single:0', 'query:0']
        cls_e = np.zeros((32, 3, len(schema('cls'))), np.float32)
        reg_e = np.zeros((32, 3, len(schema('reg'))), np.float32)
        cls_a, reg_a = cls_e.copy(), reg_e.copy()
        primitive, factor = SCORE_PRODUCTS[field]
        for cls, reg, current in ((cls_e, reg_e, False), (cls_a, reg_a, True)):
            x = (3e-6 if current else 1e-6) if factor.endswith('_count') else (.80003 if current else .8)
            y = 30 if factor.endswith('_count') else (.80003 if current else .8)
            cls[..., schema('cls').index(primitive)] = x
            for task, value in (('cls', cls), ('reg', reg)):
                value[..., schema(task).index(factor)] = y
                value[..., schema(task).index(field)] = x*y
        return cls_e, cls_a, reg_e, reg_a, names

    def test_51_all_score_products_propagate_only_original_primitive_tolerance(self):
        from .reproduction import SCORE_PRODUCTS, output_reproduction
        for field in SCORE_PRODUCTS:
            ce, ca, re, ra, names = self.score_product_fixture(field)
            for task, a, e in (('cls', ca, ce), ('reg', ra, re)):
                before = e.copy()
                check = output_reproduction(a, e, task, names, classification_actual=ca, classification_expected=ce)
                self.assertFalse(check['numeric']['allclose_2e5'])
                self.assertTrue(check['accepted'], check['reason'])
                self.assertTrue(check['score_products'][field]['accepted'])
                np.testing.assert_array_equal(e, before)
                self.assertFalse(output_reproduction(a, e, task, names, True)['accepted'])

    def test_52_fabricated_products_and_primitive_drift_rejected(self):
        from .reproduction import SCORE_PRODUCTS, output_reproduction
        from local_fusion_action_utility_audit.features import schema
        for field, (primitive, factor) in SCORE_PRODUCTS.items():
            ce, ca, re, ra, names = self.score_product_fixture(field)
            for changed in (field, primitive, factor):
                broken = ca.copy()
                broken[0, 1, schema('cls').index(changed)] += .01
                self.assertFalse(output_reproduction(broken, ce, 'cls', names)['accepted'], changed)
            self.assertFalse(output_reproduction(ra, re, 'reg', names)['accepted'])
            broken = ra.copy()
            broken[0, 1, schema('reg').index(field)] += 1e-4
            self.assertFalse(output_reproduction(broken, re, 'reg', names,
                classification_actual=ca, classification_expected=ce)['accepted'])

    def test_53_count_change_not_hidden_by_score_product(self):
        from .reproduction import output_reproduction
        from local_fusion_action_utility_audit.features import schema
        field = 'competition_roi_delta_x_higher_count'
        ce, ca, _, _, names = self.score_product_fixture(field)
        ca[..., schema('cls').index('competition_higher_score_overlap_count')] = 29
        ca[..., schema('cls').index(field)] = 3e-6*29
        self.assertFalse(output_reproduction(ca, ce, 'cls', names)['accepted'])

    def test_54_source_geometry_and_unexplained_source_counts_still_fail(self):
        from .reproduction import output_reproduction
        from local_fusion_action_utility_audit.features import schema
        names = [KEEP, 'single:0', 'query:0']
        for task, field in (('cls', 'competition_source_higher_overlap_count'),
                            ('reg', 'source_shared_pred_iou'), ('reg', 'delta_x')):
            value = np.zeros((3, 3, len(schema(task))), np.float32)
            changed = value.copy()
            changed[0, 1, schema(task).index(field)] = .01
            self.assertFalse(output_reproduction(changed, value, task, names)['accepted'])

    def test_55_preaudit_collects_failures_across_all_frames_and_groups(self):
        import json
        import tempfile
        from unittest.mock import patch
        from types import SimpleNamespace
        from .preaudit import audit_frames
        indices = [457, 2660]
        batches = [{'ego': {'communication_sample_index': [i]}} for i in indices]
        runtime = SimpleNamespace(protocol={k+'_indices': indices for k in ('train', 'validation')},
            target=None, loader=lambda *_: (None, batches), predict=lambda *_args, **_kw: ({}, None, {}))
        def check_frame(_run, split, weather, index, *_args):
            return None, {'split': split, 'weather': weather, 'frame': index, 'reasons': ['synthetic failure']}
        with tempfile.TemporaryDirectory(dir=PACKAGE) as directory, \
             patch('builtins.print'), \
             patch('local_fusion_source_evidence_audit.preaudit.base_frame', return_value={}), \
             patch('local_fusion_source_evidence_audit.preaudit.reference_pool', return_value=({}, {}, [])), \
             patch('local_fusion_source_evidence_audit.preaudit.check_frame', side_effect=check_frame) as check:
            with self.assertRaisesRegex(ValueError, '16 failed frames'):
                audit_frames(directory, runtime, None, lambda b, _: b, lambda *_: 'unchanged')
            report = json.loads((Path(directory)/'REPRODUCTION_AUDIT.json').read_text())
        self.assertEqual(check.call_count, 16)
        self.assertEqual(report['checked_frames'], 16)
        self.assertEqual(len(report['groups']), 8)
        self.assertTrue(report['complete'])
        self.assertFalse(report['accepted'])
        self.assertEqual({x['frame'] for x in report['failures']}, set(indices))

    def test_56_preaudit_pass_and_stage_gate(self):
        import json
        import tempfile
        from unittest.mock import patch
        from types import SimpleNamespace
        from .preaudit import audit_frames
        from .pipeline import stages, artifacts
        indices = [457]
        runtime = SimpleNamespace(protocol={k+'_indices': indices for k in ('train', 'validation')},
            target=None, loader=lambda *_: (None, [{'ego': {'communication_sample_index': indices}}]),
            predict=lambda *_args, **_kw: ({}, None, {}))
        with tempfile.TemporaryDirectory(dir=PACKAGE) as directory, \
             patch('builtins.print'), \
             patch('local_fusion_source_evidence_audit.preaudit.base_frame', return_value={}), \
             patch('local_fusion_source_evidence_audit.preaudit.reference_pool', return_value=({}, {}, [])), \
             patch('local_fusion_source_evidence_audit.preaudit.check_frame', return_value=(None, None)):
            report = audit_frames(directory, runtime, None, lambda b, _: b, lambda *_: 'unchanged')
            self.assertEqual(report, json.loads((Path(directory)/'REPRODUCTION_AUDIT.json').read_text()))
        self.assertTrue(report['accepted'])
        order = [name for name, _ in stages('/run')]
        self.assertLess(order.index('R0'), order.index('reproduction-audit'))
        self.assertLess(order.index('reproduction-audit'), order.index('evidence-train'))
        self.assertEqual(artifacts('/run', 'reproduction-audit'), [Path('/run')/'REPRODUCTION_AUDIT.json'])

    def test_57_repeated_raw_drift_is_checked_before_snapshot_restoration(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .extract import fixed_proposals
        reference, scores, corners, names = self.proposal_fixture()
        cached = {'proposals': dict(reference, corners=ArrayTensor(reference['corners'])), 'source_names': names}
        proxies = {name: {'scores': scores.copy(), 'corners': corners.copy()} for name in names}
        proxies[KEEP]['scores'][2] += .01
        with patch('local_fusion_action_utility_audit.features.extract_features') as feature_call:
            result, check = fixed_proposals({}, proxies, {}, SimpleNamespace(spec={'shared_top_k': 3}), cached)
        self.assertIsNone(result)
        self.assertFalse(check['accepted'])
        feature_call.assert_not_called()

    def test_58_all_product_diagnostics_survive_unrelated_count_failure(self):
        from .reproduction import output_reproduction, SCORE_PRODUCTS
        from local_fusion_action_utility_audit.features import schema
        ce, ca, re, ra, names = self.product_fixture()
        for task, value in (('cls', ca), ('reg', ra)):
            value[0, 0, schema(task).index('competition_source_higher_overlap_count')] += 1
        check = output_reproduction(ca, ce, 'cls', names, regression_actual=ra, regression_expected=re,
                                    classification_actual=ca, classification_expected=ce)
        self.assertFalse(check['accepted'])
        self.assertTrue(check['geometry_product']['accepted'])
        self.assertEqual(set(check['score_products']), set(SCORE_PRODUCTS))
        self.assertIsNotNone(check['rank_ties'])

    def test_59_exact_products_reconstructed_in_full_audit(self):
        from .reproduction import output_reproduction, SCORE_PRODUCTS
        from local_fusion_action_utility_audit.features import schema
        ce, _, re, _, names = self.product_fixture()
        check = output_reproduction(ce, ce, 'cls', names, regression_actual=re, regression_expected=re,
                                    classification_actual=ce, classification_expected=ce)
        self.assertTrue(check['accepted'])
        self.assertEqual(set(check['score_products']), set(SCORE_PRODUCTS))
        broken = ce.copy()
        field = next(iter(SCORE_PRODUCTS))
        broken[0, 1, schema('cls').index(field)] = 1e-7
        # Allclose alone passes; the product is inconsistent with zero factors.
        check = output_reproduction(broken, ce, 'cls', names, regression_actual=re, regression_expected=re,
                                    classification_actual=broken, classification_expected=ce)
        self.assertTrue(check['numeric']['allclose_2e5'])
        self.assertFalse(check['accepted'])

    def test_60_original_corners_preserve_all_shared_iou_threshold_branches(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .extract import fixed_proposals, check_outputs
        from local_fusion_action_utility_audit.features import extract_features, schema
        def iou(left, right):
            a, b = np.asarray(left), np.asarray(right)
            al, ah, bl, bh = a[..., :2].min(1), a[..., :2].max(1), b[..., :2].min(1), b[..., :2].max(1)
            intersection = np.maximum(0, np.minimum(ah[:, None], bh)-np.maximum(al[:, None], bl)).prod(-1)
            return intersection/((ah-al).prod(-1)[:, None]+(bh-bl).prod(-1)-intersection)
        with patch('local_fusion_action_utility_audit.features.pred_iou', side_effect=iou):
            for threshold in (.1, .3, .5):
                cached, proxies, spec, masks = self.full_output_fixture()
                d = np.nextafter(np.float32(4*(1-threshold)/(1+threshold)), np.float32(0))
                corners = np.stack([box(), box()+[float(d), 0, 0], box()+[20, 0, 0]]).astype(np.float32)
                for proxy in proxies.values():
                    proxy['corners'] = corners.copy()
                _, expected = extract_features(proxies, np.arange(3), proxies[KEEP]['scores'].astype(np.float64), corners, masks, spec)
                cached['proposals']['corners'] = ArrayTensor(corners.copy())
                cached['features'] = {k: ArrayTensor(v) for k, v in expected.items()}
                proxies[KEEP]['corners'][1, :, 0] += 4e-6
                _, raw = extract_features(proxies, np.arange(3), proxies[KEEP]['scores'], proxies[KEEP]['corners'], masks, spec)
                field = f'competition_iou_over_0p{int(threshold*10)}_count'
                j = schema('cls').index(field)
                self.assertFalse(np.array_equal(raw['cls'][..., j], expected['cls'][..., j]), field)
                with patch('qa_local_intervention.operators.masks_for_frame', return_value=(masks[0], [], {})):
                    result, check = fixed_proposals({'ego': {'anchor_box': None}}, proxies, {'levels': []},
                        SimpleNamespace(spec=spec, lidar_range=[0]*6), cached)
                self.assertTrue(check['accepted'])
                self.assertEqual(check_outputs(result, cached, 'train', {}), [])

    def test_61_validation_raw_check_records_every_source_and_head(self):
        from unittest.mock import patch
        from .extract import reference_pool
        names = [KEEP, 'single:0', 'query:0']
        pool = {name: {k: ArrayTensor(np.zeros(2, np.float32)) for k in ('psm', 'rm')} for name in names}
        generated = {name: {k: ArrayTensor(np.ones(2, np.float32)) for k in ('psm', 'rm')} for name in names}
        with patch('local_fusion_action_utility_audit.reproducibility.load_reference', return_value=(pool, {})):
            result, diagnostics, failures = reference_pool(None, None, 'clean', 457, 'validation', generated)
        self.assertIs(result, pool)
        self.assertEqual(len(diagnostics), 6)
        self.assertEqual(len(failures), 6)

    def test_62_feature_computation_error_is_recorded_as_failed_frame(self):
        import json
        import tempfile
        from unittest.mock import patch
        from .extract import check_frame
        with tempfile.TemporaryDirectory(dir=PACKAGE) as directory, \
             patch('local_fusion_source_evidence_audit.extract.fixed_proposals', side_effect=ValueError('Nonfinite IoU')):
            result, failure = check_frame(directory, 'train', 'clean', 457, None, None, None, None,
                                          {'source_names': [KEEP]}, {})
            diagnostic = json.loads(Path(failure['diagnostics']).read_text())
        self.assertIsNone(result)
        self.assertIn('Nonfinite IoU', failure['reasons'][0])
        self.assertEqual(diagnostic['frame'], 457)

    def run_order_audit(self, train, validation, expected=(457, 2660, 1343)):
        """Run the whole preaudit loop with observable train/validation orders."""
        import tempfile
        from unittest.mock import patch
        from types import SimpleNamespace
        from .preaudit import audit_frames
        def loader(split, _weather):
            ids = train if split == 'train' else validation
            return None, [{'ego': {'communication_sample_index': [i]}} for i in ids]
        runtime = SimpleNamespace(protocol={k+'_indices': list(expected) for k in ('train', 'validation')},
                                  target=None, loader=loader,
                                  predict=lambda *_args, **_kw: ({}, None, {}))
        with tempfile.TemporaryDirectory(dir=PACKAGE) as directory, patch('builtins.print'), \
             patch('local_fusion_source_evidence_audit.preaudit.base_frame', return_value={}) as frames, \
             patch('local_fusion_source_evidence_audit.preaudit.reference_pool', return_value=({}, {}, [])), \
             patch('local_fusion_source_evidence_audit.preaudit.check_frame', return_value=(None, None)):
            report = audit_frames(directory, runtime, None, lambda b, _: b, lambda *_: 'unchanged')
        return report, [(call.args[1], call.args[2], call.args[3]) for call in frames.call_args_list]

    def test_63_shuffled_train_is_checked_by_frame_id_without_reordering(self):
        expected = [457, 2660, 1343]
        shuffled = [1343, 457, 2660]
        report, visits = self.run_order_audit(shuffled, expected, expected)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['checked_frames'], 24)
        for weather in ('clean', 'fog', 'rain', 'snow'):
            self.assertEqual([i for split, w, i in visits if split == 'train' and w == weather], shuffled)
            self.assertEqual([i for split, w, i in visits if split == 'validation' and w == weather], expected)
        self.assertEqual(expected, [457, 2660, 1343])
        self.assertEqual(shuffled, [1343, 457, 2660])

    def test_64_validation_order_still_strict(self):
        with self.assertRaisesRegex(RuntimeError, 'validation/clean.*expected=457 actual=1343'):
            self.run_order_audit([1343, 457, 2660], [1343, 457, 2660])

    def test_65_train_duplicate_missing_and_unknown_frames_are_rejected(self):
        for train, message in (([1343, 457, 457], 'duplicate.*frame=457'),
                               ([1343, 457], 'missing=\\[2660\\]'),
                               ([1343, 457, 9999], 'unexpected.*frame=9999')):
            with self.subTest(train=train), self.assertRaisesRegex(RuntimeError, message):
                self.run_order_audit(train, [457, 2660, 1343])

    def test_66_duplicate_protocol_indices_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Duplicate train frame indices in S0 protocol'):
            self.run_order_audit([457, 457], [457, 457], [457, 457])

    def test_67_s0_fp64_candidate_accumulation_matches_exact_validation(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .extract import fixed_proposals, check_outputs
        from local_fusion_action_utility_audit.features import extract_features, schema
        scores = np.array([.9, .8, .7, .6, .5, .4], np.float32)
        corners = np.stack([box()+[i*.2, 0, 0] for i in range(6)]).astype(np.float32)
        names = [KEEP, 'single:0', 'query:0']
        proxies = {name: {'scores': scores.copy(), 'corners': corners.copy(), 'anchor_num': 1} for name in names}
        masks = [np.ones((1, 6), bool) for _ in scores]
        spec = {'shared_top_k': 6, 'roi_expansion': 1.5, 'local_top_k': 2, 'nearby_distance_m': 10.}
        # Independent S0 boundary: top_shared_proposals promotes to FP64,
        # whereas the decoded source pool remains FP32.
        with patch('local_fusion_action_utility_audit.features.pred_iou', side_effect=lambda a, b: np.full((len(a), len(b)), .7)):
            _, expected = extract_features(proxies, np.arange(6), scores.astype(np.float64), corners, masks, spec)
            _, broken = extract_features(proxies, np.arange(6), scores, corners, masks, spec)
            j = schema('cls').index('competition_local_score_density')
            self.assertFalse(np.array_equal(expected['cls'][..., j], broken['cls'][..., j]))
            cached = {'source_names': names, 'proposals': {'anchor_ids': list(range(6)), 'scores': scores.tolist(),
                'corners': ArrayTensor(corners), 'ranks': list(range(1, 7))},
                'features': {k: ArrayTensor(v.copy()) for k, v in expected.items()}}
            with patch('qa_local_intervention.operators.masks_for_frame', return_value=(masks[0], [], {})):
                result, check = fixed_proposals({'ego': {'anchor_box': None}}, proxies, {'levels': []},
                    SimpleNamespace(spec=spec, lidar_range=[0]*6), cached, True)
        self.assertTrue(check['accepted'])
        self.assertEqual(result[1].dtype, np.float64)
        self.assertEqual(check_outputs(result, cached, 'validation', {}), [])
        for task in ('cls', 'reg'):
            self.assertEqual(result[-1][task].tobytes(), expected[task].tobytes())

    def test_68_source_threshold_snapshot_prevents_near_score_count_flip(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .extract import fixed_proposals, check_outputs
        from local_fusion_action_utility_audit.features import extract_features, schema
        with patch('local_fusion_action_utility_audit.features.pred_iou', side_effect=lambda a, b: np.full((len(a), len(b)), .7)):
            cached, proxies, spec, masks = self.full_output_fixture()
            proxies['single:0']['scores'][1] = proxies[KEEP]['scores'][1]
            _, expected = extract_features(proxies, np.arange(3), proxies[KEEP]['scores'].astype(np.float64),
                                            proxies[KEEP]['corners'], masks, spec)
            cached['features'] = {k: ArrayTensor(v.copy()) for k, v in expected.items()}
            # Raw source score moves across an original Shared score by 2 ULPs.
            proxies['single:0']['scores'][1] += np.float32(1.2e-7)
            before = {name: {k: v.copy() for k, v in p.items() if isinstance(v, np.ndarray)} for name, p in proxies.items()}
            _, raw = extract_features(proxies, np.arange(3), proxies[KEEP]['scores'].astype(np.float64),
                                     proxies[KEEP]['corners'], masks, spec)
            j = schema('cls').index('competition_source_higher_overlap_count')
            self.assertNotEqual(raw['cls'][1, 1, j], expected['cls'][1, 1, j])
            with patch('qa_local_intervention.operators.masks_for_frame', return_value=(masks[0], [], {})):
                result, check = fixed_proposals({'ego': {'anchor_box': None}}, proxies, {'levels': []},
                    SimpleNamespace(spec=spec, lidar_range=[0]*6), cached)
        self.assertTrue(check['accepted'])
        self.assertFalse(check['source_anchor_scores']['sources']['single:0']['anchor_probability']['exact'])
        self.assertEqual(check_outputs(result, cached, 'train', {}), [])
        for name, values in before.items():
            for key, value in values.items():
                np.testing.assert_array_equal(proxies[name][key], value)
        self.assertEqual(cached['labels'], {'do_not_read': 7})

    def test_69_source_score_and_delta_drift_rejected_before_restore(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .extract import fixed_proposals
        with patch('local_fusion_action_utility_audit.features.pred_iou', side_effect=lambda a, b: np.full((len(a), len(b)), .7)):
            cached, proxies, spec, masks = self.full_output_fixture()
        proxies['single:0']['scores'][0] += .01
        with patch('qa_local_intervention.operators.masks_for_frame', return_value=(masks[0], [], {})), \
             patch('local_fusion_action_utility_audit.features.extract_features') as extract:
            result, check = fixed_proposals({'ego': {'anchor_box': None}}, proxies, {'levels': []},
                SimpleNamespace(spec=spec, lidar_range=[0]*6), cached)
        self.assertIsNone(result)
        self.assertFalse(check['accepted'])
        extract.assert_not_called()
        from .reproduction import source_score_reproduction
        names = [KEEP, 'single:0']
        expected = np.array([[.8, .8]], np.float32)
        proxies = {name: {'scores': expected[:, i].copy()} for i, name in enumerate(names)}
        proxies['single:0']['scores'][0] += 3e-5
        check = source_score_reproduction(proxies, [0], names, expected)
        self.assertTrue(check['sources']['single:0']['anchor_probability']['allclose_2e5'])
        self.assertFalse(check['sources']['single:0']['anchor_delta_shared']['allclose_2e5'])
        self.assertFalse(check['accepted'])

    def test_70_validation_source_score_remains_exact(self):
        from .reproduction import source_score_reproduction
        names = [KEEP, 'single:0']
        expected = np.array([[.6, .7]], np.float32)
        proxies = {name: {'scores': expected[:, i].copy()} for i, name in enumerate(names)}
        self.assertTrue(source_score_reproduction(proxies, [0], names, expected, True)['accepted'])
        proxies['single:0']['scores'][0] += 1e-7
        self.assertTrue(source_score_reproduction(proxies, [0], names, expected)['accepted'])
        self.assertFalse(source_score_reproduction(proxies, [0], names, expected, True)['accepted'])

    def test_71_source_geometry_count_drift_still_rejected(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .extract import fixed_proposals, check_outputs
        from local_fusion_action_utility_audit.features import extract_features, schema
        def iou(left, right):
            a, b = np.asarray(left), np.asarray(right)
            al, ah, bl, bh = a[..., :2].min(1), a[..., :2].max(1), b[..., :2].min(1), b[..., :2].max(1)
            intersection = np.maximum(0, np.minimum(ah[:, None], bh)-np.maximum(al[:, None], bl)).prod(-1)
            return intersection/((ah-al).prod(-1)[:, None]+(bh-bl).prod(-1)-intersection)
        with patch('local_fusion_action_utility_audit.features.pred_iou', side_effect=iou):
            cached, proxies, spec, masks = self.full_output_fixture()
            d = np.nextafter(np.float32(4*(1-.1)/(1+.1)), np.float32(0))
            proxies['single:0']['corners'][2] = (box()+[float(d), 0, 0]).astype(np.float32)
            _, expected = extract_features(proxies, np.arange(3), proxies[KEEP]['scores'].astype(np.float64),
                                            proxies[KEEP]['corners'], masks, spec)
            cached['features'] = {k: ArrayTensor(v.copy()) for k, v in expected.items()}
            # Within the original continuous geometry tolerance, but crossing
            # the strict IoU=.1 boundary must still reject the changed count.
            proxies['single:0']['corners'][2, :, 0] += 4e-6
            with patch('qa_local_intervention.operators.masks_for_frame', return_value=(masks[0], [], {})):
                result, check = fixed_proposals({'ego': {'anchor_box': None}}, proxies, {'levels': []},
                    SimpleNamespace(spec=spec, lidar_range=[0]*6), cached)
            differences = {}
            failures = check_outputs(result, cached, 'train', differences)
        self.assertTrue(check['accepted'])
        self.assertTrue(any('competition_source_higher_overlap_count' in f for f in failures))
        for field in ('delta_x', 'delta_y', 'delta_z', 'center_distance'):
            column = differences['output_only/reg']['columns'].get(field)
            self.assertTrue(column is None or column['allclose_2e5'])

    def test_72_recheck_selects_all_train_failures_and_each_validation_weather(self):
        from .preaudit import recheck_targets
        rows = [{'split': 'train', 'weather': w, 'frame': i} for w, i in (('fog', 4309), ('fog', 2941), ('snow', 2941))]
        rows += [{'split': 'validation', 'weather': w, 'frame': i} for w in ('clean', 'fog', 'rain', 'snow') for i in (0, 13)]
        protocol = {'train_indices': [2941, 4309], 'validation_indices': [0, 13]}
        result = recheck_targets({'complete': True, 'accepted': False, 'failures': rows}, protocol)
        self.assertEqual(result['train/fog'], {4309, 2941})
        self.assertEqual(result['train/snow'], {2941})
        self.assertEqual(sum(map(len, result.values())), 7)
        for w in ('clean', 'fog', 'rain', 'snow'):
            self.assertEqual(result['validation/'+w], {0})

    def test_73_recheck_preserves_loader_prefix_and_does_not_pass_full_audit(self):
        import json
        import tempfile
        from unittest.mock import patch
        from types import SimpleNamespace
        from .preaudit import audit_frames
        from .pipeline import stages, artifacts
        order = [457, 1343, 2660, 2941]
        consumed = []
        def loader(split, weather):
            def batches():
                for i in order:
                    consumed.append((split, weather, i))
                    yield {'ego': {'communication_sample_index': [i]}}
            return None, batches()
        # Match DataLoader's length without changing the iterator's order.
        class Loader:
            def __init__(self, iterator):
                self.iterator = iterator
            def __len__(self):
                return len(order)
            def __iter__(self):
                return self.iterator
        runtime = SimpleNamespace(protocol={s+'_indices': order for s in ('train', 'validation')}, target=None,
            loader=lambda s, w: (None, Loader(loader(s, w)[1])),
            predict=lambda *_a, **_k: ({}, None, {}))
        with tempfile.TemporaryDirectory(dir=PACKAGE) as directory, patch('builtins.print'), \
             patch('local_fusion_source_evidence_audit.preaudit.base_frame', return_value={}) as frames, \
             patch('local_fusion_source_evidence_audit.preaudit.reference_pool', return_value=({}, {}, [])), \
             patch('local_fusion_source_evidence_audit.preaudit.check_frame', return_value=(None, None)):
            report = audit_frames(directory, runtime, None, lambda b, _: b, lambda *_: 'same',
                                  targets={'train/fog': {2660}, 'validation/clean': {457}})
            self.assertTrue(report['accepted'])
            self.assertFalse(report['full_audit'])
            self.assertEqual(report['checked_frames'], 2)
            self.assertFalse((Path(directory)/'REPRODUCTION_AUDIT.json').exists())
            self.assertEqual(report, json.loads((Path(directory)/'REPRODUCTION_RECHECK.json').read_text()))
        self.assertEqual([c.args[3] for c in frames.call_args_list], [2660, 457])
        self.assertEqual(consumed, [('train', 'fog', i) for i in order[:3]]+[('validation', 'clean', 457)])
        scheduled = stages('/run', '/previous/REPRODUCTION_AUDIT.json')
        names = [name for name, _ in scheduled]
        self.assertLess(names.index('server-tests'), names.index('repair-recheck'))
        self.assertLess(names.index('repair-recheck'), names.index('R0'))
        self.assertLess(names.index('R0'), names.index('reproduction-audit'))
        self.assertIn('--recheck-audit', dict(scheduled)['repair-recheck'])
        self.assertEqual(artifacts('/run', 'repair-recheck'), [Path('/run')/'REPRODUCTION_RECHECK.json'])

    def test_74_invalid_recheck_reports_rejected(self):
        from .preaudit import recheck_targets
        protocol = {'train_indices': [457], 'validation_indices': [0]}
        for report in ({}, {'complete': True, 'accepted': True, 'failures': []},
                       {'complete': True, 'accepted': False, 'failures': [{'split': 'train', 'weather': 'fog', 'frame': 99}]}):
            with self.assertRaises(ValueError):
                recheck_targets(report, protocol)

    def test_75_residual_source_count_diagnostics_explain_both_score_thresholds(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .extract import fixed_proposals, source_count_diagnostics
        from local_fusion_action_utility_audit.features import schema
        with patch('local_fusion_action_utility_audit.features.pred_iou', side_effect=lambda a, b: np.full((len(a), len(b)), .100001)):
            cached, proxies, spec, masks = self.full_output_fixture()
            with patch('qa_local_intervention.operators.masks_for_frame', return_value=(masks[0], [], {})):
                result, check = fixed_proposals({'ego': {'anchor_box': None}}, proxies, {'levels': []},
                    SimpleNamespace(spec=spec, lidar_range=[0]*6), cached)
            result[-1]['cls'][0, 1, schema('cls').index('competition_source_higher_overlap_count')] += 1
            diagnostic = source_count_diagnostics(result, cached, proxies)
        self.assertTrue(check['accepted'])
        self.assertEqual(diagnostic['different_rows'], 1)
        row = diagnostic['rows'][0]
        self.assertEqual(row['source'], 'single:0')
        self.assertEqual(row['iou_within_2e5_of_boundary'], [1, 2])
        self.assertEqual(row['count_current_geometry_original_probability'], row['expected_count'])
        self.assertEqual(len(row['current_prediction_iou']), 3)

    def test_76_recheck_binds_base_and_report_hash(self):
        import tempfile
        from .common import atomic_json
        from .preaudit import recheck_specification
        with tempfile.TemporaryDirectory(dir=PACKAGE) as directory:
            path = Path(directory)/'REPRODUCTION_AUDIT.json'
            base = {'base_run': 'original', 'files_hash': 'original_hash'}
            protocol = {'train_indices': [457], 'validation_indices': [0]}
            atomic_json(path.parent/'protocol.json', {'base': base})
            report = {'complete': True, 'accepted': False,
                      'failures': [{'split': 'train', 'weather': 'fog', 'frame': 457}]}
            atomic_json(path, report)
            first = recheck_specification(path, base, protocol)
            self.assertEqual(first['targets'], {'train/fog': [457]})
            report['note'] = 'changed'
            atomic_json(path, report)
            second = recheck_specification(path, base, protocol)
            self.assertNotEqual(first['sha256'], second['sha256'])
            with self.assertRaisesRegex(ValueError, 'different BASE_RUN'):
                recheck_specification(path, dict(base, files_hash='different'), protocol)

    def test_77_recheck_failure_has_separate_report_and_full_audit_stays_unfinished(self):
        import json
        import tempfile
        from unittest.mock import patch
        from types import SimpleNamespace
        from .preaudit import audit_frames
        runtime = SimpleNamespace(protocol={'train_indices': [457], 'validation_indices': [0]}, target=None,
            loader=lambda *_: (None, [{'ego': {'communication_sample_index': [457]}}]),
            predict=lambda *_a, **_k: ({}, None, {}))
        failure = {'split': 'train', 'weather': 'fog', 'frame': 457, 'reasons': ['unexplained geometry']}
        with tempfile.TemporaryDirectory(dir=PACKAGE) as directory, patch('builtins.print'), \
             patch('local_fusion_source_evidence_audit.preaudit.base_frame', return_value={}), \
             patch('local_fusion_source_evidence_audit.preaudit.reference_pool', return_value=({}, {}, [])), \
             patch('local_fusion_source_evidence_audit.preaudit.check_frame', return_value=(None, failure)):
            with self.assertRaisesRegex(ValueError, 'recheck found 1 failed frames'):
                audit_frames(directory, runtime, None, lambda b, _: b, lambda *_: 'same', targets={'train/fog': {457}})
            report = json.loads((Path(directory)/'REPRODUCTION_RECHECK.json').read_text())
            self.assertTrue(report['complete'])
            self.assertFalse(report['accepted'])
            self.assertFalse((Path(directory)/'REPRODUCTION_AUDIT.json').exists())


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
    # Exercise the actual inherited selected_loader/RandomSampler contract.
    # Synthetic indices and a tiny dataset only; no files/checkpoints/models.
    import tempfile
    from unittest.mock import patch
    from torch.utils.data import RandomSampler, SequentialSampler
    from local_fusion_task_split_pilot.pipeline import selected_loader
    from .preaudit import audit_frames
    class IndexDataset:
        def __len__(self):
            return 32
        def __getitem__(self, index):
            return {'ego': {'communication_sample_index': [index]}}
        @staticmethod
        def collate_batch_test(items):
            return items[0]
    indices = [i*2 for i in range(12)]
    options = {'seed': 17, 'workers': 0}
    def loader(split, weather):
        return selected_loader({}, options, split, weather, indices)
    runtime_order = SimpleNamespace(protocol={k+'_indices': indices for k in ('train', 'validation')},
        target=None, loader=loader, predict=lambda *_args, **_kw: ({}, None, {}))
    with tempfile.TemporaryDirectory() as directory, patch('builtins.print'), \
         patch('local_fusion_task_split_pilot.pipeline.er.make_loader', return_value=(IndexDataset(), None, None)), \
         patch('local_fusion_source_evidence_audit.preaudit.base_frame', return_value={}) as visited, \
         patch('local_fusion_source_evidence_audit.preaudit.reference_pool', return_value=({}, {}, [])), \
         patch('local_fusion_source_evidence_audit.preaudit.check_frame', return_value=(None, None)):
        _, train_loader = loader('train', 'clean')
        _, val_loader = loader('validation', 'clean')
        assert isinstance(train_loader.sampler, RandomSampler)
        assert isinstance(val_loader.sampler, SequentialSampler)
        report_order = audit_frames(directory, runtime_order, None, lambda b, _: b, lambda *_: 'unchanged')
        assert report_order['accepted'] and report_order['checked_frames'] == 96
        train_order = [c.args[3] for c in visited.call_args_list if c.args[1:3] == ('train', 'clean')]
        val_order = [c.args[3] for c in visited.call_args_list if c.args[1:3] == ('validation', 'clean')]
        assert sorted(train_order) == indices and train_order != indices
        assert val_order == indices
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
    # Same candidate boundary using actual OpenCOOD IoU/masks and real Torch
    # cached tensors. This block is server-only, before any dataset extraction.
    from .extract import fixed_proposals, check_outputs
    runtime.spec = {'shared_top_k': 2, 'roi_expansion': 1.5,
                    'local_top_k': 2, 'nearby_distance_m': 10.}
    # Independent baseline from the real S0 proposals(), including its FP64
    # promotion, OpenCOOD filters/decoder/IoU and validation's empty levels.
    # Never generate the expected cache using the S4 function under test.
    from local_fusion_action_utility_audit.s0_counterfactual import proposals as s0_proposals
    from local_fusion_action_utility_audit.features import extract_features as old_extract_features
    from qa_local_intervention.operators import masks_for_frame
    six_scores = torch.full((1, 4, 4, 2), .1)
    six_scores.reshape(-1)[20:26] = torch.tensor([.9, .8, .7, .6, .5, .4])
    six_pool = dict(pool)
    six_pool[KEEP] = dict(pool[KEEP], psm=torch.logit(six_scores).permute(0, 3, 1, 2).contiguous())
    runtime.spec['shared_top_k'] = 6
    s0 = s0_proposals(dataset, batch, six_pool[KEEP], six_pool, {'levels': []}, runtime)
    assert s0[0].tolist() == list(range(20, 26)) and s0[1].dtype == np.float64
    six_cached = {'source_names': s0[4], 'proposals': {'anchor_ids': s0[0].tolist(), 'scores': s0[1].tolist(),
        'corners': torch.from_numpy(s0[2].copy()), 'ranks': list(range(1, 7))},
        'features': {task: torch.from_numpy(value.copy()) for task, value in s0[5].items()}}
    six_proxies = {name: _proxy_cache(dataset, batch, value) for name, value in six_pool.items()}
    six, check = fixed_proposals(batch, six_proxies, {'levels': levels}, runtime, six_cached, True)
    assert check['accepted'] and not check_outputs(six, six_cached, 'validation', {})
    assert all(six[-1][task].tobytes() == s0[5][task].tobytes() for task in ('cls', 'reg'))
    runtime.spec['shared_top_k'] = 2
    audit_proxies = dict(proxies)
    audit_proxies[KEEP] = dict(proxies[KEEP], scores=proxies[KEEP]['scores'].copy(),
                                corners=proxies[KEEP]['corners'].copy())
    audit_proxies[KEEP]['scores'][:] = .1
    audit_proxies[KEEP]['scores'][[20, 21]] = np.array([.60000006, .6], np.float32)
    audit_proxies['single:0'] = dict(proxies['single:0'], scores=proxies['single:0']['scores'].copy())
    audit_proxies['single:0']['scores'][21] = np.float32(.6)
    cached = {'source_names': source_names(proxies), 'proposals': {
        'anchor_ids': [20, 21], 'ranks': [1, 2],
        'scores': audit_proxies[KEEP]['scores'][[20, 21]].tolist(),
        'corners': torch.from_numpy(audit_proxies[KEEP]['corners'][[20, 21]].copy())}}
    old_ids = np.array([20, 21])
    old_corners = cached['proposals']['corners'].numpy()
    old_masks = [masks_for_frame(anchors, corner, [], runtime.lidar_range, runtime.spec['roi_expansion'])[0] for corner in old_corners]
    _, old_features = old_extract_features(audit_proxies, old_ids, np.asarray(cached['proposals']['scores'], np.float64),
                                            old_corners, old_masks, runtime.spec)
    cached['features'] = {task: torch.from_numpy(value.copy()) for task, value in old_features.items()}
    original, check = fixed_proposals(batch, audit_proxies, {'levels': levels}, runtime, cached, True)
    assert check['accepted']
    assert not check_outputs(original, cached, 'validation', {})
    audit_proxies[KEEP]['scores'][[20, 21]] = audit_proxies[KEEP]['scores'][[21, 20]]
    audit_proxies[KEEP]['corners'][20, :, 0] += 1e-6
    audit_proxies['single:0']['scores'][21] += np.float32(1.2e-7)
    repeated, check = fixed_proposals(batch, audit_proxies, {'levels': levels}, runtime, cached)
    assert check['accepted'] and check['reordered_positions_in_repeated_scores'] == 2
    assert not check_outputs(repeated, cached, 'train', {})
    assert not check['source_anchor_scores']['sources']['single:0']['anchor_probability']['exact']
    _, exact_check = fixed_proposals(batch, audit_proxies, {'levels': levels}, runtime, cached, True)
    assert not exact_check['accepted']
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
