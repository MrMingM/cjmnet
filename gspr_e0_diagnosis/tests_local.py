"""Synthetic tests only. No Torch, OpenCOOD, real point clouds or GPU imports."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from e0_numeric import (array_hash, author_mask_proxy, covariance_summary, coverage, distribution,
    merge_distribution, nearest_proxy, range_group, replace_weights)
from e0_common import atomic, finish, frame_commit, frame_done, identity, read, receipt, safe, sha
from e0_inference import compare_predictions, finalize_observations


class Interventions(unittest.TestCase):
    def setUp(self):
        self.r = np.asarray([[.1, .7, 0], [.4, 0, 0], [.8, .3, .6]], np.float32)
        self.valid = np.asarray([[1, 1, 0], [1, 0, 0], [1, 1, 1]], bool)
        self.agent = np.asarray([0, 0, 1], np.int32)

    def run_arm(self, arm, **kwargs):
        return replace_weights(self.r, self.valid, self.agent, arm, ['snow', 'scene', '001'], **kwargs)

    def test_r0_exact_nonmutating(self):
        x, audit = self.run_arm('R0')
        self.assertTrue(np.array_equal(x, self.r))
        x[0, 0] = 1
        self.assertEqual(float(self.r[0, 0]), float(np.float32(.1)))

    def test_unit_only_valid(self):
        x, _ = self.run_arm('R1')
        np.testing.assert_array_equal(x[self.valid], 1)
        np.testing.assert_array_equal(x[~self.valid], 0)
        self.assertEqual(x.dtype, self.r.dtype)

    def test_mean_is_per_vehicle_over_points_not_pillars(self):
        x, _ = self.run_arm('R2')
        self.assertTrue(np.all(x[:2][self.valid[:2]] == np.float32(.4)))
        self.assertTrue(np.all(x[2] == self.r[2].mean(dtype=np.float64).astype(np.float32)))
        self.assertEqual(x[1, 1], 0)

    def test_permutation_distribution_preserved_per_cav(self):
        x, audit = self.run_arm('R3')
        for a in (0, 1):
            mask = self.valid & (self.agent[:, None] == a)
            np.testing.assert_array_equal(np.sort(x[mask]), np.sort(self.r[mask]))
        self.assertTrue(all(a['permutation_sha256'] for a in audit))

    def test_permutation_reproducible_without_global_rng_use(self):
        np.random.seed(99)
        before = np.random.get_state()
        a, first = self.run_arm('R3')
        b, second = self.run_arm('R3')
        after = np.random.get_state()
        np.testing.assert_array_equal(a, b)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_actual_vehicle_id_seed(self):
        _, a = self.run_arm('R3', agent_ids=['1', '9'])
        _, b = self.run_arm('R3', agent_ids=['1', '10'])
        self.assertEqual(a[0]['seed_sha256'], b[0]['seed_sha256'])
        self.assertNotEqual(a[1]['seed_sha256'], b[1]['seed_sha256'])

    def test_reject_bad_masks_and_values(self):
        for wrong in (self.r.astype(np.int32), self.r+np.float32(.1), self.r*np.nan):
            with self.assertRaises(ValueError):
                replace_weights(wrong, self.valid, self.agent, 'R0', [])
        with self.assertRaises(ValueError):
            replace_weights(self.r, self.valid.astype(np.int8), self.agent, 'R0', [])
        with self.assertRaises(ValueError):
            self.run_arm('R4')

    def test_empty_slots_stay_empty(self):
        x, _ = replace_weights(np.zeros((0, 32), np.float32), np.zeros((0, 32), bool), np.zeros(0, int), 'R3', [])
        self.assertEqual(x.shape, (0, 32))


class Statistics(unittest.TestCase):
    def test_author_color_does_not_force_ambiguous_or_cross_vehicle_matches(self):
        raw = np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
        labels = np.asarray([False, True, False, True])
        source = np.asarray([0, 0, 0, 1])
        query = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0]], float)
        status = author_mask_proxy(query, np.asarray([0, 0, 0, 1]), raw, source, labels)
        np.testing.assert_array_equal(status, [2, 0, -1, 1])

    def test_no_range_gaps_or_overlap(self):
        np.testing.assert_array_equal(range_group([0, 19.99, 20, 39.99, 40, 69.99, 70, 200]), [0, 0, 1, 1, 2, 2, 3, 3])
        with self.assertRaises(ValueError):
            range_group([np.nan])

    def test_pooled_moments_not_mean_of_frame_means(self):
        x = np.asarray([.1, .4, .9, .9, .9])
        d = merge_distribution([distribution(x[:2]), distribution(x[2:])])
        self.assertAlmostEqual(d['mean'], x.mean())
        self.assertAlmostEqual(d['std'], x.std())
        self.assertEqual(sum(d['histogram']), len(x))
        self.assertLess(abs(d['quantiles']['0.5']-np.median(x)), d['quantile_bin_width'])

    def test_incremental_merge_and_empty_groups(self):
        a, b = distribution([.1, .2]), distribution([])
        d = merge_distribution([merge_distribution([a, b]), distribution([.6])])
        self.assertEqual(d['n'], 3)
        self.assertAlmostEqual(d['mean'], .3)
        self.assertIsNone(merge_distribution([b])['mean'])

    def test_outside_histogram_not_silently_quantiled(self):
        d = merge_distribution([distribution([0, 50], upper=32)])
        self.assertEqual(d['outside_histogram'], 1)
        self.assertIsNone(d['quantiles'])

    def test_coverage_uses_common_origin_and_union(self):
        a = np.asarray([[.01, .01, 0], [.2, .2, 0], [.7, .1, 0]])
        b = np.asarray([[.05, .05, 0], [1.1, .1, 0]])
        result = coverage(a, b)
        self.assertEqual((result['clean_occupied'], result['intersection'], result['missing_clean_cells'], result['additional_weather_cells']), (2, 1, 1, 1))

    def test_nearest_proxy_threshold_sensitivity_and_empty_clean(self):
        a = np.asarray([[0., 0, 0], [1., 0, 0]])
        b = np.asarray([[.08, 0, 0]])
        result, distance = nearest_proxy(a, b)
        self.assertEqual(result['0.05']['weather_without_near_clean'], 1)
        self.assertEqual(result['0.1']['weather_without_near_clean'], 0)
        empty, _ = nearest_proxy(np.zeros((0, 3)), b)
        self.assertEqual(empty['0.1']['weather_without_near_clean'], 1)

    def test_conditioning_density_removes_linear_confound(self):
        rng = np.random.default_rng(7)
        d = rng.normal(size=10000)
        x = np.column_stack([d+.1*rng.normal(size=len(d)), d+.1*rng.normal(size=len(d)), d,
                             rng.normal(size=len(d)), rng.normal(size=len(d))])
        result = covariance_summary(['r', 'u', 'density', 'sensor_range', 'intensity'], len(x), x.sum(0), x.T@x)
        self.assertGreater(result['correlation'][0][1], .9)
        self.assertLess(abs(result['partial_correlation']['u']), .05)

    def test_constant_quality_no_nan_correlation(self):
        x = np.ones((20, 5))
        result = covariance_summary(['r', 'u', 'density', 'sensor_range', 'intensity'], 20, x.sum(0), x.T@x)
        self.assertIsNone(result['partial_correlation']['u'])
        json.dumps(result, allow_nan=False)


class Integrity(unittest.TestCase):
    def test_adopt_pending_observation_after_prediction_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for weather in ('clean', 'fog', 'snow'):
                atomic(root/'predictions'/weather/'000000.json', dict(prediction_is_none=True, prediction_boxes=None, scores=None))
                atomic(root/'pending'/weather/'000000.json', dict(final_prediction_identity=identity([None, None, True])))
            finalize_observations(root, 'request', 1)
            finalize_observations(root, 'request', 1)
            observed = read(root/'observations/clean/000000.json')
            self.assertEqual(observed['prediction_sha256'], sha(root/'predictions/clean/000000.json'))
            atomic(root/'predictions/clean/000000.json', dict(prediction_is_none=False, prediction_boxes=[], scores=[]))
            with self.assertRaises(ValueError):
                finalize_observations(root, 'request', 1)

    def test_full_intervention_summary_uses_saved_target_id_sets(self):
        from e0_analysis import arm_folder, interventions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weather_names = ('clean', 'fog', 'snow')
            for arm in ('R0', 'R1', 'R2', 'R3'):
                folder = arm_folder(root, arm, smoke=True)
                result = {w: dict(AP={p: dict(ap30=.1, ap50=.1, ap70=.1) for p in ('O', 'O*', 'F')}) for w in weather_names}
                atomic(folder/'summary.json', dict(status='complete', frame_count=12, conditions=list(weather_names),
                    checkpoint_sha256='same', results=result))
                metrics = []
                for weather in weather_names:
                    for index in range(12):
                        objects = [1] if arm != 'R1' else [2]
                        item = dict(index=index, condition=weather, protocols={p: dict(thresholds={str(t): dict(tp=1, fp=0, fn=1)
                            for t in (.3, .5, .7)}) for p in ('O', 'O*', 'F')})
                        item['protocols']['F']['matched_pairs'] = {str(t): [dict(object_id=o) for o in objects] for t in (.3, .5, .7)}
                        metrics.append(item)
                (folder/'per_frame_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics), encoding='utf-8')
            summary = interventions(root, smoke=True)
            self.assertEqual(summary['fixed_GT_target_changes_from_R0']['R1']['snow']['0.7'], dict(recovered=12, lost=12))
            self.assertEqual(summary['detection_counts']['R0']['fog']['F']['0.7']['tp'], 12)
            self.assertEqual(summary['delta_from_R0_pp']['R2']['snow']['F']['ap70'], 0)

    def test_pending_observation_cannot_attach_to_different_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic(root/'predictions/clean/000000.json', dict(prediction_is_none=True, prediction_boxes=None, scores=None))
            atomic(root/'pending/clean/000000.json', dict(final_prediction_identity='wrong'))
            with self.assertRaises(ValueError):
                finalize_observations(root, 'request', 1)

    def test_conclusions_handles_empty_proxy_groups_without_inventing_means(self):
        from e0_report import conclusions
        from e0_common import HERE
        source = read(HERE/'original_FINAL_RESULTS.json')['results']
        baseline = {v: dict(results=source[v]['results']) for v in ('C', 'D')}
        delta = {a: {w: dict(F=dict(ap30=0., ap50=0., ap70=0.)) for w in ('fog', 'snow')} for a in ('R1', 'R2', 'R3')}
        empty = dict(r=dict(mean=None, n=0))
        reliability = dict(weather=dict(snow=dict(distributions={k: empty for k in
            ('near_clean_proxy', 'unmatched_clean_proxy', 'author_color/zero', 'author_color/nonzero')})))
        point = dict(aggregate=dict(snow=dict(zero_retained_point_targets=0, fixed_targets=1,
            missing_clean_cells=0, clean_occupied_cells=1)))
        errors = dict(weather=dict(snow=dict(counts={'0.7': dict(C_tp=1, D_tp=1, C_fp=0, D_fp=0)})))
        text = conclusions(dict(delta_from_R0_pp=delta), reliability, point, errors, baseline)
        self.assertIn('无样本', text)
        self.assertIn('E4', text)
        self.assertIn('尚不能确定', text)

    def test_receipt_rejects_changed_successful_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic(root/'x.json', {'v': 3})
            finish(root, 'req')
            self.assertIsNotNone(receipt(root, 'req'))
            atomic(root/'x.json', {'v': 4})
            with self.assertRaises(ValueError):
                receipt(root, 'req')

    def test_immutable_request_cannot_be_rebound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'request.json'
            atomic(path, {'a': 1}, immutable=True)
            atomic(path, {'a': 1}, immutable=True)
            with self.assertRaises(ValueError):
                atomic(path, {'a': 2}, immutable=True)

    def test_frame_commit_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frame.json'
            frame_commit(path, {'n': 1}, 'id')
            self.assertTrue(frame_done(path, 'id'))
            with self.assertRaises(ValueError):
                frame_done(path, 'other')
            atomic(path, {'n': 2})
            with self.assertRaises(ValueError):
                frame_done(path, 'id')

    def test_test_path_refused_before_open(self):
        with self.assertRaises(ValueError):
            safe('/data/dataset/test/scene/pcd')

    def test_array_hash_shape_and_dtype_bound(self):
        x = np.asarray([0, 1], dtype=np.float32)
        self.assertNotEqual(array_hash(x), array_hash(x.astype(np.float64)))
        self.assertNotEqual(array_hash(x), array_hash(x.reshape(1, 2)))

    def test_final_prediction_gate_never_ignores_discrete_changes(self):
        protocol = {'box_atol_m': .0001, 'score_atol': .00001}
        row = dict(prediction_is_none=False, prediction_boxes=np.zeros((1, 8, 3)).tolist(),
                   scores=[.9], box_dtype='float32', score_dtype='float32')
        compare_predictions(np.zeros((1, 8, 3), np.float32), np.asarray([.9], np.float32), row, protocol)
        with self.assertRaises(ValueError):
            compare_predictions(None, None, row, protocol)
        with self.assertRaises(AssertionError):
            compare_predictions(np.ones((1, 8, 3), np.float32), np.asarray([.9], np.float32), row, protocol)


if __name__ == '__main__':
    unittest.main()
