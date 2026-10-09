"""Audit continuous outputs and discontinuous source ranks without changing inputs."""
import numpy as np
from local_fusion_action_utility_audit.common import KEEP
from local_fusion_action_utility_audit.features import schema
from local_fusion_action_utility_audit.reproducibility import (
    NUMERIC_ATOL, NUMERIC_RTOL, numeric_difference)


POLICY = {
    'schema': 2,
    'continuous_atol': NUMERIC_ATOL,
    'continuous_rtol': NUMERIC_RTOL,
    'validation': 'exact_original_snapshot_features',
    'train_rank_exception': 'source_relative_rank_only; reconstruct both ranks from roi_max; '
                            'every flipped comparison must be inside propagated score tolerance',
    'train_geometry_product': 'unchanged exact overlap counts; center distance and xyz deltas within '
                              'original tolerance; reconstruct products within FP32 storage rounding; '
                              'propagate the original distance bound through multiplication',
    'model_inputs': 'unchanged_original_S0_features_including_original_ranks',
}


PRODUCT = 'competition_geometry_distance_x_overlap_count'


def geometry_product_reproduction(actual, expected, task, regression_actual, regression_expected):
    """Verify distance * count using cached primitive columns, never a new epsilon."""
    columns, reg_columns = schema(task), schema('reg')
    result = {'accepted': False}
    if regression_actual is None or regression_expected is None:
        return dict(result, reason='regression primitive features required for product audit')
    reg_a, reg_e = np.asarray(regression_actual), np.asarray(regression_expected)
    expected_shape = actual.shape[:2]+(len(reg_columns),)
    if (reg_a.shape != expected_shape or reg_e.shape != expected_shape
            or any(x.dtype != np.float32 for x in (actual, expected, reg_a, reg_e))
            or not np.isfinite(reg_a).all() or not np.isfinite(reg_e).all()):
        return dict(result, reason='invalid FP32 regression primitive arrays')
    for name in ('center_distance', 'delta_x', 'delta_y', 'delta_z'):
        j = reg_columns.index(name)
        if not np.isclose(reg_a[..., j], reg_e[..., j], atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL).all():
            return dict(result, reason='primitive geometry exceeds original tolerance: '+name)
    count_col = columns.index('competition_iou_over_0p1_count')
    count_a, count_e = actual[..., count_col], expected[..., count_col]
    if (not np.array_equal(count_a, count_e) or (count_e < 0).any()
            or (count_e > actual.shape[0]-1).any() or (count_e != np.floor(count_e)).any()
            or not np.array_equal(count_e, np.broadcast_to(count_e[:, :1], count_e.shape))):
        return dict(result, reason='overlap counts changed or are invalid')
    # Both tasks contain the same product/count columns. A cls comparison must
    # be tied to the companion reg row, not to arbitrary supplied distances.
    for name in (PRODUCT, 'competition_iou_over_0p1_count'):
        if (not np.array_equal(actual[..., columns.index(name)], reg_a[..., reg_columns.index(name)])
                or not np.array_equal(expected[..., columns.index(name)], reg_e[..., reg_columns.index(name)])):
            return dict(result, reason='classification/regression product rows do not align')
    count = count_e.astype(np.float64)
    d_a = reg_a[..., reg_columns.index('center_distance')].astype(np.float64)
    d_e = reg_e[..., reg_columns.index('center_distance')].astype(np.float64)
    p_a = actual[..., columns.index(PRODUCT)].astype(np.float64)
    p_e = expected[..., columns.index(PRODUCT)].astype(np.float64)
    if (d_a < 0).any() or (d_e < 0).any():
        return dict(result, reason='negative center distance')
    # Distance and its product were separately rounded from float64 to FP32.
    # One FP32 relative unit + one subnormal unit is a conservative storage
    # rounding bound, separate from the unchanged primitive inference tolerance.
    def rounding(value):
        return np.abs(value)*np.finfo(np.float32).eps+float(np.nextafter(np.float32(0), np.float32(1)))
    storage_a = count*rounding(d_a)+rounding(p_a)
    storage_e = count*rounding(d_e)+rounding(p_e)
    if (not (np.abs(p_a-count*d_a) <= storage_a).all()
            or not (np.abs(p_e-count*d_e) <= storage_e).all()
            or (p_a[count == 0] != 0).any() or (p_e[count == 0] != 0).any()):
        return dict(result, reason='product cannot be reconstructed from distance and overlap count')
    bound = count*(NUMERIC_ATOL+NUMERIC_RTOL*np.abs(d_e))+storage_a+storage_e
    difference = np.abs(p_a-p_e)
    result.update(accepted=bool((difference <= bound).all()),
                  reason='exact counts and verified FP32 products; original distance error propagated',
                  counts_exact=True, distance=numeric_difference(d_a, d_e),
                  product_max_absolute_difference=float(difference.max()) if difference.size else 0.,
                  outside_propagated_bound=int((difference > bound).sum()))
    return result


def output_reproduction(actual, expected, task, names, require_exact=False,
                        regression_actual=None, regression_expected=None):
    """Return a JSON-safe decision, field diagnostics, and any proven rank ties.

    A rank is a count of strict comparisons, not a continuous measurement. The
    Rank and distance-product audits are restricted to their exact definitions,
    never applied to counts/IoUs or validation. Cached model inputs stay intact.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    columns = schema(task)
    result = {'numeric': numeric_difference(actual, expected), 'accepted': False,
              'policy': POLICY, 'columns': {}, 'rank_ties': None, 'geometry_product': None}
    if (actual.shape != expected.shape or actual.ndim != 3
            or actual.shape[1:] != (len(names), len(columns))
            or len(set(names)) != len(names) or names.count(KEEP) != 1
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        result['reason'] = 'shape/schema/source names or finite check failed'
        return result
    close = np.isclose(actual, expected, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL)
    failures = []
    for j, name in enumerate(columns):
        delta = numeric_difference(actual[..., j], expected[..., j])
        if not delta['exact']:
            index = np.unravel_index(np.abs(actual[..., j].astype(float)
                                           - expected[..., j]).argmax(), actual.shape[:2])
            result['columns'][name] = dict(delta, outside_tolerance=int((~close[..., j]).sum()),
                worst={'proposal_position': int(index[0]), 'source': names[index[1]],
                       'expected': float(expected[index+(j,)]), 'actual': float(actual[index+(j,)])})
            if not close[..., j].all():
                error = np.abs(actual[..., j].astype(float)-expected[..., j])
                index = np.unravel_index(np.where(~close[..., j], error, -1).argmax(), error.shape)
                result['columns'][name]['worst_outside_tolerance'] = {
                    'proposal_position': int(index[0]), 'source': names[index[1]],
                    'expected': float(expected[index+(j,)]), 'actual': float(actual[index+(j,)])}
        if not close[..., j].all():
            failures.append(name)
    if require_exact:
        result['accepted'] = result['numeric']['exact']
        result['reason'] = 'validation requires exact original feature bytes'
        return result
    if not failures:
        result.update(accepted=True, reason='all columns within unchanged numeric tolerance')
        return result
    permitted = {PRODUCT} | ({'source_relative_rank'} if task == 'cls' else set())
    if set(failures)-permitted:
        result['reason'] = 'out-of-tolerance columns: '+', '.join(failures)
        return result
    if PRODUCT in failures:
        check = geometry_product_reproduction(actual, expected, task,
            actual if task == 'reg' else regression_actual,
            expected if task == 'reg' else regression_expected)
        result['geometry_product'] = check
        if not check['accepted']:
            return dict(result, reason='geometry product: '+check['reason'])
        failures.remove(PRODUCT)
    if not failures:
        return dict(result, accepted=True, reason='verified distance-product error propagation; cached inputs retained')

    rank_column, score_column = columns.index('source_relative_rank'), columns.index('roi_max')
    evidence = [i for i, name in enumerate(names) if name != KEEP]
    if not evidence:
        result['reason'] = 'no single/query sources'
        return result
    before = expected[..., score_column].astype(np.float64)
    after = actual[..., score_column].astype(np.float64)
    # [proposal, source being ranked, source it is compared with]. Scores were
    # FP32 sigmoid values in S0; roi_max preserves them exactly in the cache.
    before_gap = before[:, None, evidence]-before[:, :, None]
    after_gap = after[:, None, evidence]-after[:, :, None]
    before_order, after_order = before_gap > 0, after_gap > 0
    before_rank = before_order.sum(-1)/len(evidence)
    after_rank = after_order.sum(-1)/len(evidence)
    if (not np.array_equal(before_rank.astype(np.float32), expected[..., rank_column])
            or not np.array_equal(after_rank.astype(np.float32), actual[..., rank_column])):
        result['reason'] = 'recorded source ranks cannot be reconstructed from roi_max'
        return result
    flipped = before_order != after_order
    # Propagate the SAME per-score error bound to a score difference. This does
    # not enlarge the allowed error of either score or any other feature.
    bound = NUMERIC_ATOL+NUMERIC_RTOL*np.abs(before)
    pair_bound = bound[:, None, evidence]+bound[:, :, None]
    ambiguous = (np.abs(before_gap) <= pair_bound) & (np.abs(after_gap) <= pair_bound)
    result['rank_ties'] = {
        'changed_rank_elements': int(np.count_nonzero(actual[..., rank_column] != expected[..., rank_column])),
        'flipped_comparisons': int(flipped.sum()),
        'flips_outside_score_bounds': int((flipped & ~ambiguous).sum()),
        'largest_flipped_score_gap': float(np.maximum(np.abs(before_gap), np.abs(after_gap))[flipped].max())
                                    if flipped.any() else 0.,
    }
    result['accepted'] = bool(ambiguous[flipped].all())
    result['reason'] = ('only verified near-tie rank comparisons changed; cached inputs retained'
                        if result['accepted'] else 'source ordering changed outside score error bounds')
    return result
