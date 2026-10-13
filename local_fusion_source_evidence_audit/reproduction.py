"""Audit fixed S0 inputs, continuous outputs and explicitly derived features."""
import numpy as np
from local_fusion_action_utility_audit.common import KEEP
from local_fusion_action_utility_audit.features import schema
from local_fusion_action_utility_audit.reproducibility import (
    NUMERIC_ATOL, NUMERIC_RTOL, numeric_difference)


POLICY = {
    'schema': 5,
    'continuous_atol': NUMERIC_ATOL,
    'continuous_rtol': NUMERIC_RTOL,
    'validation': 'exact_original_snapshot_features',
    'train_rank_exception': 'source_relative_rank_only; reconstruct both ranks from roi_max; '
                            'every flipped comparison must be inside propagated score tolerance',
    'train_geometry_product': 'unchanged exact overlap counts; center distance and xyz deltas within '
                              'original tolerance; reconstruct products within FP32 storage rounding; '
                              'propagate the original distance bound through multiplication',
    'model_inputs': 'unchanged_original_S0_features_including_original_ranks',
    'proposal_pool': 'original_S0_anchor_ids_order_and_ROIs; no repeated_topk_selection; '
                     'source_names_exact; redecoded_reference_anchors_checked_with_original_tolerance',
    'candidate_inputs': 'after raw verification, use original S0 candidate scores/corners in '
                        'output reproduction, including KEEP at candidate anchors; '
                        'do not mutate generated outputs or evidence proxies',
    'score_inputs': 'candidate scores are FP64 as in S0 top_shared_proposals; source anchor '
                    'probabilities are saved S0 FP32 values after checking raw probabilities '
                    'and anchor deltas with the original tolerance (exact for validation); '
                    'source geometry and noncandidate ROI scores remain current',
    'competition': 'all counts and threshold decisions remain exact; no count-change exception; '
                   'all three explicit products reconstructed from their primitive columns '
                   'with FP32 storage bounds and original primitive tolerance',
    'preaudit': 'all train/validation frames before expensive evidence/LOO; collect every '
                'reproduction failure; repeat the same checks during extraction',
}


PRODUCT = 'competition_geometry_distance_x_overlap_count'
SCORE_PRODUCTS = {
    'competition_roi_delta_x_higher_count': ('roi_delta_shared', 'competition_higher_score_overlap_count'),
    'competition_anchor_delta_x_max_iou': ('anchor_delta_shared', 'competition_max_pred_iou'),
}


def candidate_snapshot_proxies(proxies, reference):
    """Local copy for output audit only; caller must first verify raw predictions.

    S0 saved the selected Shared predictions, not every train anchor. Restore
    exactly those saved values, keeping the remaining ROI predictions current.
    New evidence/LOO continues to see the untouched generated proxy pool.
    """
    result = dict(proxies)
    shared = dict(proxies[KEEP])
    ids = np.asarray(reference['anchor_ids'], np.int64)
    for key in ('scores', 'corners'):
        shared[key] = np.array(shared[key], copy=True)
        shared[key][ids] = reference[key]
    result[KEEP] = shared
    return result


def source_score_reproduction(proxies, ids, names, expected, require_exact=False):
    """Verify the saved inference scores before using them as fixed thresholds.

    S0 cls stores every source's anchor probability exactly in FP32. Comparing
    original Shared scores against a repeated source probability mixes two
    forwards at a strict > boundary. Both the raw score and raw Shared delta
    are checked first; the generated evidence pool is never modified.
    """
    result = {'accepted': False, 'sources': {}}
    expected = np.asarray(expected)
    if (expected.dtype != np.float32 or expected.shape != (len(ids), len(names))
            or not np.isfinite(expected).all() or (expected < 0).any() or (expected > 1).any()):
        return dict(result, reason='invalid original source anchor probability snapshot')
    actual = np.stack([proxies[name]['scores'][ids] for name in names], axis=1)
    if actual.dtype != np.float32 or not np.isfinite(actual).all() or (actual < 0).any() or (actual > 1).any():
        return dict(result, reason='invalid repeated FP32 source anchor probabilities')
    # KEEP has already been independently checked and restored from proposals.
    if not np.array_equal(actual[:, 0], expected[:, 0]):
        return dict(result, reason='S0 Shared proposal and anchor probability snapshots disagree')
    failures = []
    for i, name in enumerate(names):
        values = {'anchor_probability': (actual[:, i], expected[:, i]),
                  'anchor_delta_shared': (actual[:, i]-actual[:, 0], expected[:, i]-expected[:, 0])}
        checks = {field: numeric_difference(a, e) for field, (a, e) in values.items()}
        result['sources'][name] = checks
        for field, check in checks.items():
            if not check['allclose_2e5'] or (require_exact and not check['exact']):
                failures.append(name+'/'+field)
                a, e = values[field]
                outside = a != e if require_exact else ~np.isclose(a, e, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL)
                position = int(np.where(outside, np.abs(a.astype(float)-e), -1).argmax())
                check['worst_outside_tolerance'] = {'proposal_position': position, 'anchor_id': int(ids[position]),
                                                    'expected': float(e[position]), 'actual': float(a[position])}
    return dict(result, accepted=not failures,
                reason='source anchor score/delta drift: '+', '.join(failures) if failures
                else 'raw source scores/deltas verified; original snapshot thresholds retained')


def source_score_snapshot_proxies(proxies, ids, names, expected):
    """Copy only selected scores; call only after source_score_reproduction."""
    result = dict(proxies)
    for i, name in enumerate(names):
        result[name] = dict(proxies[name], scores=np.array(proxies[name]['scores'], copy=True))
        result[name]['scores'][ids] = expected[:, i]
    return result


def feature_roles(task):
    """Inventory every discrete branch and product in the frozen extractor."""
    from local_fusion_action_utility_audit.features import SHARED, IDENTITY, CLASSIFICATION, REGRESSION, COMPETITION
    fixed = {'shared_rank', 'source_count', *IDENTITY,
             'competition_nearby_count', 'competition_iou_over_0p1_count',
             'competition_iou_over_0p3_count', 'competition_iou_over_0p5_count',
             'competition_higher_score_overlap_count', 'competition_source_higher_overlap_count'}
    derived = {PRODUCT, *SCORE_PRODUCTS}
    continuous = (set(SHARED+CLASSIFICATION+REGRESSION+COMPETITION)
                  - fixed - derived - {'source_relative_rank'})
    roles = {name: ('exact_discrete' if name in fixed else 'verified_product' if name in derived
                    else 'verified_rank' if name == 'source_relative_rank' else 'continuous')
             for name in schema(task)}
    if set(roles) - fixed - derived - continuous - {'source_relative_rank'}:
        raise ValueError('Unreviewed feature in reproduction schema')
    return roles


def proposal_reproduction(reference, expected_names, actual_names, scores, corners,
                          top_k, require_exact=False):
    """Check current predictions at the original proposals, without reselecting them.

    Only the original prediction-derived proposal fields cross this boundary.
    Outcomes/GT/labels never determine the anchors, their order or their ROIs.
    """
    result = {'accepted': False, 'policy': POLICY['proposal_pool'],
              'expected_source_names': list(expected_names), 'actual_source_names': list(actual_names)}
    if list(expected_names) != list(actual_names):
        return dict(result, reason='source family/order/cardinality differs from S0')
    ids = np.asarray(reference['anchor_ids'])
    old_scores, old_corners = np.asarray(reference['scores']), np.asarray(reference['corners'])
    scores, corners = np.asarray(scores), np.asarray(corners)
    n = len(ids) if ids.ndim else 0
    if (ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer))
            or n > top_k or len(np.unique(ids)) != n
            or old_scores.shape != (n,) or old_corners.shape != (n, 8, 3)
            or scores.ndim != 1 or corners.shape != (len(scores), 8, 3)
            or (ids < 0).any() or (ids >= len(scores)).any()
            or not np.isfinite(old_scores).all() or not np.isfinite(old_corners).all()
            or (old_scores <= 0).any() or reference['ranks'] != list(range(1, n+1))):
        return dict(result, reason='invalid reference proposal IDs/ranks/shapes/values')
    ids = ids.astype(np.int64)
    if not np.array_equal(np.lexsort((ids, -old_scores)), np.arange(n)):
        return dict(result, reason='S0 proposal scores/order are inconsistent')
    score_check = numeric_difference(scores[ids], old_scores)
    corner_check = numeric_difference(corners[ids], old_corners)
    result.update(proposals=n, scores=score_check, corners=corner_check,
                  reference_anchor_ids=ids.tolist(),
                  reordered_positions_in_repeated_scores=int(np.count_nonzero(
                      np.lexsort((ids, -scores[ids])) != np.arange(n))))
    checks = (score_check, corner_check)
    if any(not check.get('allclose_2e5', False) for check in checks):
        return dict(result, reason='redecoded S0 proposal score/geometry exceeds original tolerance')
    if require_exact and any(not check['exact'] for check in checks):
        return dict(result, reason='validation reference proposal values must be exact')
    return dict(result, accepted=True, reason='original S0 pool retained; reference anchor outputs verified')


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


def score_product_reproduction(actual, expected, task, classification_actual,
                               classification_expected, field):
    """Propagate original primitive bounds through the two score interactions."""
    columns, cls_columns = schema(task), schema('cls')
    if classification_actual is None or classification_expected is None:
        return {'accepted': False, 'reason': 'classification primitives required for score product'}
    cls_a, cls_e = np.asarray(classification_actual), np.asarray(classification_expected)
    if (cls_a.shape != actual.shape[:2]+(len(cls_columns),) or cls_e.shape != cls_a.shape
            or any(x.dtype != np.float32 for x in (actual, expected, cls_a, cls_e))
            or not np.isfinite(cls_a).all() or not np.isfinite(cls_e).all()):
        return {'accepted': False, 'reason': 'invalid FP32 classification primitives'}
    score_field, factor_field = SCORE_PRODUCTS[field]
    for name in (field, factor_field):
        if (not np.array_equal(actual[..., columns.index(name)], cls_a[..., cls_columns.index(name)])
                or not np.array_equal(expected[..., columns.index(name)], cls_e[..., cls_columns.index(name)])):
            return {'accepted': False, 'reason': 'classification/regression score product rows do not align'}
    x_a, x_e = (value[..., cls_columns.index(score_field)].astype(np.float64) for value in (cls_a, cls_e))
    y_a, y_e = (value[..., columns.index(factor_field)].astype(np.float64) for value in (actual, expected))
    p_a, p_e = (value[..., columns.index(field)].astype(np.float64) for value in (actual, expected))
    if not np.isclose(x_a, x_e, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL).all():
        return {'accepted': False, 'reason': 'score delta exceeds original tolerance'}
    count = factor_field.endswith('_count')
    if count:
        if (not np.array_equal(y_a, y_e) or (y_e < 0).any()
                or (y_e > actual.shape[0]-1).any() or (y_e != np.floor(y_e)).any()
                or not np.array_equal(y_e, np.broadcast_to(y_e[:, :1], y_e.shape))):
            return {'accepted': False, 'reason': 'higher-overlap count changed or is invalid'}
    elif ((y_a < 0).any() or (y_a > 1).any() or (y_e < 0).any() or (y_e > 1).any()
          or not np.isclose(y_a, y_e, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL).all()):
        return {'accepted': False, 'reason': 'IoU primitive invalid or exceeds original tolerance'}
    def rounding(value):
        return np.abs(value)*np.finfo(np.float32).eps+float(np.nextafter(np.float32(0), np.float32(1)))
    def storage(x, y, p):
        dx, dy = rounding(x), np.zeros_like(y) if count else rounding(y)
        return np.abs(y)*dx+np.abs(x)*dy+dx*dy+rounding(p)
    storage_a, storage_e = storage(x_a, y_a, p_a), storage(x_e, y_e, p_e)
    if (not (np.abs(p_a-x_a*y_a) <= storage_a).all()
            or not (np.abs(p_e-x_e*y_e) <= storage_e).all()
            or (p_a[y_a == 0] != 0).any() or (p_e[y_e == 0] != 0).any()):
        return {'accepted': False, 'reason': 'score product cannot be reconstructed from primitives'}
    dx = NUMERIC_ATOL+NUMERIC_RTOL*np.abs(x_e)
    dy = np.zeros_like(y_e) if count else NUMERIC_ATOL+NUMERIC_RTOL*np.abs(y_e)
    bound = np.abs(y_e)*dx+np.abs(x_e)*dy+dx*dy+storage_a+storage_e
    outside = int((np.abs(p_a-p_e) > bound).sum())
    return {'accepted': outside == 0, 'reason': 'verified score product; original primitive bounds propagated',
            'outside_propagated_bound': outside, 'count_exact': count}


def output_reproduction(actual, expected, task, names, require_exact=False,
                        regression_actual=None, regression_expected=None,
                        classification_actual=None, classification_expected=None):
    """Return a JSON-safe decision, field diagnostics, and any proven rank ties.

    A rank is a count of strict comparisons, not a continuous measurement. The
    Rank and product audits are restricted to their exact definitions,
    never applied to counts/IoUs or validation. Cached model inputs stay intact.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    columns = schema(task)
    result = {'numeric': numeric_difference(actual, expected), 'accepted': False,
              'policy': POLICY, 'columns': {}, 'rank_ties': None, 'geometry_product': None,
              'score_products': {}, 'feature_roles': feature_roles(task)}
    if (actual.shape != expected.shape or actual.ndim != 3
            or actual.shape[1:] != (len(names), len(columns))
            or len(set(names)) != len(names) or names.count(KEEP) != 1
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        result['reason'] = 'shape/schema/source names or finite check failed'
        return result
    close = np.isclose(actual, expected, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL)
    # Counts/identity/ranks are exact decisions, independent of floating epsilon.
    for j, name in enumerate(columns):
        if result['feature_roles'][name] == 'exact_discrete':
            close[..., j] = actual[..., j] == expected[..., j]
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
    reasons = []
    # The full extractor provides both companion tasks: verify every product,
    # including products inside allclose, and keep all independent diagnostics.
    audit_all_products = regression_actual is not None and classification_actual is not None
    if PRODUCT in failures or audit_all_products:
        check = geometry_product_reproduction(actual, expected, task,
            actual if task == 'reg' else regression_actual,
            expected if task == 'reg' else regression_expected)
        result['geometry_product'] = check
        if not check['accepted']:
            reasons.append('geometry product: '+check['reason'])
        elif PRODUCT in failures:
            failures.remove(PRODUCT)
    for field in SCORE_PRODUCTS:
        if field in failures or audit_all_products:
            check = score_product_reproduction(actual, expected, task,
                actual if task == 'cls' else classification_actual,
                expected if task == 'cls' else classification_expected, field)
            result['score_products'][field] = check
            if not check['accepted']:
                reasons.append(field+': '+check['reason'])
            elif field in failures:
                failures.remove(field)
    if 'source_relative_rank' in failures:
        check = rank_reproduction(actual, expected, names)
        result['rank_ties'] = check.get('details')
        if not check['accepted']:
            reasons.append(check['reason'])
        else:
            failures.remove('source_relative_rank')
    if failures:
        reasons.insert(0, 'out-of-tolerance columns: '+', '.join(failures))
    result['accepted'] = not reasons
    result['reason'] = '; '.join(reasons) if reasons else 'all primitive/discrete checks and derived audits passed; cached inputs retained'
    return result


def rank_reproduction(actual, expected, names):
    """Reconstruct both ranks, then explain every changed comparison."""
    columns = schema('cls')
    rank_column, score_column = columns.index('source_relative_rank'), columns.index('roi_max')
    evidence = [i for i, name in enumerate(names) if name != KEEP]
    if not evidence:
        return {'accepted': False, 'reason': 'no single/query sources'}
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
        return {'accepted': False, 'reason': 'recorded source ranks cannot be reconstructed from roi_max'}
    flipped = before_order != after_order
    # Propagate the SAME per-score error bound to a score difference. This does
    # not enlarge the allowed error of either score or any other feature.
    bound = NUMERIC_ATOL+NUMERIC_RTOL*np.abs(before)
    pair_bound = bound[:, None, evidence]+bound[:, :, None]
    ambiguous = (np.abs(before_gap) <= pair_bound) & (np.abs(after_gap) <= pair_bound)
    details = {
        'changed_rank_elements': int(np.count_nonzero(actual[..., rank_column] != expected[..., rank_column])),
        'flipped_comparisons': int(flipped.sum()),
        'flips_outside_score_bounds': int((flipped & ~ambiguous).sum()),
        'largest_flipped_score_gap': float(np.maximum(np.abs(before_gap), np.abs(after_gap))[flipped].max())
                                    if flipped.any() else 0.,
    }
    accepted = bool(ambiguous[flipped].all())
    return {'accepted': accepted, 'details': details,
            'reason': 'verified near-tie ranks' if accepted else 'source ordering changed outside score error bounds'}
