"""Original labels are imported verbatim; new labels use four detection counts."""
from .common import KEEP
from local_fusion_action_utility_audit.counterfactual import (
    action_keys as original_keys, pairwise_preferences as original_pairs,
    outcome_ranks as original_ranks,
)


def detection_key(outcome):
    return (outcome['action_tp'], -outcome['lost'], -outcome['new_fp'], -outcome['action_fp'])


def keys(outcomes, names, simplify=True):
    # Reuse the original alignment/KEEP-first guard, independently of label values.
    original_keys(outcomes, names)
    return [detection_key(o) + ((int(n == KEEP),) if simplify else ())
            for o, n in zip(outcomes, names)]


def ordered_pairs(values):
    return [(i, j) if values[i] > values[j] else (j, i)
            for i in range(len(values)) for j in range(i + 1, len(values)) if values[i] != values[j]]


def detection_pairs(outcomes, names, simplify=True):
    return ordered_pairs(keys(outcomes, names, simplify))


def detection_ranks(outcomes, names):
    values = keys(outcomes, names, False)
    distinct = sorted(set(values))
    denominator = max(1, len(distinct) - 1)
    return [distinct.index(k) / denominator for k in values]


def category(o, keep):
    """Exclusive hierarchy. Full equivalence refers only to saved observations."""
    if o['action_tp'] > keep['action_tp']:
        return 'tp_count_increase'
    if o['action_tp'] < keep['action_tp']:
        return 'tp_count_decrease'
    if o['recovered'] or o['lost']:
        return 'gt_identity_change_same_tp_count'
    if o['action_fp'] != keep['action_fp'] or o['new_fp']:
        return 'fp_change_same_tp_and_gt_ids'
    if o['focal_detected_after'] != keep['focal_detected_after']:
        return 'focal_assignment_change'
    if any(o[k] != keep[k] for k in ('focal_score_after', 'focal_iou_after')):
        return 'score_or_localization_only'
    return 'equivalent_saved_outcome'


def focal_only_pair(a, b):
    from local_fusion_action_utility_audit.counterfactual import path_independent_action_key
    ka = path_independent_action_key(a, KEEP, False)
    kb = path_independent_action_key(b, KEEP, False)
    return ka[:-2] == kb[:-2] and ka[-2:] != kb[-2:]
