"""Label/evaluation boundary; every trial receives the immutable Shared state."""
from .common import KEEP


def path_independent_action_key(outcome, source, simplicity=True):
    key = (outcome['action_tp'], -outcome['lost'], -outcome['new_fp'],
           int(outcome['focal_detected_after'] or False), outcome['recovered'],
           -outcome['action_fp'],
           outcome['focal_score_after'] if outcome['focal_score_after'] is not None else 0.,
           outcome['focal_iou_after'] if outcome['focal_iou_after'] is not None else 0.)
    return key + (int(source == KEEP),) if simplicity else key


def action_keys(outcomes, names, simplicity=True):
    if not names or names[0] != KEEP or len(set(names)) != len(names) or len(outcomes) != len(names):
        raise ValueError('Expected one outcome per unique source, KEEP first')
    return [path_independent_action_key(row, name, simplicity) for row, name in zip(outcomes, names)]


def pairwise_preferences(outcomes, names):
    keys = action_keys(outcomes, names)
    return [(i, j) if keys[i] > keys[j] else (j, i)
            for i in range(len(keys)) for j in range(i + 1, len(keys)) if keys[i] != keys[j]]


def outcome_ranks(outcomes, names):
    keys = action_keys(outcomes, names, simplicity=False)
    ordered = sorted(set(keys))
    denominator = max(1, len(ordered) - 1)
    return [ordered.index(key) / denominator for key in keys]


def outcome_state(post, threshold):
    from local_fusion_utility_v2.outcomes import detection_outcome
    matched, fp = detection_outcome(*post, threshold)
    return {'matched': matched, 'fp': fp, 'post': post}


def structured_outcome(baseline, action, focal, spec):
    from local_fusion_utility_v2.outcomes import count_new_false_positives
    from local_fusion_task_source_oracle.oracle import _focal_quality
    row = dict(baseline_tp=len(baseline['matched']), action_tp=len(action['matched']),
               recovered=len(action['matched'] - baseline['matched']),
               lost=len(baseline['matched'] - action['matched']),
               baseline_fp=len(baseline['fp']), action_fp=len(action['fp']),
               new_fp=count_new_false_positives(action['fp'], baseline['fp'], spec['fp_identity_iou']))
    for suffix, state in (('before', baseline), ('after', action)):
        quality = _focal_quality(state['post'], focal, spec['target_iou']) if focal is not None else None
        row['focal_detected_' + suffix] = focal in state['matched'] if focal is not None else None
        row['focal_score_' + suffix] = float(quality['score']) if quality is not None else None
        row['focal_iou_' + suffix] = float(quality['iou']) if quality is not None else None
    return row


def independent_prediction(shared, pool, mask, task, source):
    from local_fusion_task_source_oracle.oracle import compose_prediction
    if task not in ('cls', 'reg'):
        raise ValueError('Unknown task')
    return compose_prediction(shared, pool, mask,
                              source if task == 'cls' else KEEP,
                              source if task == 'reg' else KEEP)


def counterfactual_outcomes(dataset, batch, shared, pool, mask, names, baseline, focal, spec):
    """Exhaust all single-task sources; no greedy shortlist/hindsight source IDs."""
    result = {}
    keep = structured_outcome(baseline, baseline, focal, spec)
    for task in ('cls', 'reg'):
        rows = []
        for source in names:
            if source == KEEP:
                rows.append(dict(keep))
            else:
                prediction = independent_prediction(shared, pool, mask, task, source)
                post = dataset.post_process(batch, {'ego': prediction})
                rows.append(structured_outcome(baseline, outcome_state(post, spec['target_iou']), focal, spec))
        result[task] = rows
    return result
