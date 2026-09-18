"""CPU-only paired mechanism bookkeeping for the extended Stage-3B audit."""
from .stage3_analysis import STAGES
from .stage3_trace import describe_target, proposal


def suppression_kind(event, target):
    """Geometric relation, deliberately NOT a final TP/FP classification."""
    if event['suppressor']['gt_iou'] >= .7:
        return 'also_qualifies_target'
    if event['suppressor_best_gt_iou'] >= .7:
        return 'qualifies_other_gt'
    if event['suppressor_best_gt'] == target:
        return 'target_nearest_but_iou_below70'
    return 'no_iou70_other_or_background'


def watched_anchors(a_rows):
    result = {}
    for j, row in a_rows.items():
        ids = set()
        for peer in row['peers']:
            ids.add(peer['trace']['matched_candidate_id'])
        for stage in row['full']['stages'].values():
            ids.update([stage['best_iou_candidate_id'], stage['highest_qualifying_score_candidate_id']])
        for e in row['full']['qualifying_nms_suppressions']:
            ids.update([e['suppressed']['candidate_id'], e['suppressor']['candidate_id']])
        result[j] = sorted(ids - {None})
    return result


def compact_targets(trace, targets, watched, threshold):
    rows = {}
    for j in targets:
        d = describe_target(trace, j)
        high = d['stages']['decoded']['highest_qualifying_score']
        d['qualifying_score_margin'] = None if high is None else high - threshold
        for e in d['qualifying_nms_suppressions']:
            e['geometric_relation'] = suppression_kind(e, j)
            e['score_gap'] = e['suppressor']['score'] - e['suppressed']['score']
        probes = {}
        for cid in watched[j]:
            p = proposal(trace, cid, j)
            p['logit'] = float(trace['logits'][cid])
            p['survives'] = {s: bool(cid in trace['ids'][s]) for s in STAGES}
            pair = trace['suppressors'].get(cid)
            p['suppressed_by'] = None if pair is None else int(pair[0])
            probes[str(cid)] = p
        d['watched_anchors'] = probes
        rows[str(j)] = d
    return rows


def compare_targets(before, after):
    rows = {}
    for j, b in before.items():
        a = after[j]
        bh = b['stages']['decoded']['highest_qualifying_score']
        ah = a['stages']['decoded']['highest_qualifying_score']
        pairs = {}
        for cid, bp in b['watched_anchors'].items():
            ap = a['watched_anchors'][cid]
            pairs[cid] = dict(score_delta=ap['score']-bp['score'],
                logit_delta=ap['logit']-bp['logit'], iou_delta=ap['gt_iou']-bp['gt_iou'],
                remains_iou70=bp['gt_iou'] >= .7 and ap['gt_iou'] >= .7,
                score_decreases=ap['score'] < bp['score'])
        rows[j] = dict(before=b['failure_stage'], after=a['failure_stage'],
            lost=b['matched'] and not a['matched'], recovered=a['matched'] and not b['matched'],
            highest_qualifying_score_delta=None if ah is None or bh is None else ah-bh,
            anchor_changes=pairs)
    return rows


def subset_edges(branches):
    """Every one-source addition, with the other sources held fixed."""
    by_set = {tuple(r['subset']): r for r in branches if r['family'] == 'subset'}
    edges = []
    for small, b in by_set.items():
        for large, a in by_set.items():
            added = set(large)-set(small)
            if set(small) < set(large) and len(added) == 1:
                edges.append(dict(before=list(small), after=list(large), added_source=added.pop(),
                                  targets=compare_targets(b['targets'], a['targets'])))
    return edges

