"""Stage-3 pure CPU bookkeeping; no model, training or test-set access."""
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np
from qa_observation_diagnostic.metrics import distance_bin

STAGES = ('decoded', 'score', 'geometry', 'nms_topk', 'nms', 'range')
FAILURES = ('no_iou70_after_decode', 'score_filtered', 'geometry_filtered',
            'nms_topk_filtered', 'nms_suppressed', 'range_filtered',
            'matching_competition', 'other')
WEATHERS = ('fog', 'rain', 'snow')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def row_key(row):
    return int(row['sample_index']), int(row['target_index'])


def unique_rows(rows):
    out = {}
    for row in rows:
        key = row_key(row)
        if key in out:
            raise ValueError(f'Duplicate target-frame occurrence: {key}')
        out[key] = row
    return out


def rate(n, d):
    return n / d if d else None


def stratify(rows):
    """Rows include ALL source-valid cases, not just the full-miss numerator."""
    def aggregate(items):
        failures = [r for r in items if r['source_valid_full_miss']]
        return dict(source_valid_occurrences=len(items), full_miss_occurrences=len(failures),
                    failure_rate=rate(len(failures), len(items)),
                    unique_frames=len({r['sample_index'] for r in items}),
                    failure_unique_frames=len({r['sample_index'] for r in failures}))
    valid = [r for r in rows if r['any_peer_alone_detected']]
    scenes, distances = defaultdict(list), defaultdict(list)
    for row in valid:
        scenes[str(row['scene'])].append(row)
        distances[distance_bin(row['distance'])].append(row)
    by_scene = {}
    for scene, items in sorted(scenes.items()):
        nested = defaultdict(list)
        for row in items:
            nested[distance_bin(row['distance'])].append(row)
        by_scene[scene] = dict(aggregate(items), by_distance={k: aggregate(v) for k, v in nested.items()})
    return dict(unit='target-frame occurrence', total=aggregate(valid),
                by_scene=by_scene, by_distance={k: aggregate(v) for k, v in distances.items()})


def greedy_assignment(scores, ious, threshold=.7):
    """Same stable score order / remaining-GT argmax as Stage-2."""
    scores, ious = np.asarray(scores), np.asarray(ious)
    remaining = list(range(ious.shape[1]))
    assignment = np.full(len(scores), -1, dtype=np.int64)
    for pi in np.argsort(-scores, kind='stable'):
        if not remaining:
            break
        values = ious[pi, remaining]
        best = int(np.argmax(values))
        if float(values[best]) >= threshold:
            assignment[pi] = remaining.pop(best)
    return assignment


def traced_nms(scores, overlap, threshold, top=1000):
    """Exact numpy ordering/top-k/strict > of box_utils.nms_rotated.

    overlap(i, js) returns the SAME polygon IoU values as original NMS.
    Suppressor is the first actually selected box that removes this proposal.
    """
    order = np.asarray(scores).argsort()[::-1]
    pool = order[:top].copy()
    top_ids = pool.copy()
    suppressors, pick = {}, []
    while len(pool):
        i = int(pool[0])
        pick.append(i)
        values = overlap(i, pool[1:])
        if not np.isfinite(values).all():
            raise ValueError('Non-finite NMS overlap')
        removed = np.where(values > threshold)[0] + 1
        for pos in removed:
            suppressors[int(pool[pos])] = (i, float(values[pos-1]))
        pool = np.delete(pool, removed)
        pool = np.delete(pool, 0)
    return np.asarray(pick, dtype=np.int64), top_ids, suppressors


def target_summary(scores, target_ious, stage_ids, final_assignment, target_index):
    # Stage-2 compares float(np.float32_iou) to Python 0.7. In particular,
    # float32(0.7) is slightly BELOW 0.7; NumPy array comparison would round
    # the scalar down and incorrectly classify that boundary as qualified.
    scores, target_ious = np.asarray(scores), np.asarray(target_ious, dtype=np.float64)
    out = {}
    for stage in STAGES:
        ids = np.asarray(stage_ids[stage], dtype=np.int64)
        qualified = ids[target_ious[ids] >= .7]
        best = int(ids[np.argmax(target_ious[ids])]) if len(ids) else None
        high = int(qualified[np.argmax(scores[qualified])]) if len(qualified) else None
        out[stage] = dict(count=len(ids), qualifying_count=len(qualified),
                          best_iou_candidate_id=best,
                          best_iou=float(target_ious[best]) if best is not None else None,
                          score_at_best_iou=float(scores[best]) if best is not None else None,
                          highest_qualifying_score_candidate_id=high,
                          highest_qualifying_score=float(scores[high]) if high is not None else None)
    final = np.asarray(stage_ids['range'], dtype=np.int64)
    assignment = np.asarray(final_assignment)
    matched = final[assignment == target_index]
    competitors = [dict(candidate_id=int(cid), assigned_gt=int(gid))
                   for cid, gid in zip(final, assignment)
                   if target_ious[cid] >= .7 and gid != target_index]
    if len(matched):
        failure = 'detected'
    else:
        failure = next((FAILURES[i] for i, name in enumerate(STAGES)
                        if out[name]['qualifying_count'] == 0), 'matching_competition')
    return dict(stages=out, matched=bool(len(matched)),
                matched_candidate_id=int(matched[0]) if len(matched) else None,
                failure_stage=failure, matching_competitors=competitors)


def enumerate_subsets(n):
    if not 1 <= n <= 5:
        raise ValueError('Expected 1..5 CAVs')
    return [(0,) + tuple(i+1 for i, on in enumerate(bits) if on)
            for bits in product((False, True), repeat=n-1)]


def choose_frame_subset(rows):
    """One realizable subset per frame. Never union target-wise detections."""
    if not rows:
        raise ValueError('Empty subset enumeration')
    # New FPs are determined by a fixed one-to-one geometry match, not net count.
    chosen = min(rows, key=lambda r: (-len(r['recovered_candidates']),
                 len(r['lost_baseline_gt']), r['new_fp_count'], tuple(r['subset'])))
    pareto = []
    for r in rows:
        a = (len(r['recovered_candidates']), -len(r['lost_baseline_gt']), -r['new_fp_count'])
        if not any(all(x >= y for x, y in zip(
                (len(s['recovered_candidates']), -len(s['lost_baseline_gt']), -s['new_fp_count']), a))
                   and (len(s['recovered_candidates']), -len(s['lost_baseline_gt']), -s['new_fp_count']) != a
                   for s in rows):
            pareto.append(r['subset'])
    union = sorted({t for r in rows for t in r['recovered_candidates']})
    return dict(chosen=chosen, pareto_subsets=pareto,
                target_wise_recoverable=union,
                warning='Target-wise union is NOT a realizable frame output or AP.')
