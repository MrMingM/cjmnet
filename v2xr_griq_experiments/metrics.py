"""Candidate calibration and frozen protocol summaries."""
from collections import Counter
from copy import deepcopy

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def binary(y, p):
    if not len(y):
        return dict(n=0)
    result = dict(n=len(y), positives=int(y.sum()), logloss=float(log_loss(y, p, labels=[0, 1])),
                  brier=float(brier_score_loss(y, p)))
    result.update(roc_auc=float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
                  pr_auc=float(average_precision_score(y, p)) if y.any() else None)
    result['calibration'] = []
    for i in range(10):
        mask = np.minimum((10*p).astype(int), 9) == i
        if mask.any():
            result['calibration'].append(dict(bin=i, count=int(mask.sum()),
                                              mean_score=float(p[mask].mean()), frequency=float(y[mask].mean())))
    return result


def combine(rows, global_sort=False):
    from opencood.utils.eval_utils_opv2v import calculate_ap

    result = {}
    for threshold in (.3, .5, .7):
        tp = [x for row in rows for x in row[threshold]['tp']]
        fp = [x for row in rows for x in row[threshold]['fp']]
        score = [x for row in rows for x in row[threshold]['score']]
        gt = sum(row[threshold]['gt'] for row in rows)
        ap = calculate_ap({threshold: dict(tp=tp.copy(), fp=fp.copy(), gt=gt, score=score)},
                          threshold, global_sort)[0] if gt else None
        result[f'{threshold:.1f}'] = dict(ap=None if ap is None else float(ap), tp=sum(tp),
                                         fp=sum(fp), fn=gt-sum(tp), gt=gt)
    return result


def scene_delta(rows, base, draws=300):
    """Resample paired within-scene contiguous blocks; report each scene too."""
    from opencood.utils.eval_utils_opv2v import calculate_ap
    rng = np.random.default_rng(20261017)
    blocks = {}
    by_scene = {}
    for scene in dict.fromkeys(r['scene'] for r in rows):
        indices = np.flatnonzero([r['scene'] == scene for r in rows])
        blocks[scene] = [indices[i:i+16] for i in range(0, len(indices), 16)]
        by_scene[scene] = indices

    def ap(items, indices):
        t = .7
        s = [items[i]['F'][t] for i in indices]
        stat = {k: [x for r in s for x in r[k]] for k in ('tp','fp','score')}
        stat['gt'] = sum(r['gt'] for r in s)
        return calculate_ap({t: stat}, t, False)[0] if stat['gt'] else np.nan

    def diff(indices):
        return 100 * (ap(rows, indices) - ap(base, indices))

    draws_pp = []
    for _ in range(draws):
        sample = np.concatenate([chunks[j] for chunks in blocks.values()
                                 for j in rng.integers(len(chunks), size=len(chunks))])
        draws_pp.append(diff(sample))
    finite = np.asarray(draws_pp)[np.isfinite(draws_pp)]
    per_scene = {s: float(d) if np.isfinite(d) else None
                 for s, ids in by_scene.items() for d in [diff(ids)]}
    return dict(block_frames=16, replicates=len(finite),
                pp_95=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
                scene_pp=per_scene)


def equal_fp_frontier(rows, base):
    """Evaluate the locked score at the baseline FP budget, never deploy this cutoff."""
    target = sum(sum(r['F'][.7]['fp']) for r in base)
    baseline = {(i, oid) for i,r in enumerate(base) for oid in r['hits']}
    entries = []
    for i,r in enumerate(rows):
        stat = r['F'][.7]
        # The traced insertion order is the actual native prediction score order.
        matched = iter(r['hits'])
        entries.extend((score, i, rank, next(matched) if tp else None)
                       for rank,(score,tp) in enumerate(zip(stat['score'],stat['tp'])))
    entries.sort(key=lambda r:(-r[0],r[1],r[2]))
    hits = set()
    fp = 0
    start = 0
    while start < len(entries):
        end = start+1
        while end < len(entries) and entries[end][0] == entries[start][0]:
            end += 1
        bucket = entries[start:end]
        nf = sum(x[3] is None for x in bucket)
        if fp+nf > target:
            break
        fp += nf
        hits.update((i, oid) for _,i,_,oid in bucket if oid is not None)
        start = end
    return dict(fp_budget=target, fp=fp, tp=len(hits), recovered=len(hits-baseline),
                lost=len(baseline-hits), offline_only=True)


def target_changes(rows, base):
    groups = {}
    for row, old in zip(rows, base):
        hit, original = set(row['hits']), set(old['hits'])
        for oid, g in row['observations'].items():
            distance = g['distance']
            for label in ('all', 'near' if distance < 30 else 'mid' if distance < 60 else 'far',
                          'radar_present' if g['radar'] else 'radar_absent',
                          'lidar_0' if g['lidar']==0 else 'lidar_1_5' if g['lidar']<=5
                          else 'lidar_6_20' if g['lidar']<=20 else 'lidar_21_plus'):
                c = groups.setdefault(label, Counter())
                c.update(gt=1, tp=int(oid in hit), recovered=int(oid in hit-original),
                         lost=int(oid in original-hit),
                         low_recovered=int(oid in hit-original and row['hits'][oid]['low']))
    return {k: dict(v) for k,v in groups.items()}
