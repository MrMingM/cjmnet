"""Read frozen prediction files and report F protocol counts."""
from collections import Counter
import json

import numpy as np

from common import read_json, sha256


def describe_scores(values):
    if not values:
        return dict(count=0, mean=None, quantiles=None, histogram=[0]*5)
    a = np.asarray(values)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite archived detection score')
    return dict(count=len(a), mean=float(a.mean()),
        quantiles=np.quantile(a, [.1, .5, .9]).tolist(),
        histogram=np.histogram(a, bins=[0, .2, .4, .6, .8, 1])[0].tolist())


def rows(path, weather):
    with path.open() as stream:
        return [r for line in stream if (r := json.loads(line))['condition'] == weather]


def summarize_f3(history):
    base = history / 'mechanism_review_20261111/F3'
    results, matches = {}, {}
    full = read_json(base / 'full/DIAGNOSTIC.json')
    off = read_json(base / 'radar_off/DIAGNOSTIC.json')
    for key in ('epoch', 'sha256', 'frames', 'fixed_gt', 'conditions'):
        if full['config'][key] != off['config'][key]:
            raise ValueError('F3 archived intervention uses different inputs')
    if (full['config']['epoch'], full['config']['frames']) != (10, 120):
        raise ValueError('Expected frozen F3 epoch 10 and 120 frames')
    for mode in ('full', 'radar_off'):
        out = base / mode
        diagnostic = full if mode == 'full' else off
        manifest = read_json(out / 'prediction_manifest.json')
        digests = {(r['condition'], r['index']):r['sha256'] for r in manifest['files']}
        target_rows = diagnostic['diagnostics']['fixed_target_rows']
        fog = rows(out/'per_frame_metrics.jsonl', 'fog')
        if [r['index'] for r in fog] != list(range(120)):
            raise ValueError('Archived Fog indices differ')
        score_groups = {k:[] for k in ('all','tp_damaged_radar','tp_remaining','fp')}
        counts = Counter()
        hit, ids = set(), set()
        for frame in fog:
            index = frame['index']
            p = out/'predictions/fog'/f'{index:06d}.json'
            if sha256(p) != digests[('fog',index)]:
                raise ValueError(f'Archived prediction bytes changed: {p}')
            pred = read_json(p)
            if frame['frame_key'] != pred['frame_key']:
                raise ValueError('Archived frame identity changed')
            scores = pred['scores'] or []
            if len(scores) != pred['prediction_count']:
                raise ValueError('Prediction count differs from score array')
            protocol = frame['protocols']['F']
            pairs = protocol.get('matched_pairs', {}).get('0.7', [])
            tp = {p['prediction_index']:p['object_id'] for p in pairs}
            if len(tp) != protocol['thresholds']['0.7']['tp']:
                raise ValueError('Duplicate prediction match')
            for pair in pairs:
                ident = f'fog/{index}/{pair["object_id"]}'
                subgroup = 'damaged_radar' if 'damaged|radar:True' in target_rows[ident]['bins'] else 'remaining'
                if subgroup == 'damaged_radar':
                    ids.add(ident)
                hit.add(ident)
                score_groups['tp_'+subgroup].append(scores[pair['prediction_index']])
            score_groups['all'].extend(scores)
            score_groups['fp'].extend(s for i,s in enumerate(scores) if i not in tp)
            counts.update({k:protocol['thresholds']['0.7'][k] for k in ('tp','fp','fn')})
        eligible = {key for key,row in target_rows.items() if key.startswith('fog/') and 'damaged|radar:True' in row['bins']}
        all_targets = {key for key in target_rows if key.startswith('fog/')}
        if (len(eligible), len(all_targets)) != (182,1347) or ids != hit & eligible:
            raise ValueError('Archived target groups differ')
        subgroups = {'damaged_radar':eligible, 'remaining':all_targets-eligible}
        for name, target_ids in subgroups.items():
            matches[(mode,name)] = hit & target_ids
        if len(score_groups['fp']) != counts['fp'] or len(hit) != counts['tp']:
            raise ValueError('Cached FP/TP differ from official replay')
        results[mode] = dict(global_counts=dict(counts, predictions=len(score_groups['all'])),
            targets={name:dict(targets=len(t), tp=len(hit & t), fn=len(t-hit)) for name,t in subgroups.items()},
            scores={k:describe_scores(v) for k,v in score_groups.items()})
    for group in ('damaged_radar','remaining'):
        a,b = (matches[(mode,group)] for mode in ('full','radar_off'))
        results[group+'_full_vs_off'] = dict(recovered=len(a-b), lost=len(b-a))
    return dict(results=results, source=str(base), score_edges=[0,.2,.4,.6,.8,1],
                protocol='F@0.7', fp_attribution='unmatched; global only')

