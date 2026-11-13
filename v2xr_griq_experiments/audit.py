"""Verify complete shared data and choose a candidate policy using train-only scenes."""
import numpy as np

from v2xr_griq_experiments.dataset import (
    DATA, FAILED_TRAIN_KEY, FLOORS, CAPS, OUT, WEATHERS, accepted_frame_contracts,
    bind, load, partitions, paths, read, sha, smoke_root,
)
from v2xr_griq_experiments.features import FEATURES


def audit(root, smoke=False):
    contract = sha(OUT / 'CONTRACT.json')
    train_source = read(DATA / 'v2xr-official-train-v2/manifest.json')['scenes']
    split = partitions([(r['scene'], None) for r in train_source])
    if smoke:
        marker = read(root / 'train_COMPLETE.json')
        scenes = sorted({k[0] for w, i, k in marker['frames']})
        split = dict(fit=scenes, select=scenes, calibrate=scenes)
    records, counts = [], {}
    totals = {f'{f}/{c}': np.zeros(4, dtype=np.int64) for f in FLOORS for c in CAPS}
    errors = {w: dict(frames=0, replay=[0., 0.], archive=[0., 0.]) for w in WEATHERS}
    identities = {}
    weather_only = []
    reused = 0
    for category in ('train', 'validate'):
        accepted = accepted_frame_contracts(root, category)
        done = read(root / f'{category}_COMPLETE.json')
        if done['contract'] != contract:
            raise ValueError('Shared extraction contract differs')
        identities[category] = sha(root / f'{category}_COMPLETE.json')
        counts[category] = {}
        for w, i, key in done['frames']:
            f, m = paths(root, category, w, i)
            row = read(m)
            if row['key'] != key or row['contract'] not in accepted or sha(f) != row['npz_sha256']:
                raise ValueError(f'Candidate or frame manifest differs {f}')
            reused += row['contract'] != contract
            if category == 'train' and row.get('weather_only_gt'):
                weather_only.append(dict(key=key, weather=w, ids=row['weather_only_gt']))
            counts[category][w] = counts[category].get(w, 0) + 1
            if category == 'validate':
                err = errors[w]
                err['frames'] += 1
                err['replay'] = np.maximum(err['replay'], row['replay_error']).tolist()
                err['archive'] = np.maximum(err['archive'], row['archive_error']).tolist()
            elif key[0] in split['select']:
                for tag, result in row['coverage'].items():
                    totals[tag] += result
            records.append(dict(split=category, weather=w, scene=key[0], index=i,
                                npz=str(f), meta=str(m), npz_sha256=row['npz_sha256'], meta_sha256=sha(m)))
    if smoke and not any(r['key'] == FAILED_TRAIN_KEY and '247' in r['ids'] for r in weather_only):
        raise ValueError('Known Fog-only training target regression was not reproduced')
    if not smoke:
        if counts['validate'] != dict.fromkeys(WEATHERS, 829) or len({r['scene'] for r in records if r['split']=='validate'}) != 6:
            raise ValueError('Formal set incomplete')
        train = {r['scene'] for r in records if r['split']=='train'}
        val = {r['scene'] for r in records if r['split']=='validate'}
        if train & val or train != set(sum(split.values(), [])) or counts['train']['clean'] != counts['train']['fog']:
            raise ValueError('Unexpected training and validation scene overlap/count')
    maximum = totals[f'{min(FLOORS)}/{max(CAPS)}']
    options = [(int(totals[f'{f}/{c}'][0]), f, c) for f in FLOORS for c in CAPS
               if np.all(totals[f'{f}/{c}'][1:] >= .995 * maximum[1:])]
    _, floor, cap = min(options)
    policy = dict(floor=floor, cap=cap, selection='Minimum count retaining 99.5% of widest observed coverage at all IoUs on selection scenes')
    fixed = {w: sum(len(read(row['meta'])['fgt']) for row in records if row['split']=='validate' and row['weather']==w)
             for w in WEATHERS}
    if not smoke and fixed != dict.fromkeys(WEATHERS, 6235):
        raise ValueError(f'Fixed Clean targets differ: {fixed}')
    for row in records:
        d = load(row, policy)
        n = len(d['ids'])
        if (len(set(d['ids'])) != n or d['x'].shape != (n, len(FEATURES)) or
                not np.isfinite(d['iou']).all() or np.isinf(d['x']).any() or
                not np.isin(d['final_ids'], d['ids']).all() or np.any(d['gt_index'] >= len(d['meta']['fgt']))):
            raise ValueError(f'Candidate and supervision misalignment {row["npz"]}')
    bind(root / 'BASELINE_REPLAY.json', dict(max_errors=errors, fixed_counts=fixed, smoke=smoke))
    bind(root / 'DATA_AUDIT.json', dict(contract=contract, feature_schema=list(FEATURES),
                                      identities=identities, counts=counts, split=split,
                                      policy=policy, coverage={k:v.tolist() for k,v in totals.items()},
                                      fixed=fixed, files=records, reused_frames=reused,
                                      training_weather_only_gt=weather_only))
    print('AUDITED', counts, policy, flush=True)


if __name__ == '__main__':
    import sys
    smoke = '--smoke' in sys.argv
    audit(smoke_root() if smoke else OUT, smoke)
