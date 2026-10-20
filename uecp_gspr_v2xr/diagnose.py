"""E1: frozen real GSPR outputs on train scenes; optional strictly sourced quality labels."""
import io
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import atomic_bytes, bound, commit, complete, identity, read, sha, stream_seed, write
from .data_adapter import Dataset
from .quality_adapter import load_quality, aggregate, regional
from .diagnostic_statistics import contrasts, distribution
from .point_origin import resolve_neighbors, aggregate_labels


def point_labels(meta, entry, points, coords, valid, tolerance):
    """Nearest coordinates identify recorded origins; unmatched/ambiguous stays unknown."""
    from scipy.spatial import cKDTree
    raw = np.asarray(meta['raw_original'])
    with np.load(entry['path'], allow_pickle=False) as data:
        labels = data['labels']
    if labels.shape != (len(raw),) or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError('Simulator-origin labels must be one -1/0/1 per original current-weather PCD row')
    distance, index = cKDTree(raw[:, :3]).query(points[valid, :3], k=2, distance_upper_bound=tolerance)
    result = resolve_neighbors(labels, distance, index)
    # Coordinates are already local source voxels. Region is exactly the 2x downsampled grid.
    return aggregate_labels(result, coords, valid)


def run(output, smoke=False):
    root = Path(output); c = read(root / 'config.json')
    dest = root / ('smoke' if smoke else '') / 'E1/quality_diagnosis'
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / 'complete.json'
    if marker.exists():
        return complete(marker)
    quality = load_quality(c).cuda().eval()
    labels_path = root / 'preflight/quality_labels.json'
    labels = read(labels_path)['files'] if labels_path.exists() else {}
    bound(dest / 'protocol.json', dict(gspr=c['gspr'], diagnosis=c['diagnosis'],
        target='simulator-recorded point origins only; unmatched paired Clean points never become noise labels',
        quality_labels=bool(labels), additional_training='none; frozen historical supervised OPV2V checkpoint',
        region_sampling='up to256 occupied 2x2 voxel regions per CAV/frame/weather, fixed independent stream',
        decision='scene-macro A-C Brier gain lower95%>0 and mean>=.001; A-D mean<=.0005; upper95%<=0 -> NOT_SUPPORTED; otherwise INCONCLUSIVE'))
    shards = []
    for split in ('train_core', 'train_dev'):
        dataset = Dataset(root, split=split, epoch=0)
        indices = dataset.indices[:c['smoke_frames']] if smoke else dataset.indices
        for index in indices:
            for w, weather in enumerate(('clean', 'fog', 'snow')):
                key = identity([split, dataset.keys[index], weather])
                path = dest / 'regions' / (key + '.npz')
                journal = path.with_suffix('.json')
                if journal.exists():
                    complete(journal); shards.append((split, path)); continue
                item = dataset.get_global(index, weather)
                data = dataset.model_inputs([item])
                p = {k: v.cuda() for k, v in data['processed_lidar'].items()}
                empty = data['empty_voxels'].cuda()
                with torch.no_grad():
                    values = quality(p['voxel_features'], p['voxel_num_points'], p['voxel_coords'])
                    grid = aggregate(values, p, len(item['_local']), empty)
                    q, observed = regional(grid, (100, 352))
                q = q.cpu().numpy(); observed = observed.cpu().numpy()[:, 0]
                all_x, all_y, knowns, slots, intensity = [], [], [], [], []
                pc = data['processed_lidar']
                for cav, meta in enumerate(item['_local']):
                    count = F.avg_pool2d(torch.from_numpy(meta['density'])[None, None].float(), 2, 2)[0, 0].numpy()*4
                    cov = F.avg_pool2d(torch.from_numpy(1-meta['target'])[None, None], 2, 2)[0, 0].numpy()
                    ids = np.flatnonzero(observed[cav])
                    rng = np.random.RandomState(stream_seed(c['seed'], 0, [meta['uid'], meta['cav'], weather], 'diagnosis_regions'))
                    ids = rng.permutation(ids)[:c['diagnosis']['regions_per_cav']]
                    features = np.c_[np.log1p(count.flatten()[ids]), cov.flatten()[ids], q[cav].reshape(4, -1)[:, ids].T]
                    target = np.full(len(ids), np.nan, np.float32); known = np.zeros(len(ids)); denominator = np.zeros(len(ids))
                    select = pc['voxel_coords'][:, 0].numpy() == cav
                    v = values['point_valid_mask'][select].cpu().numpy() & ~empty[select].cpu().numpy()[:, None]
                    vf = pc['voxel_features'][select].numpy(); coords = pc['voxel_coords'][select].numpy()
                    flat = (coords[:, 2]//2)*352+coords[:, 3]//2
                    total = np.bincount(flat, weights=v.sum(1), minlength=100*352)
                    denominator = total[ids]
                    if meta['relative'] in labels and len(meta['raw_original']):
                        counts, sums = point_labels(meta, labels[meta['relative']], vf, coords, v,
                            c['diagnosis']['point_match_tolerance_m'])
                        known = counts.flatten()[ids]
                        fraction = sums.flatten()[ids]/np.maximum(known, 1)
                        eligible = (known >= 4) & (known/np.maximum(denominator, 1) >= .9)
                        target[eligible] = fraction[eligible]
                    all_x.append(features); all_y.append(target); knowns.append(known); slots.append(denominator)
                    intensity.append(vf[..., 3][v])
                x = np.concatenate(all_x); y = np.concatenate(all_y)
                raw_i = np.concatenate(intensity)
                blob = io.BytesIO()
                np.savez_compressed(blob, x=x, y=y, known=np.concatenate(knowns), slots=np.concatenate(slots),
                    intensity_summary=np.array([len(raw_i), raw_i.mean() if len(raw_i) else 0,
                        raw_i.min() if len(raw_i) else 0, raw_i.max() if len(raw_i) else 0]),
                    scene=np.full(len(x), meta['uid'][0]), weather=np.full(len(x), w))
                atomic_bytes(path, blob.getvalue())
                commit(journal, [path], parent_contract=read(root / 'contract.json')['identity'], split=split, weather=weather,
                    uid=dataset.keys[index], regions=len(x), labeled=int(np.isfinite(y).sum()))
                shards.append((split, path))
            if len(shards) % 30 == 0:
                print('E1 extracted frame', dataset.keys[index], 'shards', len(shards), flush=True)
    collected = {}
    for split in ('train_core', 'train_dev'):
        arrays = {k: [] for k in ('x', 'y', 'known', 'slots', 'scene', 'weather')}
        intensity_stats = []
        for s, path in shards:
            if s != split:
                continue
            with np.load(path, allow_pickle=False) as data:
                for k in arrays:
                    arrays[k].append(data[k])
                intensity_stats.append(data['intensity_summary'].tolist())
        collected[split] = {k: np.concatenate(v) for k, v in arrays.items()}
        collected[split]['intensity_summary_per_cav_frame'] = intensity_stats
    stats, labeled = {}, {}
    for split, data in collected.items():
        x = data['x']; bins = np.floor(np.log2(np.maximum(np.expm1(x[:, 0]), 1))).astype(int)
        strata = []
        for w in range(3):
            for b in np.unique(bins[data['weather'] == w]):
                mask = (data['weather'] == w) & (bins == b)
                strata.append(dict(weather=('clean', 'fog', 'snow')[w], density_bin=int(b), statistics=distribution(x[mask])))
        valid = np.isfinite(data['y'])
        calibration = []
        for w in range(3):
            for b in range(10):
                selected = valid & (data['weather'] == w) & (np.minimum((x[:, 2]*10).astype(int), 9) == b)
                if selected.any():
                    calibration.append(dict(weather=('clean', 'fog', 'snow')[w], reliability_bin=b,
                        n=int(selected.sum()), mean_reliability=float(x[selected, 2].mean()),
                        mean_recorded_quality=float(data['y'][selected].mean()),
                        brier=float(((x[selected, 2]-data['y'][selected])**2).mean())))
        stats[split] = dict(all=distribution(x), strata=strata, regions=len(x), labeled_regions=int(valid.sum()),
            sampled_point_label_coverage=float(data['known'].sum()/max(data['slots'].sum(), 1)),
            labeled_region_fraction=float(valid.mean()) if len(valid) else 0.,
            intensity_summary_per_cav_frame=data['intensity_summary_per_cav_frame'],
            raw_quality_calibration=calibration if calibration else None,
            calibration_status='measured on provenance-bound labels' if calibration else 'unverified; no sufficiently covered trustworthy labels',
            saturation_fraction=float((x[:, 2] > .99).mean()) if len(x) else None)
        labeled[split] = {k: data[k][valid] for k in ('x', 'y', 'scene', 'weather')}
    if set(labeled['train_core']['scene']) & set(labeled['train_dev']['scene']):
        raise ValueError('Quality fit/diagnostic scene leakage')
    result = contrasts(labeled['train_core'], labeled['train_dev'], c['diagnosis'], c['seed'])
    write(dest / 'feature_statistics.json', dict(columns=['log1p_current_point_count', 'density_coverage',
        'mean_reliability', 'low_reliability_fraction', 'mean_uncertainty', 'mean_total_evidence'], splits=stats))
    write(dest / 'ABCD.json', result)
    write(dest / 'shard_manifest.json', [dict(path=str(p), sha256=sha(p), commit_sha256=sha(p.with_suffix('.json'))) for _, p in shards])
    text = '# E1 quality diagnosis\n\nConclusion: '+result['conclusion']+'\n\n'
    text += ('Recorded point-origin labels unavailable/insufficient. No independent noise classification claim.\n'
             if result['conclusion'] == 'INCONCLUSIVE' else 'See ABCD.json for pre-registered scene-bootstrap decision.\n')
    text += '\nFrozen supervised OPV2V GSPR is cross-domain. E2/E3 remain exploratory if E1 is inconclusive.\n'
    atomic_bytes(dest / 'RESULTS.md', text.encode())
    return commit(marker, [dest / n for n in ('protocol.json', 'feature_statistics.json', 'ABCD.json', 'shard_manifest.json', 'RESULTS.md')],
        conclusion=result['conclusion'], smoke_only=smoke, checkpoint_sha256=c['gspr']['checkpoint_sha256'],
        quality_model_compatible=True, extra_quality_training=False)
