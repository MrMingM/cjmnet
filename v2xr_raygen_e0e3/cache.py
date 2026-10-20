"""Shared per-frame immutable ray arrays and per-Clean-target candidate indices, on data disk."""
import io
from pathlib import Path
import time

import numpy as np

from .common import atomic_bytes, bound, identity, progress, read, sha, write
from .geometry import candidate_indices, labels, nearest_returns, transform, windows


def path_for(root, index):
    return Path(root) / 'labels/samples' / f'{index:06d}.npz'


def load_sample(root, index):
    p = path_for(root, index)
    journal = read(p.with_suffix('.json'))
    if journal['sha256'] != sha(p):
        raise ValueError('Immutable sample cache changed: ' + str(p))
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def window_file(root, index, epoch):
    return Path(root) / 'labels/windows' / f'{epoch:02d}' / f'{index:06d}.npz'


def prepare_windows(root, indices=None, epochs=None):
    root = Path(root)
    c = read(root / 'config.json')
    manifest = read(root / 'preflight/train_manifest.json')
    uids = [[s['scene'], t, s['ego']] for s in manifest['scenes'] for t in s['timestamps']]
    selected = list(range(len(uids))) if indices is None else list(indices)
    epochs = list(range(1, c['epochs'] + 1)) if epochs is None else list(epochs)
    count, total, start = 0, len(selected) * len(epochs), time.monotonic()
    for index in selected:
        cache = None
        base_hash = sha(path_for(root, index).with_suffix('.json'))
        for epoch in epochs:
            path = window_file(root, index, epoch)
            journal = path.with_suffix('.json')
            expected = identity(dict(base=base_hash, config=c, uid=uids[index], epoch=epoch))
            if journal.exists():
                old = read(journal)
                if old['identity'] != expected or old['sha256'] != sha(path):
                    raise ValueError('Prepared window cache identity/bytes differ: ' + str(path))
            else:
                if cache is None:
                    cache = load_sample(root, index)
                ids, centers = windows(cache['boxes'], c['seed'], epoch, uids[index], c)
                fields = {k: [] for k in ('O_raw', 'E_raw', 'ego', 'peer')}
                counts = dict(O=0, E=0, U=0, ego=0, peer=0)
                for k, center in zip(ids, centers):
                    pick = cache['candidate_' + str(k)]
                    lab = labels(cache['points'][pick], cache['origins'][pick], cache['source'][pick],
                                 cache['nearest'][pick], center, c,
                                 dict(projection=cache['projection'], image_wh=cache['image_wh']))
                    for key in fields:
                        fields[key].append(np.packbits(lab[key].reshape(-1)))
                    for key in counts:
                        counts[key] += int(lab[key].sum())
                arrays = dict(ids=ids, centers=centers, **{k: np.asarray(v, dtype=np.uint8).reshape(len(ids), 1024) for k, v in fields.items()})
                stream = io.BytesIO(); np.savez_compressed(stream, **arrays)
                atomic_bytes(path, stream.getvalue())
                write(journal, dict(status='complete', identity=expected, sha256=sha(path),
                                    bytes=path.stat().st_size, counts=counts, windows=len(ids)))
            count += 1
            if count % 20 == 0 or count == total:
                elapsed = time.monotonic() - start
                progress(root / 'labels/window_progress.json', status='running', sample_uid=uids[index],
                         epoch=epoch, completed=count, total=total, seconds=elapsed)
                print(f'Window labels {count}/{total} uid={uids[index]} epoch={epoch} elapsed={elapsed:.1f}s', flush=True)
    if indices is None and epochs == list(range(1, c['epochs'] + 1)):
        write(root / 'labels/windows_complete.json', dict(status='complete', sample_epochs=total, seconds=time.monotonic()-start))


def load_windows(root, index, epoch):
    p = window_file(root, index, epoch)
    if sha(p) != read(p.with_suffix('.json'))['sha256']:
        raise ValueError('Prepared window bytes changed: ' + str(p))
    result = []
    with np.load(p, allow_pickle=False) as z:
        for j, (k, center) in enumerate(zip(z['ids'], z['centers'])):
            lab = {key: np.unpackbits(z[key][j], count=8192).reshape(8, 32, 32).astype(bool)
                   for key in ('O_raw', 'E_raw', 'ego', 'peer')}
            lab.update(O=lab['O_raw'] & ~lab['E_raw'], E=lab['E_raw'] & ~lab['O_raw'])
            lab.update(U=~(lab['O'] | lab['E']), x0=np.where(lab['O_raw'], 1., -1.).astype(np.float32))
            result.append(dict(center=center, target=lab, object_index=int(k)))
    return result


def rays_from_base(dataset):
    from opencood.utils.pcd_utils import mask_ego_points
    base = dataset.last_base
    ego = next(x for x in base.values() if x['ego'])
    ego_pose = np.asarray(ego['params']['lidar_pose'])
    points, origins, source, nearest = [], [], [], []
    cavs = []
    c = dataset.config
    limits = np.asarray(dataset.native.params['preprocess']['cav_lidar_range'])
    for cav, row in base.items():
        if np.linalg.norm(np.asarray(row['params']['lidar_pose'])[:2] - ego_pose[:2]) > 70:
            continue
        raw = np.asarray(row['lidar_np'])
        if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all():
            raise ValueError('Invalid paired Clean points: ' + cav)
        # De-duplicate before identifying the nearest recorded return; duplicate rows cannot change evidence.
        raw = np.unique(raw, axis=0)
        raw = raw[np.linalg.norm(raw[:, :3], axis=1) > c['rays']['endpoint_tolerance_m']]
        near = nearest_returns(raw[:, :3], np.zeros(3), c['rays']['angular_bin_rad'])
        indexed = np.concatenate([raw, near[:, None]], axis=1)
        indexed = mask_ego_points(indexed)
        keep = np.all((indexed[:, :3] > limits[:3]) & (indexed[:, :3] < limits[3:]), axis=1)
        indexed = indexed[keep]
        matrix = np.asarray(row['params']['transformation_matrix'])
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('Invalid current-frame sensor transform')
        indexed[:, :3] = transform(indexed[:, :3], matrix)
        indexed = dataset.native.mask_ego_fov_flag(row, indexed, ego['params'])
        # Restrict endpoints to the same ego feature support. No ray is extended behind its return.
        keep = np.all((indexed[:, :3] >= limits[:3]) & (indexed[:, :3] < limits[3:]), axis=1)
        indexed = indexed[keep]
        points.append(indexed[:, :3])
        origins.append(np.repeat(matrix[None, :3, 3], len(indexed), axis=0))
        source.append(np.full(len(indexed), 0 if row['ego'] else len(cavs) + 1, dtype=np.int16))
        nearest.append(indexed[:, 4].astype(bool))
        cavs.append(str(cav))
    projection = np.asarray(ego['params']['camera0']['intrinsic']) @ np.linalg.inv(
        np.asarray(ego['params']['c2e_transformation_matrix']))[:3, :4]
    return dict(points=np.concatenate(points).astype(np.float64), origins=np.concatenate(origins).astype(np.float64),
                source=np.concatenate(source), nearest=np.concatenate(nearest),
                projection=projection, image_wh=np.asarray(ego['camera0'].size)), cavs


def build(output, indices=None):
    from .data import Dataset
    root = Path(output)
    dataset = Dataset(root, auxiliary=False)
    c = dataset.config
    contract = read(root / 'contract.json')
    cache_identity = identity(dict(input=contract['inputs'], config=c, sources=contract['sources'],
                                   geometry=read(root / 'protocol.json')['geometry']))
    bound(root / 'labels/contract.json', dict(identity=cache_identity, scheme='per sample raw finite rays + candidates'))
    selected = list(range(len(dataset))) if indices is None else list(indices)
    entries, totals = [], dict(O=0, E=0, U=0, ego=0, peer=0, conflict=0)
    start = time.monotonic()
    for offset, index in enumerate(selected):
        p = path_for(root, index)
        record = p.with_suffix('.json')
        if record.exists():
            row = read(record)
            if row['cache_identity'] != cache_identity or row['uid'] != dataset.keys[index] or sha(p) != row['sha256']:
                raise ValueError('Existing ray cache identity/bytes differ')
        else:
            sample = dataset.get(index, force='clean')
            arrays, cavs = rays_from_base(dataset)
            e = sample['ego']
            boxes = e['object_bbx_center'][e['object_bbx_mask'] == 1].copy()
            arrays['boxes'] = boxes
            counts = {k: 0 for k in totals}
            for k, box in enumerate(boxes):
                center = np.array([box[0], box[1], -1.])
                candidate = candidate_indices(arrays['points'], arrays['origins'], center,
                                              c['window']['size'], c['window']['jitter'])
                arrays['candidate_' + str(k)] = candidate.astype(np.int32)
                lab = labels(arrays['points'][candidate], arrays['origins'][candidate],
                             arrays['source'][candidate], arrays['nearest'][candidate], center, c,
                             dict(projection=arrays['projection'], image_wh=arrays['image_wh']))
                for name in ('O', 'E', 'U', 'ego', 'peer'):
                    counts[name] += int(lab[name].sum())
                counts['conflict'] += int((lab['O_raw'] & lab['E_raw']).sum())
            stream = io.BytesIO()
            np.savez_compressed(stream, **arrays)
            atomic_bytes(p, stream.getvalue())
            row = dict(status='complete', index=index, uid=dataset.keys[index], cavs=cavs,
                       cache_identity=cache_identity, sha256=sha(p), bytes=p.stat().st_size,
                       rays=len(arrays['points']), targets=len(boxes), counts=counts,
                       target_ids=list(map(str, e['object_ids'])), clean_target_source='native train Clean final centers/IDs')
            write(record, row)
        entries.append(row)
        for k in totals:
            totals[k] += row['counts'][k]
        if offset % 10 == 0 or offset + 1 == len(selected):
            elapsed = time.monotonic() - start
            progress(root / 'labels/progress.json', status='running', sample_uid=row['uid'],
                     completed=offset + 1, total=len(selected), seconds=elapsed)
            print(f'Ray cache {offset+1}/{len(selected)} uid={row["uid"]} bytes={row["bytes"]} elapsed={elapsed:.1f}s', flush=True)
    if indices is None:
        missing = [k for k in ('O', 'E', 'U', 'ego', 'peer') if totals[k] == 0]
        report = dict(status='failed' if missing else 'complete', identity=cache_identity,
                      samples=len(entries), bytes=sum(r['bytes'] for r in entries), counts=totals,
                      missing_required_categories=missing, entries=entries,
                      label_scope='ray-sampling approximation; counts over unjittered target audit windows, not physical occupancy truth')
        write(root / 'labels/manifest.json', report)
        write(root / 'label_check.json', report)
        if missing:
            raise ValueError('Full labels lack necessary measured categories: ' + ', '.join(missing))
    return entries
