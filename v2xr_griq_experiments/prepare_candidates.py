"""Extract one common pre-NMS cache with frozen F0."""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from v2xr_griq_experiments.dataset import (
    CAPS, F0, FAILED_TRAIN_KEY, FLOORS, OUT, TRAIN_PER_SCENE, WEATHERS, accepted_frame_contracts,
    bind, import_runtime, paths, pool_mask, read, sha, smoke_root,
)
from v2xr_griq_experiments.features import check_voxel_alignment, extract
from v2xr_griq_experiments.native import replay


def sample_keys(keys):
    by_scene = {}
    for i, key in enumerate(keys):
        by_scene.setdefault(key[0], []).append(i)
    return sorted(i for scene in by_scene.values()
                  for i in np.asarray(scene)[np.linspace(0, len(scene)-1, min(TRAIN_PER_SCENE, len(scene)), dtype=int)])


def check_predictions(pred, score, old_boxes, old_scores, tag):
    if (pred is None) != (old_boxes is None):
        raise ValueError(f'None state differs {tag}')
    if pred is None:
        return 0., 0.
    boxes, scores = np.asarray(pred.cpu()).reshape(-1, 8, 3), np.asarray(score.cpu())
    old_boxes, old_scores = np.asarray(old_boxes).reshape(-1, 8, 3), np.asarray(old_scores)
    if boxes.shape != old_boxes.shape or scores.shape != old_scores.shape:
        raise ValueError(f'Detection count differs {tag}: {len(boxes)} versus {len(old_boxes)}')
    if not len(boxes):
        return 0., 0.
    # Score ties may reorder different boxes. Match geometries and scores, then
    # separately require the official per-frame TP/FP traces in the audit.
    for b, s in zip(boxes, scores):
        if not np.any((np.max(np.abs(old_boxes-b), axis=(1, 2)) <= 1e-3) &
                      (np.abs(old_scores-s) <= 2e-4)):
            raise ValueError(f'F0 detection geometry or score differs {tag}')
    return float(np.max(np.min(np.max(np.abs(boxes[:, None]-old_boxes), axis=(2, 3)), axis=1))), float(
        np.max(np.abs(np.sort(scores)-np.sort(old_scores))))


def check_weather_gt(fgt, ogt, split, key):
    clean = {str(r['object_id']): r for r in fgt}
    weather = {str(r['object_id']): r for r in ogt}
    if len(clean) != len(fgt) or len(weather) != len(ogt):
        raise ValueError(f'Duplicate GT IDs: {key}')
    extra = sorted(weather.keys() - clean.keys())
    if split == 'validate' and extra:
        raise ValueError(f'Formal weather GT IDs do not align: {key}: {extra}')
    for oid in clean.keys() & weather.keys():
        if not np.allclose(clean[oid]['corners'], weather[oid]['corners'], rtol=0, atol=1e-4):
            raise ValueError(f'GT corner mismatch for {oid}: {key}')
    return extra


def collect(split, smoke):
    run = import_runtime()
    from context import load, frozen_clean
    from collect import dev_data, dev_gt, feature_context, inside, inspect_head, iou_matrix, native_post
    from diagnostics import input_context
    from checkpoint import seed
    from opencood.tools.train_utils import to_device
    from v2xr_weather_l012.data import tensor_digest

    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError('Extraction requires one visible ROCm GPU')
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * 2**30:
        raise RuntimeError('Extraction needs at least 6 GiB free VRAM')
    model, config, weight = load(run, 'F0')
    model.eval().requires_grad_(False)
    before = tensor_digest(model.state_dict())
    root = smoke_root() if smoke else OUT
    contract = sha(OUT / 'CONTRACT.json')
    accepted = accepted_frame_contracts(root, split)
    conditions = WEATHERS if split == 'validate' else ('clean', 'fog')
    seen = []
    archived = read(F0/'final_evaluation/prediction_manifest.json')['files'] if split=='validate' else []
    archived_hashes = {str(item['path']):item['sha256'] for item in archived}
    for weather in conditions:
        if split == 'train':
            seed(20261016)
            data, _ = dev_data(config, weather)
            chosen = sample_keys(data.keys)
        else:
            data, _ = run.build(config, condition=weather, visualize=True)
            chosen = list(range(len(data)))
        if smoke:
            chosen = chosen[:3]
            if split == 'train':
                chosen = sorted(set(chosen) | {data.keys.index(FAILED_TRAIN_KEY)})
        pp = data.native.post_processor
        threshold = float(pp.params['target_args']['score_threshold'])
        if not data.native.proj_first or pp.params['order'] != 'hwl' or threshold <= max(FLOORS):
            raise ValueError('Unexpected native coordinate/box/threshold convention')
        loader = DataLoader(Subset(data, chosen), batch_size=1, num_workers=16 if split == 'validate' else 6,
                            collate_fn=data.native.collate_batch_test)
        for index, batch in zip(chosen, loader):
            key = data.keys[index]
            path, meta = paths(root, split, weather, index)
            if path.exists() and meta.exists():
                old = read(meta)
                if old['key'] != key or old['contract'] not in accepted or old['npz_sha256'] != sha(path):
                    raise ValueError(f'Committed frame changed {path}')
                seen.append([weather, int(index), key])
                continue
            device = to_device(batch, torch.device('cuda:0'))
            ego = device['ego']
            with torch.inference_mode():
                out = model(ego)
                prob, decoded, corners = inspect_head(ego, out, pp)
                logit = out['psm'].permute(0, 2, 3, 1).reshape(-1).cpu().numpy()
                p, d, c = (x.cpu().numpy() for x in (prob, decoded, corners))
                t = ego['transformation_matrix'].cpu().numpy()
                anchor = ego['anchor_box'].reshape(-1, 7).cpu().numpy()
                if len(anchor) != len(p) or not np.allclose(t, np.eye(4), atol=1e-6):
                    raise ValueError('Candidate geometry not in ego coordinates')
                if not np.isfinite(p).all():
                    raise ValueError('Nonfinite frozen classification output')
                native, scores, _ = native_post(device, out, pp, threshold)
                if split == 'validate':
                    archive_file = F0 / 'final_evaluation/predictions' / weather / f'{index:06}.json'
                    if sha(archive_file) != archived_hashes.get(str(archive_file)):
                        raise ValueError(f'Formal F0 prediction artifact changed: {archive_file}')
                    archive = read(archive_file)
                    if archive['frame_key'] != key:
                        raise ValueError('Validation archive time identity differs')
                    fgt, ogt = frozen_clean(index, key), archive['official_gt']
                    archived_error = check_predictions(native, scores, archive['prediction_boxes'],
                                                       archive['scores'], (weather, key))
                elif weather == 'clean':
                    fgt = ogt = dev_gt(pp, device)
                    archived_error = None
                else:
                    _, clean_meta = paths(root, split, 'clean', index)
                    clean_record = read(clean_meta)
                    if clean_record['key'] != key:
                        raise ValueError(f'Clean/Fog training frame key mismatch: {key}')
                    fgt, ogt = clean_record['fgt'], dev_gt(pp, device)
                    archived_error = None
                extra = check_weather_gt(fgt, ogt, split, key)
                if extra:
                    print('TRAIN WEATHER-ONLY GT', key, extra, flush=True)
                rank = np.empty(len(p), np.int32)
                rank[np.argsort(-p, kind='stable')] = np.arange(len(p))
                ids = np.flatnonzero(pool_mask(p, rank, threshold, min(FLOORS), max(CAPS)))
                if not np.isfinite(d[ids]).all() or (d[ids, 3:6] <= 0).any():
                    raise ValueError('Invalid native geometry inside shared pool')
                rboxes, rscores, stage, final = replay(pp, d[ids], logit[ids], t, threshold, trace=True)
                replay_error = check_predictions(rboxes, rscores,
                                                 None if native is None else native.cpu().numpy(),
                                                 None if scores is None else scores.cpu().numpy(), (split, weather, key))
                ctx = feature_context(batch['ego'], model)
                check_voxel_alignment(batch['ego'], ctx)
                x = extract(d[ids], p[ids], logit[ids], ctx, inside)
                ious = iou_matrix(c[ids], [g['corners'] for g in fgt])
                if len(fgt):
                    best = ious.argmax(1)
                    values = ious[np.arange(len(ids)), best]
                    best[values == 0] = -1
                else:
                    best = np.full(len(ids), -1)
                    values = np.zeros(len(ids))
                coverage = {}
                for floor in FLOORS:
                    for cap in CAPS:
                        keep = pool_mask(p[ids], rank[ids], threshold, floor, cap)
                        coverage[f'{floor}/{cap}'] = [int(keep.sum()), *[
                            int((ious[keep] >= j).any(0).sum()) for j in (.3, .5, .7)]]
                observed = input_context(batch, fgt)['objects']
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, ids=ids.astype(np.int32), x=x, decoded=d[ids], corners=c[ids],
                                prob=p[ids], logit=logit[ids], anchor=anchor[ids], stage=stage,
                                y30=values>=.3, y50=values>=.5, y70=values>=.7,
                                raw_rank=rank[ids], iou=values, gt_index=best, transform=t,
                                native_threshold=np.float32(threshold), final_ids=ids[final])
            tmp.replace(path)
            bind(meta, dict(contract=contract, key=key, weather=weather, split=split, index=int(index),
                            npz_sha256=sha(path), fgt=fgt, ogt=ogt, observations=observed,
                            coverage=coverage, candidates=len(ids), raw_candidates=len(p),
                            native_none=native is None, replay_error=replay_error,
                            archive_error=archived_error, weather_only_gt=extra))
            seen.append([weather, int(index), key])
            if (len(seen) % 50 == 0) or smoke:
                print('FRAME', split, weather, index, 'candidates', len(ids), 'replay', replay_error, flush=True)
        print('WEATHER COMPLETE', split, weather, len(chosen), flush=True)
    if before != tensor_digest(model.state_dict()) or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen parameters or BatchNorm buffers changed')
    bind(root / f'{split}_COMPLETE.json', dict(weight=weight, state_digest=before, contract=contract, frames=seen))


if __name__ == '__main__':
    collect(sys.argv[1], '--smoke' in sys.argv)
