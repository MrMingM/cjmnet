"""Offline train only the original-author CIA-SSD IoU branch on official-train positives."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .extract import WEATHERS
from .model import make_head, patch_logits, weighted_smooth_l1


def load_positive_patches(candidate_root, patch_root):
    candidate_root, patch_root = Path(candidate_root).resolve(), Path(patch_root).resolve()
    meta = json.loads((candidate_root / 'candidate_audit.json').read_text(encoding='utf-8'))
    patches_meta = json.loads((patch_root / 'manifest.json').read_text(encoding='utf-8'))
    if meta.get('split') != 'train' or patches_meta['split'] != 'train':
        raise ValueError('IoU head training requires official-train extraction only')
    if patches_meta['arm_sha256'] != meta['arm_sha256']['F']:
        raise ValueError('Patch and candidate F checkpoints differ')
    digest = hashlib.sha256((candidate_root / 'candidate_audit.json').read_bytes()).hexdigest()
    if patches_meta['candidate_audit_sha256'] != digest:
        raise ValueError('Patch cache was extracted from another candidate audit')
    samples = [int(value) for value in meta['sample_indices']]
    scene_map = {int(key): int(value) for key, value in meta['scene_map'].items()}
    if set(samples) != set(scene_map):
        raise ValueError('Incomplete official-train scene map')
    blocks, targets, anchors, scenes, frames, weights = [], [], [], [], [], []
    channels = None
    for weather in WEATHERS:
        for sample in samples:
            path = patch_root / weather / f'F_patches_{sample}.npz'
            with np.load(path, allow_pickle=False) as packed:
                patch = np.asarray(packed['patch'], dtype=np.float32)
                positive = np.asarray(packed['positive'], dtype=bool)
                target = np.asarray(packed['target_iou3d'], dtype=np.float32)
                ids = np.asarray(packed['candidate_id'], dtype=np.int64)
            if (patch.ndim != 4 or patch.shape[2:] != (3, 3)
                    or len(patch) != len(positive) or len(ids) != len(patch)
                    or len(target) != len(patch)):
                raise ValueError(f'{path}: invalid patch dimensions')
            if channels is None:
                channels = patch.shape[1]
            elif channels != patch.shape[1]:
                raise ValueError('Detector feature width changed between frames')
            if (not np.isfinite(patch).all() or not np.isfinite(target[positive]).all()
                    or np.any((target[positive] < 0) | (target[positive] > 1))):
                raise ValueError(f'{path}: invalid feature or 3D IoU target')
            if not positive.any():
                continue
            count = int(positive.sum())
            blocks.append(patch[positive])
            targets.append(target[positive])
            anchors.append(ids[positive] % int(meta.get('anchor_count', 2)))
            scenes.extend([scene_map[sample]] * count)
            frames.extend([f'{weather}:{sample}'] * count)
            weights.extend([1. / count] * count)
    if not blocks:
        raise ValueError('No positive anchors in top256 official-train cache')
    return (np.concatenate(blocks), np.concatenate(targets),
            np.concatenate(anchors), np.asarray(scenes), np.asarray(frames), np.asarray(weights,
            dtype=np.float32), meta, patches_meta)


def _frame_batches(frames, selected, max_candidates, rng=None):
    """Keep each frame's positives intact so source-style 1/num_pos is exact."""
    groups = {name: selected[frames[selected] == name]
              for name in np.unique(frames[selected])}
    names = np.asarray(list(groups))
    if rng is not None:
        names = rng.permutation(names)
    pending, count = [], 0
    for name in names:
        group = groups[name]
        if pending and count + len(group) > max_candidates:
            yield np.concatenate(pending), len(pending)
            pending, count = [], 0
        pending.append(group)
        count += len(group)
    if pending:
        yield np.concatenate(pending), len(pending)


def run(args):
    import torch

    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.holdout_fraction < .5:
        raise ValueError('Invalid training schedule or official-train holdout fraction')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    patch, target, anchor, scene, frame, weight, meta, patch_meta = load_positive_patches(
        args.train_root, args.patch_root)
    scenes = np.unique(scene)
    if len(scenes) < 2:
        raise ValueError('Need at least two official-train scenes for internal holdout')
    rng = np.random.default_rng(args.seed)
    holdout_count = max(1, min(len(scenes)-1,
                               int(round(len(scenes)*args.holdout_fraction))))
    holdout_scenes = set(rng.permutation(scenes)[:holdout_count].tolist())
    holdout = np.isin(scene, list(holdout_scenes))
    fit_ids = np.flatnonzero(~holdout)
    holdout_ids = np.flatnonzero(holdout)
    if not len(fit_ids) or not len(holdout_ids):
        raise ValueError('Empty fit or train-held-out positive anchors')
    anchors = int(max(anchor.max()+1, 2))
    if anchors != 2:
        raise ValueError('CIA-SSD source head and frozen F require two anchors per cell')
    torch.manual_seed(args.seed)
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head = make_head(patch.shape[1], anchors).to(target_device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    output.mkdir(parents=True)
    history, best, best_epoch = [], float('inf'), None

    def batch_loss(ids, training):
        x = torch.as_tensor(patch[ids], device=target_device)
        a = torch.as_tensor(anchor[ids], device=target_device, dtype=torch.long)
        y = torch.as_tensor(2.*target[ids]-1., device=target_device)
        w = torch.as_tensor(weight[ids], device=target_device)
        prediction = patch_logits(head, x, a)
        loss = weighted_smooth_l1(prediction, y)
        result = (loss*w).sum()/w.sum()
        if training:
            optimizer.zero_grad(set_to_none=True)
            result.backward()
            optimizer.step()
        return float(result.detach())

    for epoch in range(args.epochs):
        head.train()
        fit_total = fit_frames = 0.
        for ids, frame_count in _frame_batches(frame, fit_ids, args.batch_size, rng):
            fit_total += batch_loss(ids, True) * frame_count
            fit_frames += frame_count
        head.eval()
        holdout_total = holdout_frames = 0.
        with torch.no_grad():
            for ids, frame_count in _frame_batches(frame, holdout_ids,
                                                   args.batch_size):
                holdout_total += batch_loss(ids, False) * frame_count
                holdout_frames += frame_count
        fit_loss = fit_total / fit_frames
        holdout_loss = holdout_total / holdout_frames
        if holdout_loss < best:
            best, best_epoch = holdout_loss, epoch+1
            torch.save({'state': {key: value.detach().cpu().clone()
                                  for key, value in head.state_dict().items()},
                        'channels': patch.shape[1], 'anchors': anchors,
                        'seed': args.seed, 'epoch': best_epoch,
                        'arm_sha256': meta['arm_sha256']['F'],
                        'train_patch_manifest': patch_meta}, output / 'best.pt')
        history.append({'epoch': epoch+1, 'fit_loss': fit_loss,
                        'train_holdout_loss': holdout_loss})
        print(f'epoch {epoch+1}/{args.epochs} fit={fit_loss:.6f} '
              f'train_holdout={holdout_loss:.6f}', flush=True)
    report = {'scope': 'official_train_only', 'positive_anchor_count': len(patch),
              'seed': args.seed, 'epochs_requested': args.epochs,
              'batch_max_candidates': args.batch_size,
              'learning_rate': args.learning_rate,
              'holdout_fraction_by_scene': args.holdout_fraction,
              'fit_positive_anchors': len(fit_ids),
              'holdout_positive_anchors': len(holdout_ids),
              'fit_frames_with_positives': int(fit_frames),
              'holdout_frames_with_positives': int(holdout_frames),
              'fit_scenes': [int(value) for value in sorted(set(scenes)-holdout_scenes)],
              'holdout_scenes': [int(value) for value in sorted(holdout_scenes)],
              'target': '2 * assigned_GT_3D_IoU - 1',
              'negative_anchors_in_loss': 0,
              'head': 'Conv2d(C,2,3,padding=1,bias=False)',
              'loss': 'author WeightedSmoothL1 sigma=3; per-frame positive weights',
              'best_epoch_by_train_holdout': best_epoch,
              'best_train_holdout_loss': best,
              'history': history}
    (output / 'train.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('IOU HEAD TRAINING COMPLETE:', output / 'best.pt', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', required=True)
    parser.add_argument('--patch-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--holdout-fraction', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=20260929)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
