"""Train one shared D2D-Rescore on full Clean/Fog/Rain/Snow fixed caches."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from gspr_communication.runtime import seed_all, sha256, write_json

from .data import FrameCache, collate_sets, make_labels
from .model import D2DRescore


def train(args):
    if args.epochs < 1 or args.batch_size != 1:
        raise ValueError('epochs must be positive; current audited batch size is 1')
    if not 0 < args.match_iou <= 1:
        raise ValueError('match IoU must be in (0,1]')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = FrameCache(args.cache, 'train')
    seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        data, batch_size=1, shuffle=True, generator=generator,
        collate_fn=collate_sets, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = D2DRescore(
        width=args.width, layers=args.layers, heads=args.heads,
        frequencies=args.frequencies,
        coordinate_scale=tuple(args.coordinate_scale),
        variant=args.variant, radius=args.radius).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    warmup = max(1, min(len(loader), total_steps // 10))

    def learning_rate(step):
        if step < warmup:
            return max(0.01, (step + 1) / warmup)
        phase = (step - warmup) / max(1, total_steps - warmup)
        return .5 * (1. + math.cos(math.pi * min(1., phase)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_count = total_positive = 0
        began = time.time()
        for step, (boxes, scores, mask, records) in enumerate(loader, 1):
            if not mask.any():
                continue
            boxes = boxes.to(device)
            scores = scores.to(device)
            mask = mask.to(device)
            logits = model(boxes, scores, mask)
            labels = torch.zeros_like(logits)
            # Assignment is non-differentiable by design: no GT enters model().
            current = torch.sigmoid(logits.detach()).cpu().numpy()
            for index, (_, frame) in enumerate(records):
                count = len(frame['scores'])
                y = make_labels(
                    frame['corners'], frame['gt'],
                    current[index, :count], args.match_iou,
                    iou_matrix=frame['gt_ious'])
                labels[index, :count] = torch.from_numpy(y).to(device)
                total_positive += int(y.sum())
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], labels[mask])
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite D2D classification loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            scheduler.step()
            count = int(mask.sum())
            total_loss += float(loss.detach()) * count
            total_count += count
            if step == 1 or step % 200 == 0:
                print(f'epoch={epoch} frame={step}/{len(loader)} '
                      f'loss={float(loss.detach()):.5f}', flush=True)
        if not total_count:
            raise RuntimeError('no valid training candidates')
        row = dict(epoch=epoch, loss=total_loss / total_count,
                   candidates=total_count, matched_positive=total_positive,
                   elapsed_seconds=time.time() - began,
                   learning_rate=optimizer.param_groups[0]['lr'])
        history.append(row)
        write_json(out / 'history.json', history)
        print(f'D2D epoch {epoch}: {json.dumps(row)}', flush=True)
    checkpoint = {
        'model': model.cpu().state_dict(),
        'model_config': model.config,
        'match_iou': float(args.match_iou),
        'extract_manifest_sha256': sha256(
            Path(args.cache) / 'manifest.json'),
        'f_checkpoint_sha256': data.metadata['f_checkpoint_sha256'],
        'frontend_sha256': data.metadata['frontend_sha256'],
        'v3_checkpoint_sha256': data.metadata['v3_checkpoint_sha256'],
        'weather_root': data.metadata['weather_root'],
        'train_split': 'train',
        'epochs': args.epochs,
        'seed': args.seed,
        'method': 'adapted_d2d_rescore_opv2v_iou_matching',
    }
    checkpoint_file = out / 'last.pt'
    temp = out / 'last.pt.tmp'
    torch.save(checkpoint, temp)
    temp.replace(checkpoint_file)
    write_json(out / 'summary.json', dict(
        checkpoint=str(checkpoint_file), history=history,
        model_config=model.config, match_iou=args.match_iou,
        train_frames=len(data)))
    print(f'D2D TRAIN COMPLETE: {checkpoint_file}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=5.5e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--match-iou', type=float, default=.7)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--layers', type=int, default=None)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--variant', choices=('d2d', 'gossip'), default='d2d')
    p.add_argument('--radius', type=float, default=5.)
    p.add_argument('--frequencies', type=int, default=10)
    p.add_argument('--coordinate-scale', type=float, nargs=3,
                   default=(140., 40., 10.))
    p.add_argument('--seed', type=int, default=20260930)
    args = p.parse_args()
    if args.layers is None:
        args.layers = 6 if args.variant == 'd2d' else 4
    train(args)


if __name__ == '__main__':
    main()
