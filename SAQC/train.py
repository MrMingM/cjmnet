"""Train the SAQC Local Spatial Quality Head on the frozen F detector.

Primary benchmark protocol: one quality head is trained on the complete OPV2V
training split under Clean + fixed physics Fog/Rain/Snow PCDs. The F detector and
all decoded geometry are frozen. Only positive training anchors supervise the
quality branch, mirroring SAQC's positive matched quality supervision as
closely as an anchor-based detector permits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from gspr_communication.runtime import seed_all, sha256
from gspr_evidence.stage3_trace import trace_branch
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device

from .adapter import (
    association_diagnostics,
    detector_feature_map,
    extract_patches,
    positive_anchor_ids,
)
from .model import SpatialQualityHead
from .offline_weather import make_loader as make_fixed_weather_loader
from .project_runtime import (
    add_common_arguments,
    load_frozen_f,
    validate_common,
)


WEATHERS = ('clean', 'fog', 'rain', 'snow')


def _parse_weathers(value: str):
    result = tuple(
        x.strip().lower()
        for x in value.split(',')
        if x.strip()
    )
    if (not result or any(x not in WEATHERS for x in result)
            or len(set(result)) != len(result)):
        raise ValueError(
            '--weathers must be a unique comma-separated subset of '
            'clean,fog,rain,snow')
    return result


def _quality(trace, ids, target):
    if len(ids) == 0:
        return torch.empty(
            0, dtype=torch.float32, device=target)
    ious = np.asarray(trace['ious'])[
        ids.detach().cpu().numpy()]
    value = (
        ious.max(1)
        if ious.shape[1]
        else np.zeros(len(ids), dtype=np.float32)
    )
    return torch.as_tensor(
        value, dtype=torch.float32, device=target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--weathers', default='clean,fog,rain,snow')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument(
        '--learning-rate', type=float, default=1e-3)
    parser.add_argument(
        '--weight-decay', type=float, default=1e-4)
    parser.add_argument(
        '--smooth-l1-beta', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument(
        '--no-coordinates',
        action='store_true',
        help='paper ablation: patch without relative-coordinate channels')
    args = parser.parse_args()
    validate_common(args)
    weathers = _parse_weathers(args.weathers)
    if (args.epochs < 1 or args.learning_rate <= 0
            or args.weight_decay < 0):
        parser.error('invalid optimizer/epoch setting')
    if args.smooth_l1_beta <= 0:
        parser.error('--smooth-l1-beta must be positive')

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    state = load_frozen_f(args)
    model = state['model']
    arm = state['arm']
    target = state['target']

    feature_channels = int(
        arm.detector.cls_head.in_channels)
    quality_head = SpatialQualityHead(
        feature_channels,
        args.hidden_channels,
        args.patch_size,
        with_coordinates=not args.no_coordinates,
    ).to(target)
    optimizer = torch.optim.AdamW(
        quality_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    total_updates = 0
    seed_all(args.seed)

    for epoch in range(1, args.epochs + 1):
        quality_head.train()
        epoch_loss = 0.0
        epoch_candidates = 0
        epoch_frames = 0
        same_cells = []
        center_distances = []
        started = time.time()

        for weather in weathers:
            dataset, loader, _ = make_fixed_weather_loader(
                state, args, split='train', weather=weather)
            branch = 'clean'

            for number, batch in enumerate(loader, 1):
                batch = to_device(batch, target)

                with torch.no_grad():
                    ctx = v3rt.context(
                        model,
                        batch['ego'],
                        branch,
                        verify=(
                            epoch == 1
                            and weather == weathers[0]
                            and number == 1
                        ),
                    )
                    feature_map, prediction, _ = (
                        detector_feature_map(
                            arm, ctx['levels']))
                    trace = trace_branch(
                        dataset, batch, prediction)
                    ids = positive_anchor_ids(
                        batch['ego']['label_dict'],
                        prediction['psm'],
                    )

                    if len(ids):
                        decoded = np.asarray(
                            trace['decoded'])[
                                ids.detach().cpu().numpy()]
                        assoc = association_diagnostics(
                            ids,
                            decoded,
                            batch['ego']['anchor_box'],
                            prediction['psm'],
                        )
                        patches = extract_patches(
                            feature_map,
                            assoc['decoded_rows'],
                            assoc['decoded_cols'],
                            args.patch_size,
                        ).detach()
                        labels = _quality(
                            trace, ids, target)
                    else:
                        patches = feature_map.new_empty((
                            0,
                            feature_channels,
                            args.patch_size,
                            args.patch_size,
                        ))
                        labels = feature_map.new_empty((0,))
                        assoc = None

                if len(labels):
                    prediction_q = quality_head(patches)
                    loss = F.smooth_l1_loss(
                        prediction_q,
                        labels,
                        beta=args.smooth_l1_beta,
                        reduction='mean',
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            'non-finite SAQC quality loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        quality_head.parameters(), 5.)
                    optimizer.step()

                    total_updates += 1
                    epoch_loss += (
                        float(loss.detach()) * len(labels))
                    epoch_candidates += len(labels)
                    same_cells.append(float(
                        assoc[
                            'decoded_to_anchor_cell_same'
                        ].float().mean().cpu()))
                    center_distances.append(float(
                        assoc[
                            'decoded_to_nearest_grid_distance'
                        ].mean().cpu()))

                epoch_frames += 1
                if number == 1 or number % 100 == 0:
                    print(
                        f'epoch={epoch} '
                        f'weather={weather} '
                        f'frame={number}/{len(loader)} '
                        f'quality_samples={epoch_candidates}',
                        flush=True,
                    )

            del dataset, loader

        if epoch_candidates == 0:
            raise RuntimeError(
                'no positive anchors available for SAQC training')

        row = dict(
            epoch=epoch,
            frames=epoch_frames,
            quality_samples=epoch_candidates,
            mean_loss=epoch_loss / epoch_candidates,
            mean_decoded_anchor_same_cell=(
                float(np.mean(same_cells))
                if same_cells else None),
            mean_decoded_grid_distance=(
                float(np.mean(center_distances))
                if center_distances else None),
            seconds=time.time() - started,
            optimizer_updates=total_updates,
        )
        history.append(row)
        print(json.dumps(row), flush=True)

    checkpoint = dict(
        schema=1,
        method='SAQC-LSQH anchor-based adaptation',
        quality_head=quality_head.state_dict(),
        feature_channels=feature_channels,
        hidden_channels=args.hidden_channels,
        patch_size=args.patch_size,
        with_coordinates=not args.no_coordinates,
        beta=args.beta,
        smooth_l1_beta=args.smooth_l1_beta,
        weathers=list(weathers),
        epochs=args.epochs,
        seed=args.seed,
        frontend_sha256=state['frontend_sha256'],
        v3_checkpoint_sha256=state[
            'v3_checkpoint_sha256'],
        f_checkpoint_sha256=state[
            'f_checkpoint_sha256'],
        weather_dataset_manifest_sha256=sha256(
            Path(args.weather_dataset_root) / 'manifest.json'),
        target=(
            'max BEV IoU of decoded positive anchor '
            'to frame GT'),
        association=(
            'decoded box xy center -> nearest cell '
            'center of frozen anchor BEV grid'),
        detector_frozen=True,
        history=history,
    )
    torch.save(
        checkpoint, output / 'saqc_quality.pth')

    (output / 'train_history.json').write_text(
        json.dumps(
            history, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    (output / 'protocol.json').write_text(
        json.dumps({
            key: value
            for key, value in checkpoint.items()
            if key not in ('quality_head', 'history')
        }, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    (output / 'checkpoint.sha256').write_text(
        sha256(output / 'saqc_quality.pth') + '\n',
        encoding='utf-8',
    )

    print(
        'SAQC TRAIN COMPLETE:',
        output / 'saqc_quality.pth',
        flush=True,
    )


if __name__ == '__main__':
    main()
