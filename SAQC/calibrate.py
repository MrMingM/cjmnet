"""Fit SAQC post-hoc Platt calibration with soft IoU targets on TRAIN data.

Calibration does not change the primary AP ranking benchmark. It is included
only to reproduce SAQC's numerical reliability stage without validation/test
leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gspr_communication.runtime import sha256
from gspr_evidence.stage3_trace import trace_branch
from local_fusion_detector_adaptation.candidate_audit import (
    candidate_ids,
    geometry_ids,
)
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device

from .adapter import (
    association_diagnostics,
    detector_feature_map,
    extract_patches,
)
from .core import paper_fused_score
from .evaluate import _load_quality
from .offline_weather import make_loader as make_fixed_weather_loader
from .project_runtime import (
    add_common_arguments,
    load_frozen_f,
    validate_common,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        '--checkpoint', required=True)
    parser.add_argument(
        '--output', required=True)
    parser.add_argument(
        '--weather',
        choices=('clean', 'fog', 'rain', 'snow'),
        default='clean',
    )
    parser.add_argument(
        '--steps', type=int, default=300)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=.05,
    )
    args = parser.parse_args()
    validate_common(args)

    if args.steps < 1 or args.learning_rate <= 0:
        parser.error(
            'invalid calibration optimizer settings')

    state = load_frozen_f(args)
    quality_head, checkpoint = _load_quality(
        args.checkpoint,
        state,
        state['target'],
    )
    if checkpoint.get('weather_dataset_manifest_sha256') != sha256(
            Path(args.weather_dataset_root) / 'manifest.json'):
        raise ValueError('SAQC training/calibration weather datasets differ')

    dataset, loader, _ = make_fixed_weather_loader(
        state, args, split='train', weather=args.weather)

    fused_scores = []
    targets = []
    branch = 'clean'

    with torch.no_grad():
        for number, batch in enumerate(loader, 1):
            batch = to_device(
                batch, state['target'])
            ctx = v3rt.context(
                state['model'],
                batch['ego'],
                branch,
                verify=number == 1,
            )
            feature_map, prediction, _ = (
                detector_feature_map(
                    state['arm'],
                    ctx['levels'],
                ))
            trace = trace_branch(
                dataset, batch, prediction)

            ids = candidate_ids(
                trace,
                0.,
                256,
                geometry_ids(trace),
            )

            if len(ids):
                assoc = association_diagnostics(
                    ids,
                    np.asarray(trace['decoded'])[ids],
                    batch['ego']['anchor_box'],
                    prediction['psm'],
                )
                patches = extract_patches(
                    feature_map,
                    assoc['decoded_rows'],
                    assoc['decoded_cols'],
                    checkpoint['patch_size'],
                )
                q = (
                    quality_head(patches)
                    .cpu()
                    .numpy()
                )
                raw = np.asarray(
                    trace['scores'])[ids]
                fused_scores.extend(
                    np.asarray(
                        paper_fused_score(
                            raw,
                            q,
                            checkpoint['beta'],
                        )
                    ).tolist()
                )
                quality = (
                    np.asarray(
                        trace['ious'])[ids].max(1)
                    if len(trace['gt'])
                    else np.zeros(
                        len(ids),
                        dtype=np.float32,
                    )
                )
                targets.extend(
                    quality.tolist())

            if number == 1 or number % 100 == 0:
                print(
                    f'calibration {args.weather} '
                    f'{number}/{len(loader)}',
                    flush=True,
                )

    x = torch.as_tensor(
        np.clip(
            fused_scores,
            1e-6,
            1 - 1e-6,
        ),
        dtype=torch.float64,
    )
    y = torch.as_tensor(
        np.clip(targets, 0, 1),
        dtype=torch.float64,
    )
    if len(x) == 0:
        raise RuntimeError(
            'no calibration candidates')

    z = torch.log(x) - torch.log1p(-x)
    a = torch.nn.Parameter(
        torch.ones((), dtype=torch.float64))
    b = torch.nn.Parameter(
        torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [a, b],
        lr=args.learning_rate,
        max_iter=args.steps,
        line_search_fn='strong_wolfe',
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        pred = torch.sigmoid(a * z + b)
        loss = F.binary_cross_entropy(
            pred, y)
        loss.backward()
        return loss

    loss = float(optimizer.step(closure))
    result = {
        'schema': 1,
        'a': float(a.detach()),
        'b': float(b.detach()),
        'training_candidates': len(x),
        'soft_target': 'max BEV IoU',
        'source_split': 'official train only',
        'weather': args.weather,
        'objective': (
            'binary cross entropy with soft IoU targets'),
        'final_loss': loss,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2),
        encoding='utf-8',
    )
    print(
        'SAQC CALIBRATION COMPLETE:',
        args.output,
        flush=True,
    )


if __name__ == '__main__':
    main()
