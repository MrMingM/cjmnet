"""Evaluate SAQC rescoring on development or the frozen historical benchmark."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ceif_audit.scoring import empty_stats
from gspr_communication.runtime import seed_all, sha256, verify_frozen
from gspr_evidence import benchmark as historical_benchmark
from gspr_evidence import runtime as er
from gspr_evidence.stage3_trace import trace_branch
from local_fusion_detector_adaptation.candidate_audit import (
    candidate_ids,
    geometry_ids,
)
from local_fusion_detector_adaptation.candidate_rescore import (
    rescore_postprocess,
)
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils

from .adapter import (
    association_diagnostics,
    detector_feature_map,
    extract_patches,
)
from .core import (
    paper_fused_score,
    platt_calibrate,
    quality_ece,
    spearman,
)
from .model import SpatialQualityHead
from .offline_weather import make_loader as make_fixed_weather_loader
from .project_runtime import (
    add_common_arguments,
    load_frozen_f,
    validate_common,
)


WEATHERS = ('clean', 'fog', 'rain', 'snow')
THRESHOLDS = (.3, .5, .7)
METHODS = (
    'original',
    'scorepass_saqc',
    'top256_raw',
    'top256_saqc',
    'oracle_top256',
)


def _ap(stats, global_sort):
    return {
        f'ap{int(t * 100)}': float(
            eval_utils.calculate_ap(
                copy.deepcopy(stats),
                t,
                global_sort,
            )[0]
        )
        for t in THRESHOLDS
    }


def _normalise_eval_output(output):
    """Convert post-processing outputs to the tensor contract expected by OpenCOOD."""
    boxes, scores, gt_boxes = output

    def tensor_or_none(value, *, dtype=torch.float32):
        if value is None or torch.is_tensor(value):
            return value
        return torch.as_tensor(value, dtype=dtype)

    return (
        tensor_or_none(boxes),
        tensor_or_none(scores),
        tensor_or_none(gt_boxes),
    )


def _update(stats, output):
    output = _normalise_eval_output(output)
    for threshold in THRESHOLDS:
        eval_utils.caluclate_tp_fp(
            *output, stats, threshold)


def _load_quality(path, state, target):
    checkpoint = torch.load(
        path, map_location='cpu', weights_only=True)
    for key, expected in (
        ('frontend_sha256', state['frontend_sha256']),
        ('v3_checkpoint_sha256',
         state['v3_checkpoint_sha256']),
        ('f_checkpoint_sha256',
         state['f_checkpoint_sha256']),
    ):
        if checkpoint.get(key) != expected:
            raise ValueError(
                f'SAQC checkpoint {key} differs '
                'from frozen F evaluation stack')
    model = SpatialQualityHead(
        int(checkpoint['feature_channels']),
        int(checkpoint['hidden_channels']),
        int(checkpoint['patch_size']),
        bool(checkpoint['with_coordinates']),
    ).to(target)
    model.load_state_dict(
        checkpoint['quality_head'], strict=True)
    return model.eval(), checkpoint


def _loader(state, args, weather):
    if args.phase == 'development':
        return make_fixed_weather_loader(
            state, args, split='validate', weather=weather)[:2]

    if args.smoke:
        raise ValueError(
            'benchmark phase deliberately forbids --smoke')

    options, hypes = historical_benchmark.load_config(
        args.v3_config,
        args.frontend_config,
        weather,
    )
    dataset, loader, _ = er.make_loader(
        hypes,
        options,
        train=False,
        weather='clean',
        smoke=0,
    )
    return dataset, loader


def evaluate_condition(
    state,
    quality_head,
    checkpoint,
    args,
    weather,
):
    dataset, loader = _loader(
        state, args, weather)
    model = state['model']
    arm = state['arm']
    target = state['target']

    stats = {
        name: empty_stats()
        for name in METHODS
    }
    raw_scores = []
    fused_scores = []
    qualities = []
    predicted_q = []
    same_cells = []
    center_distances = []
    frames = 0

    beta = float(
        args.beta
        if args.beta_override
        else checkpoint['beta']
    )

    calibration = None
    if args.calibration:
        calibration = json.loads(
            Path(args.calibration).read_text(
                encoding='utf-8'))

    with torch.no_grad():
        for number, batch in enumerate(loader, 1):
            batch = to_device(batch, target)
            branch = 'clean'
            ctx = v3rt.context(
                model,
                batch['ego'],
                branch,
                verify=number == 1,
            )
            feature_map, prediction, _ = (
                detector_feature_map(
                    arm, ctx['levels']))
            trace = trace_branch(
                dataset, batch, prediction)
            original = dataset.post_process(
                batch, {'ego': prediction})

            valid = geometry_ids(trace)
            ids = candidate_ids(
                trace, 0., 256, valid)

            if len(ids):
                decoded = np.asarray(
                    trace['decoded'])[ids]
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
                    checkpoint['patch_size'],
                )
                q = (
                    quality_head(patches)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                score = np.asarray(
                    trace['scores'])[ids].astype(
                        np.float32)
                fused = np.asarray(
                    paper_fused_score(
                        score, q, beta),
                    dtype=np.float32,
                )
                gtq = (
                    np.asarray(
                        trace['ious'])[ids]
                    .max(1)
                    .astype(np.float32)
                    if len(trace['gt'])
                    else np.zeros(
                        len(ids),
                        dtype=np.float32)
                )

                if calibration is not None:
                    calibrated = np.asarray(
                        platt_calibrate(
                            fused,
                            calibration['a'],
                            calibration['b'],
                        ),
                        dtype=np.float32,
                    )
                else:
                    calibrated = fused

                top_raw = rescore_postprocess(
                    trace,
                    ids,
                    score,
                    dataset.post_processor,
                    trace['gt'],
                )
                saqc = rescore_postprocess(
                    trace,
                    ids,
                    fused,
                    dataset.post_processor,
                    trace['gt'],
                )
                oracle = rescore_postprocess(
                    trace,
                    ids,
                    gtq,
                    dataset.post_processor,
                    trace['gt'],
                )

                scorepass_ids = np.asarray(
                    trace['ids']['geometry'],
                    dtype=np.int64,
                )
                if len(scorepass_ids):
                    positions = {
                        int(cid): j
                        for j, cid in enumerate(ids)
                    }
                    missing = [
                        int(cid)
                        for cid in scorepass_ids
                        if int(cid) not in positions
                    ]
                    if missing:
                        raise RuntimeError(
                            'top256 does not contain '
                            'original scorepass candidate')
                    local = np.asarray([
                        positions[int(cid)]
                        for cid in scorepass_ids
                    ], dtype=np.int64)
                    scorepass_saqc = rescore_postprocess(
                        trace,
                        scorepass_ids,
                        fused[local],
                        dataset.post_processor,
                        trace['gt'],
                    )
                else:
                    scorepass_saqc = rescore_postprocess(
                        trace,
                        scorepass_ids,
                        np.empty(
                            0, dtype=np.float32),
                        dataset.post_processor,
                        trace['gt'],
                    )

                raw_scores.extend(score.tolist())
                fused_scores.extend(
                    calibrated.tolist())
                qualities.extend(gtq.tolist())
                predicted_q.extend(q.tolist())
                same_cells.append(float(
                    assoc[
                        'decoded_to_anchor_cell_same'
                    ].float().mean().cpu()))
                center_distances.append(float(
                    assoc[
                        'decoded_to_nearest_grid_distance'
                    ].mean().cpu()))
            else:
                empty_box = torch.empty(
                    (0, 8, 3),
                    dtype=torch.float32,
                    device=target,
                )
                empty_score = torch.empty(
                    (0,),
                    dtype=torch.float32,
                    device=target,
                )
                gt = torch.as_tensor(
                    trace['gt'],
                    dtype=torch.float32,
                    device=target,
                )
                top_raw = (
                    empty_box, empty_score, gt)
                saqc = (
                    empty_box, empty_score, gt)
                oracle = (
                    empty_box, empty_score, gt)
                scorepass_saqc = (
                    empty_box, empty_score, gt)

            budget = (
                0
                if original[0] is None
                else int(len(original[0]))
            )

            def fixed_budget(value):
                boxes, scores, gt_boxes = value
                return (
                    boxes[:budget],
                    scores[:budget],
                    gt_boxes,
                )

            _update(
                stats['original'], original)
            _update(
                stats['scorepass_saqc'],
                fixed_budget(scorepass_saqc),
            )
            _update(
                stats['top256_raw'],
                fixed_budget(top_raw),
            )
            _update(
                stats['top256_saqc'],
                fixed_budget(saqc),
            )
            _update(
                stats['oracle_top256'],
                fixed_budget(oracle),
            )

            frames += 1
            if number == 1 or number % 100 == 0:
                print(
                    f'{weather} {args.phase} '
                    f'{number}/{len(loader)}',
                    flush=True,
                )

    if not frames:
        raise RuntimeError(
            'empty SAQC evaluation')

    raw_scores = np.asarray(
        raw_scores, dtype=np.float64)
    fused_scores = np.asarray(
        fused_scores, dtype=np.float64)
    qualities = np.asarray(
        qualities, dtype=np.float64)
    predicted_q = np.asarray(
        predicted_q, dtype=np.float64)

    frame_ap = {
        name: _ap(value, False)
        for name, value in stats.items()
    }
    global_ap = {
        name: _ap(value, True)
        for name, value in stats.items()
    }

    base = global_ap['original']['ap70']
    ceiling = global_ap[
        'oracle_top256']['ap70']
    gain = (
        global_ap['top256_saqc']['ap70']
        - base
    )
    recovery = (
        gain / (ceiling - base)
        if ceiling > base + 1e-12
        else None
    )

    return {
        'weather': weather,
        'phase': args.phase,
        'frames': frames,
        'beta': beta,
        'frame_order_ap': frame_ap,
        'global_sort_ap': global_ap,
        'score_quality': {
            'raw_spearman': spearman(
                raw_scores, qualities),
            'saqc_spearman': spearman(
                fused_scores, qualities),
            'predicted_quality_spearman': spearman(
                predicted_q, qualities),
            'raw_qece': quality_ece(
                raw_scores, qualities),
            'saqc_qece': quality_ece(
                fused_scores, qualities),
        },
        'association': {
            'decoded_anchor_same_cell_rate': (
                float(np.mean(same_cells))
                if same_cells else None),
            'mean_decoded_to_nearest_grid_distance': (
                float(np.mean(center_distances))
                if center_distances else None),
        },
        'oracle_recovery_ratio_global_ap70': recovery,
        'candidate_scores_calibrated_for_quality_metrics': (
            calibration is not None),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        '--checkpoint', required=True)
    parser.add_argument(
        '--output', required=True)
    parser.add_argument(
        '--phase',
        choices=('development', 'benchmark'),
        default='development',
    )
    parser.add_argument(
        '--calibration',
        help='optional train-only Platt calibration JSON',
    )
    parser.add_argument(
        '--beta-override',
        action='store_true',
        help='use command-line --beta instead of checkpoint beta',
    )
    args = parser.parse_args()
    validate_common(args)

    output = Path(args.output).resolve()
    output.mkdir(
        parents=True, exist_ok=False)

    state = load_frozen_f(args)
    quality_head, checkpoint = _load_quality(
        args.checkpoint,
        state,
        state['target'],
    )
    if args.phase == 'development' and checkpoint.get(
            'weather_dataset_manifest_sha256') != sha256(
                Path(args.weather_dataset_root) / 'manifest.json'):
        raise ValueError('SAQC training/development weather datasets differ')

    reports = {}
    for weather in WEATHERS:
        seed_all(20260929)
        result = evaluate_condition(
            state,
            quality_head,
            checkpoint,
            args,
            weather,
        )
        reports[weather] = result
        (output / f'{weather}.json').write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )

    report = {
        'method': 'SAQC anchor-based adaptation',
        'phase': args.phase,
        'checkpoint_sha256': sha256(
            args.checkpoint),
        'frontend_sha256': state[
            'frontend_sha256'],
        'v3_checkpoint_sha256': state[
            'v3_checkpoint_sha256'],
        'f_checkpoint_sha256': state[
            'f_checkpoint_sha256'],
        'candidate_pool': (
            'geometry-valid top256 by original F score; '
            'fixed before SAQC rescoring'),
        'geometry': 'unchanged decoded F boxes',
        'score': "paper SAQC ranking score s'=s*q^beta",
        'fixed_budget': (
            'per-frame final box count equals original F output count'),
        'conditions': reports,
    }
    (output / 'saqc_results.json').write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )

    verify_frozen()
    print(
        'SAQC EVALUATION COMPLETE:',
        output / 'saqc_results.json',
        flush=True,
    )


if __name__ == '__main__':
    main()
