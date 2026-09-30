"""Extract top256 pre-NMS detection sets from *fixed* PCDs, exactly once.

This module intentionally does not call any weather simulator, augmentor or
OPV2V-W test set. SAQC.offline_weather selects the materialized PCD roots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gspr_communication.runtime import seed_all, sha256, write_json
from gspr_evidence.stage3_trace import polygon_ious
from local_fusion_detector_adaptation.candidate_audit import candidate_ids
from local_fusion_detector_adaptation.candidate_rescore import (
    assert_scorepass_replay, extract_candidates, rescore_postprocess,
)
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from SAQC.offline_weather import make_loader as make_fixed_loader
from SAQC.project_runtime import (
    add_common_arguments, load_frozen_f, validate_common,
)

from .data import WEATHERS


def _numpy(value, shape_tail):
    if value is None:
        return np.empty((0,) + shape_tail, dtype=np.float32)
    value = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    return np.asarray(value, dtype=np.float32).reshape((-1,) + shape_tail)


def extract(args):
    validate_common(args)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    state = load_frozen_f(args)
    model, arm, device = state['model'], state['arm'], state['target']

    manifest = {
        'status': 'in_progress',
        'method': 'adapted_d2d_rescore',
        'candidate_pool': 'geometry_valid_top256',
        'network_box_order': 'xyz_lwh_yaw (adapted from OpenCOOD hwl)',
        'source': 'fixed physics-weather PCD; no online augmentation',
        'weather_root': str(Path(args.weather_dataset_root).resolve()),
        'frontend_config_sha256': sha256(args.frontend_config),
        'frontend_sha256': state['frontend_sha256'],
        'v3_checkpoint_sha256': state['v3_checkpoint_sha256'],
        'f_checkpoint_sha256': state['f_checkpoint_sha256'],
        'extractor_sha256': sha256(Path(__file__).resolve()),
        'smoke': args.smoke,
        'seed': args.seed,
        'conditions': {'train': {}, 'validate': {}},
    }
    write_json(out / 'manifest.json', manifest)
    for split in ('train', 'validate'):
        for weather in WEATHERS:
            folder = out / split / weather
            folder.mkdir(parents=True, exist_ok=False)
            dataset, loader, _ = make_fixed_loader(
                state, args, split=split, weather=weather)
            pp = dataset.post_processor
            entries = []
            with torch.no_grad():
                for ordinal, batch in enumerate(loader):
                    batch = to_device(batch, device)
                    ctx = v3rt.context(
                        model, batch['ego'], 'clean',
                        verify=(split == 'train' and weather == 'clean'
                                and ordinal == 0))
                    prediction, _ = arm.predict(
                        model.engine.base, ctx['levels'])
                    original = dataset.post_process(batch, {'ego': prediction})
                    trace, pools = extract_candidates(
                        pp, batch['ego'], prediction)
                    ids = pools['top256']
                    scorepass = pools['scorepass']
                    # Must reproduce the original detector for every frame;
                    # no learned comparison is allowed if this gate fails.
                    replay = rescore_postprocess(
                        trace, scorepass, trace['scores'][scorepass],
                        pp, original[2])
                    assert_scorepass_replay(original, replay)
                    decoded = pp.delta_to_boxes3d(
                        prediction['rm'], batch['ego']['anchor_box'])[0]
                    top_boxes = _numpy(decoded[ids], (7,))
                    # OpenCOOD VoxelPostprocessor emits xyz-h-w-l-yaw;
                    # the paper's D2D embedding expects xyz-l-w-h-yaw.
                    if pp.params['order'] != 'hwl':
                        raise ValueError('unexpected decoded box order')
                    top_boxes = top_boxes[:, [0, 1, 2, 5, 4, 3, 6]]
                    top_corners = _numpy(trace['corners'][ids], (8, 3))
                    gt = _numpy(original[2], (8, 3))
                    original_corners = _numpy(original[0], (8, 3))
                    original_scores = (
                        _numpy(original[1], ())
                        if original[1] is not None
                        else np.empty(0, dtype=np.float32))
                    index_tensor = batch['ego'].get(
                        'communication_sample_index')
                    sample = (
                        int(index_tensor[0])
                        if index_tensor is not None else ordinal)
                    filename = f'{ordinal:08d}.npz'
                    np.savez_compressed(
                        folder / filename,
                        candidate_ids=ids.astype(np.int64),
                        boxes=top_boxes,
                        corners=top_corners,
                        scores=np.asarray(
                            trace['scores'][ids], dtype=np.float32),
                        gt=gt,
                        gt_ious=polygon_ious(top_corners, gt),
                        original_corners=original_corners,
                        original_scores=original_scores,
                        nms_threshold=np.float32(pp.params['nms_thresh']),
                        sample_index=np.int64(sample),
                    )
                    entries.append({
                        'file': filename, 'sample_index': sample,
                        'candidates': int(len(ids)),
                        'original_count': int(len(original_scores)),
                        'original_scorepass_count': int(len(scorepass)),
                        'truncated_original_pool': bool(
                            not set(scorepass.tolist()).issubset(set(ids.tolist()))),
                    })
                    if ordinal == 0 or (ordinal + 1) % 100 == 0:
                        print(f'extract {split}/{weather} '
                              f'{ordinal + 1} frames', flush=True)
            manifest['conditions'][split][weather] = entries
            write_json(out / 'manifest.json', manifest)
            del dataset, loader
    manifest['status'] = 'complete'
    write_json(out / 'manifest.json', manifest)
    print(f'D2D EXTRACTION COMPLETE: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
