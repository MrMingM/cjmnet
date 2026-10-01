"""Extract top256 pre-NMS detection sets from fixed physics-weather PCDs.

The extractor supports a provenance-checked resume mode. Existing frame caches
are never overwritten: they are opened and validated, filenames must form one
contiguous prefix (00000000.npz ...), and their sample IDs are checked against
the restarted deterministic loader before new GPU inference begins.

No online weather simulator or OPV2V-W test set is invoked here.
"""
from __future__ import annotations

import argparse
import json
import os
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

from .data import WEATHERS, read_frame


def _numpy(value, shape_tail):
    if value is None:
        return np.empty((0,) + shape_tail, dtype=np.float32)
    value = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    return np.asarray(value, dtype=np.float32).reshape((-1,) + shape_tail)


def _expected_manifest(args, state):
    return {
        'method': 'adapted_d2d_rescore',
        'candidate_pool': 'geometry_valid_top256',
        'network_box_order': 'xyz_lwh_yaw (adapted from OpenCOOD hwl)',
        'source': 'fixed physics-weather PCD; no online augmentation',
        'weather_root': str(Path(args.weather_dataset_root).resolve()),
        'frontend_config_sha256': sha256(args.frontend_config),
        'frontend_sha256': state['frontend_sha256'],
        'v3_checkpoint_sha256': state['v3_checkpoint_sha256'],
        'f_checkpoint_sha256': state['f_checkpoint_sha256'],
        'smoke': args.smoke,
        'seed': args.seed,
    }


def _new_manifest(args, state):
    result = _expected_manifest(args, state)
    result.update({
        'status': 'in_progress',
        'extractor_sha256': sha256(Path(__file__).resolve()),
        'resume_history': [],
        'conditions': {'train': {}, 'validate': {}},
    })
    return result


def _load_resume_manifest(path, expected):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('status') not in ('in_progress', 'complete'):
        raise ValueError(
            f'Cannot resume cache with status={manifest.get("status")!r}')
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            'Refusing to resume cache from a different experiment: '
            + json.dumps(mismatches, sort_keys=True))
    if set(manifest.get('conditions', {})) != {'train', 'validate'}:
        raise ValueError('Resume manifest has invalid split structure')
    old_hash = manifest.get('extractor_sha256')
    new_hash = sha256(Path(__file__).resolve())
    if old_hash != new_hash:
        history = list(manifest.get('resume_history', []))
        migration = {
            'from_extractor_sha256': old_hash,
            'to_extractor_sha256': new_hash,
            'reason': (
                'resume-capable extractor; existing NPZ files are validated '
                'and never rewritten'),
        }
        if not history or history[-1] != migration:
            history.append(migration)
        manifest['resume_history'] = history
        manifest['resume_extractor_sha256'] = new_hash
    return manifest


def _cache_files(folder):
    """Return a validated contiguous prefix of cached frame paths."""
    if not folder.is_dir():
        return []
    files = sorted(folder.glob('*.npz'))
    expected_names = [f'{index:08d}.npz' for index in range(len(files))]
    actual_names = [path.name for path in files]
    if actual_names != expected_names:
        raise RuntimeError(
            f'Resume cache is not a contiguous prefix in {folder}: '
            f'expected first mismatch among 0..{len(files)-1}')
    return files


def _entry_from_existing(path):
    frame = read_frame(path)
    return {
        'file': path.name,
        'sample_index': int(frame['sample_index']),
        'candidates': int(len(frame['scores'])),
        'original_count': int(frame['original_budget']),
        # The first extractor version did not save the two pre-NMS audit
        # fields inside NPZ. They are not consumed by training/evaluation.
        'original_scorepass_count': None,
        'truncated_original_pool': None,
        'resumed_existing': True,
    }


def _write_frame_atomic(path, **arrays):
    temporary = path.with_suffix('.tmp')
    if path.exists() or temporary.exists():
        raise FileExistsError(
            f'Refusing to overwrite extraction cache: {path}')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sample_index(batch, ordinal):
    index_tensor = batch['ego'].get('communication_sample_index')
    return int(index_tensor[0]) if index_tensor is not None else ordinal


def extract(args):
    validate_common(args)
    out = Path(args.output).resolve()
    manifest_path = out / 'manifest.json'

    if args.resume:
        if not out.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(
                f'--resume requires an existing cache manifest: {manifest_path}')
    else:
        out.mkdir(parents=True, exist_ok=False)

    seed_all(args.seed)
    state = load_frozen_f(args)
    model, arm, device = state['model'], state['arm'], state['target']
    expected = _expected_manifest(args, state)

    if args.resume:
        manifest = _load_resume_manifest(manifest_path, expected)
        if manifest.get('status') == 'complete':
            print(f'D2D EXTRACTION ALREADY COMPLETE: {out}', flush=True)
            return
        print(f'RESUME D2D EXTRACTION: {out}', flush=True)
    else:
        manifest = _new_manifest(args, state)
        write_json(manifest_path, manifest)

    for split in ('train', 'validate'):
        for weather in WEATHERS:
            folder = out / split / weather
            folder.mkdir(parents=True, exist_ok=True)
            existing = _cache_files(folder)

            dataset, loader, indices = make_fixed_loader(
                state, args, split=split, weather=weather)
            total = len(indices)
            if len(existing) > total:
                raise RuntimeError(
                    f'{split}/{weather}: {len(existing)} cached frames '
                    f'exceed loader length {total}')

            # Open every saved file before trusting it. This detects an
            # interrupted/corrupt final npz rather than silently reusing it.
            entries = [_entry_from_existing(path) for path in existing]

            if len(existing) == total:
                manifest['conditions'][split][weather] = entries
                write_json(manifest_path, manifest)
                print(
                    f'reuse complete {split}/{weather}: {total}/{total}',
                    flush=True)
                del dataset, loader
                continue

            pp = dataset.post_processor
            print(
                f'resume {split}/{weather}: existing={len(existing)} '
                f'total={total}',
                flush=True)

            with torch.no_grad():
                for ordinal, batch in enumerate(loader):
                    sample = _sample_index(batch, ordinal)

                    if ordinal < len(existing):
                        saved = entries[ordinal]
                        if int(saved['sample_index']) != sample:
                            raise RuntimeError(
                                f'{split}/{weather} frame {ordinal}: '
                                f'saved sample={saved["sample_index"]} '
                                f'but restarted loader sample={sample}')
                        if ordinal == 0 or (ordinal + 1) % 100 == 0:
                            print(
                                f'validate existing {split}/{weather} '
                                f'{ordinal + 1}/{len(existing)}',
                                flush=True)
                        continue

                    batch = to_device(batch, device)
                    ctx = v3rt.context(
                        model, batch['ego'], 'clean',
                        verify=(split == 'train' and weather == 'clean'
                                and ordinal == len(existing)))
                    prediction, _ = arm.predict(
                        model.engine.base, ctx['levels'])
                    original = dataset.post_process(
                        batch, {'ego': prediction})
                    trace, pools = extract_candidates(
                        pp, batch['ego'], prediction)
                    ids = pools['top256']
                    scorepass = pools['scorepass']

                    replay = rescore_postprocess(
                        trace, scorepass, trace['scores'][scorepass],
                        pp, original[2])
                    assert_scorepass_replay(original, replay)

                    decoded = pp.delta_to_boxes3d(
                        prediction['rm'], batch['ego']['anchor_box'])[0]
                    top_boxes = _numpy(decoded[ids], (7,))
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

                    filename = f'{ordinal:08d}.npz'
                    target = folder / filename
                    truncated = bool(
                        not set(scorepass.tolist()).issubset(
                            set(ids.tolist())))
                    _write_frame_atomic(
                        target,
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
                        'file': filename,
                        'sample_index': sample,
                        'candidates': int(len(ids)),
                        'original_count': int(len(original_scores)),
                        'original_scorepass_count': int(len(scorepass)),
                        'truncated_original_pool': truncated,
                        'resumed_existing': False,
                    })

                    # Record progress periodically; file discovery can still
                    # reconstruct frames saved after the last manifest flush.
                    if (ordinal + 1) % 20 == 0:
                        manifest['conditions'][split][weather] = entries
                        write_json(manifest_path, manifest)
                    if ordinal == 0 or (ordinal + 1) % 100 == 0:
                        print(
                            f'extract {split}/{weather} '
                            f'{ordinal + 1}/{total}',
                            flush=True)

            if len(entries) != total:
                raise RuntimeError(
                    f'{split}/{weather}: extraction ended with '
                    f'{len(entries)}/{total} frames')
            manifest['conditions'][split][weather] = entries
            write_json(manifest_path, manifest)
            del dataset, loader

    manifest['status'] = 'complete'
    manifest['completed_extractor_sha256'] = sha256(
        Path(__file__).resolve())
    write_json(manifest_path, manifest)
    print(f'D2D EXTRACTION COMPLETE: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument(
        '--resume', action='store_true',
        help='validate and continue an existing in-progress cache')
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
