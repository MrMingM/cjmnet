"""Export frozen F-arm top256 candidates from official TRAIN scenes only.

The output uses the candidate_audit row/cache format. It does not perform
leave-one-source-out inference, train a detector, or touch validation/test GT.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gspr_communication.runtime import (device, seed_all, seed_worker, sha256,
                                        verify_frozen, write_json)
from gspr_evidence import runtime as er
from gspr_evidence.stage3_trace import trace_branch
from local_fusion_utility_v2.fusion import predict_each_source
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device

from .candidate_audit import (_candidate_rows, _fixed_candidate_features,
                              _load_arm, candidate_ids, geometry_ids)
from .pipeline import choose_frames


WEATHERS = ('clean', 'fog', 'rain', 'snow')


def _scene_map(indices, boundaries):
    ends = np.asarray(boundaries, dtype=np.int64)
    if not len(ends) or any(index < 0 or index >= ends[-1] for index in indices):
        raise ValueError('Selected training index outside scene boundaries')
    return {str(index): int(np.searchsorted(ends, index, side='right'))
            for index in indices}


def export(args):
    verify_frozen()
    run = Path(args.run).resolve()
    pilot = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    target = device()
    model, frontend_hash = er.load_model(
        hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    if frontend_hash != pilot['frontend_sha256']:
        raise ValueError('Frontend differs from the frozen pilot')
    contract = v3rt.contract(options, args.frontend_config, frontend_hash)
    v3_hash = sha256(args.v3_checkpoint)
    if contract != pilot['v3_contract'] or v3_hash != pilot['v3_checkpoint_sha256']:
        raise ValueError('v3 configuration/checkpoint differs from the pilot')
    source, checkpoint = v3rt.load(args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected residual v3 checkpoint')
    f_path = run / 'F.pth'
    arm = _load_arm(f_path, source, model.engine.base, target, False)
    arm.requires_grad_(False).eval()
    del source

    train_ds, _, _ = er.make_loader(hypes, options, train=True, weather='clean')
    scenes, indices = choose_frames(train_ds.len_record, args.seed,
                                    args.train_scenes, args.frames_per_scene)
    expected_map = _scene_map(indices, train_ds.len_record)
    del train_ds
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'split': 'train', 'audit_pool': 'top_256', 'stage0_only': False,
        'candidate_feature_cache': True, 'ablation_arm': None,
        'source_interpretation': 'standalone source predictions; no peer removal',
        'sample_indices': indices, 'scene_map': expected_map,
        'selected_scenes': scenes, 'selection_seed': args.seed,
        'frames_per_scene': args.frames_per_scene,
        'train_root': str(Path(hypes['root_dir']).resolve()),
        'frontend_sha256': frontend_hash,
        'v3_checkpoint_sha256': v3_hash,
        'arm_sha256': {'F': sha256(f_path)},
        'implementation_sha256': sha256(Path(__file__).resolve()),
        'row_builder_sha256': sha256(Path(__file__).with_name('candidate_audit.py')),
        'conditions': {},
    }
    write_json(output / 'protocol.json', report)
    for weather in WEATHERS:
        seed_all(args.seed + WEATHERS.index(weather))
        dataset, _, _ = er.make_loader(hypes, options, train=True,
                                       weather=weather)
        if abs(float(dataset.post_processor.params['target_args'][
                'score_threshold']) - .2) > 1e-8:
            raise ValueError('Training postprocessor score threshold differs from 0.2')
        if _scene_map(indices, dataset.len_record) != expected_map:
            raise RuntimeError('Weather dataset changed training scene boundaries')
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(Subset(dataset, indices), batch_size=1,
                            shuffle=False, num_workers=int(options['workers']),
                            collate_fn=dataset.collate_batch_test,
                            worker_init_fn=seed_worker, generator=generator)
        folder = output / weather
        folder.mkdir()
        top_count = original_count = incomplete_frames = 0
        with torch.no_grad(), (folder / 'candidate_rows.jsonl').open(
                'w', encoding='utf-8') as row_stream, (
                folder / 'frame_targets.jsonl').open(
                'w', encoding='utf-8') as target_stream:
            for position, batch in enumerate(loader):
                batch = to_device(batch, target)
                sample = int(batch['ego']['communication_sample_index'][0])
                if sample != indices[position]:
                    raise RuntimeError('Training loader order differs from selected indices')
                branch = 'clean' if weather == 'clean' else 'weather'
                ctx = v3rt.context(model, batch['ego'], branch,
                                   verify=position == 0)
                prediction, _ = arm.predict(model.engine.base, ctx['levels'])
                trace = trace_branch(dataset, batch, prediction)
                local = predict_each_source(model.engine.base, ctx['levels'])
                valid_ids = geometry_ids(trace)
                ids = candidate_ids(trace, 0., 256, valid_ids)
                original_ids = candidate_ids(trace, .2, None, valid_ids)
                top_count += len(ids)
                original_count += len(original_ids)
                incomplete_frames += int(len(original_ids) > len(ids))
                feature_name = f'F_features_{sample}.npz'
                if len(ids):
                    features = _fixed_candidate_features(
                        arm, ctx['levels'], prediction, ids).astype(np.float32)
                    np.savez_compressed(folder / feature_name,
                                        candidate_id=ids, feature=features)
                else:
                    feature_name = None
                target_stream.write(json.dumps({
                    'weather': weather, 'sample_index': sample,
                    'gt_bev_corners': np.asarray(trace['gt'])[:, :4, :2].tolist(),
                }, ensure_ascii=False) + '\n')
                for row in _candidate_rows(trace, ids, local,
                                           dataset.post_processor, batch['ego'],
                                           weather, 'F', sample,
                                           feature_cache=feature_name):
                    row_stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                if position == 0 or (position + 1) % 20 == 0:
                    print(f'{weather} train extraction {position + 1}/{len(indices)}',
                          flush=True)
        report['conditions'][weather] = {
            'frames': len(indices),
            'original_score_threshold': float(dataset.post_processor.params[
                'target_args']['score_threshold']),
            'nms_iou_threshold': float(dataset.post_processor.params['nms_thresh']),
            'pools': {'F': {
                'top_256': {'total_candidates': top_count},
                'original': {'total_candidates': original_count},
            }},
            'original_pool_may_exceed_top256_frames': incomplete_frames,
        }
        write_json(folder / 'summary.json', report['conditions'][weather])
        del dataset, loader
    if sha256(Path(__file__).resolve()) != report['implementation_sha256']:
        raise RuntimeError('Extraction source changed during execution')
    write_json(output / 'candidate_audit.json', report)
    verify_frozen()
    print(f'TRAIN CANDIDATE EXTRACTION COMPLETE: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-scenes', type=int, default=32)
    parser.add_argument('--frames-per-scene', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260929)
    args = parser.parse_args()
    if args.train_scenes < 1 or args.frames_per_scene < 1:
        parser.error('Train scenes and frames per scene must be positive')
    export(args)


if __name__ == '__main__':
    main()
