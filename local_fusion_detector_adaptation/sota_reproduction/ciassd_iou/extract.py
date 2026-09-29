"""Add only missing 3x3 patches and positive-anchor 3D IoU labels to an existing top256 cache.

The old candidate rows, IDs, geometry, and scores are never changed. This
module replays frozen F once per cached frame and refuses mismatched caches.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from gspr_communication.runtime import device, seed_all, seed_worker, sha256, verify_frozen
from gspr_evidence import runtime as er
from gspr_evidence.stage3_trace import trace_branch
from local_fusion_detector_adaptation.candidate_audit import _load_arm, candidate_ids, geometry_ids
from local_fusion_detector_adaptation.pipeline import selected_loader
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import box_utils, common_utils


WEATHERS = ('clean', 'fog', 'rain', 'snow')


def candidate_patches(joined, ids, anchors_per_cell):
    """[N,C,3,3] in the exact Conv2d kernel order, zero padded at borders."""
    if joined.ndim != 4 or joined.shape[0] != 1:
        raise ValueError('Expected one fused detector feature map')
    h, w = joined.shape[-2:]
    ids = torch.as_tensor(ids, device=joined.device, dtype=torch.long)
    if len(ids) and (ids.min() < 0 or ids.max() >= h*w*anchors_per_cell):
        raise ValueError('Candidate ID outside fused detector grid')
    cell = ids // anchors_per_cell
    y, x = cell // w, cell % w
    padded = F.pad(joined, (1, 1, 1, 1))[0]
    values = [padded[:, y+dy, x+dx].T for dy in range(3) for dx in range(3)]
    return torch.stack(values, dim=-1).reshape(len(ids), joined.shape[1], 3, 3)


def project_gt_corners(gt_corners, transformation_matrix):
    """Project GT corners with the frozen pipeline transform dtype/device.

    OPV2V object_bbx_center may arrive as float64 while the frozen model
    transformation matrix is float32. torch.matmul requires identical dtypes,
    so keep this benchmark on the transform/model dtype before projection.
    """
    gt_corners = gt_corners.to(device=transformation_matrix.device,
                               dtype=transformation_matrix.dtype)
    return box_utils.project_box3d(gt_corners, transformation_matrix)


def aligned_iou3d(pred_corners, gt_corners):
    """Exact oriented BEV overlap times vertical overlap, for aligned box pairs."""
    if len(pred_corners) != len(gt_corners):
        raise ValueError('Aligned 3D IoU needs equal pair counts')
    if not len(pred_corners):
        return np.empty(0, dtype=np.float32)
    pred_polygons = common_utils.convert_format(pred_corners)
    gt_polygons = common_utils.convert_format(gt_corners)
    answer = np.empty(len(pred_corners), dtype=np.float32)
    for index, (pred, gt) in enumerate(zip(pred_polygons, gt_polygons)):
        intersection = pred.intersection(gt).area
        bottom = max(float(pred_corners[index, :, 2].min()),
                     float(gt_corners[index, :, 2].min()))
        top = min(float(pred_corners[index, :, 2].max()),
                  float(gt_corners[index, :, 2].max()))
        overlap = intersection * max(0., top-bottom)
        pv = pred.area * float(np.ptp(pred_corners[index, :, 2]))
        gv = gt.area * float(np.ptp(gt_corners[index, :, 2]))
        answer[index] = overlap / (pv + gv - overlap) if overlap else 0.
    if not np.isfinite(answer).all() or (answer < 0).any() or (answer > 1+1e-5).any():
        raise ValueError('Invalid aligned 3D IoU')
    return np.clip(answer, 0., 1.)


def _cached_frames(root, weather):
    frames = {}
    path = root / weather / 'candidate_rows.jsonl'
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['arm'] == 'F':
                frames.setdefault(int(row['sample_index']), []).append(row)
    return frames


def _loader(split, metadata, hypes, options, weather, pilot):
    indices = ([int(value) for value in metadata['sample_indices']] if split == 'train'
               else [int(value) for value in metadata['validation_indices']])
    if split == 'validation':
        seed_all(int(pilot['pilot']['seed']) + 1)
        dataset, loader = selected_loader(hypes, options, 'validation', weather, indices)
    else:
        seed = int(metadata['selection_seed'])
        seed_all(seed + WEATHERS.index(weather))
        dataset, _, _ = er.make_loader(hypes, options, train=True, weather=weather)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                            num_workers=int(options['workers']),
                            collate_fn=dataset.collate_batch_test,
                            worker_init_fn=seed_worker, generator=generator)
    return dataset, loader, indices


def run(args):
    verify_frozen()
    root = Path(args.candidate_root).resolve()
    if not (root / 'candidate_audit.json').is_file():
        root = root / 'extraction'
    metadata = json.loads((root / 'candidate_audit.json').read_text(encoding='utf-8'))
    if metadata.get('audit_pool') != 'top_256' or metadata.get('stage0_only'):
        raise ValueError('Requires the existing complete top256 extraction')
    if not metadata.get('candidate_feature_cache'):
        raise ValueError('Requires original candidate center-feature caches')
    if (args.split == 'train') != (metadata.get('split') == 'train'):
        raise ValueError('Candidate extraction split differs from requested split')
    if (args.split == 'validation' and metadata.get('scope') !=
            'same saved validation frames; online synthetic weather; no training or OPV2V-W test'):
        raise ValueError('Validation patch extraction requires the frozen development audit')
    pilot = json.loads((Path(args.run) / 'protocol.json').read_text(encoding='utf-8'))
    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    target = device()
    model, frontend_hash = er.load_model(hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    contract = v3rt.contract(options, args.frontend_config, frontend_hash)
    if (frontend_hash != pilot['frontend_sha256'] or contract != pilot['v3_contract']
            or sha256(args.v3_checkpoint) != pilot['v3_checkpoint_sha256']):
        raise ValueError('Frozen frontend/v3 differs from F pilot')
    f_path = Path(args.run) / 'F.pth'
    if any(metadata[key] != value for key, value in (
            ('frontend_sha256', frontend_hash),
            ('v3_checkpoint_sha256', sha256(args.v3_checkpoint)))):
        raise ValueError('Candidate cache frozen frontend/v3 identity differs')
    if metadata['arm_sha256']['F'] != sha256(f_path):
        raise ValueError('Candidate cache F checkpoint differs')
    source, checkpoint = v3rt.load(args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Expected frozen residual v3')
    arm = _load_arm(f_path, source, model.engine.base, target, False)
    arm.requires_grad_(False).eval()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'split': args.split, 'candidate_root': str(root),
                'candidate_audit_sha256': sha256(root / 'candidate_audit.json'),
                'arm_sha256': sha256(f_path), 'frontend_sha256': frontend_hash,
                'v3_checkpoint_sha256': sha256(args.v3_checkpoint),
                'implementation_sha256': sha256(Path(__file__)),
                'head': 'author 3x3 zero-padded bias-free 2-anchor convolution',
                'conditions': {}}
    for weather in WEATHERS:
        dataset, loader, indices = _loader(args.split, metadata, hypes, options, weather, pilot)
        cached = _cached_frames(root, weather)
        if set(cached) != set(indices):
            raise ValueError(f'{weather}: existing cached frames differ from split')
        folder = output / weather
        folder.mkdir()
        counts = {'frames': 0, 'candidates': 0, 'positive_anchors': 0,
                  'legacy_center_feature_mismatch_frames': 0,
                  'legacy_center_feature_max_abs': 0.0}
        with torch.no_grad():
            for position, batch in enumerate(loader):
                batch = to_device(batch, target)
                sample = int(batch['ego']['communication_sample_index'][0])
                if sample != indices[position]:
                    raise RuntimeError('Frozen loader frame order changed')
                ctx = v3rt.context(model, batch['ego'],
                                   'clean' if weather == 'clean' else 'weather',
                                   verify=position == 0)
                # Use one fusion result for both detector outputs and the 3x3
                # quality patches. This avoids requiring a second independent
                # fusion call to reproduce an old intermediate feature cache.
                fused, _ = arm.fusion(ctx['levels'])
                joined = torch.cat([deblock(value) for deblock, value in
                                    zip(arm.detector.backbone.deblocks, fused)], 1)
                prediction = {
                    'psm': arm.detector.cls_head(joined),
                    'rm': arm.detector.reg_head(joined),
                }
                trace = trace_branch(dataset, batch, prediction)
                rows = cached[sample]
                ids = np.asarray([row['candidate_id'] for row in rows], dtype=np.int64)
                expected = candidate_ids(trace, 0., 256, geometry_ids(trace))
                if not np.array_equal(ids, expected):
                    raise RuntimeError(f'{weather}/{sample}: top256 candidate IDs drifted')
                scores = np.asarray([row['score'] for row in rows])
                if not np.allclose(scores, trace['scores'][ids], atol=1e-6, rtol=1e-6):
                    raise RuntimeError(f'{weather}/{sample}: original score drifted')
                for field, current in (
                        ('fused_regression_deltas', trace['regression_deltas'][ids]),
                        ('fused_decoded_box', trace['decoded'][ids]),
                        ('fused_bev_corners', trace['corners'][ids, :4, :2])):
                    saved = np.asarray([row[field] for row in rows], dtype=np.float32)
                    if not np.allclose(saved, current, atol=2e-5, rtol=1e-5):
                        raise RuntimeError(f'{weather}/{sample}: {field} drifted')
                anchors = prediction['psm'].shape[1]
                patches_t = candidate_patches(joined, ids, anchors)
                h, w = joined.shape[-2:]
                flattened = joined.permute(0, 2, 3, 1).reshape(h*w, joined.shape[1])
                current_center = flattened[torch.as_tensor(
                    ids // anchors, device=joined.device)]
                torch.testing.assert_close(
                    patches_t[:, :, 1, 1], current_center,
                    atol=0., rtol=0.,
                    msg=lambda msg: '3x3 patch center does not match current fused feature: ' + msg)
                patches = patches_t.cpu().numpy().astype(np.float32)

                # The historical candidate cache stored an intermediate center
                # feature from a separate fusion call. Candidate identity and
                # detector outputs above remain hard gates; this legacy
                # intermediate is diagnostic only because a second GPU fusion
                # replay need not be bitwise identical across runs.
                if len(ids):
                    with np.load(root / weather / f'F_features_{sample}.npz',
                                 allow_pickle=False) as packed:
                        if not np.array_equal(packed['candidate_id'], ids):
                            raise RuntimeError('Center feature cache IDs drifted')
                        legacy = np.asarray(packed['feature'], dtype=np.float32)
                    center = patches[:, :, 1, 1]
                    if not np.allclose(legacy, center, atol=1e-6, rtol=1e-5):
                        counts['legacy_center_feature_mismatch_frames'] += 1
                        counts['legacy_center_feature_max_abs'] = max(
                            counts['legacy_center_feature_max_abs'],
                            float(np.max(np.abs(legacy-center))))
                if args.split == 'train':
                    labels = batch['ego']['label_dict']
                    positive = labels['pos_equal_one'].reshape(-1).cpu().numpy()[ids] > 0
                    assigned = labels['pos_gt_index'].reshape(-1).cpu().numpy()[ids].astype(np.int64)
                    if np.any(positive & (assigned < 0)):
                        raise RuntimeError('Positive anchor has no assigned GT')
                    iou = np.full(len(ids), np.nan, dtype=np.float32)
                    if np.any(positive):
                        centers = batch['ego']['object_bbx_center'][0]
                        mask = batch['ego']['object_bbx_mask'][0]
                        if np.any(assigned[positive] >= len(mask)) or not torch.all(mask[assigned[positive]] == 1):
                            raise RuntimeError('Positive anchor points to padded GT')
                        gt_corners = box_utils.boxes_to_corners_3d(
                            centers[assigned[positive]], order=dataset.post_processor.params['order'])
                        gt_corners = project_gt_corners(
                            gt_corners, batch['ego']['transformation_matrix']).cpu().numpy()
                        iou[positive] = aligned_iou3d(trace['corners'][ids[positive]], gt_corners)
                    np.savez_compressed(folder / f'F_patches_{sample}.npz',
                                        candidate_id=ids, patch=patches,
                                        positive=positive, target_iou3d=iou)
                    counts['positive_anchors'] += int(positive.sum())
                else:
                    # Validation GT never enters the inference feature cache.
                    np.savez_compressed(folder / f'F_patches_{sample}.npz',
                                        candidate_id=ids, patch=patches)
                counts['frames'] += 1
                counts['candidates'] += len(ids)
                if position == 0 or (position+1) % 20 == 0:
                    print(f'{weather} patch extraction {position+1}/{len(indices)}', flush=True)
        manifest['conditions'][weather] = counts
        del dataset, loader
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    verify_frozen()
    print('PATCH EXTRACTION COMPLETE:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--candidate-root', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
