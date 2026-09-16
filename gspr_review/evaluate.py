"""Identical-budget review/protocol/A0B0 controls using actual serialized messages."""
import argparse
import copy
import json
import time
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='gspr_review/experiment.yaml')
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    p.add_argument('--reviewer')
    p.add_argument('--fixed-bev-adjustment', type=float, default=None,
                   help='Offline rule control: feature*(1+c) only at the original supported BEV cells')
    p.add_argument('--packet-cache')
    p.add_argument('--packet-cache-mode', choices=('record', 'replay'))
    p.add_argument('--correction-scale', type=float, default=1.,
                   help='Evaluation-only multiplier after tanh; retains original adjustment and weight bounds')
    p.add_argument('--evidence-intervention', choices=('normal', 'neutral', 'permuted'), default='normal',
                   help='Offline fixed-eligibility content control; packets/bytes unchanged')
    p.add_argument('--mode', choices=('review', 'protocol', 'shuffled', 'a0b0', 'none', 'full'), default='review')
    p.add_argument('--lidar-key', choices=('processed_lidar', 'processed_lidar_weather'))
    p.add_argument('--data-root')
    p.add_argument('--all-scenes', action='store_true')
    p.add_argument('--global-sort-detections', action='store_true')
    p.add_argument('--compare-ego', action='store_true')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    if bool(args.packet_cache) != bool(args.packet_cache_mode):
        p.error('Specify both packet cache and its mode')
    if args.packet_cache and args.mode not in ('review', 'protocol'):
        p.error('Packet replay supports review/protocol only')
    if not math.isfinite(args.correction_scale) or args.correction_scale < 0:
        p.error('--correction-scale must be finite and nonnegative')
    if args.correction_scale != 1 and args.mode != 'review':
        p.error('Amplitude scanning requires --mode review')
    import torch
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from gspr_communication.evaluate import matched_objects
    from . import runtime as rt
    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    if args.data_root:
        hypes['validate_dir'] = args.data_root
    if args.all_scenes:
        options.update(validation_scene_indices=None, frame_stride=1)
    rt.seed_all(int(options['seed']))
    target = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    if args.mode in ('review', 'shuffled') and not args.reviewer:
        raise ValueError('Learned review requires --reviewer')
    if args.reviewer:
        rt.load_reviewer(model, args.reviewer, options, digest, args.frontend_config)
    model.mode = args.mode
    if args.evidence_intervention != 'normal' and args.mode != 'review':
        raise ValueError('Evidence intervention requires review mode')
    model.reviewer.intervention = args.evidence_intervention
    model.eval()
    if args.fixed_bev_adjustment is not None:
        if (not model.bev_location or args.mode != 'review' or args.correction_scale != 1
            or args.evidence_intervention != 'normal' or not math.isfinite(args.fixed_bev_adjustment)
            or abs(args.fixed_bev_adjustment) > min(.25, float(options['review']['max_adjustment']))):
            raise ValueError('Fixed scaling requires normal BEV review, scale=1, and the original adjustment bound')
        model.fixed_bev_adjustment = args.fixed_bev_adjustment
    model.reviewer.evaluation_scale = args.correction_scale
    rt.seed_all(int(options['seed']))
    ds, loader, indices = rt.make_loader(hypes, options, test=True)
    out = rt.new_output(args.output_dir)
    rt.write_json(out/'protocol.json', {'options': options, 'mode': args.mode,
        'evidence_intervention': args.evidence_intervention,
        'correction_scale': args.correction_scale,
        'fixed_bev_adjustment': args.fixed_bev_adjustment,
        'packet_cache': args.packet_cache, 'packet_cache_mode': args.packet_cache_mode,
        'lidar_key': args.lidar_key or options['lidar_key'], 'root': hypes['validate_dir'],
        'sample_indices': indices, 'frontend_sha256': digest,
        'reviewer_sha256': rt.sha256(args.reviewer) if args.reviewer else None,
        'global_sort': args.global_sort_detections,
        'metric': 'OpenCOOD planar polygon IoU; not height-aware 3D IoU',
        'latency': 'Model + actual CPU packet roundtrip + receiver re-encoding, excludes loading/postprocess; first frame warmup'})
    result = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for iou in (.3, .5, .7)}
    rows, durations = [], []
    with torch.no_grad(), (out/'frames.jsonl').open('w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            inp = rt.input_branch(batch['ego'], options, args.lidar_key)
            cache = None
            if args.packet_cache:
                from .packet_replay import PacketReplay, fingerprint
                frame_id = int(batch['ego']['communication_sample_index'][0])
                cache = PacketReplay(Path(args.packet_cache)/f'{frame_id}.json', args.packet_cache_mode, fingerprint(inp))
                model.packet_exchange = cache.exchange
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(inp)
            if cache is not None:
                cache.finish()
                del model.packet_exchange
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            diag = copy.deepcopy(model.last_diagnostics[0])
            if cache is not None:
                diag['candidate_packet_differences'] = cache.differences
            boxes, scores, gt = ds.post_process(batch, {'ego': output})
            for iou in result:
                eval_utils.caluclate_tp_fp(boxes, scores, gt, result, iou)
            diag.update(sample_index=int(batch['ego']['communication_sample_index'][0]), model_packet_ms=elapsed*1000, warmup=not rows)
            if args.compare_ego:
                model.mode = 'none'
                ego_output = model(inp)
                model.mode = args.mode
                eb, es, eg = ds.post_process(batch, {'ego': ego_output})
                torch.testing.assert_close(gt, eg)
                cm, cf = matched_objects(boxes, scores, gt)
                em, ef = matched_objects(eb, es, eg)
                diag.update(corrected_misses=len(cm-em), lost_detections=len(em-cm), fp_count_change=cf-ef)
            if rows:
                durations.append(elapsed*1000)
            rows.append(diag)
            stream.write(json.dumps(diag)+'\n')
            if len(rows) % 20 == 0:
                print(f'{len(rows)}/{len(indices)} frames; mean bytes {sum(r["total_bytes"] for r in rows)/len(rows):.0f}', flush=True)
    if not rows or not result[.7]['gt']:
        raise RuntimeError('No frames/GT')
    eval_utils.eval_final_results(result, str(out), args.global_sort_detections)
    summary = {'frames': len(rows), 'mean_total_bytes': sum(r['total_bytes'] for r in rows)/len(rows),
               'mean_model_packet_ms': sum(durations)/len(durations) if durations else None}
    for key in ('review_bytes', 'feature_bytes', 'query_blocks', 'eligible_points', 'changed_points', 'evidence_coverage', 'mean_abs_weight_change', 'bev_changed_cells', 'bev_abs_adjustment_sum'):
        if key in rows[0]:
            summary['mean_'+key] = sum(r[key] for r in rows)/len(rows)
    for key in ('corrected_misses', 'lost_detections', 'fp_count_change'):
        summary[key] = sum(r[key] for r in rows) if args.compare_ego else None
    rt.write_json(out/'communication.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    rt.verify_frozen()


if __name__ == '__main__':
    main()
