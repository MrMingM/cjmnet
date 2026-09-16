"""Evaluate fixed selectors or internal rules; serialize every message."""
import argparse
import copy
import json
import time
from pathlib import Path


def matched_objects(boxes, scores, gt, threshold=.7):
    """Same planar polygon IoU/greedy matching as this checkout's eval_utils."""
    import numpy as np
    from opencood.utils import common_utils as cu
    gt_polygons = list(cu.convert_format(gt.detach().cpu().numpy()))
    remaining = list(range(len(gt_polygons)))
    matched, false_positives = set(), 0
    if boxes is None:
        return matched, 0
    polygons = list(cu.convert_format(boxes.detach().cpu().numpy()))
    for index in np.argsort(-scores.detach().cpu().numpy(), kind='stable'):
        ious = cu.compute_iou(polygons[index], gt_polygons)
        if not len(ious) or np.max(ious) < threshold:
            false_positives += 1
        else:
            best = int(np.argmax(ious))
            matched.add(remaining.pop(best))
            gt_polygons.pop(best)
    return matched, false_positives


def main():
    parser = argparse.ArgumentParser(description='Communication AP, actual bytes and ego-relative diagnostics')
    parser.add_argument('--config', default='gspr_communication/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--selectors', help='Required for learned A/B variants')
    parser.add_argument('--variant', choices=['a0b0', 'a1b0', 'a0b1', 'a1b1', 'residual', 'quality', 'random', 'none', 'full'])
    parser.add_argument('--data-root', help='Override validation root for a frozen-protocol evaluation')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--global-sort-detections', action='store_true')
    parser.add_argument('--compare-ego', action='store_true')
    parser.add_argument('--all-scenes', action='store_true', help='Evaluate all frames; disable development scene/stride filters')
    args = parser.parse_args()
    import torch
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from . import runtime as rt
    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    if args.variant:
        options['communication']['variant'] = args.variant
    if args.data_root:
        hypes['validate_dir'] = args.data_root
    if args.all_scenes:
        options['validation_scene_indices'] = None
        options['frame_stride'] = 1
    rt.seed_all(int(options['seed']))
    target = rt.device()
    model, frontend_hash = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    learned = model.variant in ('a1b0', 'a0b1', 'a1b1', 'residual')
    if learned and not args.selectors:
        raise ValueError('Learned variants require trained --selectors')
    if args.selectors:
        state = torch.load(args.selectors, map_location='cpu', weights_only=True)
        if state['frontend_sha256'] != frontend_hash or state['communication'] != options['communication']:
            raise ValueError('Selector checkpoint frontend/variant/budget settings differ from evaluation config')
        if state['frontend_config_sha256'] != rt.sha256(args.frontend_config):
            raise ValueError('Use the exact frontend config saved with selector training; override data via --data-root')
        model.request.load_state_dict(state['request'], strict=True)
        model.response.load_state_dict(state['response'], strict=True)
        if model.correction is not None:
            model.correction.load_state_dict(state['correction'], strict=True)
    model.eval()
    ds, loader, indices = rt.make_loader(hypes, options, test=True)
    out = rt.new_output(args.output_dir)
    rt.write_json(out/'protocol.json', {'options': options, 'root': hypes['validate_dir'],
        'sample_indices': indices, 'frontend_sha256': frontend_hash,
        'selectors_sha256': rt.sha256(args.selectors) if args.selectors else None,
        'global_sort': args.global_sort_detections,
        'metric': 'Existing OpenCOOD eval_utils planar polygon IoU; not height-aware 3D IoU',
        'latency': 'Model + CPU packet roundtrip, excludes data loading and postprocessing; first frame is warmup'})
    result = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for iou in (.3, .5, .7)}
    frames, byte_sum, duration_sum, timed_frames = 0, 0, 0., 0
    corrected = lost = fp_change = 0
    with torch.no_grad(), (out/'frames.jsonl').open('w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            inp = rt.model_input(batch['ego'], options)
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(inp)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            diag = copy.deepcopy(model.last_diagnostics[0])
            boxes, scores, gt = ds.post_process(batch, {'ego': output})
            for iou in result:
                eval_utils.caluclate_tp_fp(boxes, scores, gt, result, iou)
            diag['sample_index'] = int(batch['ego']['communication_sample_index'][0])
            diag['model_packet_ms'] = elapsed*1000
            diag['warmup'] = frames == 0
            if args.compare_ego:
                saved = model.variant
                model.variant = 'none'
                ego_output = model(inp)
                model.variant = saved
                eb, es, eg = ds.post_process(batch, {'ego': ego_output})
                if gt.shape != eg.shape or not torch.allclose(gt, eg):
                    raise RuntimeError('GT mismatch in paired evaluation')
                cm, cf = matched_objects(boxes, scores, gt)
                em, ef = matched_objects(eb, es, eg)
                diag.update({'corrected_misses': len(cm-em), 'lost_detections': len(em-cm),
                             'communication_fp': cf, 'ego_fp': ef, 'fp_count_change': cf-ef})
                corrected += len(cm-em)
                lost += len(em-cm)
                fp_change += cf-ef
            stream.write(json.dumps(diag)+'\n')
            byte_sum += diag['total_bytes']
            if frames:
                duration_sum += elapsed
                timed_frames += 1
            frames += 1
            if frames % 20 == 0:
                print(f'{frames}/{len(indices)} frames; mean bytes {byte_sum/frames:.0f}', flush=True)
    if not frames or result[.7]['gt'] == 0:
        raise RuntimeError('No frames/GT; AP is undefined')
    eval_utils.eval_final_results(result, str(out), args.global_sort_detections)
    rt.write_json(out/'communication.json', {'frames': frames, 'mean_total_bytes': byte_sum/frames,
        'mean_model_packet_ms': duration_sum/max(timed_frames, 1)*1000 if timed_frames else None,
        'corrected_misses': corrected if args.compare_ego else None,
        'lost_detections': lost if args.compare_ego else None,
        'fp_count_change': fp_change if args.compare_ego else None})
    rt.verify_frozen()


if __name__ == '__main__':
    main()
