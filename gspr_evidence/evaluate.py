"""Historical test benchmark by default; development validation must be explicit."""
import argparse
import bisect
import copy
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    p.add_argument('--heads', required=True)
    p.add_argument('--weather', choices=('clean', 'fog', 'rain', 'snow'), required=True)
    p.add_argument('--evaluation-protocol', choices=('benchmark', 'development'), default='benchmark',
                   help='benchmark: historical test point clouds, no augmentation, non-global AP')
    p.add_argument('--modes', nargs='+', default=['none', 'a0b0', 'protocol', 'learned', 'full'],
                   choices=('none', 'a0b0', 'protocol', 'learned', 'full'))
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    import torch
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from gspr_communication.evaluate import matched_objects
    from .supervision import dense_targets
    from . import runtime as rt
    rt.verify_frozen()
    benchmark = args.evaluation_protocol == 'benchmark'
    if benchmark:
        from .benchmark import load_config
        options, hypes = load_config(args.config, args.frontend_config, args.weather)
    else:
        options, hypes = rt.load_config(args.config, args.frontend_config)
    rt.seed_all(options['seed'])
    target = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    state = rt.load_heads(model, args.heads, rt.contract(options, args.frontend_config, digest))
    model.eval()
    rt.seed_all(options['seed'])
    # OPV2V-W files are already weather-degraded: use processed_lidar, never augment again.
    branch = 'clean' if benchmark else args.weather
    ds, loader, indices = rt.make_loader(hypes, options, weather=branch)
    out = rt.new_output(args.output_dir)
    modes = list(dict.fromkeys(['none']+args.modes))
    if benchmark and 'full' in modes:
        modes.append('original_full')
    results = {m: {iou: dict(tp=[], fp=[], gt=0, score=[]) for iou in (.3, .5, .7)} for m in modes}
    sums = {m: dict(frames=0, total_bytes=0, request_bytes=0, feature_bytes=0, selected_blocks=0,
                    corrected_misses=0, lost_detections=0, fp_count_change=0, replaced_blocks=0,
                    risk_target_mass=0., requested_risk_mass=0.) for m in modes}
    rt.write_json(out/'protocol.json', dict(contract=state['contract'], variant=state['variant'],
        heads_sha256=rt.sha256(args.heads), weather=args.weather, sample_indices=indices, scene_ends=ds.len_record,
        modes=modes, full_validation=not benchmark, full_test=benchmark, global_sort=not benchmark,
        evaluation_protocol=args.evaluation_protocol, data_root=hypes['validate_dir'],
        online_weather_augmentation=not benchmark and args.weather != 'clean',
        input_key='processed_lidar' if branch == 'clean' else 'processed_lidar_weather',
        frontend_sha256=digest,
        metric='OpenCOOD planar polygon IoU; historical non-global AP' if benchmark else 'OpenCOOD globally sorted planar AP',
        timing='Shared encoding reported separately; selection+packet+fusion excludes postprocessing/data loading'))
    start = time.perf_counter()
    print(f'{args.evaluation_protocol} {args.weather}: {hypes["validate_dir"]}; ALL {len(indices)} frames, '
          f'{len(ds.len_record)} scenes; global_sort={not benchmark}; modes {modes}', flush=True)
    with torch.no_grad(), (out/'frames.jsonl').open('w', encoding='utf-8') as stream:
        for frame, batch in enumerate(loader, 1):
            batch = to_device(batch, target)
            ego = batch['ego']
            torch.cuda.synchronize()
            t = time.perf_counter()
            inp = rt.input_branch(ego, branch)
            encoded = model.encode(inp)
            torch.cuda.synchronize()
            encode_ms = (time.perf_counter()-t)*1000
            t = time.perf_counter()
            exchange = model.request(encoded)
            torch.cuda.synchronize()
            request_ms = (time.perf_counter()-t)*1000
            baseline_ids = None
            ego_matches, ego_fp, ego_gt = None, None, None
            truth_risk = None
            for mode in modes:
                torch.cuda.synchronize()
                t = time.perf_counter()
                if mode == 'original_full':
                    reference = model.engine.base(inp)
                    prediction = {k: reference[k] for k in ('psm', 'rm')}
                    for key in prediction:
                        torch.testing.assert_close(prediction[key], full_prediction[key], atol=2e-4, rtol=2e-4)
                    diag = copy.deepcopy(full_diag)
                    diag['original_full_output_checked'] = True
                else:
                    prediction, diag = model.run(encoded, mode, exchange=exchange if mode in ('learned', 'protocol', 'none') else None)
                    if mode == 'full':
                        full_prediction, full_diag = prediction, copy.deepcopy(diag)
                torch.cuda.synchronize()
                ms = (time.perf_counter()-t)*1000
                boxes, scores, gt = ds.post_process(batch, {'ego': prediction})
                per_frame = {iou: dict(tp=[], fp=[], gt=0, score=[]) for iou in results[mode]}
                for iou in results[mode]:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, per_frame, iou)
                    results[mode][iou]['gt'] += per_frame[iou]['gt']
                    for key in ('tp', 'fp', 'score'):
                        results[mode][iou][key].extend(per_frame[iou][key])
                matched, false = matched_objects(boxes, scores, gt)
                if mode == 'none':
                    ego_matches, ego_fp, ego_gt = matched, false, gt
                    truth_risk, _ = dense_targets(prediction, ego['label_dict'], model.grid)
                else:
                    torch.testing.assert_close(gt, ego_gt)
                if mode == 'protocol':
                    baseline_ids = diag['selected_ids']
                replaced = 0
                if mode == 'learned' and baseline_ids is not None:
                    replaced = sum(len(set(a)-set(b)) for a, b in zip(diag['selected_ids'], baseline_ids))
                risk_map = truth_risk.amax(1).flatten()
                diag.update(sample_index=int(ego['communication_sample_index'][0]),
                    scene=bisect.bisect_right(ds.len_record, int(ego['communication_sample_index'][0])),
                    mode=mode, encode_ms=encode_ms, request_ms=request_ms if mode in ('learned', 'protocol') else 0.,
                    selection_packet_fusion_ms=ms, total_model_packet_ms=encode_ms+ms+(request_ms if mode in ('learned', 'protocol') else 0.), warmup=frame == 1,
                    corrected_misses=len(matched-ego_matches), lost_detections=len(ego_matches-matched),
                    fp_count_change=false-ego_fp, replaced_blocks=replaced,
                    risk_target_mass=float(risk_map.sum()), requested_risk_mass=float(risk_map[exchange[2]].sum()),
                    ap_inputs=per_frame)
                stream.write(json.dumps(diag)+'\n')
                sums[mode]['frames'] += 1
                for key in sums[mode]:
                    if key != 'frames':
                        sums[mode][key] += diag[key]
            if frame == 1 or frame % 20 == 0:
                elapsed = time.perf_counter()-start
                print(f'{args.weather}: {frame}/{len(indices)}; {elapsed/frame:.2f}s/frame; '
                      f'ETA {(len(indices)-frame)*elapsed/frame/3600:.2f}h', flush=True)
    for mode in modes:
        folder = out/mode
        folder.mkdir()
        if sums[mode]['frames'] != len(indices) or not results[mode][.7]['gt']:
            raise RuntimeError('Incomplete or no-GT evaluation')
        # Non-global OpenCOOD AP mutates TP/FP lists; retain the raw statistics.
        eval_utils.eval_final_results(copy.deepcopy(results[mode]), str(folder), not benchmark)
        # Save sufficient statistics for per-scene regrouping/bootstrap without another GPU pass.
        rt.write_json(folder/'ap_inputs.json', results[mode])
        summary = sums[mode]
        for key in ('total_bytes', 'request_bytes', 'feature_bytes', 'selected_blocks'):
            summary['mean_'+key] = summary[key]/summary['frames']
        summary['replacement_fraction'] = summary['replaced_blocks']/max(summary['selected_blocks'], 1)
        summary['requested_risk_fraction'] = summary['requested_risk_mass']/max(summary['risk_target_mass'], 1e-12)
        rt.write_json(folder/'communication.json', summary)
    rt.write_json(out/'summary.json', sums)
    rt.verify_frozen()
    print(f'Complete {args.evaluation_protocol}: {out}', flush=True)


if __name__ == '__main__':
    main()
