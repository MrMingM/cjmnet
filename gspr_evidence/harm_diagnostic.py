"""Offline deletion interventions on FULL development validation, never a test selector."""
import argparse
import bisect
import copy
import json
import math
import time
from pathlib import Path
import torch


def deletion_orders(obs, seed):
    """Only regions with retained point observations have defined point quality/u."""
    result = []
    generator = torch.Generator(device=obs.device).manual_seed(seed)
    for peer in range(1, len(obs)):
        valid = (obs[peer, 3].flatten() > 0).nonzero().flatten()
        p = obs[peer, 0].flatten()[valid]
        u = obs[peer, 1].flatten()[valid]
        result.append(dict(low_r=valid[torch.argsort(p, stable=True)],
                           high_u=valid[torch.argsort(u, descending=True, stable=True)],
                           random=valid[torch.randperm(len(valid), generator=generator, device=obs.device)]))
    return result


def filter_mask(full, orders, method, fraction):
    mask = full.clone()
    for peer, order in enumerate(orders, 1):
        count = min(len(order[method]), math.ceil(len(order[method])*fraction))
        mask[peer].view(-1)[order[method][:count]] = 0
    return mask


def candidates(orders, per_source=2):
    rows = {}
    for peer, order in enumerate(orders, 1):
        for source, ids in order.items():
            for block in ids[:per_source].cpu().tolist():
                rows.setdefault((peer, block), []).append(source)
    return [(peer, block, sources) for (peer, block), sources in rows.items()]


def empty_stats():
    return {iou: dict(tp=[], fp=[], gt=0, score=[]) for iou in (.3, .5, .7)}


def merge_stats(total, addition):
    for iou in total:
        total[iou]['gt'] += addition[iou]['gt']
        for key in ('tp', 'fp', 'score'):
            total[iou][key].extend(addition[iou][key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gspr_evidence/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--weather', required=True, choices=('clean', 'fog', 'rain', 'snow'))
    parser.add_argument('--stage', choices=('filters', 'single', 'both'), default='both')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fractions', type=float, nargs='+', default=[.01, .05, .10])
    parser.add_argument('--candidates-per-source', type=int, default=2)
    args = parser.parse_args()
    if (len(set(args.fractions)) != len(args.fractions) or not args.fractions
            or any(not 0 < f < 1 for f in args.fractions) or args.candidates_per_source < 1):
        parser.error('Use distinct fractions in (0,1) and positive candidate count')
    from opencood.tools.train_utils import to_device
    from opencood.loss.point_pillar_loss import PointPillarLoss
    from opencood.utils import eval_utils
    from gspr_communication.evaluate import matched_objects
    from . import runtime as rt
    rt.verify_frozen()
    options, hypes = rt.load_config(args.config, args.frontend_config)
    # This is a validation diagnostic; never permit a historical test root here.
    if Path(hypes['validate_dir']).resolve() != Path('/data/scd/datasets/opv2v_official_data_dumping/validate').resolve():
        raise ValueError('Diagnostic requires the fixed development validation root, not test')
    rt.seed_all(options['seed'])
    device = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    criterion = PointPillarLoss(hypes['loss']['args'])
    rt.seed_all(options['seed'])
    ds, loader, indices = rt.make_loader(hypes, options, weather=args.weather)
    out = rt.new_output(args.output_dir)
    runs = [('full', None, 0.)]
    if args.stage in ('filters', 'both'):
        runs += [(f'{method}_{f:g}', method, f) for f in args.fractions for method in ('low_r', 'high_u', 'random')]
    all_stats = {name: empty_stats() for name, _, _ in runs}
    scene_stats = {}
    totals = {name: dict(corrected=0, lost=0, fp_change=0, removed=0, loss_delta=0.) for name, _, _ in runs}
    group_totals = {}
    candidate_frames = set()
    rt.write_json(out/'protocol.json', dict(stage=args.stage, weather=args.weather, options=options,
        data_root=hypes['validate_dir'], sample_indices=indices, scene_ends=ds.len_record,
        frontend_sha256=digest, frontend_config_sha256=rt.sha256(args.frontend_config),
        source_sha256=rt.sha256(__file__), global_sort=True,
        online_weather=args.weather != 'clean', development_only=True,
        fractions=args.fractions, candidates_per_source=args.candidates_per_source,
        eligible='Each sender region with retained points; empty regions excluded from quality ranking',
        denominator='Removal fraction is per-sender eligible regions, not all transmitted grid cells',
        intervention='Frozen full communication; one frame encoded once; hard missing-block attention; no retraining',
        payload='Offline virtual block deletion; no request protocol or deployable communication saving claimed',
        candidate_sign='positive loss_delta = full loss minus loss after removing one block',
        no_oracle_ap=True))
    start = time.perf_counter()
    with torch.no_grad(), (out/'frames.jsonl').open('w') as frames, (out/'candidates.jsonl').open('w') as candidate_log:
        for number, batch in enumerate(loader, 1):
            batch = to_device(batch, device)
            ego = batch['ego']
            index = int(ego['communication_sample_index'][0])
            scene = bisect.bisect_right(ds.len_record, index)
            encoded = model.encode(rt.input_branch(ego, args.weather))
            full = torch.ones_like(model.empty_masks(encoded))
            orders = deletion_orders(encoded['obs'], options['seed']+index)
            scene_stats.setdefault(scene, {name: empty_stats() for name, _, _ in runs})
            for name, method, fraction in runs:
                mask = full if method is None else filter_mask(full, orders, method, fraction)
                prediction, _ = model.detect(encoded, mask, serialize=False)
                if number == 1:
                    # Verify the virtual replay against actual three-scale packet decoding.
                    wire, _ = model.detect(encoded, mask, serialize=True)
                    for key in prediction:
                        torch.testing.assert_close(prediction[key], wire[key], atol=2e-4, rtol=2e-4)
                loss = float(criterion(prediction, ego['label_dict']))
                if not math.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at frame {index}')
                boxes, scores, gt = ds.post_process(batch, {'ego': prediction})
                matched, fp = matched_objects(boxes, scores, gt)
                if method is None:
                    base_loss, base_matched, base_fp = loss, matched, fp
                    base_gt = gt
                    repeat, _ = model.detect(encoded, full, serialize=False)
                    repeat_loss = float(criterion(repeat, ego['label_dict']))
                    noise = abs(base_loss-repeat_loss)
                    epsilon = max(1e-4, 5*noise)
                else:
                    torch.testing.assert_close(gt, base_gt)
                stats = empty_stats()
                for iou in stats:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, stats, iou)
                merge_stats(all_stats[name], stats)
                merge_stats(scene_stats[scene][name], stats)
                row = dict(sample_index=index, scene=scene, mode=name, corrected=len(matched-base_matched),
                           lost=len(base_matched-matched), fp_change=fp-base_fp,
                           removed=int((full-mask).sum()), loss_delta=base_loss-loss,
                           eligible=sum(len(x['random']) for x in orders), repeat_loss_noise=noise, ap_inputs=stats)
                for key in totals[name]:
                    totals[name][key] += row[key]
                frames.write(json.dumps(row)+'\n')
            if args.stage in ('single', 'both'):
                for peer, block, sources in candidates(orders, args.candidates_per_source):
                    mask = full.clone()
                    mask[peer].view(-1)[block] = 0
                    prediction, _ = model.detect(encoded, mask, serialize=False)
                    loss = float(criterion(prediction, ego['label_dict']))
                    if not math.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite candidate loss at frame {index}')
                    boxes, scores, gt = ds.post_process(batch, {'ego': prediction})
                    torch.testing.assert_close(gt, base_gt)
                    matched, fp = matched_objects(boxes, scores, gt)
                    row = dict(sample_index=index, scene=scene, peer=peer, block=block, sources=sources,
                        reliability=float(encoded['obs'][peer, 0].flatten()[block]),
                        uncertainty=float(encoded['obs'][peer, 1].flatten()[block]),
                        loss_delta=base_loss-loss, loss_sign_epsilon=epsilon, corrected=len(matched-base_matched),
                        lost=len(base_matched-matched), fp_change=fp-base_fp)
                    candidate_log.write(json.dumps(row)+'\n')
                    candidate_frames.add(index)
                    for source in ['all_unique']+sources:
                        accumulator = group_totals.setdefault(source, dict(count=0, loss_delta_sum=0.,
                            loss_improved=0, loss_worsened=0, detection_improved=0, detection_worsened=0))
                        accumulator['count'] += 1
                        accumulator['loss_delta_sum'] += row['loss_delta']
                        accumulator['loss_improved'] += int(row['loss_delta'] > epsilon)
                        accumulator['loss_worsened'] += int(row['loss_delta'] < -epsilon)
                        good = row['corrected'] >= row['lost'] and row['fp_change'] <= 0
                        bad = row['corrected'] <= row['lost'] and row['fp_change'] >= 0
                        accumulator['detection_improved'] += int(good and (row['corrected'] > row['lost'] or row['fp_change'] < 0))
                        accumulator['detection_worsened'] += int(bad and (row['corrected'] < row['lost'] or row['fp_change'] > 0))
            if number == 1 or number % 20 == 0:
                elapsed = time.perf_counter()-start
                print(f'{args.stage}/{args.weather} {number}/{len(indices)}; {elapsed/number:.2f}s/frame; '
                      f'ETA {(len(indices)-number)*elapsed/number/3600:.2f}h', flush=True)
                frames.flush()
                candidate_log.flush()
    if number != len(indices) or not all_stats['full'][.7]['gt']:
        raise RuntimeError('Incomplete/no-GT diagnostic')
    report = dict(frames=number, candidate_frames=len(candidate_frames), candidate_groups=group_totals, modes={})
    for row in group_totals.values():
        row['mean_loss_delta'] = row['loss_delta_sum']/row['count']
        for key in ('loss_improved', 'loss_worsened', 'detection_improved', 'detection_worsened'):
            row[key+'_fraction'] = row[key]/row['count']
    base_ap = eval_utils.calculate_ap(copy.deepcopy(all_stats['full']), .7, True)[0]
    lines = ['# Development validation deletion diagnostic', '',
             'Online simulated weather on validation; NOT historical OPV2V-W test AP.', '',
             '| Mode | AP30 | AP50 | AP70 | Delta AP70 | Corrected/lost | FP change | Mean removed |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in all_stats:
        folder = out/name
        folder.mkdir()
        eval_utils.eval_final_results(copy.deepcopy(all_stats[name]), str(folder), True)
        ap = [float(eval_utils.calculate_ap(copy.deepcopy(all_stats[name]), iou, True)[0]) for iou in (.3, .5, .7)]
        row = dict(totals[name], ap30=ap[0], ap50=ap[1], ap70=ap[2], delta_ap70=ap[2]-base_ap)
        report['modes'][name] = row
        lines.append(f'| {name} | {ap[0]:.6f} | {ap[1]:.6f} | {ap[2]:.6f} | {ap[2]-base_ap:+.6f} | '
                     f'{row["corrected"]}/{row["lost"]} | {row["fp_change"]} | {row["removed"]/number:.2f} |')
    per_scene = {}
    for scene, variants in scene_stats.items():
        if variants['full'][.7]['gt']:
            per_scene[scene] = {name: float(eval_utils.calculate_ap(copy.deepcopy(stats), .7, True)[0]) for name, stats in variants.items()}
    rt.write_json(out/'scene_ap70.json', per_scene)
    rt.write_json(out/'summary.json', report)
    lines += ['', 'Single-block interventions (overlapping source categories; all_unique is deduplicated):', '',
              '```json', json.dumps(group_totals, indent=2), '```', '',
              'Positive loss_delta means deletion improved the frozen detection loss. This is not AP gain. '
              'Detection events use IoU .7 and the fixed postprocessing threshold. Candidate search is partial, '
              'not an oracle ceiling. Inspect per-scene consistency and random controls before deciding.']
    (out/'diagnosis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    rt.verify_frozen()
    print(f'Send back: {out}/diagnosis.md', flush=True)


if __name__ == '__main__':
    main()
