"""Frozen detector feasibility audit, full split by default; never trains."""
import argparse
import bisect
import copy
import json
import time
from pathlib import Path
import numpy as np
from .core import Settings, Source, make_actions
from .scoring import empty_stats, merge, assignments, choose_hindsight, ap_values

MODES = ('baseline', 'full_reference', 'generic_fp_oracle70', 'evidence_fp_oracle70',
         'pool_random_feature', 'evidence_rule_feature', 'evidence_hindsight_feature', 'pool_hindsight_feature')


def numpy_detection(boxes, scores):
    if boxes is None:
        return np.empty((0, 8, 3), np.float32), np.empty(0, np.float32)
    return boxes.detach().cpu().numpy(), scores.detach().cpu().numpy()


def build_sources(base, inp, masks, grid, extent, settings):
    processed = inp['processed_lidar']
    rel = base.gspr(processed['voxel_features'], processed['voxel_num_points'], processed['voxel_coords'])
    valid = rel['point_valid_mask'].bool()
    evidence = rel['point_evidence']
    strength = evidence.sum(-1)+2
    import torch
    opinion = torch.stack((evidence[..., 0]/strength, evidence[..., 1]/strength, 2/strength), -1)
    xyz = processed['voxel_features'][..., :3]
    owners = processed['voxel_coords'][:, 0].long()
    sources = []
    for peer in range(len(masks)):
        keep = valid & (owners[:, None] == peer)
        sources.append(Source(xyz[keep].cpu().numpy(), opinion[keep].cpu().numpy(),
            inp['clouds'][peer].cpu().numpy(), inp['transforms'][peer].cpu().numpy(),
            masks[peer].cpu().numpy(), grid, extent, settings))
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='gspr_evidence/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--phase', choices=('development', 'benchmark-audit'), default='development')
    parser.add_argument('--weather', choices=('clean', 'fog', 'rain', 'snow'), required=True)
    parser.add_argument('--baseline', choices=('full', 'a0b0'), default='full')
    parser.add_argument('--budget-bytes', type=int)
    parser.add_argument('--settings', help='Frozen JSON settings; defaults are preregistered, no sweep')
    parser.add_argument('--smoke', type=int, default=0, help='Plumbing check only; 0 means all frames')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.smoke < 0 or (args.baseline == 'a0b0' and (args.budget_bytes is None or args.budget_bytes <= 0)):
        parser.error('A0B0 requires an explicit positive budget; smoke must be nonnegative')
    if args.baseline == 'full' and args.budget_bytes is not None:
        parser.error('full has no sparse budget; omit --budget-bytes')
    settings = Settings(**(json.loads(Path(args.settings).read_text(encoding='utf-8')) if args.settings else {}))
    settings.validate()
    import torch
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from gspr_evidence import runtime as rt
    from gspr_evidence.benchmark import load_config as benchmark_config
    from .replay import received_levels, fused_levels, apply_action, detect
    rt.verify_frozen()
    formal_data = args.phase == 'benchmark-audit'
    options, hypes = (benchmark_config(args.config, args.frontend_config, args.weather) if formal_data
                      else rt.load_config(args.config, args.frontend_config))
    if not formal_data and Path(hypes['validate_dir']).resolve() != Path('/data/scd/datasets/opv2v_official_data_dumping/validate').resolve():
        raise ValueError('Development audit requires the fixed full validation split')
    if formal_data and args.settings:
        # Explicitly recorded, never auto-selected using test metrics.
        print('Using externally frozen settings for exploratory benchmark audit; no automatic tuning.', flush=True)
    rt.seed_all(options['seed'])
    device = rt.device()
    model, digest = rt.load_model(hypes, options, args.frontend_checkpoint, device)
    if args.baseline == 'a0b0':
        model.engine.budget = args.budget_bytes
    branch = 'clean' if formal_data else args.weather
    rt.seed_all(options['seed'])
    ds, loader, indices = rt.make_loader(hypes, options, weather=branch, smoke=args.smoke)
    if not indices:
        raise ValueError('Empty audit split')
    out = rt.new_output(args.output_dir)
    contract = rt.contract(options, args.frontend_config, digest)
    protocol = dict(phase=args.phase, weather=args.weather, data_root=hypes['validate_dir'],
        online_weather=not formal_data and args.weather != 'clean', global_sort=False,
        baseline=args.baseline, budget_bytes=args.budget_bytes, settings=settings.validate(),
        sample_indices=indices, scene_ends=ds.len_record, smoke=bool(args.smoke),
        comparison_contract=contract,
        audit_sources={p.name: rt.sha256(p) for p in Path(__file__).parent.glob('*.py')},
        evidence_access='Only retained endpoints in selected blocks; queries also masked by sender receipt. '
            'Extra ideal geometry side-channel, explicitly costed, NOT within original communication budget.',
        geometry='Cross-source local endpoint vs actual reliable first-return segment; finite beam tolerance. '
            'No-return is unknown. Box interior is NOT declared solid or free.',
        interventions='One local received-source feature substitution per frame, all original heads frozen.',
        oracle='GT is used only by retrospective evaluation/FP oracle; no GT box or score inserted. '
            'Hindsight selector preserves baseline matched IDs and adds no FP at IoU .7; NOT AP-maximizing.',
        limitation='Finite existing-proposal pool; cannot exclude gains from learned reconstruction. '
            'Benchmark-audit is exploratory test-set diagnosis, not trained-method benchmark performance.')
    rt.write_json(out/'protocol.json', protocol)
    stats = {m: empty_stats() for m in MODES}
    scene_stats = {}
    totals = dict(frames=0, baseline_tp=0, baseline_fp=0, baseline_fn=0, evidence_flagged_boxes=0,
        flagged_true_positives=0, flagged_false_positives=0, evaluated_actions=0, eligible_actions=0,
        useful_pool_actions=0, useful_eligible_actions=0, harmful_eligible_actions=0,
        feature_bytes=0, geometry_extra_bytes=0, no_received_peer_frames=0,
        screened_base_boxes=0, received_endpoints=0, valid_rays=0, frames_with_eligible_action=0)
    tick = time.perf_counter()

    def measure(boxes, scores, gt):
        frame = empty_stats()
        for threshold in frame:
            eval_utils.caluclate_tp_fp(boxes, scores, gt, frame, threshold)
        return frame

    with torch.no_grad(), (out/'frames.jsonl').open('w', encoding='utf-8') as frames, \
            (out/'actions.jsonl').open('w', encoding='utf-8') as action_log:
        for number, batch in enumerate(loader, 1):
            batch = to_device(batch, device)
            ego = batch['ego']
            sample = int(ego['communication_sample_index'][0])
            scene = bisect.bisect_right(ds.len_record, sample)
            scene_stats.setdefault(scene, {m: empty_stats() for m in MODES})
            inp = rt.input_branch(ego, branch)
            encoded = model.encode(inp)
            base_prediction, diag = model.run(encoded, args.baseline)
            masks = model.empty_masks(encoded)
            for peer, ids in enumerate(diag['selected_ids'], 1):
                masks[peer].view(-1)[ids] = 1
            received = received_levels(encoded, masks, model.engine.value_bytes)
            fused = fused_levels(received, masks)
            replay = detect(model.engine.base, fused)
            for key in replay:
                torch.testing.assert_close(replay[key], base_prediction[key], atol=2e-4, rtol=2e-4)
            boxes, scores, gt = ds.post_process(batch, {'ego': base_prediction})
            b, s = numpy_detection(boxes, scores)
            g = gt.cpu().numpy()
            baseline_stats = measure(boxes, scores, gt)
            assigned = assignments(b, s, g)
            base_match = set(assigned[assigned >= 0].tolist())
            base_fp = int((assigned < 0).sum())
            if len(base_match) != sum(baseline_stats[.7]['tp']) or base_fp != sum(baseline_stats[.7]['fp']):
                raise AssertionError('Hindsight matcher does not match official evaluator')
            if args.baseline == 'full':
                full_stats = copy.deepcopy(baseline_stats)
                full_prediction = base_prediction
            else:
                full_prediction, _ = model.run(encoded, 'full')
                fb, fs, fg = ds.post_process(batch, {'ego': full_prediction})
                torch.testing.assert_close(gt, fg)
                full_stats = measure(fb, fs, fg)
            if number == 1:
                original = model.engine.base(inp)
                for key in ('psm', 'rm'):
                    torch.testing.assert_close(original[key], full_prediction[key], atol=2e-4, rtol=2e-4)
                del original
            sources = build_sources(model.engine.base, inp, masks, model.grid, model.lidar_range, settings)
            donors = []
            for peer in range(1, len(masks)):
                if not masks[peer].any():
                    continue
                pred = detect(model.engine.base, [x[peer:peer+1] for x in received])
                db, dscores, dg = ds.post_process(batch, {'ego': pred})
                torch.testing.assert_close(gt, dg)
                nb, ns = numpy_detection(db, dscores)
                donors.append((peer, nb, ns))
            mask_array = masks.cpu().numpy()
            actions, evidence_rows = make_actions(b, s, donors, sources, mask_array, model.lidar_range, settings)
            flagged = np.zeros(len(b), bool)
            for row in evidence_rows:
                if row['kind'] == 'correct':
                    flagged[row['prediction_index']] = any(c >= settings.min_conflicts and f >= settings.conflict_fraction
                        for c, f in zip(row['conflict_counts'], row['conflict_fractions']))
            mode_stats = dict(baseline=baseline_stats, full_reference=full_stats)
            for mode, remove in [('generic_fp_oracle70', assigned < 0),
                                 ('evidence_fp_oracle70', flagged & (assigned < 0))]:
                if boxes is None:
                    mode_stats[mode] = copy.deepcopy(baseline_stats)
                else:
                    keep = torch.as_tensor(~remove, device=boxes.device)
                    mode_stats[mode] = measure(boxes[keep], scores[keep], gt)
            candidates = []
            for action_index, action in enumerate(actions):
                prediction = detect(model.engine.base, apply_action(fused, received, masks, action))
                cb, cs, cg = ds.post_process(batch, {'ego': prediction})
                torch.testing.assert_close(gt, cg)
                nb, ns = numpy_detection(cb, cs)
                assignment = assignments(nb, ns, g)
                matched, fp = set(assignment[assignment >= 0].tolist()), int((assignment < 0).sum())
                candidate_stats = measure(cb, cs, cg)
                if len(matched) != sum(candidate_stats[.7]['tp']) or fp != sum(candidate_stats[.7]['fp']):
                    raise AssertionError('Candidate matcher differs from official evaluator')
                gained, lost = len(matched-base_match), len(base_match-matched)
                useful = not lost and fp <= base_fp and (gained > 0 or fp < base_fp)
                harmful = lost > 0 or fp > base_fp
                candidates.append(dict(matched=matched, fp=fp, eligible=action['eligible'], stats=candidate_stats))
                serial = {key: value for key, value in action.items() if key != 'region'}
                action_log.write(json.dumps(dict(sample_index=sample, action_index=action_index,
                    **serial, block_ids=np.flatnonzero(action['region']).tolist(),
                    gained=gained, lost=lost, fp_change=fp-base_fp, useful=useful,
                    ap_inputs=candidate_stats))+'\n')
                totals['useful_pool_actions'] += int(useful)
                totals['useful_eligible_actions'] += int(useful and action['eligible'])
                totals['harmful_eligible_actions'] += int(harmful and action['eligible'])
            rule_options = [i for i, a in enumerate(actions) if a['eligible']]
            rule = max(rule_options, key=lambda i: actions[i]['priority']) if rule_options else None
            chosen = dict(evidence_rule_feature=rule,
                pool_random_feature=(int(np.random.default_rng(int(options['seed'])+sample).integers(len(actions)))
                                     if actions else None),
                evidence_hindsight_feature=choose_hindsight(base_match, base_fp, candidates, True),
                pool_hindsight_feature=choose_hindsight(base_match, base_fp, candidates, False))
            for mode, index in chosen.items():
                mode_stats[mode] = baseline_stats if index is None else candidates[index]['stats']
            for mode in MODES:
                merge(stats[mode], mode_stats[mode]); merge(scene_stats[scene][mode], mode_stats[mode])
            extra_bytes = sum(source.certificate_bytes for source in sources[1:])
            frames.write(json.dumps(dict(sample_index=sample, scene=scene, selected=chosen,
                evidence_rows=evidence_rows, feature_message_bytes=diag['total_bytes'],
                ideal_geometry_extra_bytes=extra_bytes,
                source_endpoints=[len(x.xyz) for x in sources], source_valid_rays=[len(x.ranges) for x in sources],
                ap_inputs=mode_stats))+'\n')
            totals['frames'] += 1
            totals['baseline_tp'] += len(base_match); totals['baseline_fp'] += base_fp
            totals['baseline_fn'] += len(g)-len(base_match)
            totals['evidence_flagged_boxes'] += int(flagged.sum())
            totals['flagged_true_positives'] += int((flagged & (assigned >= 0)).sum())
            totals['flagged_false_positives'] += int((flagged & (assigned < 0)).sum())
            totals['evaluated_actions'] += len(actions)
            totals['eligible_actions'] += sum(a['eligible'] for a in actions)
            totals['screened_base_boxes'] += sum(row['kind'] == 'correct' for row in evidence_rows)
            totals['received_endpoints'] += sum(len(x.xyz) for x in sources)
            totals['valid_rays'] += sum(len(x.ranges) for x in sources)
            totals['frames_with_eligible_action'] += int(bool(rule_options))
            totals['feature_bytes'] += diag['total_bytes']; totals['geometry_extra_bytes'] += extra_bytes
            totals['no_received_peer_frames'] += int(not masks[1:].any())
            if number == 1 or number % 20 == 0:
                elapsed = time.perf_counter()-tick
                print(f'{args.phase}/{args.weather} {number}/{len(indices)}, {elapsed/number:.2f}s/frame, '
                      f'ETA {(len(indices)-number)*elapsed/number/3600:.2f}h, '
                      f'eligible/useful={totals["eligible_actions"]}/{totals["useful_eligible_actions"]}', flush=True)
                frames.flush(); action_log.flush()
    if totals['frames'] != len(indices) or not stats['baseline'][.7]['gt']:
        raise RuntimeError('Incomplete/no-GT audit; do not interpret as feasibility result')
    results = {}
    for mode, values in stats.items():
        folder = out/mode
        folder.mkdir()
        eval_utils.eval_final_results(copy.deepcopy(values), str(folder), False)
        results[mode] = ap_values(values, eval_utils)
        results[mode].update(tp70=sum(values[.7]['tp']), fp70=sum(values[.7]['fp']),
                             fn70=values[.7]['gt']-sum(values[.7]['tp']))
    summary = dict(protocol=protocol, totals=totals, results=results,
        scene_results={scene: {mode: ap_values(values, eval_utils) for mode, values in rows.items()}
                       for scene, rows in scene_stats.items()})
    rt.verify_frozen()
    if rt.sha256(args.frontend_checkpoint) != digest:
        raise RuntimeError('Checkpoint changed during audit')
    rt.write_json(out/'summary.json', summary)
    from .report import write_weather
    write_weather(summary, out/'audit.md')
    print(f'Completed: {out}/audit.md', flush=True)


if __name__ == '__main__':
    main()
