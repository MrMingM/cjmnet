"""Stage-3C frozen local oracle, one focal target per sampled frame."""
import copy
import json
import time
from pathlib import Path
import numpy as np
from gspr_evidence import stage3_runtime as sr
from gspr_evidence.stage3_analysis import read_json, write_json
from .common import action_grid, global_key, choose_action, compare_full_replay
from .operators import WeightCache, masks_for_frame, replace_output


def baseline_row(full, failures):
    return dict(name='KEEP_FULL', family='baseline', scope='baseline', matched_gt=full['matched_gt'],
        recovered_candidates=[], lost_gt=[], new_fp_count=0, fp_count_delta=0, net_matched_gt=0,
        focal_detected=True, retained_gt_score_deltas={}, source='current_full')


def enrich(row, baseline, focus, failures, gt):
    before, after = set(baseline['matched_gt']), set(row['matched_gt'])
    row['recovered_candidates'] = sorted(set(failures) & after)
    row['lost_gt'] = sorted(before-after)
    row['newly_matched_gt'] = sorted(after-before)
    row['net_matched_gt'] = len(after)-len(before)
    row['focal_detected'] = focus in after
    centers = gt[:, :4, :2].mean(1)
    nearby = {j for j, c in enumerate(centers) if np.linalg.norm(c-centers[focus]) <= 10.}
    row['lost_within10m_gt'] = sorted((before-after)&nearby)
    row['lost_beyond10m_gt'] = sorted((before-after)-nearby)
    def scores(r):
        return {int(j): float(s) for j, s in zip(r['final_assigned_gt'], r['final_scores']) if j >= 0}
    bs, now = scores(baseline), scores(row)
    row['retained_gt_score_deltas'] = {str(j): now[j]-bs[j] for j in sorted(before & after)}
    row['retained_gt_drop_over_005'] = [int(j) for j, d in row['retained_gt_score_deltas'].items() if d < -.05]
    return row


def main():
    p = sr.parser(__doc__)
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    plan_path = Path(args.plan).resolve(); plan = read_json(plan_path)
    if plan['stage2_root'] != str(Path(args.stage2_root).resolve()):
        raise ValueError('Sampling/Stage-2 root mismatch')
    rows = plan['weathers'][args.weather]['rows']
    smoke = args.smoke
    if smoke < 0:
        raise ValueError('Negative smoke count')
    if smoke:
        rows = ([r for r in rows if r['cohort']=='candidate'][:smoke]
                + [r for r in rows if r['cohort']=='control'][:1])
    selected = {r['sample_index']: r for r in rows}
    if len(selected) != len(rows):
        raise ValueError('Duplicate sampled frame')
    for name, old in plan['input_files'].items():
        if sr.sha(name) != old:
            raise ValueError('Sampling input changed: '+name)
    for name, old in plan['code_sha256'].items():
        if sr.source_sha(sr.ROOT/name) != old:
            raise ValueError('Stage-3C code changed since sampling: '+name)
    broot = Path(plan['stage3b_root']); bprotocol = read_json(broot/args.weather/'protocol.json')
    # Preserve the complete deterministic weather queue; smoke selects inference only.
    args.smoke = 0
    rt, lineage, model, device, ds, loader, _, out, protocol = sr.prepare(args)
    for key in ('frontend_sha256','frontend_config_sha256','experiment_config_sha256','software','audited_sources','postprocessing'):
        if protocol[key] != bprotocol[key]:
            raise ValueError('Stage-3B/C mismatch: '+key)
    for name, old in bprotocol['implementation_sha256'].items():
        if protocol['implementation_sha256'].get(name) != old:
            raise ValueError('Stage-3B implementation drift: '+name)
    lineage['input_files'].update(plan['input_files'])
    lineage['input_files'][str(plan_path)] = sr.sha(plan_path)
    lineage['implementation'].update(plan['code_sha256'])
    protocol.update(stage='3C', smoke=smoke, plan_sha256=sr.sha(plan_path), plan=str(plan_path),
        selected_candidate_frames=sorted(selected), candidate_targets=sum(r['cohort']=='candidate' for r in rows),
        sampled_rows=rows, new_implementation_sha256=plan['code_sha256'],
        control_definition=plan['controls'], scientific_scope='GT-localized diagnostic, fixed features; not AP or deployable selection',
        selection_rule='max candidate recovery, max net matched GT, min lost GT, min new FP; prefer KEEP_FULL on ties',
        action_grid=list(action_grid()), near_distance_m=10., score_drop_reporting_threshold=.05,
        global_comparison='Same peer/family/scales/alpha; reuse Stage-3B exact actions; recompute missing controls, mixed weights and scales01',
        localization='Feature change spatially masked before deblocks. Downstream receptive fields and NMS may affect outside ROI; measured, not masked away.')
    write_json(out/'protocol.json', protocol)
    previous = {}
    with (broot/args.weather/'diagnostics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            if r['sample_index'] in selected:
                previous[r['sample_index']] = dict(input_sha256=r['input_sha256'], candidate_targets=r['candidate_targets'],
                    baseline_targets=r['baseline_targets'], branches=[b for b in r['branches'] if b['family'] in
                        ('subset','peer_score_full_geometry','full_score_peer_geometry','peer_query_scale_0','peer_query_scale_1')])
    import torch
    from opencood.tools.train_utils import to_device
    from gspr_evidence.stage3_trace import trace_branch, describe_target
    from gspr_evidence.stage3_diagnostic_runtime import FrameAudit
    start = time.monotonic(); completed = []
    (out/'frames').mkdir()
    with torch.no_grad():
        for batch in loader:
            index = int(batch['ego']['communication_sample_index'][0])
            if index not in selected:
                continue
            spec = selected[index]; focus = spec['target_index']
            batch = to_device(batch, device); inp = rt.input_branch(batch['ego'], args.weather)
            encoded = model.encode(inp)
            predictions, observed, gt = sr.replay_frame(model, ds, batch, encoded, inp, index, lineage)
            input_hash = sr.tensor_digest(dict(inp=inp, gt=gt))
            if spec['input_sha256'] is not None and input_hash != spec['input_sha256']:
                raise ValueError('Stage-3B input hash mismatch')
            torch.testing.assert_close(batch['ego']['transformation_matrix'],
                                       torch.eye(4, device=device), rtol=0, atol=1e-6)
            full_trace = trace_branch(ds, batch, predictions['full'])
            gt_np = gt.detach().cpu().numpy()
            if bool(full_trace['details'][focus]['matched']) != (spec['cohort']=='control'):
                raise ValueError('Focal baseline state changed')
            failures = previous[index]['candidate_targets'] if index in previous else []
            if spec['cohort']=='candidate' and focus not in failures:
                raise ValueError('Focal failure not in historical candidates')
            if spec['cohort']=='control' and any(r['source_valid_full_miss'] for (i,j),r in lineage['rows'].items() if i==index):
                raise ValueError('Control frame contains a Stage-3B candidate')
            traced_peers = {}
            for peer in spec['valid_peers']:
                tr = trace_branch(ds, batch, predictions['peer_'+str(peer)])
                if not tr['details'][focus]['matched']:
                    raise ValueError('Valid focal peer changed')
                traced_peers[peer] = describe_target(tr, focus)
            target_ids = sorted(set(failures) | {focus})
            arows = {}
            for j in target_ids:
                d = describe_target(full_trace, j)
                arows[j] = dict(full=d, failure_stage=d['failure_stage'],
                    peers=[dict(trace=t) for t in traced_peers.values()] if j==focus else [])
                if index in previous and j in failures and d['failure_stage'] != previous[index]['baseline_targets'][str(j)]['failure_stage']:
                    raise ValueError('Stage-3B failure stage changed')
            audit = FrameAudit(ds, batch, predictions, arows, out, index)
            full = audit.add('full', 'baseline', predictions['full'], trace=full_trace)
            replay_comparison = None
            if index in previous:
                old_full = max((b for b in previous[index]['branches'] if b['family']=='subset'), key=lambda b: len(b['subset']))
                replay_comparison = compare_full_replay(full, old_full)
                print(f'frame={index} Stage-3B full replay: '+json.dumps(replay_comparison), flush=True)
            baseline = baseline_row(full, failures); baseline['focal_detected'] = focus in full['matched_gt']
            cache = WeightCache(model, encoded)
            identity = cache.decode(cache.original)
            for key in ('psm','rm'):
                torch.testing.assert_close(identity[key], predictions['full'][key], atol=2e-4, rtol=2e-4)
            identity_trace = trace_branch(ds, batch, identity)
            np.testing.assert_array_equal(identity_trace['ids']['range'], full_trace['ids']['range'])
            np.testing.assert_array_equal(identity_trace['assignment'], full_trace['assignment'])
            del identity_trace, identity
            masks = {radius: masks_for_frame(batch['ego']['anchor_box'], gt_np[focus], encoded['levels'], model.lidar_range, radius)
                     for radius in (1.,1.5)}
            anchor_shape = tuple(full_trace['anchor_shape'][:-1])
            for hm, fm, stats in masks.values():
                stats['focal_watched_anchor_inside_output_roi'] = {
                    str(cid): bool(hm[np.unravel_index(cid, anchor_shape)[:2]]) for cid in audit.watched[focus]}
            local_rows, global_rows, pair_rows = [], [], []
            for peer in spec['valid_peers']:
                global_cache = {}
                def evaluate(name, family, pred, scope, **meta):
                    r = audit.add(name, family, pred, **meta)
                    r.update(scope=scope, peer=peer, focal_target=focus, source='current_inference')
                    return enrich(r, full, focus, failures, gt_np)
                for action in action_grid():
                    family, radius, scales, alpha = (action[k] for k in ('family','radius','scales','alpha'))
                    key = global_key(family, scales, alpha)
                    if key not in global_cache:
                        old_family = {'score':'peer_score_full_geometry', 'geometry':'full_score_peer_geometry'}.get(family)
                        if family=='weights' and alpha==1 and len(scales)==1:
                            old_family = 'peer_query_scale_'+str(scales[0])
                        old = next((b for b in previous.get(index, {}).get('branches', [])
                                    if b['family']==old_family and b.get('peer')==peer), None)
                        if old is not None:
                            global_r = copy.deepcopy(old)
                            global_r.update(name=f'global_p{peer}_{key}', family=family, scope='global',
                                peer=peer, scales=scales, alpha=alpha, source='verified_stage3b_cache')
                            enrich(global_r, full, focus, failures, gt_np)
                        else:
                            pred = (replace_output(predictions['full'], predictions['peer_'+str(peer)],
                                    np.ones_like(masks[radius][0]), family) if family!='weights'
                                    else cache.predict(peer, scales, alpha))
                            global_r = evaluate(f'global_p{peer}_{key}', family, pred, 'global', scales=scales, alpha=alpha)
                        global_cache[key] = global_r; global_rows.append(global_r)
                    hm, fm, mask_stats = masks[radius]
                    pred = (replace_output(predictions['full'], predictions['peer_'+str(peer)], hm, family)
                            if family!='weights' else cache.predict(peer, scales, alpha, fm))
                    local = evaluate(f'local_p{peer}_{key}_r{radius}', family, pred, 'local', radius=radius,
                                     scales=scales, alpha=alpha, mask_stats=mask_stats)
                    local_rows.append(local)
                    global_r = global_cache[key]
                    pair_rows.append(dict(local=local['name'], global_action=global_r['name'],
                        family=family, radius=radius, scales=scales, alpha=alpha, peer=peer,
                        local_focus_detected=local['focal_detected'], global_focus_detected=global_r['focal_detected'],
                        local_lost=len(local['lost_gt']), global_lost=len(global_r['lost_gt']),
                        local_new_fp=local['new_fp_count'], global_new_fp=global_r['new_fp_count']))
            decisions = {}
            for scope, actions in (('local', local_rows),('global',global_rows)):
                for family in ('all','score','geometry','weights'):
                    eligible = actions if family=='all' else [r for r in actions if r['family']==family]
                    for safe in (False,True):
                        r = choose_action(eligible, baseline, safe=safe)
                        decisions[f'{scope}/{family}/'+('zero_cost' if safe else 'recovery_first')] = dict(
                            action=r['name'], recovered_candidates=r['recovered_candidates'], lost_gt=r['lost_gt'],
                            new_fp_count=r['new_fp_count'], net_matched_gt=r['net_matched_gt'],
                            focal_detected=r['focal_detected'])
            frame = dict(sample_index=index, weather=args.weather, cohort=spec['cohort'], focal_target=focus,
                sampling=spec, input_sha256=input_hash, gt_corners=gt_np.tolist(), candidate_targets=failures,
                baseline=full, replay_comparison=replay_comparison,
                local_actions=local_rows, global_actions=global_rows, paired_actions=pair_rows,
                frame_decisions=decisions, roi_stats={str(k):v[2] for k,v in masks.items()},
                control_note='All control interventions are reported, not only GT-selected safe actions.')
            write_json(out/'frames'/f'{index}.json', frame)
            completed.append(index)
            elapsed=time.monotonic()-start
            print(f'{args.weather} {len(completed)}/{len(selected)}, elapsed={elapsed/60:.1f}min, ETA={elapsed/len(completed)*(len(selected)-len(completed))/60:.1f}min', flush=True)
            del audit, full_trace, cache, encoded, predictions, frame, local_rows, global_rows
            if len(completed)==len(selected):
                break
    if sorted(completed)!=sorted(selected):
        raise ValueError('Incomplete sampled frames')
    sr.final_guards(lineage)
    write_json(out/'summary.json', dict(complete=True, stage='3C', smoke=smoke, frames=len(completed),
        frame_sha256={str(i):sr.sha(out/'frames'/f'{i}.json') for i in completed}, plan_sha256=sr.sha(plan_path)))


if __name__ == '__main__':
    main()
