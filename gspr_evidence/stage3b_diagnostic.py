"""Extended Stage-3B: paired source, head-output and attention interventions."""
import json
from pathlib import Path
import numpy as np
from . import stage3_runtime as sr
from .stage3_analysis import read_json, read_rows, unique_rows, write_json, enumerate_subsets, choose_frame_subset, greedy_assignment


def fp_boxes(boxes, scores, gt):
    """Actual final false positives under the fixed Stage-2 GT matcher."""
    from .stage3_trace import polygon_ious
    b = np.zeros((0, 8, 3), dtype=np.float32) if boxes is None else boxes.detach().cpu().numpy()
    s = np.zeros(0, dtype=np.float32) if scores is None else scores.detach().cpu().numpy()
    assignment = greedy_assignment(s, polygon_ious(b, gt.detach().cpu().numpy()))
    ids = np.flatnonzero(assignment == -1)
    return b[ids], s[ids], ids


def main():
    p = sr.parser(__doc__)
    p.add_argument('--stage3a-root', required=True)
    p.add_argument('--save-dense', action='store_true', help='Save all decoded proposals for every intervention; large disk cost')
    args = p.parse_args()
    aroot = sr.safe_output(args.stage3a_root)
    aresult = read_json(aroot/'stage3a_results.json')
    if aresult['smoke'] and not args.smoke:
        raise ValueError('A smoke run cannot authorize full Stage-3B')
    aprotocol = read_json(aroot/args.weather/'protocol.json')
    if (aprotocol['stage'] != '3A' or not aprotocol['development_only'] or
            aprotocol['test_data_used'] or Path(aprotocol['stage2_root']).resolve() != Path(args.stage2_root).resolve()):
        raise ValueError('Stage-3A lineage mismatch')
    atargets = unique_rows(read_rows(aroot/args.weather/'targets.jsonl'))
    afiles = {str(aroot/n): sr.sha(aroot/n) for n in
              ('stage3a_results.json', args.weather+'/protocol.json', args.weather+'/targets.jsonl')}
    if sr.sha(aroot/args.weather/'targets.jsonl') != aresult['weathers'][args.weather]['targets_sha256']:
        raise ValueError('Stage-3A target log has changed since its report')
    import torch
    from opencood.tools.train_utils import to_device
    from .stage3_trace import polygon_ious
    rt, lineage, model, device, ds, loader, candidates, out, protocol = sr.prepare(args)
    for field in ('frontend_sha256', 'frontend_config_sha256', 'experiment_config_sha256',
                  'audited_sources', 'postprocessing', 'software'):
        if protocol[field] != aprotocol[field]:
            raise ValueError('Stage-3A/B mismatch: '+field)
    # New orchestration is expected. Historical replay/tracer/math must be identical.
    for name in ('stage3_runtime.py', 'stage3_trace.py', 'stage3_analysis.py'):
        key = 'gspr_evidence/'+name
        if protocol['implementation_sha256'].get(key) != aprotocol['implementation_sha256'].get(key):
            raise ValueError('Historical A core implementation changed: '+key)
    protocol.update(diagnostic_schema=1, save_dense=args.save_dense,
        intervention_contract='All subsets; peer/full psm-rm swaps; fixed keys/values query changes all and individual scales; uniform weights; original weights with zero ego value; full-output score x NMS sweep',
        historical_core_verified=['stage3_runtime.py', 'stage3_trace.py', 'stage3_analysis.py'])
    from .stage3_diagnostic_runtime import FrameAudit
    import time
    started = time.monotonic()
    protocol.update(stage='3B', stage3a_root=str(aroot),
        stage3a_input_files=afiles,
        selection_rule='max recovered candidates; min lost baseline GT; min new FP; lexicographic subset',
        fp_definition='Match subset final FPs to full final FPs one-to-one, stable descending subset score, planar IoU>=0.7. Unmatched subset FP is new. This is geometric persistence, not physical object identity; signed count delta is also reported.',
        scope='Fixed-feature vehicle subsets plus explicitly labeled artificial head/attention/postprocess interventions; NOT a fusion theoretical upper bound')
    write_json(out/'protocol.json', protocol)
    frames, total, opportunity, realizable, lost_total, fp_delta, new_fp_total = 0, 0, 0, 0, 0, 0, 0
    with torch.no_grad(), (out/'frames.jsonl').open('w', encoding='utf-8') as stream, (out/'diagnostics.jsonl').open('w', encoding='utf-8') as diagnostic_stream:
        for batch in loader:
            index = int(batch['ego']['communication_sample_index'][0])
            if index not in candidates:
                continue
            batch = to_device(batch, device)
            inp = rt.input_branch(batch['ego'], args.weather)
            encoded = model.encode(inp)
            predictions, observed, gt = sr.replay_frame(model, ds, batch, encoded, inp, index, lineage)
            input_hash = sr.tensor_digest(dict(inp=inp, gt=gt))
            for j in candidates[index]:
                if (index, j) not in atargets or atargets[index, j]['input_sha256'] != input_hash:
                    raise ValueError('Stage-3A input hash mismatch or missing target')
            audit = FrameAudit(ds, batch, predictions, {j: atargets[index, j] for j in candidates[index]}, out, index, args.save_dense)
            baseline, baseline_fp = observed['full']
            bboxes, bscores, _ = ds.post_process(batch, {'ego': predictions['full']})
            baseline_fp_boxes, _, baseline_fp_ids = fp_boxes(bboxes, bscores, gt)
            if len(baseline_fp_boxes) != baseline_fp:
                raise AssertionError('FP assignment differs from Stage-2')
            baseline_ids = {j for j, d in enumerate(baseline) if d['matched']}
            cand_ids = set(candidates[index])
            subsets = []
            for subset in enumerate_subsets(len(encoded['levels'][0])):
                masks = model.empty_masks(encoded)
                for a in subset:
                    masks[a] = 1
                if len(subset) == len(encoded['levels'][0]):
                    pred = predictions['full']
                else:
                    pred, _ = model.detect(encoded, masks, serialize=False)
                traced = audit.add('subset_'+'_'.join(map(str, subset)), 'subset', pred, subset=list(subset),
                    trace=audit.full_trace if len(subset) == len(encoded['levels'][0]) else None)
                # trace_branch already checks original postprocessor and Stage-2
                # matching. Reuse those final outputs instead of running NMS again.
                assigned = np.asarray(traced['final_assigned_gt'], dtype=np.int64)
                final_boxes = np.asarray(traced['final_boxes'], dtype=np.float32).reshape(-1, 8, 3)
                final_scores = np.asarray(traced['final_scores'], dtype=np.float32)
                fids = np.flatnonzero(assigned < 0)
                fboxes, fscores = final_boxes[fids], final_scores[fids]
                correspondence = greedy_assignment(fscores, polygon_ious(fboxes, baseline_fp_boxes))
                new_ids = fids[correspondence < 0].tolist()
                if len(new_ids) != traced['new_fp_count']:
                    raise AssertionError('Diagnostic/standard FP correspondence mismatch')
                subsets.append(dict(subset=list(subset),
                    **{k: traced[k] for k in ('recovered_candidates', 'lost_baseline_gt', 'total_matched_gt',
                        'matched_gt', 'frame_fp', 'fp_count_delta', 'new_fp_count')},
                    new_fp_final_indices=new_ids,
                    fp_final_indices=fids.tolist(), fp_ego_corners=fboxes.tolist(), fp_scores=fscores.tolist(),
                    fp_matched_baseline_final_indices=[int(baseline_fp_ids[k]) if k >= 0 else -1 for k in correspondence]))
            audit.interventions(model, encoded)
            diagnostic_stream.write(json.dumps(audit.result(args.weather, input_hash), allow_nan=False)+'\n')
            diagnostic_stream.flush()
            del audit
            decision = choose_frame_subset(subsets)
            target_opportunities = {str(j): [r['subset'] for r in subsets if j in r['recovered_candidates']]
                                    for j in sorted(cand_ids)}
            row = dict(sample_index=index, weather=args.weather, input_sha256=input_hash,
                       candidate_targets=sorted(cand_ids), subsets=subsets,
                       target_wise_subsets=target_opportunities, frame_oracle=decision,
                       baseline_matched_gt=sorted(baseline_ids), baseline_fp=baseline_fp,
                       baseline_fp_final_indices=baseline_fp_ids.tolist(), baseline_fp_ego_corners=baseline_fp_boxes.tolist())
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            frames += 1; total += len(cand_ids)
            opportunity += len(decision['target_wise_recoverable'])
            realizable += len(decision['chosen']['recovered_candidates'])
            lost_total += len(decision['chosen']['lost_baseline_gt'])
            fp_delta += decision['chosen']['fp_count_delta']
            new_fp_total += decision['chosen']['new_fp_count']
            print(f'{args.weather} diagnostic frames {frames}/{len(candidates)}, elapsed={(time.monotonic()-started)/60:.1f}min, ETA={(time.monotonic()-started)/frames*(len(candidates)-frames)/60:.1f}min', flush=True)
            if args.smoke and frames == len(candidates):
                break
    if frames != len(candidates) or total != protocol['candidate_targets']:
        raise RuntimeError('Incomplete Stage-3B')
    sr.final_guards(lineage)
    for path, digest in afiles.items():
        if sr.sha(path) != digest:
            raise RuntimeError('Stage-3A input changed during Stage-3B')
    write_json(out/'summary.json', dict(complete=True, smoke=args.smoke, frames=frames, occurrences=total,
        target_wise_recoverable=opportunity, frame_wise_recovered=realizable,
        frame_wise_lost_baseline_gt=lost_total, frame_wise_fp_count_delta=fp_delta,
        frame_wise_new_fp=new_fp_total, diagnostics_sha256=sr.sha(out/'diagnostics.jsonl'), diagnostic_schema=1,
        warning='Not AP, not a deployable selector, not a theoretical fusion upper bound.'))


if __name__ == '__main__':
    main()
