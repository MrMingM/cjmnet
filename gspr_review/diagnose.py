"""Replay a trained reviewer on fixed received evidence/support; no optimization or AP claims."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', help='Completed review training directory; auto-select only if exactly one exists')
    parser.add_argument('--output-dir')
    parser.add_argument('--max-frames', type=int, default=0, help='Per branch; 0 uses saved development selection')
    parser.add_argument('--target-margin', type=float, default=1., help='Meters outside GT BEV footprint')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--interaction-points', type=int, default=2048,
                        help='Max fixed-eligibility point samples per frame for two-source interaction; 0 disables')
    args = parser.parse_args()
    if args.max_frames < 0 or args.target_margin < 0 or args.epsilon <= 0 or args.interaction_points < 0:
        parser.error('Invalid diagnostic limits')
    if args.run_dir:
        run = Path(args.run_dir).resolve()
    else:
        choices = sorted(p.parent for p in Path('/data/cjm/datasets/logs').glob('gspr_review_seed*/reviewer_best.pth'))
        if len(choices) != 1:
            parser.error('Specify --run-dir: expected one completed review run, found ' + str([str(p) for p in choices]))
        run = choices[0]
    import numpy as np
    import torch
    from opencood.tools.train_utils import to_device
    from . import runtime as rt
    from .evidence import point_cells
    from .diagnostic_utils import (near_boxes_xy, altered_inputs, fixed_mask_weights,
                                   point_totals, pillar_totals, summarize_rows)
    rt.verify_frozen()
    options, hypes = rt.load_config(run/'experiment.yaml', run/'frontend_config.yaml')
    if options['review'].get('architecture') == 'bev_contrast':
        raise ValueError('Point/PFN replay does not measure BEV correction; use run_location_experiment.sh diagnostics')
    manifest = json.loads((run/'manifest.json').read_text())
    target = rt.device()
    rt.seed_all(int(options['seed']))
    model, digest = rt.load_model(hypes, options, manifest['frontend_checkpoint'], target)
    if digest != manifest['frontend_sha256']:
        raise ValueError('Training manifest frontend hash mismatch')
    checkpoint = run/'reviewer_best.pth'
    reviewer_hash = rt.sha256(checkpoint)
    rt.load_reviewer(model, checkpoint, options, digest, run/'frontend_config.yaml')
    model.eval()
    out = rt.new_output(args.output_dir or str(run)+'_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    print('Read-only diagnostics:', run, '->', out, flush=True)
    rt.write_json(out/'protocol.json', {'training_run': str(run), 'options': options,
        'frontend_sha256': digest, 'reviewer_sha256': reviewer_hash, 'margin': args.target_margin,
        'epsilon': args.epsilon, 'max_frames_per_branch': args.max_frames,
        'interaction_points': args.interaction_points,
        'interaction_reference': 'Local: mean seven inputs among eligible points in this frame; peer: equal reliable/noise mass, preserving actual u/support. Fixed eligibility; bounded pre-clamp update. Not dataset-level InterSHAP or AP.',
        'box_order': hypes['postprocess']['order'],
        'scope': 'Fixed eligibility and one real packet roundtrip per frame. GT only groups statistics after inference. No retraining; no intervention AP.',
        'source_sha256': {str(p.relative_to(rt.ROOT)): rt.sha256(p) for p in (rt.ROOT/'gspr_review').glob('*.py')}})
    captured = {}
    def capture_review(module, inputs, output):
        captured['args'], captured['result'] = inputs, output
    def capture_net(module, inputs):
        captured['net_input'] = inputs[0]
    handles = [model.reviewer.register_forward_hook(capture_review), model.reviewer.net.register_forward_pre_hook(capture_net)]
    rows = []
    try:
        with torch.no_grad(), (out/'frames.jsonl').open('w', encoding='utf-8') as stream:
            for branch in ('processed_lidar', 'processed_lidar_weather'):
                rt.seed_all(int(options['seed']))
                _, loader, indices = rt.make_loader(hypes, options, test=True)
                for i, batch in enumerate(loader):
                    if args.max_frames and i >= args.max_frames:
                        break
                    batch = to_device(batch, target)
                    captured.clear()
                    model(rt.input_branch(batch['ego'], options, branch))
                    inputs = captured.get('net_input')
                    processed, rel, ids, received, grid, z_range, z_bins = captured['args']
                    normal, eligible = captured['result']
                    normal = normal.clone()
                    original = rel['point_reliability']
                    valid = rel['point_valid_mask']
                    _, cell = point_cells(processed, grid, z_range, z_bins, ids)
                    def pillars(weights):
                        local = dict(processed, point_reliability=weights)
                        return model.base.pillar_vfe(local)['pillar_features']
                    baseline = pillars(original).clone()
                    repeated = pillars(original)
                    repeat_error = float((baseline-repeated).abs().max()) if baseline.numel() else 0.
                    # GT enters only this analysis code, never model/reviewer inputs.
                    ego = batch['ego']
                    boxes = ego['object_bbx_center'][0][ego['object_bbx_mask'][0].bool()].cpu().numpy()
                    xyz = processed['voxel_features'][..., :3].cpu().numpy()
                    near = near_boxes_xy(xyz, boxes, hypes['postprocess']['order'], args.target_margin)
                    coords = processed['voxel_coords'].cpu().numpy()
                    voxel_size = hypes['model']['args']['voxel_size']
                    extent = hypes['model']['args']['lidar_range']
                    centers = np.stack(((coords[:, 3]+.5)*voxel_size[0]+extent[0], (coords[:, 2]+.5)*voxel_size[1]+extent[1]), -1)
                    pillar_near = near_boxes_xy(centers, boxes, hypes['postprocess']['order'], args.target_margin)
                    eligible_np, valid_np = eligible.cpu().numpy(), valid.cpu().numpy()
                    row = {'branch': branch, 'sample_index': int(ego['communication_sample_index'][0]),
                           'baseline_repeat_max_error': repeat_error, 'communication': dict(model.last_diagnostics[0]), 'variants': {}}
                    for variant in ('normal', 'permuted', 'constant', 'zero'):
                        if inputs is None:
                            changed_inputs, weights, peer_change = None, original, 0.
                        else:
                            changed_inputs = altered_inputs(inputs, cell, eligible, variant)
                            weights = fixed_mask_weights(model.reviewer, changed_inputs, original, eligible)
                            peer_change = float((changed_inputs[..., -4:][eligible]-inputs[..., -4:][eligible]).abs().sum())
                        if variant == 'normal':
                            torch.testing.assert_close(weights, normal, atol=1e-7, rtol=1e-6)
                        torch.testing.assert_close(weights[~eligible], original[~eligible], atol=0, rtol=0)
                        features = pillars(weights)
                        groups = {'peer_input_abs_change_sum': peer_change}
                        for name, region, pillar_region in (
                            ('all', np.ones_like(near), np.ones_like(pillar_near)),
                            ('target_near', near, pillar_near),
                            ('outside_target_near', ~near, ~pillar_near)):
                            groups[name] = point_totals(original.cpu().numpy(), weights.cpu().numpy(), normal.cpu().numpy(), eligible_np, valid_np, region, args.epsilon)
                            groups[name].update(pillar_totals(baseline.cpu().numpy(), features.cpu().numpy(), eligible_np.any(1), pillar_region, args.epsilon))
                        row['variants'][variant] = groups
                    if inputs is not None and eligible.any() and args.interaction_points:
                        from .counterfactual import neutral_reference, two_source_interactions
                        selected = inputs[eligible]
                        local_mean = selected[:, :7].mean(0)
                        sample_ids = torch.linspace(0, len(selected)-1,
                            min(len(selected), args.interaction_points), device=selected.device).long()
                        x11 = selected[sample_ids]
                        x10 = neutral_reference(x11)
                        x01 = x11.clone()
                        x01[:, :7] = local_mean
                        x00 = neutral_reference(x01)
                        def value(x):
                            return model.reviewer.maximum*model.reviewer.net(x).squeeze(-1).tanh()
                        f00, f10, f01, f11 = [value(x) for x in (x00, x10, x01, x11)]
                        local_main, peer_main, off = two_source_interactions(f00, f10, f01, f11)
                        row['interaction'] = {'points': len(x11),
                            'local_main_abs_sum': float(local_main.abs().sum()),
                            'peer_main_abs_sum': float(peer_main.abs().sum()),
                            'off_diagonal_abs_sum': float(2*off.abs().sum()),
                            'message_effect_abs_sum': float((f11-f10).abs().sum()),
                            'completeness_max_error': float((local_main+peer_main+2*off-(f11-f00)).abs().max())}
                        if hasattr(model.reviewer.net, 'components'):
                            contrast, gain = model.reviewer.net.components(x11)
                            row['interaction'].update(contrast_abs_sum=float(contrast.abs().sum()),
                                                       gain_sum=float(gain.sum()))
                    rows.append(row)
                    stream.write(json.dumps(row)+'\n')
                    if (i+1) % 10 == 0:
                        print(f'{branch}: {i+1}/{min(len(indices), args.max_frames or len(indices))} frames', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if not rows:
        raise RuntimeError('No diagnostic frames')
    summary = summarize_rows(rows)
    for branch, data in summary.items():
        entries = [r['interaction'] for r in rows if r['branch'] == branch and 'interaction' in r]
        if entries:
            n = sum(e['points'] for e in entries)
            data['two_source_reference_interaction'] = {'points': n,
                **{key.removesuffix('_sum')+'_mean': sum(e[key] for e in entries)/n
                   for key in entries[0] if key.endswith('_sum')},
                'completeness_max_error': max(e['completeness_max_error'] for e in entries)}
    rt.write_json(out/'summary.json', summary)
    if rt.sha256(checkpoint) != reviewer_hash:
        raise RuntimeError('Reviewer checkpoint changed during diagnostics')
    rt.verify_frozen()
    print(json.dumps(summary, indent=2), flush=True)
    print('Send back:', out/'summary.json', flush=True)


if __name__ == '__main__':
    main()
