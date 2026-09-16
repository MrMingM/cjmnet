"""Cached message-deletion diagnostic on training probes or full validation."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import torch
import yaml
from .core import capture, predict, drop_mask, groups, random_deletions, loss_values, classify
from .config import fusion_args_for_probe


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, dict):
            for k in sorted(x):
                digest.update(k.encode())
                visit(x[k])
        elif isinstance(x, torch.Tensor):
            a = x.detach().cpu().contiguous().numpy()
            digest.update(str((a.shape, a.dtype)).encode())
            digest.update(a.tobytes())
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


def sample_indices(ds, scenes, stride, count):
    if not scenes or len(scenes) != len(set(scenes)) or stride < 1:
        raise ValueError('Require unique training scenes and positive stride')
    streams = []
    for scene in scenes:
        if scene < 0 or scene >= len(ds.len_record):
            raise ValueError('Training scene index out of range')
        streams.append(range(ds.len_record[scene-1] if scene else 0, ds.len_record[scene], stride))
    indices = [i for row in itertools.zip_longest(*streams) for i in row if i is not None][:count]
    if len(indices) < count:
        raise ValueError('Not enough selected training frames; reduce --frames')
    return indices


def identity(ds, index, sample):
    scene = next(i for i, end in enumerate(ds.len_record) if index < end)
    offset = index - (ds.len_record[scene-1] if scene else 0)
    database = ds.scenario_database[scene]
    timestamp = ds.return_timestamp_key(database, offset)
    cav_ids = sample['ego']['cav_ids']
    sources = {str(cav): database[cav][timestamp]['lidar'] for cav in database
               if str(cav) in cav_ids}
    scene_path = str(Path(next(iter(sources.values()))).parent.parent)
    return {'sample_index': index, 'scene_index': scene, 'scene_path': scene_path,
            'timestamp': timestamp, 'cav_ids_in_feature_order': cav_ids,
            'ego_id': cav_ids[0], 'source_lidar': sources}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='gspr_review/experiment.yaml')
    p.add_argument('--frontend-config', required=True)
    p.add_argument('--frontend-checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--full-validation', action='store_true', help='All validate_dir scenes/frames, no subsampling')
    p.add_argument('--calibration-run', help='Completed TRAIN probe supplying frozen epsilon; required for full validation')
    p.add_argument('--frames', type=int, default=8, help='Measured TRAIN frames per weather; calibration excluded')
    p.add_argument('--calibration-frames', type=int, default=2)
    p.add_argument('--scenes', type=int, nargs='+', default=[0, 1])
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--weather', nargs='+', choices=['fog', 'rain', 'snow'], default=['fog', 'rain', 'snow'])
    p.add_argument('--region-blocks', type=int, default=1, help='Square side in native 8-pillar physical blocks')
    p.add_argument('--random-repeats', type=int, default=3)
    p.add_argument('--epsilon-abs', type=float, default=1e-6)
    p.add_argument('--epsilon-rel', type=float, default=1e-5)
    args = p.parse_args()
    if args.full_validation != bool(args.calibration_run):
        p.error('--full-validation and --calibration-run must be used together')
    if min(args.frames, args.calibration_frames, args.random_repeats, args.region_blocks) < 1:
        p.error('Frame counts, random repeats and region size must be positive')
    if args.epsilon_abs <= 0 or args.epsilon_rel < 0:
        p.error('epsilon-abs must be positive; epsilon-rel nonnegative')
    from gspr_communication import runtime as rt, codec
    from gspr_communication.dataset_adapter import CommunicationDataset
    from gspr_review.resources import resolve_weather
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.loss.point_pillar_loss import PointPillarLoss
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from gspr_communication.evaluate import matched_objects

    frozen = rt.verify_frozen()
    original = load_yaml(args.frontend_config)
    fusion_args = fusion_args_for_probe(original['fusion'])
    options, hypes = rt.load_config(args.config, args.frontend_config)
    validation_scenes = None
    if args.full_validation:
        from .validation import check_scene_split
        validation_scenes = check_scene_split(hypes['root_dir'], hypes['validate_dir'])
    # Shared runtime replaces fusion.args; preserve all validated saved options here.
    hypes['fusion']['args'] = fusion_args
    print(f"Fusion config: saved args={original['fusion'].get('args', {})!r}; "
          f"effective args={fusion_args!r}", flush=True)
    options['communication']['variant'] = 'a0b0'
    options['lidar_key'] = 'processed_lidar_weather'
    seed = int(options['seed'])
    rt.seed_all(seed)
    target = rt.device()
    model, checkpoint_hash = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    model.eval().requires_grad_(False)
    criterion = PointPillarLoss(hypes['loss']['args']).to(target).eval()
    before_state = fingerprint(model.state_dict())
    out = rt.new_output(args.output_dir)
    voxel = hypes['model']['args']['voxel_size']
    audit = {'purpose': ('FULL VALIDATION diagnostic; GT-assisted, not independent test' if args.full_validation
                         else 'TRAIN-scene feasibility diagnostic; no learned gate; not independent test'),
        'args': vars(args), 'seed': seed, 'checkpoint_sha256': checkpoint_hash,
        'frontend_config_sha256': rt.sha256(args.frontend_config), 'strict_checkpoint_load': True,
        'actual_model_class': type(model.base).__name__, 'actual_backbone_class': type(model.base.backbone).__name__,
        'saved_model_config': original['model'], 'saved_fusion': original['fusion'],
        'effective_fusion': hypes['fusion'],
        'loss': hypes['loss'], 'preprocess': hypes['preprocess'], 'postprocess': hypes['postprocess'],
        'communication': options['communication'], 'train_root': hypes['root_dir'],
        'evaluation_root': hypes['validate_dir'] if args.full_validation else hypes['root_dir'],
        'validation_scenes': validation_scenes,
        'native_block_xy_metres': [8*float(v) for v in voxel[:2]],
        'region_side_xy_metres': [8*args.region_blocks*float(v) for v in voxel[:2]],
        'frozen_sources': frozen, 'source_hashes': {str(f.relative_to(rt.ROOT)): rt.sha256(f)
            for folder in ('gspr_harm', 'gspr_communication', 'gspr_review')
            for f in (rt.ROOT/folder).glob('*.py')},
        'effect': 'fixed perception pipeline only; no delay or downstream behavior effects',
        'request': 'dense uint8 ego 1-confidence; candidates are ACTUALLY SENT A0B0 blocks, not all nonzero requests',
        'epsilon': 'max(abs floor, rel floor * median calibration total loss, 10 * max repeated component drift)',
        'ap': 'existing planar polygon IoU, global score sort; see purpose for split',
        'bytes': 'actual application packets, request and response headers retained; no refill; excludes radio overhead'}
    # Config parsers can add NumPy arrays (e.g. grid_size).
    audit = json.loads(json.dumps(audit, default=lambda x: x.tolist()))
    rt.write_json(out/'audit.json', audit)
    (out/'frontend_config.yaml').write_bytes(Path(args.frontend_config).read_bytes())
    summaries = {}
    with torch.inference_mode():
        for weather in args.weather:
            wh = copy.deepcopy(hypes)
            # Read TRAIN data using evaluation transforms/collation; never train on the original test split.
            if not args.full_validation:
                wh['validate_dir'] = wh['root_dir']
            cfg = copy.deepcopy(wh.get('weather_augmentation', {}))
            cfg.update(mode='physics_'+weather, seed=seed, apply_to_validation=True)
            wh['weather_augmentation'] = resolve_weather(cfg, rt.ROOT)
            ds = CommunicationDataset(wh, train=False)
            calibration_count = 0 if args.full_validation else args.calibration_frames
            if args.full_validation:
                from .validation import validation_indices, frozen_epsilon
                indices = validation_indices(ds)
                epsilon, epsilon_record = frozen_epsilon(args.calibration_run, weather, checkpoint_hash,
                    rt.sha256(args.frontend_config), options['communication'], seed, wh['weather_augmentation'])
            else:
                indices = sample_indices(ds, args.scenes, args.stride, args.frames+calibration_count)
            frame_count = len(indices)-calibration_count
            wd = out/weather
            wd.mkdir()
            if args.full_validation:
                rt.write_json(wd/'epsilon.json', epsilon_record)
            print(f'{weather}: root={wh["validate_dir"]}; measured frames={frame_count}; '
                  f'calibration frames={calibration_count}', flush=True)
            rt.write_json(wd/'data_protocol.json', {'weather': wh['weather_augmentation'],
                'weather_source': 'simulator mode, NOT a deployed weather classifier',
                'root': wh['validate_dir'], 'full_validation': args.full_validation,
                'calibration_indices': indices[:calibration_count], 'probe_indices': indices[calibration_count:]})
            controls = ['a0b0', 'gt_single_deletion_joint'] + ['random_'+str(i) for i in range(args.random_repeats)]
            metrics = {name: {i: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for i in (.3, .5, .7)} for name in controls}
            totals = {name: {'bytes': 0, 'loss': 0., 'recovered_vs_a0b0': 0, 'lost_vs_a0b0': 0,
                             'fp_change_vs_a0b0': 0} for name in controls}
            counts = {'harmful': 0, 'beneficial': 0, 'neutral': 0}
            scene_results = {}
            cal_losses, drift_max = [], 0.
            frames_with_harm = selected_count = joint_removed = 0
            with (wd/'regions.jsonl').open('w', encoding='utf-8') as records, (wd/'frames.jsonl').open('w', encoding='utf-8') as frames_log:
                for position, index in enumerate(indices):
                    # Seed before preprocessing; the branches below never call the dataset again.
                    rt.seed_all(seed+index)
                    sample = ds[index]
                    meta = identity(ds, index, sample)
                    batch = to_device(ds.collate_batch_test([sample]), target)
                    inp = rt.model_input(batch['ego'], options)
                    labels = batch['ego']['label_dict']
                    encoded = model.encode(inp)
                    messages = capture(model, encoded)
                    baseline = predict(model, messages)
                    if position == 0:
                        # Compare with production A0B0 using IDENTICAL cached encoder tensors.
                        with patch.object(model, 'encode', return_value=encoded):
                            reference = model(inp)
                        for key in baseline:
                            torch.testing.assert_close(baseline[key], reference[key], atol=1e-5, rtol=1e-5)
                        if messages.total_bytes != model.last_diagnostics[0]['total_bytes']:
                            raise RuntimeError('Capture byte count differs from production A0B0')
                        rt.write_json(wd/'tensor_shapes.json', {'encoded_levels': [list(x.shape) for x in encoded[0]],
                            'confidence': list(encoded[3].shape), 'mask': list(messages.mask.shape),
                            'output': {k: list(v.shape) for k, v in baseline.items()},
                            'labels': {k: list(v.shape) for k, v in labels.items() if isinstance(v, torch.Tensor)}})
                        print(weather+': PASS cached production A0B0 parity and byte accounting', flush=True)
                    send = loss_values(criterion, baseline, labels)
                    drift = 0.
                    for _ in range(3):
                        repeat = loss_values(criterion, predict(model, messages), labels)
                        drift = max(drift, max(abs(send[k]-repeat[k]) for k in send))
                    if position < calibration_count:
                        cal_losses.append(send['total_loss'])
                        drift_max = max(drift_max, drift)
                        if position+1 == calibration_count:
                            epsilon = max(args.epsilon_abs, args.epsilon_rel*statistics.median(cal_losses), 10*drift_max)
                            rt.write_json(wd/'epsilon.json', {'epsilon': epsilon, 'calibration_losses': cal_losses,
                                'max_component_repeat_drift': drift_max, 'indices': indices[:calibration_count]})
                            print(f'{weather}: calibration complete, epsilon={epsilon:.8g}', flush=True)
                        continue
                    if drift > epsilon/10:
                        raise RuntimeError('Probe numerical drift exceeds calibrated bound; results incomplete, do not interpret')
                    frame_dir = wd/('frame_'+str(index))
                    frame_dir.mkdir()
                    (frame_dir/'request.bin').write_bytes(messages.request)
                    for peer, packet in messages.responses.items():
                        (frame_dir/f'sender_{peer}.bin').write_bytes(packet)
                    selected = [(p, b) for group in groups(messages.mask) for p, b in group]
                    selected_count += len(selected)
                    meta.update(weather=weather, epsilon=epsilon, seed=seed+index,
                        selected_sender_blocks=selected, input_sha256=fingerprint(inp), labels_sha256=fingerprint(labels),
                        packet_sha256={f.name: rt.sha256(f) for f in frame_dir.glob('*.bin')},
                        positive_anchor_count=int((labels['pos_equal_one'] > 0).sum()),
                        checkpoint_sha256=checkpoint_hash, total_bytes=messages.total_bytes)
                    rt.write_json(frame_dir/'metadata.json', meta)
                    # GT/targets saved separately as diagnostic supervision, never predictor input.
                    torch.save({k: v.cpu() for k, v in labels.items() if isinstance(v, torch.Tensor)}, frame_dir/'labels.pt')
                    base_boxes, base_scores, gt = ds.post_process(batch, {'ego': baseline})
                    base_match, base_fp = matched_objects(base_boxes, base_scores, gt)
                    scene_result = scene_results.setdefault(meta['scene_path'], {
                        'frames': 0, 'region_classes': {'harmful': 0, 'beneficial': 0, 'neutral': 0},
                        'harmful_single_deletions_losing_targets': 0,
                        'metrics': {name: {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
                            for iou in (.3, .5, .7)} for name in controls},
                        'controls': {name: {'bytes': 0, 'loss': 0., 'lost_targets_iou70': 0,
                                           'recovered_targets_iou70': 0} for name in controls}})
                    scene_result['frames'] += 1
                    harmful = []
                    candidate_groups = groups(messages.mask, args.region_blocks)
                    for gi, pairs in enumerate(candidate_groups):
                        dropped = predict(model, messages, drop_mask(messages.mask, pairs))
                        drop = loss_values(criterion, dropped, labels)
                        delta = {k: send[k]-drop[k] for k in send}
                        label = classify(delta['total_loss'], epsilon)
                        counts[label] += 1
                        scene_result['region_classes'][label] += 1
                        if label == 'harmful':
                            harmful.extend(pairs)
                        boxes, scores, drop_gt = ds.post_process(batch, {'ego': dropped})
                        torch.testing.assert_close(gt, drop_gt, atol=0, rtol=0)
                        match, fp = matched_objects(boxes, scores, gt)
                        if label == 'harmful' and base_match-match:
                            scene_result['harmful_single_deletions_losing_targets'] += 1
                        peer = pairs[0][0]
                        row = {'sample_index': index, 'weather': weather, 'scene': meta['scene_path'],
                            'timestamp': meta['timestamp'], 'ego_id': meta['ego_id'],
                            'sender_id': meta['cav_ids_in_feature_order'][peer], 'sender_index': peer,
                            'blocks_yx': [[b//model.grid[1], b % model.grid[1]] for _, b in pairs],
                            'group_side_native_blocks': args.region_blocks, 'removed_native_blocks': len(pairs),
                            'L_send': send, 'L_drop': drop, 'delta_send_minus_drop': delta, 'class': label,
                            'epsilon': epsilon, 'background': str(frame_dir.relative_to(out)/'metadata.json'),
                            'recovered_after_drop_iou70': len(match-base_match),
                            'lost_after_drop_iou70': len(base_match-match), 'fp_change_after_drop': fp-base_fp,
                            'recovered_gt_indices_iou70': sorted(match-base_match),
                            'lost_gt_indices_iou70': sorted(base_match-match)}
                        records.write(json.dumps(row)+'\n')
                        if (gi+1) % 10 == 0:
                            print(f'{weather} frame {position-calibration_count+1}/{frame_count}: deletion {gi+1}/{len(candidate_groups)}', flush=True)
                    frames_with_harm += bool(harmful)
                    joint_removed += len(harmful)
                    pairs_by_control = {'a0b0': [], 'gt_single_deletion_joint': harmful}
                    for i in range(args.random_repeats):
                        pairs_by_control['random_'+str(i)] = random_deletions(messages.mask, harmful, seed+index+100003*(i+1))
                    frame_controls = {}
                    for name, pairs in pairs_by_control.items():
                        prediction = baseline if not pairs else predict(model, messages, drop_mask(messages.mask, pairs))
                        vals = loss_values(criterion, prediction, labels)
                        boxes, scores, control_gt = ds.post_process(batch, {'ego': prediction})
                        torch.testing.assert_close(gt, control_gt, atol=0, rtol=0)
                        match, fp = matched_objects(boxes, scores, gt)
                        # Repack to audit ACTUAL length. Original request + empty reply headers are retained.
                        mask = drop_mask(messages.mask, pairs)
                        actual_bytes = len(messages.request)*(len(mask)-1)
                        for peer in messages.responses:
                            ids = mask[peer].flatten().nonzero().flatten().cpu().numpy()
                            packet = codec.pack_response([x[peer].cpu().numpy() for x in messages.levels], ids, model.grid, model.value_bytes)
                            actual_bytes += len(packet)
                        assert actual_bytes == messages.total_bytes-len(pairs)*model.cost
                        for iou in metrics[name]:
                            eval_utils.caluclate_tp_fp(boxes, scores, gt, metrics[name], iou)
                            eval_utils.caluclate_tp_fp(boxes, scores, gt, scene_result['metrics'][name], iou)
                        scene_control = scene_result['controls'][name]
                        scene_control['bytes'] += actual_bytes
                        scene_control['loss'] += vals['total_loss']
                        scene_control['lost_targets_iou70'] += len(base_match-match)
                        scene_control['recovered_targets_iou70'] += len(match-base_match)
                        stats = totals[name]
                        stats['bytes'] += actual_bytes
                        stats['loss'] += vals['total_loss']
                        stats['recovered_vs_a0b0'] += len(match-base_match)
                        stats['lost_vs_a0b0'] += len(base_match-match)
                        stats['fp_change_vs_a0b0'] += fp-base_fp
                        frame_controls[name] = {'removed_sender_blocks': pairs, 'bytes': actual_bytes, 'loss': vals,
                            'recovered_gt_indices_iou70': sorted(match-base_match),
                            'lost_gt_indices_iou70': sorted(base_match-match), 'fp_change_vs_a0b0': fp-base_fp}
                    frames_log.write(json.dumps({'sample_index': index, 'scene': meta['scene_path'],
                        'repeat_drift': drift, 'controls': frame_controls})+'\n')
                    frames_log.flush()
                    records.flush()
                    print(f'{weather}: frame {position-calibration_count+1}/{frame_count} done; {len(harmful)} harmful native blocks removed jointly', flush=True)
            if not selected_count:
                raise RuntimeError('No transmitted regions in probe; cannot assess harmful messages')
            summary = {'frames': frame_count, 'full_validation': args.full_validation, 'region_classes': counts,
                'harmful_group_fraction': counts['harmful']/sum(counts.values()),
                'frames_with_harmful_groups': frames_with_harm, 'selected_native_blocks': selected_count,
                'gt_joint_removed_native_blocks': joint_removed, 'controls': {},
                'warning': 'GT-assisted single-deletion joint control is not an optimal oracle or deployable result; '
                    + ('full validation split' if args.full_validation else 'training scenes only')}
            for name in controls:
                folder = wd/name
                folder.mkdir()
                if metrics[name][.7]['gt'] == 0:
                    raise RuntimeError('No GT; AP undefined')
                eval_utils.eval_final_results(metrics[name], str(folder), True)
                ap = yaml.safe_load((folder/'eval_global_sort.yaml').read_text())
                summary['controls'][name] = dict(totals[name], mean_bytes=totals[name]['bytes']/frame_count,
                    mean_loss=totals[name]['loss']/frame_count, ap50=ap['ap_50'], ap70=ap['ap_70'])
            rt.write_json(wd/'summary.json', summary)
            for scene_result in scene_results.values():
                scene_metrics = scene_result.pop('metrics')
                for name, values in scene_result['controls'].items():
                    values['mean_bytes'] = values['bytes']/scene_result['frames']
                    values['mean_loss'] = values['loss']/scene_result['frames']
                    for iou, key in ((.5, 'ap50'), (.7, 'ap70')):
                        values[key] = (eval_utils.calculate_ap(scene_metrics[name], iou, True)[0]
                                       if scene_metrics[name][iou]['gt'] else None)
            rt.write_json(wd/'scenes.json', scene_results)
            summaries[weather] = summary
    if fingerprint(model.state_dict()) != before_state or rt.sha256(args.frontend_checkpoint) != checkpoint_hash:
        raise RuntimeError('Frozen model/checkpoint changed')
    rt.verify_frozen()
    rt.write_json(out/'summary.json', {'status': 'complete', 'frozen_model_unchanged': True, 'weather': summaries})
    print('Send back: '+str(out/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
