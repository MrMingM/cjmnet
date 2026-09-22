"""Independent validation/test evaluation with frozen calibrated thresholds."""
import argparse
import copy
import json
import time
from pathlib import Path


def method_scores(method, predictor, descriptors, settings):
    from .fusion import confidence_scores
    from .network import utility_score
    if method == 'confidence':
        return confidence_scores(descriptors)
    output = predictor(descriptors)
    if method.startswith('utility'):
        return utility_score(output, settings['lost_weight'], settings['fp_weight'])
    if method == 'loss_gain':
        return output['loss_gain'][:, 0]
    raise ValueError(method)


def evaluate_condition(model, predictors, thresholds, dataset, loader, indices, branch,
                       settings, target, output, scale_ablation=False, no_keep=False,
                       policies=None):
    import torch
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from ceif_audit.scoring import empty_stats, ap_values
    from . import runtime as rt
    from .fusion import prediction_for_actions, select_actions
    from .outcomes import compare_predictions

    methods = ['baseline', 'confidence', 'utility']
    if 'loss_gain' in predictors:
        methods.append('loss_gain')
    if scale_ablation:
        methods += ['utility_s0', 'utility_s1']
    if no_keep:
        methods += ['utility_no_keep']
    if policies is not None:
        methods = ['baseline', *policies]
    stats = {name: empty_stats() for name in methods}
    diagnostics = {name: {'recovered': 0, 'lost': 0, 'new_fp': 0,
                          'selected_tiles': 0, 'model_ms': 0.} for name in methods if name != 'baseline'}
    frames = raw_bytes = 0
    baseline_tp = baseline_fp = 0
    started = time.perf_counter()
    with torch.no_grad(), (output/'frames.jsonl').open('w', encoding='utf-8') as stream:
        for batch in loader:
            batch = to_device(batch, target)
            ego = batch['ego']
            context = rt.prepare_context(model, ego, branch, settings, verify=frames == 0)
            predictions = {'baseline': context['baseline_prediction']}
            action_maps = {}
            score_cache = {}
            for method in methods:
                if method == 'baseline':
                    continue
                source, threshold = (policies[method] if policies is not None else
                    ('utility' if method.startswith('utility') else method,
                     thresholds['utility' if method.startswith('utility') else method]))
                if threshold is None and method != 'utility_no_keep':
                    predictions[method] = context['baseline_prediction']
                    action_maps[method] = torch.zeros(context['grid'], dtype=torch.long, device=target)
                    continue
                if source not in score_cache:
                    score_cache[source] = method_scores(
                        source, predictors.get(source), context['descriptors'], settings)
                allow_keep = method != 'utility_no_keep'
                action = select_actions(score_cache[source], threshold, allow_keep)
                scales = ((0,) if method == 'utility_s0' else
                          (1,) if method == 'utility_s1' else settings['changed_scales'])
                if torch.any(action):
                    tick = time.perf_counter()
                    predictions[method] = prediction_for_actions(
                        model.engine.base, context, action, scales)
                    if target.type == 'cuda':
                        torch.cuda.synchronize()
                    diagnostics[method]['model_ms'] += (time.perf_counter()-tick)*1000
                else:
                    predictions[method] = context['baseline_prediction']
                action_maps[method] = action
                diagnostics[method]['selected_tiles'] += int((action > 0).sum())

            post = {}
            for method, prediction in predictions.items():
                post[method] = dataset.post_process(batch, {'ego': prediction})
                boxes, scores, gt = post[method]
                for threshold in stats[method]:
                    eval_utils.caluclate_tp_fp(boxes, scores, gt, stats[method], threshold)
            for method in diagnostics:
                outcome = compare_predictions(post['baseline'], post[method],
                                              settings['target_iou'], settings['fp_identity_iou'])
                for key in ('recovered', 'lost', 'new_fp'):
                    diagnostics[method][key] += outcome[key]
            from .outcomes import detection_outcome
            matched, false_boxes = detection_outcome(*post['baseline'], settings['target_iou'])
            baseline_tp += len(matched)
            baseline_fp += len(false_boxes)
            row = {
                'sample_index': int(ego['communication_sample_index'][0]),
                'raw_feature_bytes': rt.raw_feature_bytes(context),
                'selected_tiles': {name: int((value > 0).sum()) for name, value in action_maps.items()},
            }
            stream.write(json.dumps(row)+'\n')
            raw_bytes += row['raw_feature_bytes']
            frames += 1
            if frames == 1 or frames % 20 == 0:
                elapsed = (time.perf_counter()-started)/frames
                print(f'{frames}/{len(indices)} frames; {elapsed:.2f}s/frame; '
                      f'ETA {(len(indices)-frames)*elapsed/3600:.2f}h', flush=True)
    if not frames or frames != len(indices) or not stats['baseline'][.7]['gt']:
        raise RuntimeError('Incomplete or empty evaluation')
    results = {name: ap_values(value, eval_utils) for name, value in stats.items()}
    for name, value in stats.items():
        destination = output/name
        destination.mkdir()
        eval_utils.eval_final_results(copy.deepcopy(value), str(destination), False)
    for value in diagnostics.values():
        value['selected_tiles_per_frame'] = value.pop('selected_tiles')/frames
        value['action_head_ms_per_frame'] = value.pop('model_ms')/frames
    return {'frames': frames, 'global_sort': False, 'results': results,
            'baseline_tp': baseline_tp, 'baseline_fp': baseline_fp,
            'diagnostics_vs_baseline': diagnostics,
            'mean_raw_full_feature_bytes': raw_bytes/frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='local_fusion_utility_v2/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--utility-checkpoint', required=True)
    parser.add_argument('--loss-gain-checkpoint')
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--phase', choices=('development', 'benchmark'), required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--scale-ablation', action='store_true')
    parser.add_argument('--no-keep-ablation', action='store_true')
    args = parser.parse_args()

    import torch
    from gspr_communication.runtime import new_output, seed_all, device, verify_frozen, write_json, sha256
    from gspr_evidence import runtime as evidence_runtime
    from gspr_evidence import benchmark
    from . import runtime as rt

    if args.phase == 'benchmark' and args.smoke:
        raise ValueError('Formal benchmark cannot use --smoke')
    calibration = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    if calibration.get('selection_data') != 'full_validation_joint_actions':
        raise ValueError('v2 requires full-scene joint-action validation calibration')
    thresholds = {name: value['threshold'] for name, value in calibration['methods'].items()}
    for name, path in [('utility', args.utility_checkpoint), ('loss_gain', args.loss_gain_checkpoint)]:
        if path and sha256(path) != calibration['checkpoints'][name]['sha256']:
            raise ValueError('Checkpoint differs from the one used to calibrate: '+name)
    required = {'confidence', 'utility'}
    if not required.issubset(thresholds):
        raise ValueError('Calibration is missing confidence/utility thresholds')
    output = new_output(args.output_dir)
    verify_frozen()
    target = device()
    conditions = ('clean', 'fog', 'rain', 'snow')
    summaries = {}
    expected_contract = calibration['cache_contract']
    for condition in conditions:
        if args.phase == 'benchmark':
            options, hypes = benchmark.load_config(args.config, args.frontend_config, condition)
            branch, loader_weather = 'clean', 'clean'
        else:
            options, hypes = rt.load_config(args.config, args.frontend_config)
            branch, loader_weather = ('clean', 'clean') if condition == 'clean' else ('weather', condition)
        if rt.settings(options) != rt.settings(expected_contract['options']):
            raise ValueError('Current method settings differ from calibrated settings')
        seed_all(int(options['seed']))
        model, digest = evidence_runtime.load_model(hypes, options, args.frontend_checkpoint, target)
        current_contract = rt.contract(options, args.frontend_config, digest)
        if current_contract != expected_contract:
            raise ValueError('Evaluation pipeline differs from calibrated training pipeline')
        predictors = {}
        predictors['utility'], _ = rt.load_predictor(
            args.utility_checkpoint, expected_contract, target, 'utility')
        if args.loss_gain_checkpoint:
            if 'loss_gain' not in thresholds:
                raise ValueError('Loss-gain checkpoint provided without calibrated threshold')
            predictors['loss_gain'], _ = rt.load_predictor(
                args.loss_gain_checkpoint, expected_contract, target, 'loss_gain')
        dataset, loader, indices = evidence_runtime.make_loader(
            hypes, options, train=False, weather=loader_weather, smoke=args.smoke)
        destination = output/condition
        destination.mkdir()
        summary = evaluate_condition(
            model, predictors, thresholds, dataset, loader, indices, branch, rt.settings(options),
            target, destination, args.scale_ablation, args.no_keep_ablation)
        summary.update({'condition': condition, 'root': hypes['validate_dir'],
                        'online_weather': args.phase == 'development' and condition != 'clean'})
        write_json(destination/'summary.json', summary)
        summaries[condition] = summary
    write_json(output/'protocol.json', {
        'phase': args.phase,
        'conditions': summaries,
        'thresholds': thresholds,
        'calibration_sha256': sha256(args.calibration),
        'utility_checkpoint_sha256': sha256(args.utility_checkpoint),
        'loss_gain_checkpoint_sha256': sha256(args.loss_gain_checkpoint) if args.loss_gain_checkpoint else None,
        'metric': 'Existing OpenCOOD planar polygon IoU and non-global sorting',
        'communication': 'raw/full float32 features; no new transmitted quality map, box, or weather label',
    })
    verify_frozen()


if __name__ == '__main__':
    main()
