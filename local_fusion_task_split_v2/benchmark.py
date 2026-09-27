"""Formal B1 benchmark on OPV2V clean test and OPV2V-W.

No training, tuning, checkpoint selection, or online weather augmentation is
allowed here.  The fixed B1 last-epoch checkpoints produced by pipeline.py are
evaluated once on the historical benchmark protocol.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from ceif_audit.scoring import ap_values, empty_stats
from gspr_communication.runtime import (
    device,
    seed_all,
    sha256,
    verify_frozen,
    write_json,
)
from gspr_evidence import runtime as er
from gspr_evidence.benchmark import ROOTS, load_config as load_benchmark_config
from local_fusion_task_split_pilot.model import TaskFusionArm
from local_fusion_v3 import runtime as v3rt
from opencood.tools.train_utils import to_device
from opencood.utils import eval_utils

from .model import TaskConflictGate, prepare_frozen_context, predict_with_gate
from .pipeline import _load_json, _load_split, _verify_b0_source_contract, settings


METHODS = ('B0-Collapse', 'B0-Split', 'Global-Gate', 'Local-Gate')


def _load_gate(path, spec, target, expected_mode, b0_split_digest):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('mode') != expected_mode:
        raise ValueError(f'Gate mode mismatch: {path}')
    if state.get('b0_split_checkpoint_sha256') != b0_split_digest:
        raise ValueError('Gate was trained from a different B0 Split checkpoint')
    if state.get('config') != spec:
        raise ValueError('Gate checkpoint config differs from current B1 config')
    gate = TaskConflictGate(
        scales=spec['scales'],
        hidden=spec['hidden'],
        mode=expected_mode,
        initial_bias=spec['initial_bias'],
    ).to(target)
    gate.load_state_dict(state['gate'], strict=True)
    if int(state.get('parameter_count', -1)) != gate.parameter_count():
        raise ValueError('Gate parameter count mismatch')
    return gate.eval()


def _gate_numbers(info):
    row = {'gate_mean': float(info['gate_mean'].detach().cpu())}
    for key, value in info.items():
        if key.startswith('gate_s') and key[6:].isdigit():
            row[key] = float(value.detach().cpu())
    return row


def evaluate_weather(model, split_arm, gates, dataset, loader, target, weather, output):
    stats = {name: empty_stats() for name in METHODS}
    gate_sums = {
        'Global-Gate': {'gate_mean': 0.0},
        'Local-Gate': {'gate_mean': 0.0},
    }
    gate_scale_sums = {'Global-Gate': {}, 'Local-Gate': {}}
    frames = 0
    started = time.time()

    with torch.no_grad(), (output / 'frames.jsonl').open(
            'w', encoding='utf-8') as stream:
        for frame, batch in enumerate(loader, 1):
            batch = to_device(batch, target)
            # Benchmark files are already clean/weather-degraded on disk.
            # Always use processed_lidar; never synthesize weather here.
            context = v3rt.context(
                model, batch['ego'], 'clean', verify=frame == 1)
            frozen = prepare_frozen_context(
                model.engine.base,
                context['levels'],
                split_arm,
                gates['Local-Gate'].scales,
            )
            global_prediction, global_info = predict_with_gate(
                model.engine.base, frozen, gates['Global-Gate'])
            local_prediction, local_info = predict_with_gate(
                model.engine.base, frozen, gates['Local-Gate'])
            predictions = {
                'B0-Collapse': frozen['collapse_prediction'],
                'B0-Split': frozen['split_prediction'],
                'Global-Gate': global_prediction,
                'Local-Gate': local_prediction,
            }

            frame_row = {
                'sample_index': int(
                    batch['ego']['communication_sample_index'][0]),
                'source_count': int(len(context['levels'][0])),
            }
            for name, prediction in predictions.items():
                boxes, scores, gt = dataset.post_process(
                    batch, {'ego': prediction})
                for threshold in stats[name]:
                    eval_utils.caluclate_tp_fp(
                        boxes, scores, gt, stats[name], threshold)

            for name, info in (
                ('Global-Gate', global_info),
                ('Local-Gate', local_info),
            ):
                numbers = _gate_numbers(info)
                frame_row[name] = numbers
                gate_sums[name]['gate_mean'] += numbers['gate_mean']
                for key, value in numbers.items():
                    if key.startswith('gate_s'):
                        gate_scale_sums[name][key] = (
                            gate_scale_sums[name].get(key, 0.0) + value)

            stream.write(json.dumps(frame_row, ensure_ascii=False) + '\n')
            frames = frame
            if frame == 1 or frame % 100 == 0:
                elapsed = time.time() - started
                print(
                    f'benchmark {weather}: {frame}/{len(loader)} '
                    f'({elapsed/frame:.2f}s/frame)',
                    flush=True,
                )

    if frames != len(loader) or not frames or not stats['B0-Split'][.7]['gt']:
        raise RuntimeError('Incomplete B1 formal benchmark')

    results = {name: ap_values(value, eval_utils) for name, value in stats.items()}
    for name, value in stats.items():
        folder = output / name
        folder.mkdir()
        eval_utils.eval_final_results(copy.deepcopy(value), str(folder), False)
        write_json(folder / 'ap_inputs.json', value)

    gate_means = {}
    for name in ('Global-Gate', 'Local-Gate'):
        gate_means[name] = {
            'gate_mean': gate_sums[name]['gate_mean'] / frames,
            **{
                key: value / frames
                for key, value in gate_scale_sums[name].items()
            },
        }

    return {
        'weather': weather,
        'frames': frames,
        'data_root': str(Path(dataset.params['validate_dir']).resolve())
            if hasattr(dataset, 'params') and 'validate_dir' in dataset.params
            else ROOTS[weather],
        'online_weather_augmentation': False,
        'input_key': 'processed_lidar',
        'global_sort': False,
        'results': results,
        'gate_means': gate_means,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='local_fusion_task_split_v2/experiment.yaml')
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    for name in (
        'frontend-config',
        'frontend-checkpoint',
        'v3-checkpoint',
        'b0-run',
        'run',
        'output',
    ):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()

    verify_frozen()
    spec = settings(args.config)
    run = Path(args.run).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    training_protocol = _load_json(run / 'protocol.json')
    training_results = _load_json(run / 'decision_results.json')
    json_spec = json.loads(json.dumps(spec))
    if training_protocol.get('config') != json_spec:
        raise ValueError('B1 run config differs from benchmark config')
    if training_protocol.get('checkpoint_selection') != 'fixed last epoch':
        raise ValueError('Formal benchmark requires fixed last-epoch B1 checkpoints')

    b0_run = Path(args.b0_run).resolve()
    if str(b0_run) != training_protocol.get('b0_run'):
        raise ValueError('Benchmark B0 run differs from B1 training protocol')
    b0_protocol = _load_json(b0_run / 'protocol.json')
    _verify_b0_source_contract(b0_protocol)

    target = device()
    conditions = {}
    common_identity = None
    for weather in ('clean', 'fog', 'rain', 'snow'):
        options, hypes = load_benchmark_config(
            args.v3_config, args.frontend_config, weather)
        seed_all(int(spec['seed']))
        model, frontend_digest = er.load_model(
            hypes, options, args.frontend_checkpoint, target)
        model.requires_grad_(False).eval()
        v3_contract = v3rt.contract(
            options, args.frontend_config, frontend_digest)
        start_digest = sha256(args.v3_checkpoint)

        identity = {
            'frontend_sha256': frontend_digest,
            'v3_contract': v3_contract,
            'v3_checkpoint_sha256': start_digest,
        }
        if common_identity is None:
            common_identity = identity
            if training_protocol.get('frontend_sha256') != frontend_digest:
                raise ValueError('Formal benchmark frontend differs from B1 training')
            if training_protocol.get('v3_contract') != v3_contract:
                raise ValueError('Formal benchmark v3 contract differs from B1 training')
            if training_protocol.get('v3_checkpoint_sha256') != start_digest:
                raise ValueError('Formal benchmark v3 checkpoint differs from B1 training')
        elif identity != common_identity:
            raise ValueError('Model identity drifted across benchmark weather conditions')

        source, checkpoint = v3rt.load(
            args.v3_checkpoint, v3_contract, target)
        if checkpoint['variant'] != 'residual':
            raise ValueError('Expected residual v3 checkpoint')
        split_arm = _load_split(
            b0_run, source, target, start_digest)
        del source

        b0_split_digest = sha256(b0_run / 'Split.pth')
        if training_protocol.get('b0_split_checkpoint_sha256') != b0_split_digest:
            raise ValueError('B0 Split checkpoint changed since B1 training')
        gates = {
            'Global-Gate': _load_gate(
                run / 'Global-Gate.pth', spec, target, 'global', b0_split_digest),
            'Local-Gate': _load_gate(
                run / 'Local-Gate.pth', spec, target, 'local', b0_split_digest),
        }

        # Full formal split: no subset and no online weather.
        dataset, loader, indices = er.make_loader(
            hypes, options, train=False, weather='clean')
        if list(indices) != list(range(len(dataset))):
            raise RuntimeError('Formal benchmark must use every frame in dataset order')

        folder = output / weather
        folder.mkdir()
        summary = evaluate_weather(
            model, split_arm, gates, dataset, loader,
            target, weather, folder)
        summary['scene_ends'] = list(dataset.len_record)
        summary['sample_indices'] = indices
        conditions[weather] = summary
        write_json(folder / 'summary.json', summary)
        del dataset, loader, model, split_arm, gates
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        'scope': (
            'formal fixed-checkpoint benchmark; OPV2V clean test plus '
            'OPV2V-W fog/rain/snow test; no online weather; no retraining'
        ),
        'test_data_used': True,
        'selection_boundary': (
            'B1 checkpoints were fixed before this benchmark. Test results '
            'must not be used to select epoch, tune thresholds, gate penalty, '
            'learning rate, architecture, or whether a run is retained.'
        ),
        'training_decision_expand': training_results['decision']['expand'],
        'model_identity': common_identity,
        'b0_split_checkpoint_sha256': sha256(b0_run / 'Split.pth'),
        'global_gate_checkpoint_sha256': sha256(run / 'Global-Gate.pth'),
        'local_gate_checkpoint_sha256': sha256(run / 'Local-Gate.pth'),
        'metric': 'OpenCOOD non-global planar polygon IoU AP30/AP50/AP70',
        'conditions': conditions,
    }
    write_json(output / 'benchmark_results.json', report)

    compact = {
        weather: {
            'frames': row['frames'],
            'data_root': row['data_root'],
            'results': row['results'],
            'gate_means': row['gate_means'],
        }
        for weather, row in conditions.items()
    }
    write_json(output / 'benchmark_summary.json', compact)
    verify_frozen()
    print('B1 FORMAL BENCHMARK COMPLETE:', output / 'benchmark_results.json', flush=True)


if __name__ == '__main__':
    main()
