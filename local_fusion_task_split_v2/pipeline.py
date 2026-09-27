"""B1 training pipeline: frozen B0 Split endpoints plus local/global gates.

This experiment reuses the exact B0 train/validation indices and its trained
Split checkpoint.  GSPR, encoders, B0 routers, deblocks and detector heads stay
frozen.  Only two equal-parameter tiny gates are trained.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import torch
import yaml

from gspr_communication.runtime import (
    ROOT,
    device,
    new_output,
    seed_all,
    sha256,
    verify_frozen,
    write_json,
)
from gspr_evidence import runtime as er
from local_fusion_task_split_pilot.model import TaskFusionArm
from local_fusion_task_split_pilot.pipeline import selected_loader
from local_fusion_v3 import runtime as v3rt
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.tools.train_utils import to_device

from .evaluate import (
    assert_b0_reproduction,
    decide,
    evaluate_condition,
)
from .model import (
    TaskConflictGate,
    prepare_frozen_context,
    predict_with_gate,
)


def settings(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('seed', 'epochs', 'hidden', 'minimum_positive_weathers'):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if value['minimum_positive_weathers'] > 3:
        raise ValueError('More than three positive weathers requested')
    scales = tuple(int(x) for x in value.get('scales', ()))
    if not scales or len(set(scales)) != len(scales):
        raise ValueError('scales must be non-empty and unique')
    if any(x not in (0, 1, 2) for x in scales):
        raise ValueError('Unsupported feature scale')
    value['scales'] = scales
    for key in (
        'learning_rate',
        'gate_sparsity_penalty',
        'minimum_weather_mean_ap70_gain_vs_global',
    ):
        if not 0 < float(value[key]) <= 1:
            raise ValueError(key + ' must be in (0,1]')
    for key in (
        'weight_decay',
        'clean_ap70_tolerance',
        'ap50_tolerance',
        'max_lost_tp_fraction',
        'max_new_fp_per_reference_tp',
        'minimum_weather_mean_ap70_gain_vs_split',
    ):
        if not 0 <= float(value[key]) <= 1:
            raise ValueError(key + ' must be in [0,1]')
    if not isinstance(value.get('initial_bias'), (int, float)):
        raise ValueError('initial_bias must be numeric')
    return value


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _state_equal(left, right):
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(
        torch.equal(a[key], b[key]) for key in a)


def _verify_b0_source_contract(protocol):
    required = (
        'local_fusion_task_split_pilot/model.py',
        'local_fusion_task_split_pilot/pipeline.py',
        'local_fusion_task_split_pilot/evaluate.py',
        'local_fusion_task_split_pilot/experiment.yaml',
    )
    recorded = protocol.get('source_hashes') or {}
    for name in required:
        expected = recorded.get(name)
        if expected is None:
            raise ValueError('B0 protocol lacks source hash: ' + name)
        if not (ROOT / name).is_file() or sha256(ROOT / name) != expected:
            raise ValueError('B0 source drift since training: ' + name)


def _load_split(run, source, target, start_digest):
    state = torch.load(run / 'Split.pth', map_location='cpu', weights_only=True)
    if state.get('mode') != 'split':
        raise ValueError('B0 Split checkpoint mode mismatch')
    if state.get('start_checkpoint_sha256') != start_digest:
        raise ValueError('B0 Split checkpoint starts from a different v3 checkpoint')
    arm = TaskFusionArm(source, 'split').to(target)
    arm.load_state_dict(state['arm'], strict=True)
    if int(state.get('parameter_count', -1)) != arm.parameter_count():
        raise ValueError('B0 Split checkpoint parameter count mismatch')
    arm.requires_grad_(False).eval()
    return arm


def train_epoch(model, split_arm, gate, loader, criterion, target,
                sparsity_penalty, optimizer):
    gate.train()
    model.eval()
    split_arm.eval()
    sums = {
        'loss': 0.0,
        'detection_loss': 0.0,
        'gate_penalty': 0.0,
        'gate_mean': 0.0,
        'gate_s0': 0.0,
        'gate_s1': 0.0,
        'frames': 0,
        'optimizer_updates': 0,
        'single_source_frames': 0,
    }
    multi_source_branches = 0

    for number, batch in enumerate(loader, 1):
        batch = to_device(batch, target)
        optimizer.zero_grad(set_to_none=True)
        frame_has_gradient = False

        for branch in ('clean', 'weather'):
            with torch.no_grad():
                context = v3rt.context(
                    model, batch['ego'], branch, verify=number == 1)
                frozen = prepare_frozen_context(
                    model.engine.base,
                    context['levels'],
                    split_arm,
                    gate.scales,
                )

            prediction, info = predict_with_gate(
                model.engine.base, frozen, gate)
            detection = criterion(
                prediction, batch['ego']['label_dict'])
            penalty = sparsity_penalty * info['gate_mean']
            loss = detection + penalty
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite B1 gate loss')

            sums['loss'] += float(loss.detach())
            sums['detection_loss'] += float(detection.detach())
            sums['gate_penalty'] += float(penalty.detach())
            sums['gate_mean'] += float(info['gate_mean'].detach())

            if len(context['levels'][0]) > 1:
                frame_has_gradient = True
                multi_source_branches += 1
                for scale in gate.scales:
                    key = f'gate_s{scale}'
                    if key in sums:
                        sums[key] += float(info[key].detach())
                if not loss.requires_grad:
                    raise RuntimeError('Multi-source B1 branch has no gate gradient')
                (loss / 2).backward()

        if frame_has_gradient:
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            sums['optimizer_updates'] += 1
        else:
            sums['single_source_frames'] += 1

        sums['frames'] = number
        if number == 1 or number % 20 == 0:
            print(
                f'train {gate.mode} {number}/{len(loader)} '
                f'loss={sums["loss"]/(2*number):.6f} '
                f'gate={sums["gate_mean"]/(2*number):.5f}',
                flush=True,
            )

    if not sums['frames']:
        raise RuntimeError('Empty B1 training split')
    branches = 2 * sums['frames']
    for key in ('loss', 'detection_loss', 'gate_penalty', 'gate_mean'):
        sums[key] /= branches
    denominator = max(1, multi_source_branches)
    for scale in gate.scales:
        key = f'gate_s{scale}'
        if key in sums:
            sums[key] /= denominator
    sums['multi_source_branches'] = multi_source_branches
    return sums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='local_fusion_task_split_v2/experiment.yaml')
    parser.add_argument(
        '--v3-config', default='local_fusion_v3/experiment.yaml')
    for name in (
        'frontend-config',
        'frontend-checkpoint',
        'v3-checkpoint',
        'b0-run',
        'run',
    ):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()

    verify_frozen()
    spec = settings(args.config)
    options, hypes = er.load_config(
        args.v3_config, args.frontend_config)
    target = device()

    model, frontend_digest = er.load_model(
        hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    v3_contract = v3rt.contract(
        options, args.frontend_config, frontend_digest)
    start_digest = sha256(args.v3_checkpoint)

    b0_run = Path(args.b0_run).resolve()
    b0_protocol = _load_json(b0_run / 'protocol.json')
    b0_results = _load_json(b0_run / 'decision_results.json')
    _verify_b0_source_contract(b0_protocol)
    if b0_protocol.get('frontend_sha256') != frontend_digest:
        raise ValueError('B0 used a different frontend checkpoint')
    if b0_protocol.get('v3_contract') != v3_contract:
        raise ValueError('B0 v3 source/config contract differs')
    if b0_protocol.get('v3_checkpoint_sha256') != start_digest:
        raise ValueError('B0 used a different v3 checkpoint')

    source, checkpoint = v3rt.load(
        args.v3_checkpoint, v3_contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Starting checkpoint must be v3 residual')
    split_arm = _load_split(
        b0_run, source, target, start_digest)
    del source

    b0_scales = tuple(int(x) for x in split_arm.router_a.scales)
    if tuple(spec['scales']) != b0_scales:
        raise ValueError(
            f'B1 scales {spec["scales"]} must equal frozen B0 scales {b0_scales}')

    seed_all(spec['seed'])
    local_gate = TaskConflictGate(
        scales=spec['scales'],
        hidden=spec['hidden'],
        mode='local',
        initial_bias=spec['initial_bias'],
    ).to(target)
    global_gate = TaskConflictGate(
        scales=spec['scales'],
        hidden=spec['hidden'],
        mode='global',
        initial_bias=spec['initial_bias'],
    ).to(target)
    global_gate.load_state_dict(local_gate.state_dict(), strict=True)
    gates = {
        'Global-Gate': global_gate,
        'Local-Gate': local_gate,
    }
    if not _state_equal(global_gate, local_gate):
        raise RuntimeError('Local and Global gates do not share identical initial weights')
    if global_gate.parameter_count() != local_gate.parameter_count():
        raise RuntimeError('Local and Global gate parameter counts differ')

    train_indices = [int(x) for x in b0_protocol['train_indices']]
    validation_indices = [
        int(x) for x in b0_protocol['validation_indices']]
    if len(train_indices) != len(set(train_indices)):
        raise ValueError('Duplicate B0 train indices')
    if len(validation_indices) != len(set(validation_indices)):
        raise ValueError('Duplicate B0 validation indices')

    run = new_output(Path(args.run).resolve())
    own_sources = [
        'local_fusion_task_split_v2/' + name
        for name in (
            '__init__.py',
            'model.py',
            'evaluate.py',
            'pipeline.py',
            'experiment.yaml',
            'run_all.sh',
            'test_core.py',
            'README.md',
        )
    ]
    b0_sources = [
        'local_fusion_task_split_pilot/model.py',
        'local_fusion_task_split_pilot/pipeline.py',
        'local_fusion_task_split_pilot/evaluate.py',
    ]
    dependencies = list(v3_contract['pipeline_sources'])
    dependencies += list(v3_contract['method_sources'])
    source_files = sorted(set(own_sources + b0_sources + dependencies))
    source_hashes = {
        name: sha256(ROOT / name) for name in source_files
    }
    for name in source_files:
        destination = run / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    write_json(
        run / 'protocol.json',
        {
            'purpose': (
                'B1 local task-conflict gating between frozen B0 Collapse and '
                'Split endpoints; Local versus equal-parameter Global control'
            ),
            'test_data_used': False,
            'v3_contract': v3_contract,
            'v3_checkpoint_sha256': start_digest,
            'frontend_sha256': frontend_digest,
            'b0_run': str(b0_run),
            'b0_protocol_sha256': sha256(b0_run / 'protocol.json'),
            'b0_split_checkpoint_sha256': sha256(b0_run / 'Split.pth'),
            'config': spec,
            'source_hashes': source_hashes,
            'train_indices': train_indices,
            'validation_indices': validation_indices,
            'checkpoint_selection': 'fixed last epoch',
            'trainable': 'only B1 gate parameters',
            'frozen': (
                'GSPR, per-agent encoders, both trained B0 Split routers, '
                'deblocks, cls_head and reg_head'
            ),
            'gate_inputs': (
                'per-cell task-weight total variation, top-source disagreement, '
                'and frozen Collapse classification activity'
            ),
            'gate_formula': (
                'w_task = w_common + g * (w_task_raw - w_common)'
            ),
            'fairness': (
                'Local and Global have identical parameter state at start, '
                'same train frames/order/epochs/optimizer settings; Global '
                'spatially averages the same three inputs before the same gate '
                'network and broadcasts one value per scale/frame'
            ),
        },
    )

    criterion = PointPillarLoss(hypes['loss']['args']).to(target)
    histories = {}
    for name in ('Global-Gate', 'Local-Gate'):
        gate = gates[name]
        optimizer = torch.optim.AdamW(
            gate.parameters(),
            lr=spec['learning_rate'],
            weight_decay=spec['weight_decay'],
        )
        history = []
        for epoch in range(1, spec['epochs'] + 1):
            seed_all(spec['seed'] + epoch)
            _, loader = selected_loader(
                hypes, options, 'train', 'mixed',
                train_indices, epoch)
            began = time.time()
            row = train_epoch(
                model,
                split_arm,
                gate,
                loader,
                criterion,
                target,
                spec['gate_sparsity_penalty'],
                optimizer,
            )
            row.update(epoch=epoch, seconds=time.time() - began)
            history.append(row)
            print(json.dumps({'arm': name, **row}), flush=True)
            del loader
        histories[name] = history
        torch.save(
            {
                'gate': gate.state_dict(),
                'mode': gate.mode,
                'epoch': spec['epochs'],
                'parameter_count': gate.parameter_count(),
                'b0_split_checkpoint_sha256': sha256(b0_run / 'Split.pth'),
                'config': spec,
            },
            run / (name + '.pth'),
        )
        write_json(run / 'train_history.json', histories)

    global_updates = [
        row['optimizer_updates'] for row in histories['Global-Gate']]
    local_updates = [
        row['optimizer_updates'] for row in histories['Local-Gate']]
    if global_updates != local_updates:
        raise RuntimeError(
            'Local and Global gates received different optimizer step counts')

    for gate in gates.values():
        gate.eval()

    summaries = {}
    for condition in ('clean', 'fog', 'rain', 'snow'):
        seed_all(spec['seed'] + 1)
        dataset, loader = selected_loader(
            hypes, options, 'validation', condition, validation_indices)
        folder = run / 'validation' / condition
        folder.mkdir(parents=True)
        summary = evaluate_condition(
            model,
            split_arm,
            gates,
            dataset,
            loader,
            validation_indices,
            'clean' if condition == 'clean' else 'weather',
            target,
            folder,
        )
        assert_b0_reproduction(
            condition, summary, b0_results, tolerance=1e-6)
        summary.update(
            root=hypes['validate_dir'],
            online_weather=condition != 'clean',
        )
        summaries[condition] = summary
        write_json(folder / 'summary.json', summary)
        del dataset, loader

    decision = decide(summaries, spec)
    write_json(
        run / 'decision_results.json',
        {
            'conditions': summaries,
            'decision': decision,
            'train_history': histories,
            'fairness': {
                'global_gate_parameters': global_gate.parameter_count(),
                'local_gate_parameters': local_gate.parameter_count(),
                'global_optimizer_updates': global_updates,
                'local_optimizer_updates': local_updates,
                'b0_split_parameters_trainable': 0,
            },
            'scope': (
                'exact B0 train/validation indices; official validation with '
                'online weather only; directional B1 pilot; no OPV2V-W'
            ),
        },
    )
    verify_frozen()
    print('TASK-SPLIT B1 COMPLETE:', run / 'decision_results.json', flush=True)


if __name__ == '__main__':
    main()
