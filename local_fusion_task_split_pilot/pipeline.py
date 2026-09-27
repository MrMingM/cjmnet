"""Paired Shared versus Task-Split source-fusion pilot.

The exact frame indices are reused from a completed detector-adaptation pilot.
No OPV2V-W path is accessed here.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from gspr_communication.runtime import (
    ROOT,
    device,
    new_output,
    seed_all,
    seed_worker,
    sha256,
    verify_frozen,
    write_json,
)
from gspr_evidence import runtime as er
from local_fusion_v3 import runtime as v3rt
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.tools.train_utils import to_device

from .evaluate import decide, evaluate_condition
from .model import TaskFusionArm


def settings(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('seed', 'epochs', 'minimum_positive_weathers'):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if value['minimum_positive_weathers'] > 3:
        raise ValueError('More than three positive weathers requested')
    for key in ('learning_rate', 'minimum_weather_mean_ap70_gain'):
        if not 0 < value[key] <= 1:
            raise ValueError(key + ' must be in (0,1]')
    for key in (
        'weight_decay',
        'clean_ap70_tolerance',
        'ap50_tolerance',
        'max_lost_tp_fraction',
        'max_new_fp_per_reference_tp',
    ):
        if not 0 <= value[key] <= 1:
            raise ValueError(key + ' must be in [0,1]')
    return value


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def selected_loader(hypes, options, split, weather, indices, epoch=0):
    dataset, _, _ = er.make_loader(
        hypes, options, train=split == 'train', weather=weather)
    if not indices or min(indices) < 0 or max(indices) >= len(dataset):
        raise RuntimeError('Selected index missing from dataset')
    generator = torch.Generator().manual_seed(options['seed'] + epoch)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=split == 'train',
        num_workers=options['workers'],
        collate_fn=dataset.collate_batch_test,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader


def _states_equal(left, right):
    a, b = left.state_dict(), right.state_dict()
    return (
        a.keys() == b.keys()
        and all(torch.equal(a[key], b[key]) for key in a)
    )


def train_epoch(model, arm, loader, criterion, target, change_penalty, optimizer):
    arm.train()
    model.eval()
    sums = {
        'loss': 0.0,
        'detection_loss': 0.0,
        'change': 0.0,
        'task_gap': 0.0,
        'task_disagreement': 0.0,
        'frames': 0,
        'optimizer_updates': 0,
        'single_source_frames': 0,
    }
    multi_source_branches = 0

    for number, batch in enumerate(loader, 1):
        batch = to_device(batch, target)
        optimizer.zero_grad(set_to_none=True)
        frame_has_multiple_sources = False

        for branch in ('clean', 'weather'):
            with torch.no_grad():
                context = v3rt.context(
                    model, batch['ego'], branch, verify=number == 1)
            prediction, info = arm.predict(
                model.engine.base, context['levels'])
            detection = criterion(prediction, batch['ego']['label_dict'])
            loss = detection + change_penalty * info['change']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite task-fusion loss')

            sums['loss'] += float(loss.detach())
            sums['detection_loss'] += float(detection.detach())
            sums['change'] += float(info['change'].detach())
            if len(context['levels'][0]) > 1:
                frame_has_multiple_sources = True
                multi_source_branches += 1
                sums['task_gap'] += float(info['task_gap'].detach())
                sums['task_disagreement'] += float(
                    info['task_disagreement'].detach())
                if not loss.requires_grad:
                    raise RuntimeError('Multi-source branch has no gradient')
                (loss / 2).backward()

        if frame_has_multiple_sources:
            torch.nn.utils.clip_grad_norm_(arm.trainable_parameters(), 5.0)
            optimizer.step()
            sums['optimizer_updates'] += 1
        else:
            sums['single_source_frames'] += 1

        sums['frames'] = number
        if number == 1 or number % 20 == 0:
            print(
                f'train {number}/{len(loader)} '
                f'loss={sums["loss"]/(2*number):.6f}',
                flush=True,
            )

    if not sums['frames']:
        raise RuntimeError('Empty train split')
    branches = 2 * sums['frames']
    for key in ('loss', 'detection_loss', 'change'):
        sums[key] /= branches
    denominator = max(1, multi_source_branches)
    sums['task_gap'] /= denominator
    sums['task_disagreement'] /= denominator
    sums['multi_source_branches'] = multi_source_branches
    return sums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pilot-config',
        default='local_fusion_task_split_pilot/experiment.yaml')
    parser.add_argument(
        '--v3-config', default='local_fusion_v3/experiment.yaml')
    for name in (
        'frontend-config',
        'frontend-checkpoint',
        'v3-checkpoint',
        'reference-run',
        'run',
    ):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()

    verify_frozen()
    pilot = settings(args.pilot_config)
    options, hypes = er.load_config(
        args.v3_config, args.frontend_config)
    v3_settings = v3rt.settings(options)
    target = device()

    model, frontend_digest = er.load_model(
        hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    v3_contract = v3rt.contract(
        options, args.frontend_config, frontend_digest)

    reference_run = Path(args.reference_run).resolve()
    reference = _load_json(reference_run / 'protocol.json')
    if reference.get('frontend_sha256') != frontend_digest:
        raise ValueError('Reference pilot used a different frontend checkpoint')
    if reference.get('v3_contract') != v3_contract:
        raise ValueError('Reference pilot v3 source/config contract differs')
    start_digest = sha256(args.v3_checkpoint)
    if reference.get('v3_checkpoint_sha256') != start_digest:
        raise ValueError('Reference pilot used a different v3 checkpoint')

    train_indices = [int(value) for value in reference['train_indices']]
    validation_indices = [
        int(value) for value in reference['validation_indices']
    ]
    if len(train_indices) != len(set(train_indices)):
        raise ValueError('Duplicate train indices in reference protocol')
    if len(validation_indices) != len(set(validation_indices)):
        raise ValueError('Duplicate validation indices in reference protocol')

    source, checkpoint = v3rt.load(
        args.v3_checkpoint, v3_contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Starting checkpoint must be v3 residual')
    start_fusion = copy.deepcopy(source).requires_grad_(False).eval()

    arms = {
        'Shared': TaskFusionArm(source, 'shared').to(target),
        'Split': TaskFusionArm(source, 'split').to(target),
    }
    del source

    if not _states_equal(arms['Shared'], arms['Split']):
        raise RuntimeError('Shared and Split do not share identical initial weights')
    for name, arm in arms.items():
        if not _states_equal(arm.router_a, arm.router_b):
            raise RuntimeError(name + ' routers do not share one identical start')
    if arms['Shared'].parameter_count() != arms['Split'].parameter_count():
        raise RuntimeError('Shared and Split parameter counts differ')

    run = new_output(Path(args.run).resolve())
    own_sources = [
        'local_fusion_task_split_pilot/' + name
        for name in (
            '__init__.py',
            'model.py',
            'evaluate.py',
            'pipeline.py',
            'experiment.yaml',
            'run_all.sh',
            'test_core.py',
        )
    ]
    dependency_sources = list(v3_contract['pipeline_sources'])
    dependency_sources += list(v3_contract['method_sources'])
    source_files = sorted(set(own_sources + dependency_sources))
    source_hashes = {name: sha256(ROOT / name) for name in source_files}
    for name in source_files:
        destination = run / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    write_json(
        run / 'protocol.json',
        {
            'purpose': (
                'paired Shared versus task-specific source fusion; '
                'frozen detector; official validation only'
            ),
            'v3_contract': v3_contract,
            'v3_checkpoint_sha256': start_digest,
            'frontend_sha256': frontend_digest,
            'reference_run': str(reference_run),
            'reference_protocol_sha256': sha256(
                reference_run / 'protocol.json'),
            'pilot': pilot,
            'source_hashes': source_hashes,
            'train_scenes': reference.get('train_scenes'),
            'train_indices': train_indices,
            'validation_scenes': reference.get('validation_scenes'),
            'validation_indices': validation_indices,
            'checkpoint_selection': 'fixed last epoch',
            'detector': (
                'frontend, deblocks, cls_head and reg_head frozen in both arms'
            ),
            'fairness': (
                'two v3-router copies in both arms; both execute two source '
                'routes and two detector decode paths; only weight sharing '
                'versus task-specific use differs'
            ),
        },
    )

    criterion = PointPillarLoss(hypes['loss']['args']).to(target)
    histories = {}
    for name, arm in arms.items():
        optimizer = torch.optim.AdamW(
            arm.trainable_parameters(),
            lr=pilot['learning_rate'],
            weight_decay=pilot['weight_decay'],
        )
        history = []
        for epoch in range(1, pilot['epochs'] + 1):
            seed_all(pilot['seed'] + epoch)
            _, loader = selected_loader(
                hypes, options, 'train', 'mixed',
                train_indices, epoch)
            started = time.time()
            row = train_epoch(
                model,
                arm,
                loader,
                criterion,
                target,
                v3_settings['change_penalty'],
                optimizer,
            )
            row.update(epoch=epoch, seconds=time.time() - started)
            history.append(row)
            print(json.dumps({'arm': name, **row}), flush=True)
            del loader
        histories[name] = history
        torch.save(
            {
                'arm': arm.state_dict(),
                'mode': arm.mode,
                'epoch': pilot['epochs'],
                'parameter_count': arm.parameter_count(),
                'start_checkpoint_sha256': start_digest,
            },
            run / (name + '.pth'),
        )
        write_json(run / 'train_history.json', histories)

    shared_updates = [
        row['optimizer_updates'] for row in histories['Shared']
    ]
    split_updates = [
        row['optimizer_updates'] for row in histories['Split']
    ]
    if shared_updates != split_updates:
        raise RuntimeError('Training arms received different optimizer step counts')

    for arm in arms.values():
        arm.eval()

    summaries = {}
    for weather in ('clean', 'fog', 'rain', 'snow'):
        seed_all(pilot['seed'] + 1)
        dataset, loader = selected_loader(
            hypes, options, 'validation', weather, validation_indices)
        folder = run / 'validation' / weather
        folder.mkdir(parents=True)
        summary = evaluate_condition(
            model,
            start_fusion,
            arms,
            dataset,
            loader,
            validation_indices,
            'clean' if weather == 'clean' else 'weather',
            target,
            folder,
        )
        summary.update(
            root=hypes['validate_dir'],
            online_weather=weather != 'clean',
        )
        summaries[weather] = summary
        write_json(folder / 'summary.json', summary)
        del dataset, loader

    decision = decide(summaries, pilot)
    write_json(
        run / 'decision_results.json',
        {
            'conditions': summaries,
            'decision': decision,
            'train_history': histories,
            'fairness': {
                'shared_parameters': arms['Shared'].parameter_count(),
                'split_parameters': arms['Split'].parameter_count(),
                'shared_optimizer_updates': shared_updates,
                'split_optimizer_updates': split_updates,
            },
            'scope': (
                'same frame indices as completed detector-adaptation pilot; '
                'selected official validation scenes; directional pilot; '
                'no OPV2V-W evaluation'
            ),
        },
    )
    verify_frozen()
    print('TASK-SPLIT PILOT COMPLETE:', run / 'decision_results.json', flush=True)


if __name__ == '__main__':
    main()
