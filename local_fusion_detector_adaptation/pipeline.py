"""Short, paired F / F+D continuation from one v3 residual checkpoint."""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from gspr_communication.runtime import (ROOT, device, new_output, seed_all,
                                        seed_worker, sha256, verify_frozen, write_json)
from gspr_evidence import runtime as er
from local_fusion_v3 import runtime as v3rt
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.tools.train_utils import to_device
from .evaluate import evaluate_condition, decide
from .model import AdaptationArm


def settings(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('seed', 'train_scenes', 'validation_scenes', 'frames_per_scene', 'epochs',
                'minimum_positive_weathers'):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if value['minimum_positive_weathers'] > 3:
        raise ValueError('More than three positive weathers requested')
    for key in ('fusion_learning_rate', 'detector_learning_rate',
                'minimum_weather_mean_ap70_gain'):
        if not 0 < value[key] <= 1:
            raise ValueError(key + ' must be in (0,1]')
    for key in ('weight_decay', 'clean_ap70_tolerance', 'ap50_tolerance',
                'max_lost_tp_fraction', 'max_new_fp_per_reference_tp'):
        if not 0 <= value[key] <= 1:
            raise ValueError(key + ' must be in [0,1]')
    return value


def choose_frames(scene_ends, seed, scene_limit, frames_per_scene):
    """Sample scenes and frames without consulting labels or weather outcomes."""
    ends = [int(x) for x in scene_ends]
    if not ends or any(b <= a for a, b in zip([0, *ends[:-1]], ends)):
        raise ValueError('Invalid scene boundaries')
    generator = torch.Generator().manual_seed(seed)
    scenes = torch.randperm(len(ends), generator=generator)[:scene_limit].sort().values.tolist()
    frames = []
    for scene in scenes:
        start = 0 if scene == 0 else ends[scene - 1]
        offsets = torch.randperm(ends[scene] - start, generator=generator)[:frames_per_scene]
        frames.extend(start + int(offset) for offset in offsets)
    if not frames:
        raise ValueError('Empty frame subset')
    return scenes, sorted(frames)


def selected_loader(hypes, options, split, weather, indices, epoch=0):
    dataset, _, _ = er.make_loader(hypes, options, train=split == 'train', weather=weather)
    if max(indices) >= len(dataset):
        raise RuntimeError('Selected index missing from dataset')
    generator = torch.Generator().manual_seed(options['seed'] + epoch)
    loader = DataLoader(Subset(dataset, indices), batch_size=1,
                        shuffle=split == 'train', num_workers=options['workers'],
                        collate_fn=dataset.collate_batch_test,
                        worker_init_fn=seed_worker, generator=generator)
    return dataset, loader


def train_epoch(model, arm, loader, criterion, target, change_penalty, optimizer):
    arm.train()
    model.eval()
    sums = dict(loss=0., detection_loss=0., change=0., frames=0,
                optimizer_updates=0, single_source_frames=0)
    for number, batch in enumerate(loader, 1):
        batch = to_device(batch, target)
        optimizer.zero_grad(set_to_none=True)
        for branch in ('clean', 'weather'):
            with torch.no_grad():
                ctx = v3rt.context(model, batch['ego'], branch, verify=number == 1)
            prediction, info = arm.predict(model.engine.base, ctx['levels'])
            detection = criterion(prediction, batch['ego']['label_dict'])
            loss = detection + change_penalty * info['change']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite paired training loss')
            sums['loss'] += float(loss.detach())
            sums['detection_loss'] += float(detection.detach())
            sums['change'] += float(info['change'].detach())
            # Both arms skip ego-only frames. Otherwise F+D would get extra
            # detector-only optimizer updates absent in F.
            if len(ctx['levels'][0]) > 1:
                if not loss.requires_grad:
                    raise RuntimeError('Multi-source branch has no gradient')
                (loss / 2).backward()
        if len(ctx['levels'][0]) > 1:
            torch.nn.utils.clip_grad_norm_(arm.trainable_parameters(), 5.)
            optimizer.step()
            sums['optimizer_updates'] += 1
        else:
            sums['single_source_frames'] += 1
        sums['frames'] = number
        if number == 1 or number % 20 == 0:
            print(f'train {number}/{len(loader)} loss={sums["loss"]/(2*number):.6f}', flush=True)
    if not sums['frames']:
        raise RuntimeError('Empty train split')
    for key in ('loss', 'detection_loss', 'change'):
        sums[key] /= 2 * sums['frames']
    return sums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-config', default='local_fusion_detector_adaptation/experiment.yaml')
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    for name in ('frontend-config', 'frontend-checkpoint', 'v3-checkpoint', 'run'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    verify_frozen()
    pilot = settings(args.pilot_config)
    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    v3_settings = v3rt.settings(options)
    target = device()
    model, frontend_digest = er.load_model(hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    v3_contract = v3rt.contract(options, args.frontend_config, frontend_digest)
    source, checkpoint = v3rt.load(args.v3_checkpoint, v3_contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Starting checkpoint must be v3 residual')
    arms = {'F': AdaptationArm(source, model.engine.base, False).to(target),
            'F+D': AdaptationArm(source, model.engine.base, True).to(target)}
    del source
    for arm in arms.values():
        arm.eval()
    if any(not torch.equal(left, right) for left, right in zip(
            arms['F'].state_dict().values(), arms['F+D'].state_dict().values())):
        raise RuntimeError('F and F+D do not share the same initial weights')
    train_ds, _, _ = er.make_loader(hypes, options, train=True)
    val_ds, _, _ = er.make_loader(hypes, options, train=False, weather='clean')
    train_scenes, train_indices = choose_frames(
        train_ds.len_record, pilot['seed'], pilot['train_scenes'], pilot['frames_per_scene'])
    val_scenes, val_indices = choose_frames(
        val_ds.len_record, pilot['seed'] + 1, pilot['validation_scenes'], pilot['frames_per_scene'])
    del train_ds, val_ds
    run = new_output(Path(args.run).resolve())
    source_files = sorted(set(
        ['local_fusion_detector_adaptation/' + name for name in
         ('__init__.py', 'model.py', 'evaluate.py', 'pipeline.py', 'experiment.yaml', 'run_all.sh')]
        + list(v3_contract['pipeline_sources']) + list(v3_contract['method_sources'])))
    source_hashes = {name: sha256(ROOT / name) for name in source_files}
    for name in source_files:
        destination = run / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write_json(run / 'protocol.json', dict(
        purpose='short paired F versus F+D continuation; official validation only',
        v3_contract=v3_contract, v3_checkpoint_sha256=sha256(args.v3_checkpoint),
        frontend_sha256=frontend_digest, pilot=pilot, source_hashes=source_hashes,
        train_scenes=train_scenes, train_indices=train_indices,
        validation_scenes=val_scenes, validation_indices=val_indices,
        checkpoint_selection='fixed last epoch',
        source_path='original frontend remains frozen for encoding and source confidence',
        train_budget='same frames, order, clean/weather branches, objective, fusion LR and optimizer steps; detector group adds parameters'))
    criterion = PointPillarLoss(hypes['loss']['args']).to(target)
    histories = {}
    for name, arm in arms.items():
        groups = [{'params': list(arm.fusion.parameters()), 'lr': pilot['fusion_learning_rate']}]
        if arm.adapt_detector:
            groups.append({'params': list(arm.detector.parameters()), 'lr': pilot['detector_learning_rate']})
        optimizer = torch.optim.AdamW(groups, weight_decay=pilot['weight_decay'])
        history = []
        for epoch in range(1, pilot['epochs'] + 1):
            seed_all(pilot['seed'] + epoch)
            _, loader = selected_loader(hypes, options, 'train', 'mixed', train_indices, epoch)
            started = time.time()
            row = train_epoch(model, arm, loader, criterion, target,
                              v3_settings['change_penalty'], optimizer)
            row.update(epoch=epoch, seconds=time.time() - started)
            history.append(row)
            print(json.dumps({'arm': name, **row}), flush=True)
            del loader
        histories[name] = history
        torch.save(dict(arm=arm.state_dict(), adapt_detector=arm.adapt_detector,
                        epoch=pilot['epochs'], start_checkpoint_sha256=sha256(args.v3_checkpoint)),
                   run / (name + '.pth'))
        write_json(run / 'train_history.json', histories)
    if [row['optimizer_updates'] for row in histories['F']] != [row['optimizer_updates'] for row in histories['F+D']]:
        raise RuntimeError('Training arms received different optimizer step counts')
    for arm in arms.values():
        arm.eval()
    summaries = {}
    for weather in ('clean', 'fog', 'rain', 'snow'):
        seed_all(pilot['seed'] + 1)
        dataset, loader = selected_loader(hypes, options, 'validation', weather, val_indices)
        folder = run / 'validation' / weather
        folder.mkdir(parents=True)
        summary = evaluate_condition(model, arms, dataset, loader, val_indices,
                                     'clean' if weather == 'clean' else 'weather', target, folder)
        summary.update(root=hypes['validate_dir'], online_weather=weather != 'clean')
        summaries[weather] = summary
        write_json(folder / 'summary.json', summary)
        del dataset, loader
    decision = decide(summaries, pilot)
    write_json(run / 'decision_results.json', dict(
        conditions=summaries, decision=decision, train_history=histories,
        scope='selected official validation scenes; directional pilot, no OPV2V-W evaluation'))
    verify_frozen()
    print('PILOT COMPLETE:', run / 'decision_results.json', flush=True)


if __name__ == '__main__':
    main()
