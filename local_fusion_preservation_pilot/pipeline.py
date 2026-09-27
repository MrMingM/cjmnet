"""Small scheme-B decision test on official train/validation scenes only."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from gspr_communication.runtime import (
    ROOT, device, new_output, seed_all, seed_worker, sha256, verify_frozen,
    write_json,
)
from gspr_evidence import runtime as er
from local_fusion_v3 import runtime as v3rt
from local_fusion_v3.evaluate import condition as evaluate_condition
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.tools.train_utils import to_device
from .preservation import correct_anchor_mask, preservation_loss


def settings(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('seed', 'train_scenes', 'validation_scenes', 'frames_per_scene', 'epochs'):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('learning_rate', 'weight_decay', 'preservation_weight'):
        if not isinstance(value.get(key), (float, int)) or value[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    if value['learning_rate'] == 0 or value['preservation_weight'] == 0:
        raise ValueError('Pilot requires nonzero learning rate and preservation weight')
    for key in ('target_iou', 'identity_iou'):
        if not 0 < value[key] <= 1:
            raise ValueError(f'{key} must be in (0,1]')
    return value


def choose_frames(scene_ends, seed, scene_limit, frames_per_scene):
    """Choose scene IDs and frame offsets without reading labels."""
    ends = [int(x) for x in scene_ends]
    if not ends or any(b <= a for a, b in zip([0, *ends[:-1]], ends)):
        raise ValueError('Invalid scene boundaries')
    generator = torch.Generator().manual_seed(int(seed))
    scenes = torch.randperm(len(ends), generator=generator)[:scene_limit].sort().values.tolist()
    frames = []
    for scene in scenes:
        start = 0 if scene == 0 else ends[scene - 1]
        count = ends[scene] - start
        offsets = torch.randperm(count, generator=generator)[:frames_per_scene]
        frames.extend(start + int(x) for x in offsets)
    if not frames:
        raise ValueError('Empty selected split')
    return scenes, sorted(frames)


def selected_loader(hypes, options, split, weather, indices, epoch=0):
    dataset, _, _ = er.make_loader(hypes, options, train=split == 'train', weather=weather)
    if max(indices) >= len(dataset):
        raise RuntimeError('Selected frame does not exist in this weather condition')
    generator = torch.Generator().manual_seed(options['seed'] + epoch)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=split == 'train',
        num_workers=options['workers'], collate_fn=dataset.collate_batch_test,
        worker_init_fn=seed_worker, generator=generator,
    )
    return dataset, loader


def backward_if_trainable(loss, source_count):
    """Skip legitimate ego-only branches; reject missing multi-source gradients."""
    if loss.requires_grad:
        (loss / 2).backward()
        return True
    if source_count != 1:
        raise RuntimeError('Multi-source fusion unexpectedly has no gradient')
    return False


def train_epoch(model, module, dataset, loader, criterion, target, config,
                change_penalty, optimizer, alpha):
    module.train()
    model.eval()
    totals = dict(detection=0., preservation=0., change=0., teacher_tp=0,
                  protected_anchors=0, frames=0, optimizer_updates=0,
                  skipped_no_grad_branches=0)
    for number, batch in enumerate(loader, 1):
        batch = to_device(batch, target)
        optimizer.zero_grad(set_to_none=True)
        for branch in ('clean', 'weather'):
            with torch.no_grad():
                context = v3rt.context(model, batch['ego'], branch, verify=number == 1)
                teacher = context['baseline_prediction']
                mask, coverage = correct_anchor_mask(
                    dataset, batch, teacher, config['target_iou'], config['identity_iou'])
            prediction, info = module.predict(model.engine.base, context['levels'])
            detection = criterion(prediction, batch['ego']['label_dict'])
            preserve = preservation_loss(prediction, teacher, mask)
            loss = detection + change_penalty * info['change'] + alpha * preserve
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite pilot training loss')
            # Ego-only input bypasses every v3 trainable router parameter.
            if not backward_if_trainable(loss, len(context['levels'][0])):
                totals['skipped_no_grad_branches'] += 1
            totals['detection'] += float(detection.detach())
            totals['preservation'] += float(preserve.detach())
            totals['change'] += float(info['change'].detach())
            totals['teacher_tp'] += coverage['teacher_tp']
            totals['protected_anchors'] += coverage['protected']
        if any(parameter.grad is not None for parameter in module.parameters()):
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5.)
            optimizer.step()
            totals['optimizer_updates'] += 1
        totals['frames'] = number
        if number == 1 or number % 20 == 0:
            print(f'train {number}/{len(loader)} alpha={alpha}', flush=True)
    if not totals['frames']:
        raise RuntimeError('Empty pilot train loader')
    for key in ('detection', 'preservation', 'change'):
        totals[key] /= 2 * totals['frames']
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-config', default='local_fusion_preservation_pilot/experiment.yaml')
    parser.add_argument('--v3-config', default='local_fusion_v3/experiment.yaml')
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--v3-checkpoint', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    verify_frozen()
    pilot = settings(args.pilot_config)
    options, hypes = er.load_config(args.v3_config, args.frontend_config)
    v3_settings = v3rt.settings(options)
    target = device()
    model, digest = er.load_model(hypes, options, args.frontend_checkpoint, target)
    model.requires_grad_(False).eval()
    contract = v3rt.contract(options, args.frontend_config, digest)
    source, checkpoint = v3rt.load(args.v3_checkpoint, contract, target)
    if checkpoint['variant'] != 'residual':
        raise ValueError('Scheme B pilot requires a v3 residual checkpoint')
    control = copy.deepcopy(source)
    protected = copy.deepcopy(source)
    del source
    modules = {'continued_v3': control, 'protected_v3': protected}
    train_ds, _, _ = er.make_loader(hypes, options, train=True, weather='mixed')
    val_ds, _, _ = er.make_loader(hypes, options, train=False, weather='clean')
    train_scenes, train_indices = choose_frames(
        train_ds.len_record, pilot['seed'], pilot['train_scenes'], pilot['frames_per_scene'])
    val_scenes, val_indices = choose_frames(
        val_ds.len_record, pilot['seed'] + 1, pilot['validation_scenes'], pilot['frames_per_scene'])
    del train_ds, val_ds

    run = new_output(Path(args.run).resolve())
    resolved = run / 'resolved_pilot.yaml'
    resolved.write_text(yaml.safe_dump(pilot, sort_keys=False), encoding='utf-8')
    files = ['local_fusion_preservation_pilot/' + x for x in
             ('pipeline.py', 'preservation.py', 'experiment.yaml')]
    files += list(contract['pipeline_sources']) + list(contract['method_sources'])
    source_hashes = {name: sha256(ROOT / name) for name in sorted(set(files))}
    for name in source_hashes:
        destination = run / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write_json(run / 'protocol.json', dict(
        purpose='small scheme-B train/validation decision; no OPV2V-W',
        v3_checkpoint_sha256=sha256(args.v3_checkpoint),
        pilot_config_sha256=sha256(args.pilot_config),
        v3_config_sha256=sha256(args.v3_config),
        frontend_sha256=digest, v3_contract=contract, pilot=pilot,
        train_scenes=train_scenes, train_indices=train_indices,
        validation_scenes=val_scenes, validation_indices=val_indices,
        source_hashes=source_hashes,
        checkpoint_selection='fixed final epoch; no validation/test selection',
    ))
    criterion = PointPillarLoss(hypes['loss']['args']).to(target)
    histories = {}
    for name, module in modules.items():
        alpha = 0. if name == 'continued_v3' else pilot['preservation_weight']
        optimizer = torch.optim.AdamW(
            module.parameters(), lr=pilot['learning_rate'],
            weight_decay=pilot['weight_decay'])
        history = []
        for epoch in range(1, pilot['epochs'] + 1):
            seed_all(pilot['seed'] + epoch)
            dataset, loader = selected_loader(
                hypes, options, 'train', 'mixed', train_indices, epoch)
            row = train_epoch(model, module, dataset, loader, criterion,
                              target, pilot, v3_settings['change_penalty'],
                              optimizer, alpha)
            row['epoch'] = epoch
            history.append(row)
            print(json.dumps({'method': name, **row}), flush=True)
            del dataset, loader
        histories[name] = history
        torch.save(dict(module=module.state_dict(), variant='residual',
                        start_checkpoint_sha256=sha256(args.v3_checkpoint),
                        epoch=pilot['epochs'], alpha=alpha), run / f'{name}.pth')
        write_json(run / 'train_history.json', histories)

    summaries = {}
    for module in modules.values():
        module.eval()
    for weather in ('clean', 'fog', 'rain', 'snow'):
        seed_all(pilot['seed'] + 1)
        dataset, loader = selected_loader(
            hypes, options, 'validation', weather, val_indices)
        folder = run / 'validation' / weather
        folder.mkdir(parents=True)
        summary = evaluate_condition(
            model, modules, dataset, loader, val_indices,
            'clean' if weather == 'clean' else 'weather', target, folder, None)
        summary['root'] = hypes['validate_dir']
        summary['online_weather'] = weather != 'clean'
        summaries[weather] = summary
        write_json(folder / 'summary.json', summary)
        del dataset, loader
    write_json(run / 'decision_results.json', dict(
        conditions=summaries, train_history=histories,
        scope='selected official validation scenes only; directional, not independent AP'))
    verify_frozen()
    print('SCHEME B PILOT COMPLETE:', run / 'decision_results.json', flush=True)


if __name__ == '__main__':
    main()
