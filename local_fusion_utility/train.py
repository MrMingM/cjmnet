"""Train either the proposed outcome head or the matched loss-gain ablation."""
import argparse
import json
import random
from pathlib import Path


def rows_from_record(record):
    for branch in ('clean', 'weather'):
        for row in record[branch]['rows']:
            yield branch, row


def frame_loss(predictor, record, variant, settings, device):
    import torch
    import torch.nn.functional as F
    descriptors, targets, gains = [], [], []
    for _, row in rows_from_record(record):
        descriptors.append(row['descriptor'].float())
        targets.append(row['target'].float())
        gains.append(float(row['loss_gain']))
    if not descriptors:
        return None, None
    x = torch.stack(descriptors).to(device)[:, :, None, None]
    target = torch.stack(targets).to(device)
    gain = torch.tensor(gains, device=device)*float(settings['loss_gain_scale'])
    prediction = predictor(x)
    if variant == 'utility':
        predicted = prediction['outcome'][:, :, 0, 0]
        element = F.smooth_l1_loss(predicted, target, reduction='none')
        weight = 1+float(settings['positive_weight'])*(target > 0).float()
        loss = (element*weight).sum()/weight.sum()
        metrics = {'mae': float((predicted-target).abs().mean().detach())}
    else:
        predicted = prediction['loss_gain'][:, 0, 0, 0]
        loss = F.smooth_l1_loss(predicted, gain)
        metrics = {'mae': float((predicted-gain).abs().mean().detach())}
    return loss, metrics


def run_epoch(predictor, paths, variant, settings, device, optimizer=None, seed=0):
    import torch
    order = list(paths)
    if optimizer is not None:
        random.Random(seed).shuffle(order)
        predictor.train()
    else:
        predictor.eval()
    loss_sum = mae_sum = count = 0
    for path in order:
        record = torch.load(path, map_location='cpu', weights_only=True)
        with torch.set_grad_enabled(optimizer is not None):
            loss, metrics = frame_loss(predictor, record, variant, settings, device)
            if loss is None:
                continue
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        loss_sum += float(loss.detach())
        mae_sum += metrics['mae']
        count += 1
    if not count:
        raise RuntimeError('No sampled actions in cache')
    return {'loss': loss_sum/count, 'mae': mae_sum/count, 'frames': count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--validation-cache', required=True)
    parser.add_argument('--variant', choices=('utility', 'loss_gain'), required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    import torch
    from gspr_communication.runtime import new_output, seed_all, device, write_json
    from .network import UtilityPredictor
    from .fusion import DESCRIPTOR_NAMES
    from . import runtime as rt

    train_manifest, train_paths = rt.cache_files(args.train_cache)
    val_manifest, val_paths = rt.cache_files(args.validation_cache)
    if train_manifest['split'] != 'train' or val_manifest['split'] != 'validation':
        raise ValueError('Use official train and validation caches in the correct roles')
    if train_manifest['contract'] != val_manifest['contract']:
        raise ValueError('Train/validation cache contracts differ')
    if train_manifest.get('smoke') or val_manifest.get('smoke'):
        print('WARNING: smoke cache; result is development-only', flush=True)
    specification = train_manifest['contract']
    options = specification['options']
    settings = rt.settings(options)
    seed_all(int(options['seed']))
    target = device()
    predictor = UtilityPredictor(len(DESCRIPTOR_NAMES), settings['hidden']).to(target)
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=float(settings['learning_rate']),
        weight_decay=float(settings['weight_decay']))
    output = new_output(args.output_dir)
    write_json(output/'protocol.json', {
        'variant': args.variant,
        'train_cache': str(Path(args.train_cache).resolve()),
        'validation_cache': str(Path(args.validation_cache).resolve()),
        'contract': specification,
        'descriptor_names': DESCRIPTOR_NAMES,
    })
    history, best = [], float('inf')
    for epoch in range(1, settings['epochs']+1):
        training = run_epoch(predictor, train_paths, args.variant, settings, target,
                             optimizer, int(options['seed'])+epoch)
        with torch.no_grad():
            validation = run_epoch(predictor, val_paths, args.variant, settings, target)
        row = {'epoch': epoch, 'train': training, 'validation': validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        rt.save_checkpoint(output/'last.pth', predictor, optimizer, epoch, validation,
                           args.variant, specification)
        if validation['loss'] < best:
            best = validation['loss']
            rt.save_checkpoint(output/'best.pth', predictor, optimizer, epoch, validation,
                               args.variant, specification)
        write_json(output/'history.json', history)
    write_json(output/'summary.json', {'best_validation_loss': best, 'epochs': settings['epochs']})


if __name__ == '__main__':
    main()
