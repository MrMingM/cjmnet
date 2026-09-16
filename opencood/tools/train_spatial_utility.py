"""Train a per-cell neighbor utility head from Oracle spatial caches."""

import argparse
import glob
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from opencood.models.sub_modules.spatial_utility_net import (
    SpatialUtilityNet, build_neighbor_input)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dirs', nargs='+', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1.0e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0e-4)
    parser.add_argument('--harm_weight', type=float, default=1.0)
    parser.add_argument('--harmful_positive_weight', type=float, default=2.0)
    parser.add_argument('--soft_eps', type=float, default=0.005)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260715)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--utility_threshold', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.02)
    parser.add_argument('--gate_floor', type=float, default=0.05)
    parser.add_argument('--foreground_threshold', type=float, default=0.05)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_samples(cache_dirs):
    samples = []
    for cache_dir in cache_dirs:
        paths = sorted(glob.glob(os.path.join(cache_dir, '*.npz')))
        if not paths:
            raise FileNotFoundError('No NPZ files in %s' % cache_dir)
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                cav_count = int(data['cav_count'])
                scenario = int(data['scenario_index'])
                domain = str(data['domain_name'].item())
            for neighbor_index in range(1, cav_count):
                samples.append({
                    'path': path,
                    'neighbor_index': neighbor_index,
                    'scenario_index': scenario,
                    'domain': domain
                })
    if not samples:
        raise RuntimeError('No neighbor samples discovered')
    return samples


def scenario_split(samples, val_ratio, seed):
    scenarios = sorted({item['scenario_index'] for item in samples})
    rng = random.Random(seed)
    rng.shuffle(scenarios)
    val_count = max(1, int(round(len(scenarios) * val_ratio)))
    val_scenarios = set(scenarios[:val_count])
    train = [item for item in samples
             if item['scenario_index'] not in val_scenarios]
    val = [item for item in samples
           if item['scenario_index'] in val_scenarios]
    if not train or not val:
        raise RuntimeError('Scenario split produced an empty partition')
    return train, val, sorted(val_scenarios)


class SpatialCacheDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        with np.load(item['path'], allow_pickle=False) as data:
            confidence = torch.from_numpy(
                data['confidence'].astype(np.float32))
            density = torch.from_numpy(data['density'].astype(np.float32))
            slot = item['neighbor_index'] - 1
            utility = torch.from_numpy(
                data['utility'][slot].astype(np.float32)).unsqueeze(0)
            valid = torch.from_numpy(
                data['valid'][slot].astype(np.bool_)).unsqueeze(0)
        inputs = build_neighbor_input(
            confidence, density, item['neighbor_index'])
        return inputs, utility, valid


def balanced_epoch_samples(samples, seed, epoch):
    by_domain = {}
    for item in samples:
        by_domain.setdefault(item['domain'], []).append(item)
    target = max(len(values) for values in by_domain.values())
    rng = random.Random(seed + epoch)
    output = []
    for values in by_domain.values():
        output.extend(values)
        output.extend(rng.choices(values, k=target - len(values)))
    rng.shuffle(output)
    return output


def masked_losses(output, utility, valid, args):
    valid_float = valid.float()
    denominator = valid_float.sum().clamp_min(1.0)
    regression = F.smooth_l1_loss(
        output['utility'], utility, reduction='none', beta=0.02)
    importance = 1.0 + utility.abs().div(0.05).clamp(max=4.0)
    regression = (regression * importance * valid_float).sum() / denominator
    harmful = (utility < -args.soft_eps).float()
    harm_weight = torch.where(
        harmful > 0.5,
        torch.full_like(harmful, args.harmful_positive_weight),
        torch.ones_like(harmful))
    classification = F.binary_cross_entropy_with_logits(
        output['harm_logit'], harmful, reduction='none')
    classification = (
        classification * harm_weight * valid_float).sum() / denominator
    return regression + args.harm_weight * classification, regression, classification


def run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(5, dtype=np.float64)
    batches = 0
    for inputs, utility, valid in loader:
        inputs = inputs.to(device)
        utility = utility.to(device)
        valid = valid.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(inputs)
            losses = masked_losses(output, utility, valid, args)
            if training:
                losses[0].backward()
                optimizer.step()
        mask = valid.bool()
        predicted_harm = output['harm_logit'] < 0.0
        target_harm = utility >= -args.soft_eps
        sign_accuracy = (predicted_harm[mask] == target_harm[mask]).float().mean()
        mae = (output['utility'][mask] - utility[mask]).abs().mean()
        totals += np.asarray([
            losses[0].detach().item(), losses[1].detach().item(),
            losses[2].detach().item(), mae.detach().item(),
            sign_accuracy.detach().item()])
        batches += 1
    return (totals / max(batches, 1)).tolist()


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    samples = discover_samples(args.cache_dirs)
    train_samples, val_samples, val_scenarios = scenario_split(
        samples, args.val_ratio, args.seed)
    val_loader = DataLoader(
        SpatialCacheDataset(val_samples), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device)
    model = SpatialUtilityNet(args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path = os.path.join(args.output_dir, 'best_spatial_utility.pth')
    history = []
    best_loss = float('inf')
    best_epoch = -1
    stale = 0
    print('Samples train=%d val=%d domains=%s val_scenarios=%s' % (
        len(train_samples), len(val_samples),
        dict(Counter(item['domain'] for item in samples)), val_scenarios))
    for epoch in range(args.epochs):
        epoch_train = balanced_epoch_samples(
            train_samples, args.seed, epoch)
        train_loader = DataLoader(
            SpatialCacheDataset(epoch_train), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers)
        train_metrics = run_epoch(
            model, train_loader, device, args, optimizer)
        val_metrics = run_epoch(model, val_loader, device, args)
        record = {
            'epoch': epoch + 1,
            'train_loss': train_metrics[0],
            'train_utility_mae': train_metrics[3],
            'train_harm_accuracy': train_metrics[4],
            'val_loss': val_metrics[0],
            'val_utility_mae': val_metrics[3],
            'val_harm_accuracy': val_metrics[4]
        }
        history.append(record)
        print('epoch %03d train %.4f val %.4f val_mae %.4f val_harm_acc %.3f' %
              (epoch + 1, train_metrics[0], val_metrics[0],
               val_metrics[3], val_metrics[4]))
        if val_metrics[0] < best_loss - 1.0e-6:
            best_loss = val_metrics[0]
            best_epoch = epoch + 1
            stale = 0
            torch.save({
                'schema_version': 'spatial-utility-v1',
                'model_state_dict': model.state_dict(),
                'hidden_channels': args.hidden_channels,
                'utility_threshold': args.utility_threshold,
                'temperature': args.temperature,
                'gate_floor': args.gate_floor,
                'foreground_threshold': args.foreground_threshold,
                'soft_eps': args.soft_eps,
                'best_epoch': best_epoch,
                'validation': record
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping at epoch %d' % (epoch + 1))
                break
    summary = {
        'schema_version': 'spatial-utility-v1',
        'cache_dirs': [os.path.abspath(path) for path in args.cache_dirs],
        'samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'val_scenarios': val_scenarios,
        'domains': dict(Counter(item['domain'] for item in samples)),
        'best_epoch': best_epoch,
        'best_val_loss': best_loss,
        'checkpoint': os.path.abspath(checkpoint_path),
        'note': 'Validation losses are spatial-label metrics; final quality is AP.'
    }
    with open(os.path.join(args.output_dir, 'training_history.json'),
              'w', encoding='utf-8') as stream:
        json.dump(history, stream, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'spatial_summary.json'),
              'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print('Training finished best_epoch=%d checkpoint=%s' %
          (best_epoch, checkpoint_path))


if __name__ == '__main__':
    main()
