"""Train a deployable coalition ranker from Oracle selection JSONL files.

Inputs are annotation-free frame/agent features exported by oracle_eval.py
(schema v3).  Ground-truth-derived candidate AP is used only as supervision.
The network scores each candidate coalition and selects the highest score.
"""

import argparse
import json
import math
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an MLP counterfactual coalition selector')
    parser.add_argument('--input_jsonl', nargs='+', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_cav', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1.0e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0e-4)
    parser.add_argument('--rank_weight', type=float, default=0.5)
    parser.add_argument('--regret_weight', type=float, default=1.0)
    parser.add_argument('--classification_weight', type=float, default=0.2)
    parser.add_argument('--negative_regret_weight', type=float, default=3.0,
                        help='Extra penalty for overvaluing harmful non-all coalitions')
    parser.add_argument('--rank_margin', type=float, default=0.002,
                        help='Ignore candidate AP differences below this value')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--group_field',
                        choices=['scenario_index', 'dataset_index'],
                        default='scenario_index')
    parser.add_argument('--safety_tolerance', type=float, default=0.1,
                        help='Choose the most conservative margin within this '
                             'fraction of best validation gain')
    parser.add_argument('--no_domain_balance', action='store_true')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=20260715)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_items(mapping, prefix=''):
    output = {}
    for key, value in mapping.items():
        name = prefix + key
        if isinstance(value, dict):
            output.update(scalar_items(value, name + '.'))
        elif isinstance(value, (bool, int, float)):
            value = float(value)
            output[name] = value if math.isfinite(value) else 0.0
    return output


def load_rows(paths):
    rows = []
    for source_index, path in enumerate(paths):
        with open(path, encoding='utf-8') as stream:
            for line_index, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                if 'deployable_features' not in row:
                    raise ValueError(
                        '%s line %d lacks schema-v3 deployable_features' %
                        (path, line_index + 1))
                row['_source_index'] = source_index
                rows.append(row)
    if not rows:
        raise RuntimeError('No training frames found')
    return rows


def discover_schema(rows, max_cav):
    agent_names = set()
    pair_names = set()
    observed_max = 0
    for row in rows:
        features = row['deployable_features']
        observed_max = max(observed_max, int(row['cav_count']))
        for agent in features['agents']:
            agent_names.update(scalar_items(
                {k: v for k, v in agent.items() if k != 'agent_index'}))
        for pair in features['pairwise']:
            pair_names.update(scalar_items({
                k: v for k, v in pair.items()
                if k not in ('first_agent', 'second_agent')
            }))
    if observed_max > max_cav:
        raise ValueError('Observed %d CAVs exceeds --max_cav %d' %
                         (observed_max, max_cav))
    return {
        'max_cav': int(max_cav),
        'agent_features': sorted(agent_names),
        'pair_features': sorted(pair_names)
    }


def pair_slots(max_cav):
    return [(first, second) for first in range(max_cav)
            for second in range(first + 1, max_cav)]


def encode_frame_base(row, schema):
    max_cav = schema['max_cav']
    agents = {
        int(item['agent_index']): scalar_items({
            k: v for k, v in item.items() if k != 'agent_index'
        }) for item in row['deployable_features']['agents']
    }
    pairs = {
        (int(item['first_agent']), int(item['second_agent'])):
        scalar_items({k: v for k, v in item.items()
                      if k not in ('first_agent', 'second_agent')})
        for item in row['deployable_features']['pairwise']
    }
    vector = [float(row['cav_count']) / max_cav]
    for index in range(max_cav):
        values = agents.get(index)
        vector.append(float(values is not None))
        for name in schema['agent_features']:
            vector.append(float(values.get(name, 0.0)) if values else 0.0)
    for slot in pair_slots(max_cav):
        values = pairs.get(slot)
        vector.append(float(values is not None))
        for name in schema['pair_features']:
            vector.append(float(values.get(name, 0.0)) if values else 0.0)
    return vector


def encode_candidate(base, candidate, schema):
    active = set(int(x) for x in candidate['active_agents'])
    mask = [float(index in active) for index in range(schema['max_cav'])]
    neighbor_fraction = max(len(active) - 1, 0) / max(schema['max_cav'] - 1, 1)
    return np.asarray(base + mask + [neighbor_fraction], dtype=np.float32)


def prepare_frames(rows, schema, group_field):
    frames = []
    for row in rows:
        base = encode_frame_base(row, schema)
        selected_name = row['selected']['frame_ap']['name']
        candidates = row['candidates']
        names = [item['name'] for item in candidates]
        if selected_name not in names:
            raise ValueError('Selected candidate %s is absent' % selected_name)
        features = np.stack([
            encode_candidate(base, item, schema) for item in candidates
        ])
        target_ap = np.asarray([
            float(item['metrics']['frame_ap']) for item in candidates
        ], dtype=np.float32)
        all_index = names.index('all')
        frames.append({
            'features': features,
            'target_ap': target_ap,
            'label': names.index(selected_name),
            'all_index': all_index,
            'names': names,
            'dataset_index': int(row['dataset_index']),
            'scenario_index': int(row.get(
                'scenario_index', row['dataset_index'])),
            'domain': str(row.get('domain_name', 'unknown')),
            # Keep all weather variants/scenes with the same base scenario
            # together. Older v3 files fall back to dataset_index.
            'group': int(row.get(group_field, row['dataset_index']))
        })
    return frames


def group_split(frames, val_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError('--val_ratio must be in (0, 1)')
    groups = sorted({item['group'] for item in frames})
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_count = max(1, int(round(len(groups) * val_ratio)))
    val_groups = set(groups[:val_count])
    train = [item for item in frames if item['group'] not in val_groups]
    val = [item for item in frames if item['group'] in val_groups]
    if not train or not val:
        raise RuntimeError('Group split produced an empty partition')
    return train, val


def fit_normalizer(frames):
    matrix = np.concatenate([item['features'] for item in frames], axis=0)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    return mean, std


def normalize_frames(frames, mean, std):
    for item in frames:
        item['features'] = (item['features'] - mean) / std


class CoalitionRanker(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(current, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features).squeeze(-1)


def frame_loss(scores, target_ap, label, all_index, args):
    classification = F.cross_entropy(
        scores.unsqueeze(0),
        torch.tensor([label], device=scores.device))
    target_regret = target_ap - target_ap[all_index]
    weights = torch.ones_like(target_regret)
    harmful = target_regret < 0.0
    harmful[all_index] = False
    weights[harmful] = args.negative_regret_weight
    regression = (
        F.smooth_l1_loss(scores, target_regret, reduction='none') *
        weights).mean()
    differences = target_ap[:, None] - target_ap[None, :]
    valid = differences > args.rank_margin
    if bool(valid.any().item()):
        score_differences = scores[:, None] - scores[None, :]
        ranking = F.softplus(-score_differences[valid]).mean()
    else:
        ranking = scores.sum() * 0.0
    total = args.regret_weight * regression + \
        args.rank_weight * ranking + \
        args.classification_weight * classification
    return total, regression, classification, ranking


def balanced_indices(frames, rng, enable):
    if not enable:
        indices = list(range(len(frames)))
        rng.shuffle(indices)
        return indices
    by_domain = {}
    for index, item in enumerate(frames):
        by_domain.setdefault(item['domain'], []).append(index)
    target = max(len(values) for values in by_domain.values())
    indices = []
    for values in by_domain.values():
        indices.extend(values)
        indices.extend(rng.choices(values, k=target - len(values)))
    rng.shuffle(indices)
    return indices


def train_epoch(model, frames, optimizer, batch_size, device, args, rng):
    model.train()
    indices = balanced_indices(frames, rng, not args.no_domain_balance)
    totals = np.zeros(4, dtype=np.float64)
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    for position, index in enumerate(indices):
        item = frames[index]
        features = torch.from_numpy(item['features']).to(device)
        target_ap = torch.from_numpy(item['target_ap']).to(device)
        scores = model(features)
        losses = frame_loss(
            scores, target_ap, item['label'], item['all_index'], args)
        (losses[0] / args.batch_size).backward()
        pending += 1
        totals += np.asarray([x.detach().item() for x in losses])
        if pending == args.batch_size or position == len(indices) - 1:
            if pending != args.batch_size:
                scale = args.batch_size / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
    return (totals / len(indices)).tolist()


@torch.no_grad()
def predict_scores(model, frames, device):
    model.eval()
    return [model(torch.from_numpy(item['features']).to(device))
            .detach().cpu().numpy() for item in frames]


def select_with_safety(item, scores, safety_margin):
    all_index = item['all_index']
    alternatives = [index for index in range(len(scores))
                    if index != all_index]
    if not alternatives:
        return all_index
    best_alternative = max(alternatives, key=lambda index: scores[index])
    if scores[best_alternative] - scores[all_index] > safety_margin:
        return best_alternative
    return all_index


def evaluate_from_scores(frames, score_list, safety_margin):
    correct = 0
    selected_ap = []
    oracle_ap = []
    all_ap = []
    selection_counts = Counter()
    harmful_switches = 0
    for item, scores in zip(frames, score_list):
        selected = select_with_safety(item, scores, safety_margin)
        correct += int(selected == item['label'])
        selected_ap.append(float(item['target_ap'][selected]))
        oracle_ap.append(float(item['target_ap'][item['label']]))
        all_index = item['all_index']
        all_ap.append(float(item['target_ap'][all_index]))
        harmful_switches += int(
            selected != all_index and
            item['target_ap'][selected] < item['target_ap'][all_index])
        selection_counts[item['names'][selected]] += 1
    selected_mean = float(np.mean(selected_ap))
    oracle_mean = float(np.mean(oracle_ap))
    all_mean = float(np.mean(all_ap))
    denominator = oracle_mean - all_mean
    return {
        'frames': len(frames),
        'safety_margin': float(safety_margin),
        'top1_accuracy': correct / len(frames),
        'mean_selected_frame_ap': selected_mean,
        'mean_oracle_frame_ap': oracle_mean,
        'mean_all_frame_ap': all_mean,
        'frame_ap_gain_vs_all': selected_mean - all_mean,
        'oracle_frame_ap_headroom': denominator,
        'headroom_recovery_ratio': (
            (selected_mean - all_mean) / denominator
            if abs(denominator) > 1.0e-12 else 0.0),
        'harmful_switch_ratio': harmful_switches / len(frames),
        'selection_counts': dict(selection_counts)
    }


def calibrate_safety_margin(frames, score_list, tolerance):
    if not 0.0 <= tolerance < 1.0:
        raise ValueError('--safety_tolerance must be in [0, 1)')
    margins = []
    for item, scores in zip(frames, score_list):
        all_index = item['all_index']
        alternatives = [scores[index] for index in range(len(scores))
                        if index != all_index]
        if alternatives:
            margins.append(float(max(alternatives) - scores[all_index]))
    if not margins:
        return 0.0, evaluate_from_scores(frames, score_list, 0.0)
    # A safety selector may leave ``all`` only for a positive predicted
    # advantage. Negative thresholds recreate the aggressive v1 behavior.
    candidates = sorted(set(
        [0.0, max(0.0, max(margins) + 1.0e-6)] +
        [max(0.0, float(x)) for x in np.quantile(
            margins, np.linspace(0.0, 1.0, 101))]))
    evaluated = [
        (margin, evaluate_from_scores(frames, score_list, margin))
        for margin in candidates
    ]
    all_mean = evaluated[0][1]['mean_all_frame_ap']
    best_gain = max(item[1]['mean_selected_frame_ap'] - all_mean
                    for item in evaluated)
    required_gain = max(0.0, best_gain * (1.0 - tolerance))
    eligible = [item for item in evaluated
                if item[1]['mean_selected_frame_ap'] - all_mean >=
                required_gain - 1.0e-10]
    # Prefer the largest threshold among near-optimal choices: it switches
    # away from all less often and is safer under domain shift.
    return max(eligible, key=lambda item: item[0])


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.input_jsonl)
    schema = discover_schema(rows, args.max_cav)
    frames = prepare_frames(rows, schema, args.group_field)
    train_frames, val_frames = group_split(
        frames, args.val_ratio, args.seed)
    mean, std = fit_normalizer(train_frames)
    normalize_frames(train_frames, mean, std)
    normalize_frames(val_frames, mean, std)

    device = torch.device(args.device)
    model = CoalitionRanker(
        int(train_frames[0]['features'].shape[1]),
        args.hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    history = []
    best_metric = -float('inf')
    best_epoch = -1
    stale = 0
    checkpoint_path = os.path.join(args.output_dir, 'best_selector.pth')

    print('Frames: train=%d val=%d input_dim=%d' % (
        len(train_frames), len(val_frames),
        train_frames[0]['features'].shape[1]))
    for epoch in range(args.epochs):
        losses = train_epoch(
            model, train_frames, optimizer, args.batch_size,
            device, args, rng)
        val_scores = predict_scores(model, val_frames, device)
        safety_margin, val_metrics = calibrate_safety_margin(
            val_frames, val_scores, args.safety_tolerance)
        train_metrics = evaluate_from_scores(
            train_frames, predict_scores(model, train_frames, device),
            safety_margin)
        record = {
            'epoch': epoch + 1,
            'loss': losses[0],
            'regression_loss': losses[1],
            'classification_loss': losses[2],
            'ranking_loss': losses[3],
            'train': train_metrics,
            'val': val_metrics
        }
        history.append(record)
        print('epoch %03d loss %.4f train_acc %.3f val_acc %.3f '
              'margin %.4f val_gain %+.4f recovery %.3f harm %.3f' % (
                  epoch + 1, losses[0], train_metrics['top1_accuracy'],
                  val_metrics['top1_accuracy'], safety_margin,
                  val_metrics['frame_ap_gain_vs_all'],
                  val_metrics['headroom_recovery_ratio'],
                  val_metrics['harmful_switch_ratio']))

        metric = val_metrics['mean_selected_frame_ap']
        if metric > best_metric + 1.0e-8:
            best_metric = metric
            best_epoch = epoch + 1
            stale = 0
            torch.save({
                'schema_version': 'coalition-ranker-v2',
                'selection_strategy': 'regret_with_all_fallback',
                'model_state_dict': model.state_dict(),
                'input_dim': int(train_frames[0]['features'].shape[1]),
                'hidden_dims': args.hidden_dims,
                'dropout': args.dropout,
                'feature_schema': schema,
                'normalizer_mean': mean,
                'normalizer_std': std,
                'seed': args.seed,
                'best_epoch': best_epoch,
                'safety_margin': safety_margin,
                'validation': val_metrics
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping at epoch %d' % (epoch + 1))
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    safety_margin = float(checkpoint['safety_margin'])
    final_train = evaluate_from_scores(
        train_frames, predict_scores(model, train_frames, device),
        safety_margin)
    final_val = evaluate_from_scores(
        val_frames, predict_scores(model, val_frames, device), safety_margin)
    summary = {
        'schema_version': 'coalition-ranker-v2',
        'input_jsonl': [os.path.abspath(x) for x in args.input_jsonl],
        'frames': len(frames),
        'train_frames': len(train_frames),
        'val_frames': len(val_frames),
        'group_split_field': args.group_field,
        'domains': dict(Counter(item['domain'] for item in frames)),
        'best_epoch': best_epoch,
        'safety_margin': safety_margin,
        'train': final_train,
        'val': final_val,
        'checkpoint': os.path.abspath(checkpoint_path),
        'notes': {
            'mean_selected_frame_ap':
                'Mean per-frame AP proxy, not dataset-level global AP.',
            'ground_truth_usage':
                'GT is used for training labels only; input features do not use GT.'
        }
    }
    with open(os.path.join(args.output_dir, 'training_history.json'),
              'w', encoding='utf-8') as stream:
        json.dump(history, stream, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'selector_summary.json'),
              'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print('\nTraining finished')
    print('Best epoch: %d' % best_epoch)
    print('Validation: %s' % json.dumps(final_val))
    print('Checkpoint: %s' % checkpoint_path)


if __name__ == '__main__':
    main()
