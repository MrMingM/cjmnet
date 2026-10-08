"""Single Linear adapter and fair detection-outcome calibration."""
from pathlib import Path
import random
import numpy as np
from .common import (GROUPS, FEATURES, WEATHERS, Manifest, atomic_json, atomic_torch,
                     cli, load_cache, protocol)
from .inputs import control_source
from .labels import detection_key, detection_pairs, original_pairs
from local_fusion_action_utility_audit.linear_ranker import LinearProbe, selection


def update_count(pair_counts, batch_size):
    updates = buffered = 0
    for count in pair_counts:
        buffered += count
        if buffered >= batch_size:
            updates += 1
            buffered = 0
    return updates + int(buffered > 0)


def all_groups(source, task, scenes):
    for weather in WEATHERS:
        yield from source.groups('train', weather, task, scenes)


def state_path(run, group, task):
    return Path(run) / 'probes' / group / f'linear_{task}.pt'


def fit_new(training, inherited, spec, run, task, identity, manifest):
    """Same zero init/AdamW/shuffle/group-buffer rule as source fit; new pairs."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(spec['seed'])
    model = torch.nn.Linear(len(inherited['feature_names']), 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['probe_learning_rate'],
                                 weight_decay=spec['probe_weight_decay'])
    epoch_start, history = 0, []
    for epoch in range(1, spec['probe_epochs'] + 1):
        if manifest.complete('fit-epoch', frame=epoch, weather=task):
            partial = load_cache(Path(run) / 'cache' / 'probes' / task / f'epoch_{epoch:03d}.pt')
            if partial['identity'] != identity:
                raise ValueError('Probe epoch identity differs')
            model.load_state_dict(partial['linear'])
            optimizer.load_state_dict(partial['optimizer'])
            epoch_start, history = epoch, partial['history']
    fallback = 'KEEP_ALL_NO_TRAINABLE_PREFERENCES' if not training else None
    for epoch in range(epoch_start, spec['probe_epochs'] if training else 0):
        order = list(range(len(training)))
        random.Random(spec['seed'] + epoch).shuffle(order)
        buffer, buffered, loss_sum, pair_count, updates = [], 0, 0., 0, 0

        def step(parts):
            delta = torch.cat(parts)
            optimizer.zero_grad(set_to_none=True)
            difference = model(delta).squeeze(-1) - model(torch.zeros_like(delta)).squeeze(-1)
            loss = torch.nn.functional.softplus(-difference).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite pairwise loss')
            loss.backward()
            optimizer.step()
            return float(loss.detach()) * len(delta), len(delta)

        for index in order:
            x, pairs = training[index]
            delta = x[pairs[:, 0]] - x[pairs[:, 1]]
            buffer.append(delta)
            buffered += len(delta)
            if buffered >= spec['pair_batch_size']:
                loss, count = step(buffer)
                loss_sum += loss
                pair_count += count
                updates += 1
                buffer, buffered = [], 0
        if buffer:
            loss, count = step(buffer)
            loss_sum += loss
            pair_count += count
            updates += 1
        history.append({'epoch': epoch + 1, 'pairs': pair_count, 'optimizer_updates': updates,
                        'loss': loss_sum / pair_count})
        path = Path(run) / 'cache' / 'probes' / task / f'epoch_{epoch+1:03d}.pt'
        atomic_torch(path, {'identity': identity, 'linear': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'history': history})
        manifest.mark('fit-epoch', 'complete', weather=task, frame=epoch + 1, artifacts=[path])
        print(f'DetectionOutcomeLabel {task} epoch={epoch+1} pairs={pair_count} updates={updates} loss={loss_sum/pair_count:.6f}', flush=True)
    return {'linear': model.state_dict(), 'history': history, 'fallback': fallback,
            'trainable_groups': len(training), 'checkpoint_selection': 'fixed last source-protocol epoch'}


def fit(run):
    import torch
    p = protocol(run)
    source = control_source(run, p)
    manifest = Manifest(run)
    for task in ('cls', 'reg'):
        if manifest.complete('fit-task', weather=task):
            continue
        with manifest.work('fit-task', weather=task):
            inherited = source.original_state(task)
            original = {**inherited, 'control_identity': p['identity'], 'label_group': GROUPS[0],
                        'historical_calibration': inherited['calibration'], 'reused_original_weights': True}
            mean = torch.tensor(inherited['mean'], dtype=torch.float32)
            std = torch.tensor(inherited['std'], dtype=torch.float32)
            training, old_pair_counts = [], []
            for group in all_groups(source, task, source.protocol['probe_fit_scenes']):
                old_pairs = original_pairs(group['outcomes'], group['names'])
                if old_pairs:
                    old_pair_counts.append(len(old_pairs))
                pairs = detection_pairs(group['outcomes'], group['names'])
                if pairs:
                    training.append(((group['x'].float() - mean) / std,
                                     torch.tensor(pairs, dtype=torch.long)))
            history = []
            for row in inherited['history']:
                if row['pairs'] != sum(old_pair_counts):
                    raise ValueError('Original fit pair count cannot be reproduced from source cache')
                order = list(range(len(old_pair_counts)))
                random.Random(source.spec['seed'] + row['epoch'] - 1).shuffle(order)
                history.append({**row, 'optimizer_updates': update_count(
                    [old_pair_counts[i] for i in order], source.spec['pair_batch_size']),
                    'updates_provenance': 'reconstructed from verified original pair groups and original batch rule'})
            original['history'] = history
            new = {k: inherited[k] for k in ('task', 'variant', 'mean', 'std', 'feature_names', 'fit_scenes', 'calibration_scenes')}
            new.update(fit_new(training, inherited, source.spec, run, task, p['identity'], manifest))
            new.update({'identity': p['identity'], 'control_identity': p['identity'],
                        'source_identity': source.protocol['identity'], 'label_group': GROUPS[1],
                        'initialization': 'torch.manual_seed(source.seed), Linear, zero weight and bias; no pretrained probe weights',
                        'reused_original_weights': False})
            for group, state in ((GROUPS[0], original), (GROUPS[1], new)):
                # Initial placeholder; all Conservative thresholds are replaced by calibration.
                state['calibration'] = {'threshold': None, 'fallback': 'NOT_CALIBRATED'}
                atomic_torch(state_path(run, group, task), state)
        manifest.mark('fit-task', 'complete', weather=task,
                      artifacts=[state_path(run, g, task) for g in GROUPS])


def calibrate_common(probe, groups, spec):
    rows = []
    for group in groups:
        chosen, margin, _ = selection(probe.scores(group['x']), group['names'])
        if chosen:
            o = group['outcomes'][chosen]
            rows.append({'margin': margin, 'beneficial': detection_key(o) > detection_key(group['outcomes'][0]),
                         'lost': o['lost'], 'new_fp': o['new_fp']})
    thresholds = sorted(set([0.] + [float(np.quantile([r['margin'] for r in rows], q))
                                   for q in spec['calibration_quantiles']])) if rows else [0.]
    trials = []
    for threshold in thresholds:
        selected = [r for r in rows if r['margin'] > threshold]
        n = len(selected)
        precision = sum(r['beneficial'] for r in selected) / max(1, n)
        lost = sum(r['lost'] for r in selected) / max(1, n)
        fp = sum(r['new_fp'] for r in selected) / max(1, n)
        feasible = (n >= spec['calibration_min_actions'] and precision >= spec['calibration_min_modify_precision']
                    and lost <= spec['calibration_max_lost_per_action'] and fp <= spec['calibration_max_new_fp_per_action'])
        trials.append({'threshold': threshold, 'actions': n, 'precision': precision,
                       'lost_per_action': lost, 'new_fp_per_action': fp, 'feasible': feasible})
    feasible = [r for r in trials if r['feasible']]
    threshold = min(r['threshold'] for r in feasible) if feasible else None
    return {'threshold': threshold, 'fallback': 'KEEP_ALL' if threshold is None else None,
            'beneficial_definition': 'detection_key(action) > detection_key(KEEP)', 'trials': trials,
            'selection': 'smallest feasible train-calibration margin; source quantiles and constraints unchanged'}


def load_probe(run, group, task, calibrated=True):
    p = protocol(run)
    state = load_cache(state_path(run, group, task))
    if state['control_identity'] != p['identity'] or state['label_group'] != group or state['task'] != task:
        raise ValueError('Control probe identity mismatch')
    if calibrated:
        from .common import read_json
        calibration = read_json(Path(run) / 'common_calibration.json')
        if calibration['identity'] != p['identity']:
            raise ValueError('Calibration identity mismatch')
        state = {**state, 'calibration': calibration['groups'][group][task]}
    return LinearProbe(state)


def calibrate(run):
    p = protocol(run)
    source = control_source(run, p)
    results = {}
    for group in GROUPS:
        results[group] = {}
        for task in ('cls', 'reg'):
            probe = load_probe(run, group, task, False)
            results[group][task] = calibrate_common(probe, all_groups(source, task,
                source.protocol['probe_calibration_scenes']), source.spec)
            print(f'CALIBRATION {group}/{task} threshold={results[group][task]["threshold"]}', flush=True)
    atomic_json(Path(run) / 'common_calibration.json', {'identity': p['identity'], 'groups': results,
        'scenes': source.protocol['probe_calibration_scenes'], 'validation_used': False})


if __name__ == '__main__':
    parser = cli(__doc__)
    parser.add_argument('--mode', choices=('fit', 'calibrate'), required=True)
    args = parser.parse_args()
    (fit if args.mode == 'fit' else calibrate)(args.run)
