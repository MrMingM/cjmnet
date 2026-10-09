"""Exactly one affine layer and the old pairwise logistic training recipe."""
from collections import Counter
from pathlib import Path
import random
import numpy as np
from .common import (KEEP, WEATHERS, Manifest, atomic_json, atomic_torch, base_frame,
                     contract, evidence_path, load_cache, read_json, report)
from .features import GROUP_BLOCKS, schema, check_names, combine
from local_fusion_action_utility_audit.counterfactual import action_keys, pairwise_preferences
from local_fusion_action_utility_audit.linear_ranker import selection, calibrate, Metrics, verdict, baseline_scores


def groups(run, split, weather, task, group, scenes=None):
    protocol = read_json(Path(run)/'protocol.json')
    manifest = Manifest(run)
    for index in protocol['base_protocol'][split+'_indices']:
        old = base_frame(run, split, weather, index)
        if scenes is not None and old['scene'] not in scenes:
            continue
        if not manifest.complete('evidence-frame', split, weather, index):
            raise ValueError('Evidence frame incomplete')
        evidence = load_cache(evidence_path(run, split, weather, index))
        if evidence['identity'] != protocol['identity'] or evidence['proposal_ids'] != old['proposals']['anchor_ids'] or evidence['source_names'] != old['source_names']:
            raise ValueError('Evidence/source/candidate identity mismatch')
        from local_fusion_action_utility_audit.reproducibility import tensor_tree_hash
        if evidence['original_labels_hash'] != tensor_tree_hash(old['labels']):
            raise ValueError('Original S0 label hash differs from evidence cache')
        matrices = {k: v.numpy() for k, v in evidence['blocks'].items()}
        for label in old['labels']:
            p = label['proposal_position']
            x = combine(old['features'][task][p].numpy(), {k: v[p] for k, v in matrices.items()}, group)
            import torch
            yield {'x': torch.from_numpy(x), 'old_x': old['features'][task][p],
                   'outcomes': label['outcomes'][task], 'names': old['source_names'],
                   'frame': index, 'scene': old['scene'], 'weather': weather,
                   'proposal': p, 'background': label['background']}


class Probe:
    def __init__(self, state):
        import torch
        self.state = state
        check_names(state['feature_names'], state['task'])
        self.model = torch.nn.Linear(len(state['feature_names']), 1)
        self.model.load_state_dict(state['linear'], strict=True)
        self.model.requires_grad_(False).eval()
        self.mean = torch.tensor(state['mean'], dtype=torch.float32)
        self.std = torch.tensor(state['std'], dtype=torch.float32)

    def scores(self, x):
        import torch
        with torch.no_grad():
            return self.model((x.cpu().float()-self.mean)/self.std).squeeze(-1).numpy()


def filename(task, group):
    if group == 'all_evidence':
        return 'linear_sem.pt' if task == 'cls' else 'linear_geo.pt'
    return f'linear_{task}_{group}.pt'


def load_probe(run, task, group='all_evidence'):
    protocol = read_json(Path(run)/'protocol.json')
    if group == 'output_only':
        from local_fusion_action_utility_audit.linear_ranker import load_probe as old_load
        return old_load(protocol['base']['base_run'], task)
    state = load_cache(Path(run)/filename(task, group))
    if state['identity'] != protocol['identity'] or state['task'] != task or state['group'] != group or state['feature_names'] != schema(task, group):
        raise ValueError('S4 model identity/schema mismatch')
    if state['fit_scenes'] != protocol['base_protocol']['probe_fit_scenes'] or state['calibration_scenes'] != protocol['base_protocol']['probe_calibration_scenes']:
        raise ValueError('S4 model split mismatch')
    return Probe(state)


def fit(run, task):
    import torch
    torch.set_num_threads(1)
    protocol = contract(run)
    base_protocol, manifest = protocol['base_protocol'], Manifest(run)
    spec = base_protocol['config']
    normalizations = {}
    for group in GROUP_BLOCKS:
        path = Path(run)/filename(task, group)
        if group == 'output_only':
            original = load_probe(run, task, group)
            normalizations[group] = {'mean': original.state['mean'], 'std': original.state['std'],
                                     'feature_names': schema(task, group), 'fit_scenes': base_protocol['probe_fit_scenes'],
                                     'model': 'exact saved S1/S2 model; no refit'}
            continue
        if manifest.complete('probe-fit', task, group):
            saved = load_probe(run, task, group).state
            normalizations[group] = {k: saved[k] for k in ('mean', 'std', 'feature_names', 'fit_scenes')}
            continue
        with manifest.work('probe-fit', task, group):
            training, total, square, count, ties = [], None, None, 0, 0
            for weather in WEATHERS:
                for row in groups(run, 'train', weather, task, group, base_protocol['probe_fit_scenes']):
                    x = row['x'].double()
                    total = x.sum(0) if total is None else total+x.sum(0)
                    square = x.square().sum(0) if square is None else square+x.square().sum(0)
                    count += len(x)
                    pairs = pairwise_preferences(row['outcomes'], row['names'])
                    ties += len(x)*(len(x)-1)//2-len(pairs)
                    if pairs:
                        training.append((row['x'].float(), torch.tensor(pairs, dtype=torch.long)))
            if not training or not count:
                raise RuntimeError('No non-tied fit pairs')
            mean = (total/count).float()
            std = (square/count-(total/count).square()).clamp_min(0).sqrt().float().clamp_min(1e-6)
            torch.manual_seed(spec['seed'])
            model = torch.nn.Linear(len(mean), 1)
            torch.nn.init.zeros_(model.weight)
            torch.nn.init.zeros_(model.bias)
            optimizer = torch.optim.AdamW(model.parameters(), lr=spec['probe_learning_rate'], weight_decay=spec['probe_weight_decay'])
            start, history = 0, []
            latest = next((epoch for epoch in range(spec['probe_epochs'], 0, -1)
                           if manifest.complete('probe-epoch', task, group, epoch)), None)
            if latest is not None:
                partial = Path(run)/'cache'/'probes'/f'{task}_{group}'/f'epoch_{latest:03d}.pt'
                state = load_cache(partial)
                if state['identity'] != protocol['identity'] or state['group'] != group or state['task'] != task:
                    raise ValueError('Probe epoch resume contract mismatch')
                model.load_state_dict(state['linear'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, history = state['epoch'], state['history']
            training = [((x-mean)/std, pairs) for x, pairs in training]
            for epoch in range(start, spec['probe_epochs']):
                order = list(range(len(training)))
                random.Random(spec['seed']+epoch).shuffle(order)
                buffer, buffered, loss_sum, pair_count = [], 0, 0., 0

                def step(parts):
                    delta = torch.cat(parts)
                    optimizer.zero_grad(set_to_none=True)
                    difference = model(delta).squeeze(-1)-model(torch.zeros_like(delta)).squeeze(-1)
                    loss = torch.nn.functional.softplus(-difference).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite pairwise loss')
                    loss.backward()
                    optimizer.step()
                    return float(loss.detach())*len(delta), len(delta)

                for index in order:
                    x, pairs = training[index]
                    delta = x[pairs[:, 0]]-x[pairs[:, 1]]
                    buffer.append(delta)
                    buffered += len(delta)
                    if buffered >= spec['pair_batch_size']:
                        loss, n = step(buffer)
                        loss_sum += loss
                        pair_count += n
                        buffer, buffered = [], 0
                if buffer:
                    loss, n = step(buffer)
                    loss_sum += loss
                    pair_count += n
                history.append({'epoch': epoch+1, 'loss': loss_sum/pair_count, 'pairs': pair_count})
                partial = Path(run)/'cache'/'probes'/f'{task}_{group}'/f'epoch_{epoch+1:03d}.pt'
                atomic_torch(partial, {'identity': protocol['identity'], 'task': task, 'group': group,
                                       'epoch': epoch+1, 'history': history, 'linear': model.state_dict(), 'optimizer': optimizer.state_dict()})
                manifest.mark('probe-epoch', 'complete', task, group, epoch+1, [partial])
                print(f'S4 {task}/{group} epoch={epoch+1} loss={loss_sum/pair_count:.6f}', flush=True)
            state = {'identity': protocol['identity'], 'task': task, 'group': group,
                     'linear': model.state_dict(), 'mean': mean.tolist(), 'std': std.tolist(),
                     'feature_names': schema(task, group), 'fit_scenes': base_protocol['probe_fit_scenes'],
                     'calibration_scenes': base_protocol['probe_calibration_scenes'], 'history': history,
                     'fit_feature_rows': count, 'ignored_ties': ties, 'checkpoint_selection': 'fixed last epoch'}
            calibration = (row for weather in WEATHERS for row in groups(run, 'train', weather, task, group, base_protocol['probe_calibration_scenes']))
            state['calibration'] = calibrate(Probe(state), calibration, spec)
            atomic_torch(path, state)
            normalizations[group] = {k: state[k] for k in ('mean', 'std', 'feature_names', 'fit_scenes')}
        manifest.mark('probe-fit', 'complete', task, group, artifacts=[path])
    atomic_json(Path(run)/('normalization_sem.json' if task == 'cls' else 'normalization_geo.json'), normalizations)


class Diagnostics:
    def __init__(self):
        self.metrics = Metrics()
        self.predicted, self.oracle = Counter(), Counter()
        self.predicted_sources, self.oracle_sources = Counter(), Counter()
        self.changed_proposals, self.proposals = set(), set()
        self.n = 0

    def add(self, row, scores, threshold=0.):
        self.metrics.add(row, scores, threshold)
        selected = selection(scores, row['names'], threshold)[0]
        keys = action_keys(row['outcomes'], row['names'])
        best = max(range(len(keys)), key=lambda i: keys[i])
        def family(name):
            return 'KEEP' if name == KEEP else name.split(':')[0]
        self.predicted[family(row['names'][selected])] += 1
        self.oracle[family(row['names'][best])] += 1
        self.predicted_sources[row['names'][selected]] += 1
        self.oracle_sources[row['names'][best]] += 1
        self.proposals.add((row['frame'], row['proposal']))
        if selected:
            self.changed_proposals.add((row['frame'], row['proposal']))
        self.n += 1

    def result(self):
        result = self.metrics.result()
        n = max(1, self.n)
        result.update(action_rate=1-self.predicted['KEEP']/n, KEEP_ratio=self.predicted['KEEP']/n,
                      predicted_action_diversity=dict(self.predicted), oracle_action_diversity=dict(self.oracle),
                      predicted_best_source_diversity=dict(self.predicted_sources), oracle_best_source_diversity=dict(self.oracle_sources),
                      predicted_changed_count=self.n-self.predicted['KEEP'],
                      unique_proposal_changed_count=len(self.changed_proposals), unique_proposals=len(self.proposals),
                      oracle_modify_count=result['counts'].get('true_modify', 0))
        return result


def evidence_verdict(conditions, output, spec, task, s4_spec):
    decision = verdict(conditions, spec, task)
    adverse = ('fog', 'rain', 'snow')
    accuracy = [conditions[w]['linear']['pairwise_ranking_accuracy'] for w in adverse]
    old_accuracy = [output[w]['linear']['pairwise_ranking_accuracy'] for w in adverse]
    gain = float(np.mean(accuracy)-np.mean(old_accuracy)) if all(v is not None for v in accuracy+old_accuracy) else None
    old_regret = float(np.mean([output[w]['linear']['mean_action_regret'] for w in adverse]))
    new_regret = float(np.mean([conditions[w]['linear']['mean_action_regret'] for w in adverse]))
    reduction = (old_regret-new_regret)/old_regret if old_regret > 0 else 0.
    checks = {**decision['checks'], 'pairwise_gain_vs_output': gain is not None and gain >= s4_spec['evidence_pairwise_gain_gate'],
              'regret_reduction_vs_output': reduction >= s4_spec['evidence_regret_reduction_gate']}
    return {**decision, 'checks': {k: bool(v) for k, v in checks.items()},
            'learnability_verdict': decision['verdict'],
            'verdict': ('SEM' if task == 'cls' else 'GEO')+('_EVIDENCE_FOUND' if all(checks.values()) else '_EVIDENCE_WEAK'),
            'pairwise_gain_vs_output': gain, 'regret_reduction_vs_output': reduction}


def attribution(probe):
    weights = np.abs(probe.model.weight.detach().cpu().numpy()[0])
    families = Counter()
    for name, weight in zip(probe.state['feature_names'], weights):
        family = next((key for key in ('geometry', 'gspr', 'semantic', 'loo', 'singlequery', 'competition') if name.startswith(key+'_')), 'output')
        families[family] += float(weight)
    return {'sum_absolute_standardized_weights': {k: families[k] for k in ('output', 'geometry', 'gspr', 'semantic', 'loo', 'singlequery', 'competition')},
            'interpretation': 'Descriptive weights on fit-standardized columns; correlated features and feature counts prevent causal attribution.'}


def evaluate(run, task):
    protocol = contract(run)
    spec = protocol['base_protocol']['config']
    report_groups = {}
    for group in GROUP_BLOCKS:
        probe = load_probe(run, task, group)
        conditions, strata_out = {}, {}
        for weather in WEATHERS:
            acc, strata = {}, {'background': {}, 'associated': {}}
            for row in groups(run, 'validation', weather, task, group):
                scores = baseline_scores({**row, 'x': row['old_x']}, task)
                scores['linear'] = probe.scores(row['x'])
                scores['linear_conservative'] = scores['linear']
                for method, values in scores.items():
                    threshold = probe.state['calibration']['threshold'] if method == 'linear_conservative' else 0.
                    acc.setdefault(method, Diagnostics()).add(row, values, threshold)
                    strata['background' if row['background'] else 'associated'].setdefault(method, Diagnostics()).add(row, values, threshold)
            conditions[weather] = {k: v.result() for k, v in acc.items()}
            strata_out[weather] = {k: {m: v.result() for m, v in rows.items()} for k, rows in strata.items()}
        if group == 'output_only':
            old_stage = 'S1' if task == 'cls' else 'S2'
            old = read_json(Path(protocol['base']['base_run'])/(old_stage+'_RESULTS.json'))['variants']['with_competition_features']['conditions']
            from .r0 import compare
            for weather in WEATHERS:
                for method, row in old[weather].items():
                    compare(row, {k: conditions[weather][method][k] for k in row})
        report_groups[group] = {'conditions': conditions, 'strata': strata_out,
                                'calibration': probe.state['calibration'], 'attribution': attribution(probe)}
        print(f'S4 evaluation {task}/{group} complete', flush=True)
    for group, value in report_groups.items():
        value['decision'] = evidence_verdict(value['conditions'], report_groups['output_only']['conditions'], spec, task, protocol['s4_config'])
    value = {'task': task, 'primary_group': 'all_evidence', 'groups': report_groups,
             'decision': report_groups['all_evidence']['decision'], 'validation_selection': False}
    name = 'S4_SEM_RESULTS' if task == 'cls' else 'S4_GEO_RESULTS'
    md = f'# {name}\n\n主结果固定 all_evidence：**{value["decision"]["verdict"]}**。\n\n'
    md += '| Group | Weather | Method | Pairwise | Top1 | Top3 | KEEP/MODIFY | Beneficial precision | Regret | Action rate | Changed | recovered/lost/newFP |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n'
    def fmt(x):
        return 'NA' if x is None else f'{x:.4f}'
    for group, result in report_groups.items():
        for weather, methods in result['conditions'].items():
            for method, row in methods.items():
                md += '| '+ ' | '.join([group, weather, method]+[fmt(row[k]) for k in ('pairwise_ranking_accuracy', 'top1_best_action_accuracy', 'top3_best_action_recall', 'KEEP_MODIFY_accuracy', 'beneficial_action_precision', 'mean_action_regret', 'action_rate')]+[str(row['predicted_changed_count']), f"{row['selected_recovered']}/{row['selected_lost']}/{row['selected_new_fp']}"])+ ' |\n'
    md += '\nJSON 包含 MODIFY precision/recall、背景/关联分层、oracle/预测来源族分布、全部校准试验和标准化权重归因。权重只供描述，不能解释为因果贡献。所有标签均来自原 S0，独立动作计数不能当作整帧收益。\n'
    report(run, name, value, md)


if __name__ == '__main__':
    from .common import parser
    cli = parser(__doc__)
    cli.add_argument('--task', choices=('cls', 'reg'), required=True)
    cli.add_argument('--mode', choices=('fit', 'evaluate'), required=True)
    args = cli.parse_args()
    (fit if args.mode == 'fit' else evaluate)(args.run, args.task)
