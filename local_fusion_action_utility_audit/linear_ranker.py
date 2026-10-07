"""One affine layer per task/ablation; scene-isolated normalization/calibration."""
from collections import Counter
import random
from pathlib import Path
import numpy as np
from .common import (KEEP, VARIANTS, WEATHERS, Manifest, atomic_json, atomic_torch,
                     cache_paths, load_cache, read_json, write_report, verify_protocol_snapshot)
from .counterfactual import action_keys, outcome_ranks, pairwise_preferences
from .features import columns, feature_leakage_check, schema


def groups(run, split, weather, task, variant, scenes=None):
    col = columns(task, variant)
    for path in cache_paths(run, split, weather):
        frame = load_cache(path)
        if scenes is not None and frame['scene'] not in scenes:
            continue
        for label in frame['labels']:
            yield {'x': frame['features'][task][label['proposal_position']][:, col],
                   'outcomes': label['outcomes'][task], 'names': frame['source_names'],
                   'frame': frame['frame'], 'scene': frame['scene'], 'weather': weather,
                   'proposal': label['proposal_position'], 'background': label['background']}


def selection(scores, names, threshold=0.):
    """KEEP wins score ties; threshold is never derived from evaluation labels."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(names) or not np.isfinite(scores).all():
        raise ValueError('Invalid source scores')
    keep = names.index(KEEP)
    order = sorted(range(len(names)), key=lambda i: (-scores[i], names[i] != KEEP, names[i]))
    best = order[0]
    margin = float(scores[best] - scores[keep])
    if threshold is None or best == keep or margin <= threshold:
        return keep, margin, order
    return best, margin, order


class LinearProbe:
    def __init__(self, state):
        import torch
        self.state = state
        feature_leakage_check(state['feature_names'])
        self.model = torch.nn.Linear(len(state['feature_names']), 1)
        self.model.load_state_dict(state['linear'], strict=True)
        self.model.requires_grad_(False).eval()
        self.mean = torch.tensor(state['mean'], dtype=torch.float32)
        self.std = torch.tensor(state['std'], dtype=torch.float32)

    def scores(self, x):
        import torch
        with torch.no_grad():
            return self.model((x.cpu().float() - self.mean) / self.std).squeeze(-1).numpy()


def load_probe(run, task, variant=None):
    protocol = read_json(Path(run) / 'protocol.json')
    variant = variant or protocol['config']['primary_features']
    filename = f'linear_{task}.pt' if variant == protocol['config']['primary_features'] else f'linear_{task}_without_competition.pt'
    state = load_cache(Path(run) / filename)
    if state['identity'] != Manifest(run).value['identity'] or state['task'] != task or state['variant'] != variant:
        raise ValueError('Probe protocol/task/ablation mismatch')
    if state['feature_names'] != [schema(task)[i] for i in columns(task, variant)]:
        raise ValueError('Probe feature schema mismatch')
    if state['fit_scenes'] != protocol['probe_fit_scenes'] or state['calibration_scenes'] != protocol['probe_calibration_scenes']:
        raise ValueError('Probe scene protocol mismatch')
    return LinearProbe(state)


def baseline_scores(group, task):
    names = [schema(task)[i] for i in columns(task, 'with_competition_features')]
    # Baselines consume the full inference-only matrix, independent of the ablation.
    x = group['x'].numpy()
    keep = np.array([1. if name == KEEP else 0. for name in group['names']])
    if task == 'cls':
        return {'Always KEEP': keep,
                'Highest local classification score': x[:, names.index('roi_max')],
                'Largest positive delta vs Shared': x[:, names.index('roi_delta_shared')]}
    return {'Always KEEP': keep,
            'Closest-to-Shared geometry': -x[:, names.index('center_distance')]
                - np.linalg.norm(x[:, [names.index('delta_' + k) for k in ('length', 'width', 'height')]], axis=1)
                - (1 - x[:, names.index('delta_yaw_cos')]),
            'Closest-to-source-consensus geometry': -x[:, names.index('center_to_mean')]
                - x[:, names.index('size_to_mean')] - x[:, names.index('yaw_to_consensus')]}


def calibrate(probe, calibration, spec):
    rows = []
    for group in calibration:
        selected, margin, _ = selection(probe.scores(group['x']), group['names'])
        if selected:
            keys = action_keys(group['outcomes'], group['names'])
            outcome = group['outcomes'][selected]
            rows.append({'margin': margin, 'beneficial': keys[selected] > keys[0],
                         'lost': outcome['lost'], 'new_fp': outcome['new_fp']})
    candidates = sorted(set([0.] + [float(np.quantile([r['margin'] for r in rows], q))
                                   for q in spec['calibration_quantiles']])) if rows else [0.]
    trials = []
    for threshold in candidates:
        selected = [r for r in rows if r['margin'] > threshold]
        count = len(selected)
        precision = sum(r['beneficial'] for r in selected) / max(1, count)
        lost = sum(r['lost'] for r in selected) / max(1, count)
        fp = sum(r['new_fp'] for r in selected) / max(1, count)
        feasible = (count >= spec['calibration_min_actions']
                    and precision >= spec['calibration_min_modify_precision']
                    and lost <= spec['calibration_max_lost_per_action']
                    and fp <= spec['calibration_max_new_fp_per_action'])
        trials.append({'threshold': threshold, 'actions': count, 'precision': precision,
                       'lost_per_action': lost, 'new_fp_per_action': fp, 'feasible': feasible})
    feasible = [r for r in trials if r['feasible']]
    chosen = min(feasible, key=lambda r: r['threshold']) if feasible else None
    return {'threshold': chosen['threshold'] if chosen else None,
            'fallback': None if chosen else 'KEEP_ALL', 'trials': trials,
            'selection': 'smallest feasible train-calibration margin; fixed single-task consequences',
            'boundary': 'single-action calibration does not guarantee multi-action AP or safety'}


def fit(run, task):
    import torch
    torch.set_num_threads(1)
    protocol = verify_protocol_snapshot(run)
    spec = protocol['config']
    manifest = Manifest(run)
    normalization = read_json(Path(run) / 'feature_normalization.json') if (Path(run) / 'feature_normalization.json').exists() else {}
    for variant in VARIANTS:
        if manifest.complete(task + '-fit', weather=variant):
            continue
        with manifest.work(task + '-fit', weather=variant):
            training = []
            total = square = None
            count = 0
            ties = 0
            for weather in WEATHERS:
                for group in groups(run, 'train', weather, task, variant, protocol['probe_fit_scenes']):
                    x = group['x'].double()
                    total = x.sum(0) if total is None else total + x.sum(0)
                    square = x.square().sum(0) if square is None else square + x.square().sum(0)
                    count += len(x)
                    pairs = pairwise_preferences(group['outcomes'], group['names'])
                    ties += len(x) * (len(x) - 1) // 2 - len(pairs)
                    if pairs:
                        training.append((group['x'].float(), torch.tensor(pairs, dtype=torch.long)))
            if not training or not count:
                raise RuntimeError('No non-tied probe-fit pairs')
            mean = (total / count).float()
            std = (square / count - (total / count).square()).clamp_min(0).sqrt().float().clamp_min(1e-6)
            names = [schema(task)[i] for i in columns(task, variant)]
            feature_leakage_check(names)
            torch.manual_seed(spec['seed'])
            model = torch.nn.Linear(len(names), 1)
            torch.nn.init.zeros_(model.weight)
            torch.nn.init.zeros_(model.bias)
            optimizer = torch.optim.AdamW(model.parameters(), lr=spec['probe_learning_rate'],
                                         weight_decay=spec['probe_weight_decay'])
            partial = Path(run) / 'cache' / 'probes' / f'{task}_{variant}_last.pt'
            epoch_start, history = 0, []
            if partial.exists():
                state = load_cache(partial)
                if state['identity'] != manifest.value['identity']:
                    raise ValueError('Probe resume identity differs')
                model.load_state_dict(state['linear'])
                optimizer.load_state_dict(state['optimizer'])
                epoch_start, history = state['epoch'], state['history']
            # Store standardized candidate features; create differences one minibatch at a time.
            training = [((x - mean) / std, pairs) for x, pairs in training]
            for epoch in range(epoch_start, spec['probe_epochs']):
                order = list(range(len(training)))
                random.Random(spec['seed'] + epoch).shuffle(order)
                buffer, buffered, loss_sum, pair_count = [], 0, 0., 0

                def step(parts):
                    delta = torch.cat(parts)
                    optimizer.zero_grad(set_to_none=True)
                    # b cancels in score(A)-score(B); still exactly one affine layer.
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
                        loss, n = step(buffer)
                        loss_sum += loss
                        pair_count += n
                        buffer, buffered = [], 0
                if buffer:
                    loss, n = step(buffer)
                    loss_sum += loss
                    pair_count += n
                history.append({'epoch': epoch + 1, 'loss': loss_sum / pair_count, 'pairs': pair_count})
                atomic_torch(partial, {'identity': manifest.value['identity'], 'linear': model.state_dict(),
                                       'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'history': history})
                print(f'{task} {variant} epoch={epoch + 1} loss={loss_sum / pair_count:.6f}', flush=True)
            state = {'identity': manifest.value['identity'], 'task': task, 'variant': variant,
                     'linear': model.state_dict(), 'mean': mean.tolist(), 'std': std.tolist(),
                     'feature_names': names, 'fit_scenes': protocol['probe_fit_scenes'],
                     'calibration_scenes': protocol['probe_calibration_scenes'], 'history': history,
                     'fit_feature_rows': count, 'ignored_ties': ties, 'checkpoint_selection': 'fixed last epoch'}
            probe = LinearProbe(state)
            calibration = (group for weather in WEATHERS for group in
                           groups(run, 'train', weather, task, variant, protocol['probe_calibration_scenes']))
            state['calibration'] = calibrate(probe, calibration, spec)
            filename = f'linear_{task}.pt' if variant == spec['primary_features'] else f'linear_{task}_without_competition.pt'
            path = Path(run) / filename
            atomic_torch(path, state)
            normalization[task + '/' + variant] = {'mean': state['mean'], 'std': state['std'],
                                                   'feature_names': names, 'fit_scenes': state['fit_scenes']}
            atomic_json(Path(run) / 'feature_normalization.json', normalization)
        manifest.mark(task + '-fit', 'complete', weather=variant, artifacts=[path])


class Metrics:
    def __init__(self):
        self.counts = Counter()
        self.regret = 0.
        self.component_regret = Counter()

    def add(self, group, scores, threshold=0.):
        outcomes, names = group['outcomes'], group['names']
        selected, _, order = selection(scores, names, threshold)
        keys = action_keys(outcomes, names)
        best_key = max(keys)
        best = next(i for i in range(len(keys)) if keys[i] == best_key)
        pairs = pairwise_preferences(outcomes, names)
        c = self.counts
        c['candidates'] += 1
        c['pairs'] += len(pairs)
        c['pair_correct'] += sum(1 if scores[i] > scores[j] else .5 if scores[i] == scores[j] else 0 for i, j in pairs)
        c['ignored_ties'] += len(names) * (len(names) - 1) // 2 - len(pairs)
        c['top1'] += keys[selected] == best_key
        c['top3'] += any(keys[i] == best_key for i in order[:3])
        true_modify, predicted_modify = best_key > keys[0], selected != 0
        c['keep_modify_correct'] += true_modify == predicted_modify
        c['true_modify'] += true_modify
        c['predicted_modify'] += predicted_modify
        c['modify_tp'] += true_modify and predicted_modify
        c['beneficial_selected'] += predicted_modify and keys[selected] > keys[0]
        ranks = outcome_ranks(outcomes, names)
        self.regret += max(ranks) - ranks[selected]
        for key in ('recovered', 'lost', 'new_fp'):
            c['selected_' + key] += outcomes[selected][key]
        self.component_regret['tp_deficit'] += outcomes[best]['action_tp'] - outcomes[selected]['action_tp']
        self.component_regret['excess_lost'] += outcomes[selected]['lost'] - outcomes[best]['lost']
        self.component_regret['excess_new_fp'] += outcomes[selected]['new_fp'] - outcomes[best]['new_fp']

    def result(self):
        c = self.counts
        n = max(1, c['candidates'])
        return {'counts': dict(c), 'pairwise_ranking_accuracy': c['pair_correct'] / c['pairs'] if c['pairs'] else None,
                'top1_best_action_accuracy': c['top1'] / n, 'top3_best_action_recall': c['top3'] / n,
                'KEEP_MODIFY_accuracy': c['keep_modify_correct'] / n,
                'MODIFY_precision': c['modify_tp'] / c['predicted_modify'] if c['predicted_modify'] else None,
                'MODIFY_recall': c['modify_tp'] / c['true_modify'] if c['true_modify'] else None,
                'beneficial_action_precision': c['beneficial_selected'] / c['predicted_modify'] if c['predicted_modify'] else None,
                'mean_action_regret': self.regret / n,
                'mean_structured_regret': {k: v / n for k, v in self.component_regret.items()},
                **{k: c[k] for k in ('selected_recovered', 'selected_lost', 'selected_new_fp')}}


def verdict(conditions, spec, task):
    prefix = 's1' if task == 'cls' else 's2'
    weather = ('fog', 'rain', 'snow')
    baseline_names = [k for k in conditions['fog'] if k not in ('linear', 'linear_conservative')]
    # Strongest comparison is per-weather minimum regret among ALL simple baselines.
    baseline = {w: min(conditions[w][k]['mean_action_regret'] for k in baseline_names) for w in weather}
    learned = {w: conditions[w]['linear']['mean_action_regret'] for w in weather}
    accuracies = [conditions[w]['linear']['pairwise_ranking_accuracy'] for w in weather]
    mean_accuracy = sum(accuracies) / 3 if all(x is not None for x in accuracies) else None
    baseline_mean, learned_mean = np.mean(list(baseline.values())), np.mean(list(learned.values()))
    reduction = float((baseline_mean - learned_mean) / baseline_mean) if baseline_mean > 0 else 0.
    checks = {'pairwise_accuracy': mean_accuracy is not None and mean_accuracy >= spec[prefix + '_pairwise_accuracy_gate'],
              'regret_reduction': reduction >= spec[prefix + '_regret_reduction_gate'],
              'two_weathers_better': sum(learned[w] < baseline[w] for w in weather) >= 2}
    return {'verdict': ('S1' if task == 'cls' else 'S2') + ('_LEARNABLE' if all(checks.values()) else '_WEAK'),
            'checks': checks, 'adverse_mean_pairwise_accuracy': mean_accuracy,
            'regret_reduction_vs_strongest_baseline': reduction,
            'strongest_baseline_regret_by_weather': baseline,
            'baseline_comparison': 'per-weather best simple baseline; reporting only, no model selection'}


def evaluate(run, task):
    protocol = verify_protocol_snapshot(run)
    spec = protocol['config']
    report = {'task': task, 'variants': {}, 'regret_definition': spec['regret_definition']}
    for variant in VARIANTS:
        probe = load_probe(run, task, variant)
        conditions = {}
        strata_conditions = {}
        for weather in WEATHERS:
            accumulators, strata = {}, {'background': {}, 'associated': {}}
            col = columns(task, variant)
            for group in groups(run, 'validation', weather, task, 'with_competition_features'):
                scores = baseline_scores(group, task)
                scores['linear'] = probe.scores(group['x'][:, col])
                scores['linear_conservative'] = scores['linear']
                for method, score in scores.items():
                    threshold = probe.state['calibration']['threshold'] if method == 'linear_conservative' else 0.
                    accumulators.setdefault(method, Metrics()).add(group, score, threshold)
                    stratum = strata['background' if group['background'] else 'associated']
                    stratum.setdefault(method, Metrics()).add(group, score, threshold)
            conditions[weather] = {k: m.result() for k, m in accumulators.items()}
            if not conditions[weather]:
                raise RuntimeError('No validation label groups: ' + weather)
            strata_conditions[weather] = {s: {k: m.result() for k, m in methods.items()} for s, methods in strata.items()}
        report['variants'][variant] = {'conditions': conditions, 'strata': strata_conditions,
                                      'decision': verdict(conditions, spec, task), 'calibration': probe.state['calibration']}
    report['decision'] = report['variants'][spec['primary_features']]['decision']
    report['competition_feature_effect'] = {weather: {
        'pairwise_accuracy_difference': (report['variants'][VARIANTS[1]]['conditions'][weather]['linear']['pairwise_ranking_accuracy'] or 0)
                                      - (report['variants'][VARIANTS[0]]['conditions'][weather]['linear']['pairwise_ranking_accuracy'] or 0),
        'regret_difference': report['variants'][VARIANTS[1]]['conditions'][weather]['linear']['mean_action_regret']
                            - report['variants'][VARIANTS[0]]['conditions'][weather]['linear']['mean_action_regret']}
        for weather in WEATHERS}
    stage = 'S1' if task == 'cls' else 'S2'
    markdown = f'# {stage}：{"分类" if task == "cls" else "回归"}来源能否学会\n\n判定：**{report["decision"]["verdict"]}**。主消融在运行前固定为 {spec["primary_features"]}。\n\n'
    markdown += '| Features | Weather | Method | Pairwise | Top1 | Top3 | KEEP/MODIFY | MODIFY P/R | 有益精度 | Regret | recovered/lost/newFP |\n|---|---|---|---:|---:|---:|---:|---|---:|---:|---|\n'
    def fmt(value):
        return 'NA' if value is None else f'{value:.4f}'
    for variant, result in report['variants'].items():
        for weather, methods in result['conditions'].items():
            for method, row in methods.items():
                markdown += (f"| {variant} | {weather} | {method} | {fmt(row['pairwise_ranking_accuracy'])} | {fmt(row['top1_best_action_accuracy'])} | "
                             f"{fmt(row['top3_best_action_recall'])} | {fmt(row['KEEP_MODIFY_accuracy'])} | {fmt(row['MODIFY_precision'])}/{fmt(row['MODIFY_recall'])} | "
                             f"{fmt(row['beneficial_action_precision'])} | {row['mean_action_regret']:.4f} | "
                             f"{row['selected_recovered']}/{row['selected_lost']}/{row['selected_new_fp']} |\n")
    markdown += '\nRegret 按结构化后果的优先级分成不同等级，再报告距离最佳等级的归一化差。JSON 同时给 TP 缺口、额外 lost/newFP；这里的 recovered/lost/newFP 是独立动作累计，整帧收益要看 S3。\n'
    markdown += '\n背景/关联候选指标分别保存在 JSON strata；没有非平局样本或预测 MODIFY 时显示 NA。标准化、权重和阈值全部来自 train 场景。\n'
    markdown += '\n竞争特征增量（pairwise差 / regret差，regret越低越好）：\n\n'
    for weather, effect in report['competition_feature_effect'].items():
        markdown += f"- {weather}: {effect['pairwise_accuracy_difference']:+.4f} / {effect['regret_difference']:+.4f}\n"
    if task == 'reg':
        cls_probe, reg_probe = load_probe(run, 'cls'), load_probe(run, 'reg')
        agreements = {}
        for weather in WEATHERS:
            best_equal = learned_equal = count = 0
            for path in cache_paths(run, 'validation', weather):
                row = load_cache(path)
                for label in row['labels']:
                    p = label['proposal_position']
                    chosen, best = {}, {}
                    for current_task, current_probe in (('cls', cls_probe), ('reg', reg_probe)):
                        x = row['features'][current_task][p][:, columns(current_task, spec['primary_features'])]
                        chosen[current_task] = selection(current_probe.scores(x), row['source_names'])[0]
                        keys = action_keys(label['outcomes'][current_task], row['source_names'])
                        best[current_task] = max(range(len(keys)), key=lambda i: keys[i])
                    best_equal += best['cls'] == best['reg']
                    learned_equal += chosen['cls'] == chosen['reg']
                    count += 1
            agreements[weather] = {'best_cls_equals_best_reg': best_equal / max(1, count),
                                   'learned_cls_equals_learned_reg': learned_equal / max(1, count), 'candidates': count}
        report['cls_reg_agreement'] = agreements
        markdown += '\n| Weather | GT最佳cls/reg相同 | learned cls/reg相同 |\n|---|---:|---:|\n'
        for weather, row in agreements.items():
            markdown += f"| {weather} | {row['best_cls_equals_best_reg']:.2%} | {row['learned_cls_equals_learned_reg']:.2%} |\n"
    write_report(run, stage, report, markdown)
