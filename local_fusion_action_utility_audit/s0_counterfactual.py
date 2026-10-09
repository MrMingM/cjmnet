"""S0: Shared proposals, independent single-task action consequences, headroom."""
from collections import Counter
import numpy as np
from .common import (KEEP, WEATHERS, Runtime, atomic_json, atomic_torch,
                     cache_paths, frame_scene, load_cache, parser, write_report)
from .counterfactual import action_keys, counterfactual_outcomes, outcome_state
from .features import extract_features, schema
from .reproducibility import assert_frame_reproduction, load_reference


def prediction_hash(prediction):
    import hashlib
    value = hashlib.sha256()
    for name in ('psm', 'rm'):
        value.update(prediction[name].detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def proposals(dataset, batch, shared, pool, context, runtime):
    from local_fusion_task_source_oracle.oracle import _proxy_cache, build_proxy_pool
    from local_fusion_task_split_pilot.proposal_constrained_oracle import top_shared_proposals
    from qa_local_intervention.operators import masks_for_frame
    proxies = build_proxy_pool(dataset, batch, pool)
    proxies[KEEP] = _proxy_cache(dataset, batch, shared)
    ids, scores, corners = top_shared_proposals(proxies[KEEP], runtime.spec['shared_top_k'])
    masks = [masks_for_frame(batch['ego']['anchor_box'], corner, context['levels'],
                             runtime.lidar_range, runtime.spec['roi_expansion'])[0]
             for corner in corners]
    names, features = extract_features(proxies, ids, scores, corners, masks, runtime.spec)
    return ids, scores, corners, masks, names, features


def background_sample(eligible, budget, seed):
    """Eligible is already sorted by decreasing Shared score; stratify thirds."""
    rng = np.random.default_rng(seed)
    strata = np.array_split(np.asarray(eligible, dtype=int), 3)
    selected = []
    for i, stratum in enumerate(strata):
        count = min(len(stratum), budget // 3 + int(i < budget % 3))
        selected.extend(int(x) for x in rng.choice(stratum, count, replace=False))
    remaining = sorted(set(eligible) - set(selected))
    if len(selected) < min(budget, len(eligible)):
        count = min(budget, len(eligible)) - len(selected)
        selected.extend(int(x) for x in rng.choice(remaining, count, replace=False))
    return sorted(selected)


def frame_ap_stats(post):
    from ceif_audit.scoring import empty_stats
    from opencood.utils import eval_utils
    stats = empty_stats()
    for threshold in stats:
        eval_utils.caluclate_tp_fp(*post, stats, threshold)
    return stats


def baseline(run):
    import torch
    from ceif_audit.scoring import ap_values, empty_stats, merge
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    runtime = Runtime(run)
    saved = runtime.protocol['b0_saved_results']
    results = {}
    with torch.no_grad():
        for weather in WEATHERS:
            folder = runtime.run / 'cache' / 'baseline' / weather
            folder.mkdir(parents=True, exist_ok=True)
            if runtime.manifest.complete('baseline-weather', 'validation', weather):
                results[weather] = load_cache(folder / 'summary.pt')
                continue
            dataset, loader = runtime.loader('validation', weather)
            stats = empty_stats()
            seen = []
            with runtime.manifest.work('baseline-weather', 'validation', weather):
                for ordinal, batch in enumerate(loader):
                    index = int(batch['ego']['communication_sample_index'][0])
                    seen.append(index)
                    if index != runtime.protocol['validation_indices'][ordinal]:
                        raise RuntimeError('Validation order differs from B0')
                    path = folder / f'{index:08d}.pt'
                    if runtime.manifest.complete('baseline-frame', 'validation', weather, index):
                        row = load_cache(path)
                    else:
                        with runtime.manifest.work('baseline-frame', 'validation', weather, index):
                            batch = to_device(batch, runtime.target)
                            _, shared, _ = runtime.predict(batch, weather, verify=ordinal == 0, sources=False)
                            post = dataset.post_process(batch, {'ego': shared})
                            row = {'stats': frame_ap_stats(post), 'prediction_hash': prediction_hash(shared)}
                            atomic_torch(path, row)
                        runtime.manifest.mark('baseline-frame', 'complete', 'validation', weather, index, [path])
                    merge(stats, row['stats'])
                    if ordinal == 0 or (ordinal + 1) % 10 == 0:
                        print(f'baseline {weather} {ordinal + 1}/{len(loader)}', flush=True)
                if seen != runtime.protocol['validation_indices']:
                    raise RuntimeError('Incomplete baseline')
                ap = ap_values(stats, eval_utils)
                differences = {}
                for metric in ('ap30', 'ap50', 'ap70'):
                    differences[metric] = abs(ap[metric] - saved['conditions'][weather]['results']['Shared'][metric])
                    if differences[metric] > 1e-6:
                        raise RuntimeError(f'Baseline reproduction failed {weather}.{metric}: {differences[metric]}')
                results[weather] = {'frames': len(seen), 'Shared': ap, 'absolute_difference': differences}
                atomic_torch(folder / 'summary.pt', results[weather])
            runtime.manifest.mark('baseline-weather', 'complete', 'validation', weather,
                                  artifacts=[folder / 'summary.pt'])
            del loader, dataset
    atomic_json(runtime.run / 'baseline_reproduction.json', results)


def build(run):
    import torch
    from opencood.tools.train_utils import to_device
    from local_fusion_task_split_pilot.proposal_constrained_oracle import assign_proposals
    runtime = Runtime(run)
    if not runtime.manifest.complete('baseline'):
        raise RuntimeError('Complete B0 baseline reproduction is required before S0')
    if not runtime.manifest.complete('validation-reference'):
        raise RuntimeError('Complete exact validation inference references are required before S0')
    atomic_json(runtime.run / 'feature_schema.json', {
        'schema': 1, 'cls': schema('cls'), 'reg': schema('reg'),
        'inference_gt_fields': 0, 'source_id_is_feature': False,
        'competition_interactions': 'fixed products of predicted evidence and predicted competition',
    })
    with torch.no_grad():
        for split in ('train', 'validation'):
            for weather in WEATHERS:
                if runtime.manifest.complete('S0-weather', split, weather):
                    print(f'S0 reuse complete {split}/{weather}', flush=True)
                    continue
                dataset, loader = runtime.loader(split, weather)
                seen = []
                folder = runtime.run / 'cache' / split / weather
                folder.mkdir(parents=True, exist_ok=True)
                with runtime.manifest.work('S0-weather', split, weather):
                    for ordinal, batch in enumerate(loader):
                        index = int(batch['ego']['communication_sample_index'][0])
                        seen.append(index)
                        if runtime.manifest.complete('S0-frame', split, weather, index):
                            continue
                        with runtime.manifest.work('S0-frame', split, weather, index):
                            batch = to_device(batch, runtime.target)
                            if split == 'validation':
                                pool, reference = load_reference(runtime, batch, weather, index)
                                shared, context = pool[KEEP], {'levels': []}
                                if prediction_hash(shared) != reference['prediction_hash']:
                                    raise RuntimeError('Exact Shared reference is corrupted')
                            else:
                                context, shared, pool = runtime.predict(batch, weather, verify=ordinal == 0)
                            shared_post = dataset.post_process(batch, {'ego': shared})
                            if split == 'validation':
                                assert_frame_reproduction(reference['evaluation_only']['stats'], frame_ap_stats(shared_post))
                            ids, scores, corners, masks, names, features = proposals(
                                dataset, batch, shared, pool, context, runtime)
                            # GT first enters AFTER the proposal masks/features have been fixed.
                            gt = shared_post[2].detach().cpu().numpy()
                            association, best_ious = assign_proposals(
                                gt, ids, scores, corners, runtime.spec['minimum_gt_proposal_iou'])
                            baseline_state = outcome_state(shared_post, runtime.spec['target_iou'])
                            positions = {int(anchor): p for p, anchor in enumerate(ids)}
                            groups = [(positions[row['anchor_id']], t) for t, row in enumerate(association) if row is not None]
                            associated = {p for p, _ in groups}
                            bg = background_sample([p for p in range(len(ids)) if p not in associated],
                                                   runtime.spec['background_candidates_per_frame'],
                                                   runtime.spec['seed'] + index + 100000 * WEATHERS.index(weather))
                            groups.extend((p, None) for p in bg)
                            labels = []
                            for number, (p, focal) in enumerate(groups, 1):
                                outcomes = counterfactual_outcomes(dataset, batch, shared, pool, masks[p], names,
                                                                  baseline_state, focal, runtime.spec)
                                labels.append({'proposal_position': p, 'target_index': focal,
                                               'background': focal is None, 'outcomes': outcomes})
                                if number == 1 or number % 10 == 0 or number == len(groups):
                                    print(f'  S0 {split}/{weather} frame={index} candidate {number}/{len(groups)}', flush=True)
                            row = {'frame': index, 'scene': frame_scene(dataset, index), 'split': split,
                                   'weather': weather, 'source_names': names, 'features': {
                                       k: torch.from_numpy(v) for k, v in features.items()},
                                   'proposals': {'anchor_ids': ids.tolist(), 'scores': scores.tolist(),
                                                 'corners': torch.from_numpy(corners), 'ranks': list(range(1, len(ids) + 1))},
                                   'labels': labels, 'evaluation_metadata': {
                                       'gt_count': len(gt), 'proposal_covered': sum(x is not None for x in association),
                                       'shared_missed': len(gt) - len(baseline_state['matched']),
                                       'shared_missed_covered': sum(x is not None and t not in baseline_state['matched']
                                                                    for t, x in enumerate(association)),
                                       'best_proposal_ious': best_ious},
                                   'prediction_hash': prediction_hash(shared)}
                            path = folder / f'{index:08d}.pt'
                            atomic_torch(path, row)
                        runtime.manifest.mark('S0-frame', 'complete', split, weather, index, [path])
                        print(f'S0 {split}/{weather} {ordinal + 1}/{len(loader)} frame={index} complete', flush=True)
                    if sorted(seen) != sorted(runtime.protocol[split + '_indices']):
                        raise RuntimeError('Incomplete or duplicate S0 frames')
                atomic_json(folder / 'complete.json', {'frames': sorted(seen)})
                runtime.manifest.mark('S0-weather', 'complete', split, weather, artifacts=[folder / 'complete.json'])
                del dataset, loader


def summarize(run):
    conditions = {}
    for split in ('train', 'validation'):
        conditions[split] = {}
        for weather in WEATHERS:
            coverage = Counter()
            strata = {s: {'cls': Counter(), 'reg': Counter(), 'disagreement': 0, 'candidates': 0}
                      for s in ('associated', 'background', 'all')}
            for path in cache_paths(run, split, weather):
                row = load_cache(path)
                meta = row['evaluation_metadata']
                coverage.update({k: meta[k] for k in ('gt_count', 'proposal_covered', 'shared_missed', 'shared_missed_covered')})
                for label in row['labels']:
                    chosen = {}
                    for task in ('cls', 'reg'):
                        outcomes = label['outcomes'][task]
                        keys = action_keys(outcomes, row['source_names'])
                        chosen[task] = max(range(len(keys)), key=lambda i: keys[i])
                    for stratum in ('background' if label['background'] else 'associated', 'all'):
                        total = strata[stratum]
                        total['candidates'] += 1
                        total['disagreement'] += chosen['cls'] != chosen['reg']
                        for task in ('cls', 'reg'):
                            outcomes = label['outcomes'][task]
                            keys = action_keys(outcomes, row['source_names'])
                            counter = total[task]
                            best = chosen[task]
                            counter['best:' + row['source_names'][best]] += 1
                            counter['keep_best'] += best == 0
                            counter['beneficial_candidates'] += keys[best] > keys[0]
                            for i, outcome in enumerate(outcomes):
                                if i:
                                    counter['actions'] += 1
                                    counter['beneficial_actions'] += keys[i] > keys[0]
                                    for key in ('recovered', 'lost', 'new_fp'):
                                        counter[f'{key}_count:{outcome[key]}'] += 1
            report = {'counts': dict(coverage),
                      'proposal_coverage': coverage['proposal_covered'] / max(1, coverage['gt_count']),
                      'shared_missed_proposal_coverage': coverage['shared_missed_covered'] / max(1, coverage['shared_missed']),
                      'strata': {}}
            for stratum, total in strata.items():
                n = total['candidates']
                dest = {'candidates': n, 'task_source_disagreement_rate': total['disagreement'] / max(1, n)}
                for task in ('cls', 'reg'):
                    counter = total[task]
                    distribution = {k[5:]: v for k, v in counter.items() if k.startswith('best:')}
                    probabilities = np.array(list(distribution.values()), float) / max(1, n)
                    dest[task] = {'beneficial_action_ratio': counter['beneficial_actions'] / max(1, counter['actions']),
                                  'beneficial_candidate_ratio': counter['beneficial_candidates'] / max(1, n),
                                  'best_source_distribution': distribution, 'KEEP_best_ratio': counter['keep_best'] / max(1, n),
                                  'source_diversity': {'distinct_best_sources': len(distribution),
                                                       'entropy': float(-(probabilities * np.log(probabilities)).sum())},
                                  'action_outcome_distribution': {k: v for k, v in counter.items() if '_count:' in k}}
                report['strata'][stratum] = dest
            report['task_source_disagreement_rate'] = report['strata']['all']['task_source_disagreement_rate']
            conditions[split][weather] = report
    markdown = '# S0：单动作后果与来源差异\n\n每个动作独立从原 Shared 出发；第一张表包含关联组和背景组，后续分开报告。关联组按 candidate–target 计数。\n\n'
    markdown += '| Split | Weather | GT覆盖 | 原漏检覆盖 | 最佳cls/reg不同 | cls有益动作率 | reg有益动作率 |\n|---|---|---:|---:|---:|---:|---:|\n'
    for split, rows in conditions.items():
        for weather, row in rows.items():
            all_rows = row['strata']['all']
            markdown += (f"| {split} | {weather} | {row['proposal_coverage']:.2%} | {row['shared_missed_proposal_coverage']:.2%} | "
                         f"{all_rows['task_source_disagreement_rate']:.2%} | {all_rows['cls']['beneficial_action_ratio']:.2%} | {all_rows['reg']['beneficial_action_ratio']:.2%} |\n")
    markdown += '\n| Split | Weather | 组别 | 样本数 | 最佳cls/reg不同 | cls有益动作率 | reg有益动作率 |\n|---|---|---|---:|---:|---:|---:|\n'
    for split, rows in conditions.items():
        for weather, row in rows.items():
            for name in ('associated', 'background'):
                stratum = row['strata'][name]
                markdown += (f"| {split} | {weather} | {name} | {stratum['candidates']} | "
                             f"{stratum['task_source_disagreement_rate']:.2%} | {stratum['cls']['beneficial_action_ratio']:.2%} | "
                             f"{stratum['reg']['beneficial_action_ratio']:.2%} |\n")
    markdown += '\n来源分布和后果分布均为全部标签组；分组版本保留在 JSON。后果分布的 `k:n` 表示有 n 个动作产生 k 个该类后果。\n\n'
    markdown += '| Split | Weather | Task | KEEP最佳 | 来源种数 / 熵 | 最佳来源次数 | recovered分布 | lost分布 | newFP分布 |\n|---|---|---|---:|---:|---|---|---|---|\n'
    for split, rows in conditions.items():
        for weather, row in rows.items():
            for task in ('cls', 'reg'):
                value = row['strata']['all'][task]
                diversity = value['source_diversity']
                best = ', '.join(f'{name}: {count}' for name, count in sorted(value['best_source_distribution'].items()))
                histograms = []
                for component in ('recovered', 'lost', 'new_fp'):
                    histograms.append(', '.join(f"{key.split(':')[1]}:{count}" for key, count in
                                               sorted(value['action_outcome_distribution'].items())
                                               if key.startswith(component + '_count:')) or '无动作')
                markdown += (f"| {split} | {weather} | {task} | {value['KEEP_best_ratio']:.2%} | "
                             f"{diversity['distinct_best_sources']} / {diversity['entropy']:.3f} | {best} | "
                             + ' | '.join(histograms) + ' |\n')
    markdown += '\n相同后果优先 KEEP；非 KEEP 同效来源按固定 family/index 顺序显示。来源不同率会受同效来源的选择影响，不能直接等同于新增 TP 或任务分离 AP 收益。原始后果保存在逐帧 cache 中。\n'
    write_report(run, 'S0', {'conditions': conditions, 'protocol': 'fixed Shared; exhaustive single-task sources'}, markdown)


def main():
    cli = parser(__doc__)
    cli.add_argument('--mode', choices=('baseline', 'build', 'summarize'), required=True)
    args = cli.parse_args()
    {'baseline': baseline, 'build': build, 'summarize': summarize}[args.mode](args.run)


if __name__ == '__main__':
    main()
