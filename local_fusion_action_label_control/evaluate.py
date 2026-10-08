"""Both probes evaluated against BOTH original and common detection labels."""
from collections import Counter
from .common import GROUPS, WEATHERS, cli, protocol, report
from .inputs import control_source
from .labels import detection_key, detection_pairs, detection_ranks
from .probes import load_probe
from local_fusion_action_utility_audit.linear_ranker import Metrics as OriginalMetrics, selection


class DetectionMetrics:
    def __init__(self):
        self.c = Counter()
        self.regret = 0.

    def add(self, group, scores, threshold=0.):
        outcomes, names = group['outcomes'], group['names']
        selected, _, order = selection(scores, names, threshold)
        values = [detection_key(o) for o in outcomes]
        best = max(values)
        best_i = values.index(best)
        pairs = detection_pairs(outcomes, names, False)
        preference_pairs = detection_pairs(outcomes, names)
        c = self.c
        c['candidates'] += 1
        c['strict_detection_pairs'] += len(pairs)
        c['pair_correct'] += sum(1 if scores[i] > scores[j] else .5 if scores[i] == scores[j] else 0. for i, j in pairs)
        c['preference_pairs_including_keep'] += len(preference_pairs)
        c['preference_pair_correct'] += sum(1 if scores[i] > scores[j] else .5 if scores[i] == scores[j] else 0. for i, j in preference_pairs)
        c['top1_best_detection'] += values[selected] == best
        c['top3_best_detection'] += any(values[i] == best for i in order[:3])
        c['has_better_than_keep'] += best > values[0]
        c['modify'] += selected != 0
        c['beneficial_modify'] += selected != 0 and values[selected] > values[0]
        c['same_detection_modify'] += selected != 0 and values[selected] == values[0]
        c['worse_detection_modify'] += selected != 0 and values[selected] < values[0]
        c['KEEP_MODIFY_correct'] += (selected != 0) == (best > values[0])
        ranks = detection_ranks(outcomes, names)
        self.regret += max(ranks) - ranks[selected]
        for component in ('recovered', 'lost', 'new_fp'):
            c['selected_' + component] += outcomes[selected][component]
        c['tp_deficit'] += outcomes[best_i]['action_tp'] - outcomes[selected]['action_tp']
        c['excess_lost'] += outcomes[selected]['lost'] - outcomes[best_i]['lost']
        c['excess_new_fp'] += outcomes[selected]['new_fp'] - outcomes[best_i]['new_fp']

    def result(self):
        c = self.c
        n, modifications = max(1, c['candidates']), c['modify']
        return {'counts': dict(c),
                'non_tie_pairwise_accuracy': c['pair_correct'] / c['strict_detection_pairs'] if c['strict_detection_pairs'] else None,
                'preference_pairwise_including_keep': c['preference_pair_correct'] / c['preference_pairs_including_keep'] if c['preference_pairs_including_keep'] else None,
                'top1_best_detection_ratio': c['top1_best_detection'] / n, 'top3_best_detection_recall': c['top3_best_detection'] / n,
                'mean_detection_outcome_regret': self.regret / n,
                'has_better_than_keep_ratio': c['has_better_than_keep'] / n,
                'beneficial_modify_precision': c['beneficial_modify'] / modifications if modifications else None,
                'modify_ratio': modifications / n, 'same_detection_modify_ratio': c['same_detection_modify'] / n,
                'same_detection_among_modifications': c['same_detection_modify'] / modifications if modifications else None,
                'worse_detection_modify_ratio': c['worse_detection_modify'] / n,
                'KEEP_MODIFY_accuracy': c['KEEP_MODIFY_correct'] / n,
                **{key: c[key] for key in ('selected_recovered', 'selected_lost', 'selected_new_fp', 'tp_deficit', 'excess_lost', 'excess_new_fp')}}


def evaluate(run):
    p = protocol(run)
    source = control_source(run, p)
    probes = {g: {t: load_probe(run, g, t) for t in ('cls', 'reg')} for g in GROUPS}
    conditions = {}
    for task in ('cls', 'reg'):
        conditions[task] = {}
        for weather in WEATHERS:
            acc = {g: {policy: {s: (OriginalMetrics(), DetectionMetrics()) for s in ('all', 'associated', 'background')}
                       for policy in ('Greedy', 'CommonConservative')} for g in GROUPS}
            for group in source.groups('validation', weather, task):
                for label_group in GROUPS:
                    probe = probes[label_group][task]
                    scores = probe.scores(group['x'])
                    for policy in ('Greedy', 'CommonConservative'):
                        threshold = 0. if policy == 'Greedy' else probe.state['calibration']['threshold']
                        for stratum in ('all', 'background' if group['background'] else 'associated'):
                            old, common = acc[label_group][policy][stratum]
                            old.add(group, scores, threshold)
                            common.add(group, scores, threshold)
            conditions[task][weather] = {g: {policy: {s: {
                'OriginalLabelCriterion': a.result(), 'DetectionOutcomeCriterion': b.result()}
                for s, (a, b) in strata.items()} for policy, strata in policies.items()} for g, policies in acc.items()}
            print(f'PROBE_RESULTS {task}/{weather} complete', flush=True)
    training = {g: {t: {k: probes[g][t].state.get(k) for k in ('history', 'fallback', 'reused_original_weights')}
                    for t in ('cls', 'reg')} for g in GROUPS}
    md = '# 动作标签离线对照\n\n两组使用相同原特征、标准化和场景；OriginalLabel 直接复用原主权重。下表统一采用检测后果标准，严格检测非平局 pair 排除 KEEP 简化偏好。原标签标准另表报告，两个标准的 regret 不直接相减。\n\n'
    md += '| Task | Weather | Label | Policy | Stratum | Pairwise | 最佳后果Top1 | Regret | 修改比例 | 检测同效仍改 | 真有益精度 | recovered/lost/newFP | TP缺口/额外lost |\n|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|\n'
    def fmt(value):
        return 'NA' if value is None else f'{value:.4f}'
    for task, weathers in conditions.items():
        for weather, groups in weathers.items():
            for group, policies in groups.items():
                for policy, strata in policies.items():
                    for stratum, criteria in strata.items():
                        r = criteria['DetectionOutcomeCriterion']
                        md += (f"| {task} | {weather} | {group} | {policy} | {stratum} | {fmt(r['non_tie_pairwise_accuracy'])} | "
                               f"{r['top1_best_detection_ratio']:.4f} | {r['mean_detection_outcome_regret']:.4f} | {r['modify_ratio']:.2%} | "
                               f"{r['same_detection_modify_ratio']:.2%} | {fmt(r['beneficial_modify_precision'])} | "
                               f"{r['selected_recovered']}/{r['selected_lost']}/{r['selected_new_fp']} | {r['tp_deficit']}/{r['excess_lost']} |\n")
    md += '\n原标签评价标准（全体标签组）：\n\n| Task | Weather | Label | Policy | 原Pairwise | 原Top1 | 原Regret |\n|---|---|---|---|---:|---:|---:|\n'
    for task, weathers in conditions.items():
        for weather, groups in weathers.items():
            for group, policies in groups.items():
                for policy, strata in policies.items():
                    r = strata['all']['OriginalLabelCriterion']
                    md += f"| {task} | {weather} | {group} | {policy} | {fmt(r['pairwise_ranking_accuracy'])} | {r['top1_best_action_accuracy']:.4f} | {r['mean_action_regret']:.4f} |\n"
    md += '\n训练预算（原标签更新次数由原分组、顺序和批规则复原）：\n\n| Label | Task | Epoch | pairs | optimizer updates | loss |\n|---|---|---:|---:|---:|---:|\n'
    for group, tasks in training.items():
        for task, row in tasks.items():
            for epoch in row['history']:
                md += f"| {group} | {task} | {epoch['epoch']} | {epoch['pairs']} | {epoch['optimizer_updates']} | {epoch['loss']:.6f} |\n"
            if row['fallback']:
                md += f"\n{group}/{task}：缺少可训练偏好，KEEP_ALL 回退；不能记作实验成功。\n"
    md += '\n独立动作的 recovered/lost/newFP 可能对同一 GT 重复计数，不能解释为整帧收益。两组 pair 数和更新次数可能不同，这是标签处理的一部分。没有套用原 S1/S2 成功门槛。\n'
    report(run, 'PROBE_RESULTS', {'conditions': conditions, 'training': training,
        'common_regret': 'normalized rank of distinct detection outcome keys; no KEEP simplicity reward',
        'original_regret': 'verbatim original outcome_ranks; never subtract across criteria'}, md)


if __name__ == '__main__':
    evaluate(cli(__doc__).parse_args().run)
