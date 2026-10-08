"""Label composition by group and equally weighted unique frame/proposal."""
from collections import Counter
from pathlib import Path
from .common import (WEATHERS, Manifest, atomic_json, cli, protocol, report)
from .inputs import control_source
from .labels import category, detection_key, detection_pairs, focal_only_pair, original_keys, original_pairs


class Composition:
    def __init__(self):
        self.c = Counter()
        self.categories = Counter()
        self.distributions = {k: Counter() for k in ('recovered', 'lost', 'new_fp')}

    def add(self, outcomes, names, weight=1.):
        old = original_keys(outcomes, names)
        new_pairs = detection_pairs(outcomes, names)
        old_pairs = original_pairs(outcomes, names)
        keep = outcomes[0]
        c = self.c
        c['candidates'] += weight
        c['original_non_tie_pairs'] += weight * len(old_pairs)
        c['detection_preference_non_tie_pairs'] += weight * len(new_pairs)
        c['strict_detection_non_tie_pairs'] += weight * len(detection_pairs(outcomes, names, False))
        c['original_keep_simplicity_pairs'] += weight * sum(
            original_keys(outcomes, names, False)[i] == original_keys(outcomes, names, False)[j]
            and (i == 0 or j == 0) for i, j in old_pairs)
        c['detection_keep_simplicity_pairs'] += weight * sum(
            detection_key(outcomes[i]) == detection_key(outcomes[j]) and (i == 0 or j == 0)
            for i, j in new_pairs)
        c['preference_conflict_pairs'] += weight * len(set(old_pairs) & {(j, i) for i, j in new_pairs})
        c['strict_detection_conflict_pairs'] += weight * len(
            set(old_pairs) & {(j, i) for i, j in detection_pairs(outcomes, names, False)})
        c['original_focal_only_pairs'] += weight * sum(focal_only_pair(outcomes[i], outcomes[j]) for i, j in old_pairs)
        for i, o in enumerate(outcomes[1:], 1):
            c['non_keep_actions'] += weight
            change = 'tp_increase' if o['action_tp'] > keep['action_tp'] else 'tp_decrease' if o['action_tp'] < keep['action_tp'] else 'tp_unchanged'
            c[change] += weight
            equal = detection_key(o) == detection_key(keep)
            c['same_detection_as_keep'] += weight * equal
            c['original_beneficial'] += weight * (old[i] > old[0])
            c['original_beneficial_same_detection'] += weight * (equal and old[i] > old[0])
            c['focal_only_vs_keep'] += weight * focal_only_pair(o, keep)
            self.categories[category(o, keep)] += weight
            for key, histogram in self.distributions.items():
                histogram[str(o[key])] += weight

    def result(self):
        c = self.c
        actions = max(1., c['non_keep_actions'])
        return {'counts': dict(c), 'exclusive_categories': dict(self.categories),
                'ratios_per_non_keep_action': {k: c[k] / actions for k in (
                    'tp_increase', 'tp_decrease', 'tp_unchanged', 'same_detection_as_keep',
                    'original_beneficial_same_detection', 'focal_only_vs_keep')},
                'focal_only_fraction_of_original_pairs': c['original_focal_only_pairs'] / max(1., c['original_non_tie_pairs']),
                'same_detection_fraction_of_original_beneficial': c['original_beneficial_same_detection'] / max(1., c['original_beneficial']),
                'outcome_distributions': {k: dict(v) for k, v in self.distributions.items()}}


def run_audit(run):
    p = protocol(run)
    source = control_source(run, p)
    manifest = Manifest(run)
    conditions = {}
    fit_summary = {task: Composition() for task in ('cls', 'reg')}
    duplicate_groups = duplicate_proposals = 0
    record_files = []
    for split in ('train', 'validation'):
        conditions[split] = {}
        for weather in WEATHERS:
            acc = {task: {stratum: {unit: Composition() for unit in ('label_group', 'unique_frame_proposal')}
                          for stratum in ('associated', 'background', 'all')} for task in ('cls', 'reg')}
            for frame in source.frames(split, weather):
                multiplicity = Counter(label['proposal_position'] for label in frame['labels'])
                duplicates = Counter(label['proposal_position'] for label in frame['labels'] if not label['background'])
                duplicate_proposals += sum(n > 1 for n in duplicates.values())
                duplicate_groups += sum(n for n in duplicates.values() if n > 1)
                best_rows = []
                for ordinal, label in enumerate(frame['labels']):
                    position = label['proposal_position']
                    best = {}
                    for task in ('cls', 'reg'):
                        outcomes = label['outcomes'][task]
                        if split == 'train' and frame['scene'] in source.protocol['probe_fit_scenes']:
                            fit_summary[task].add(outcomes, frame['source_names'])
                        best[task] = list(max(map(detection_key, outcomes)))
                        for stratum in ('background' if label['background'] else 'associated', 'all'):
                            acc[task][stratum]['label_group'].add(outcomes, frame['source_names'])
                            acc[task][stratum]['unique_frame_proposal'].add(outcomes, frame['source_names'], 1 / multiplicity[position])
                    best_rows.append({'label_ordinal': ordinal, 'proposal_position': position,
                                      'anchor_id': frame['proposals']['anchor_ids'][position],
                                      'background': label['background'], 'target_index': label['target_index'],
                                      'best_detection_key': best, 'proposal_label_multiplicity': multiplicity[position]})
                path = Path(run) / 'cache' / 'label_audit' / split / weather / f"{frame['frame']:08d}.json"
                if not manifest.complete('audit-frame', split, weather, frame['frame']):
                    atomic_json(path, {'frame': frame['frame'], 'labels': best_rows,
                                      'multi_gt_proposals': sum(n > 1 for n in duplicates.values()),
                                      'groups_on_multi_gt_proposals': sum(n for n in duplicates.values() if n > 1)})
                    manifest.mark('audit-frame', 'complete', split, weather, frame['frame'], [path])
                record_files.append(str(path.relative_to(run)))
            conditions[split][weather] = {task: {s: {u: a.result() for u, a in units.items()}
                                               for s, units in strata.items()} for task, strata in acc.items()}
            print(f'LABEL_AUDIT {split}/{weather} complete', flush=True)
    value = {'conditions': conditions, 'probe_fit_composition': {t: a.result() for t, a in fit_summary.items()},
             'multi_gt_proposal_count': duplicate_proposals,
             'label_groups_on_multi_gt_proposals': duplicate_groups, 'best_key_records': record_files,
             'unique_proposal_weighting': 'Each frame/proposal contributes total weight 1, shared equally across its target labels',
             'training_weighting': 'Original candidate-target label groups retained, no deduplication',
             'exclusive_category_order': ['tp_count_increase', 'tp_count_decrease', 'gt_identity_change_same_tp_count',
                 'fp_change_same_tp_and_gt_ids', 'focal_assignment_change', 'score_or_localization_only', 'equivalent_saved_outcome']}
    md = '# 标签组成审计\n\n所有比例来自原 S0 缓存；不重新生成标签。互斥类别按 TP 数量变化→目标身份变化→FP 变化→focal 匹配→仅分数/定位→保存后果同效依次判断，每个动作只进入一类。\n\n'
    md += '| Split | Weather | Task | Stratum | Unit | TP增加/减少/不变 | 与KEEP检测同效 | 原有益但检测同效 | 原pair仅focal打破 | 原/新训练pair | KEEP简化pair原/新 | 冲突pair |\n|---|---|---|---|---|---|---:|---:|---:|---|---|---:|\n'
    for split, weathers in conditions.items():
        for weather, tasks in weathers.items():
            for task, strata in tasks.items():
                for stratum, units in strata.items():
                    for unit, row in units.items():
                        r, c = row['ratios_per_non_keep_action'], row['counts']
                        md += (f"| {split} | {weather} | {task} | {stratum} | {unit} | "
                               f"{r['tp_increase']:.2%}/{r['tp_decrease']:.2%}/{r['tp_unchanged']:.2%} | "
                               f"{r['same_detection_as_keep']:.2%} | {r['original_beneficial_same_detection']:.2%} | "
                               f"{row['focal_only_fraction_of_original_pairs']:.2%} | {c.get('original_non_tie_pairs', 0):.1f}/{c.get('detection_preference_non_tie_pairs', 0):.1f} | "
                               f"{c.get('original_keep_simplicity_pairs', 0):.1f}/{c.get('detection_keep_simplicity_pairs', 0):.1f} | {c.get('preference_conflict_pairs', 0):.1f} |\n")
    md += (f'\n多GT关联 proposal 共 {duplicate_proposals} 个，涉及标签组 {duplicate_groups} 个。唯一 proposal 统计对其各 GT 标签等权分摊；训练仍保留原标签组权重。\n\n'
           '仅分数/定位变化仍可能影响 AP，不能称为无效动作或救回目标。保存后果同效不代表密集输出逐字节相同。原/新训练 pair 包含各自 KEEP 简化偏好；严格检测后果非平局 pair 另存 JSON。每组最佳后果键、后果直方图和互斥组成详见 JSON 及 best_key_records。\n')
    report(run, 'LABEL_AUDIT', value, md)


if __name__ == '__main__':
    run_audit(cli(__doc__).parse_args().run)
