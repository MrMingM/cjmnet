"""Chinese interpretation without tuning, causal overclaim or new success gates."""
from pathlib import Path
from .common import GROUPS, WEATHERS, Manifest, atomic_json, atomic_text, cli, protocol, read_json


def classify_execution(old, new, gain, almost_keep):
    ratio = new['changed_proposals'] / max(1, new['proposals'])
    if ratio <= almost_keep and new['recovered'] == 0:
        return '新标签使策略更保守，尚未证明提高动作选择能力。'
    if (gain > 0 and new['recovered'] >= old['recovered'] and new['lost'] <= old['lost']
            and new['new_fp'] <= old['new_fp'] and (new['lost'] < old['lost'] or new['new_fp'] < old['new_fp'])):
        return '检测后果标签帮助学习器把来源偏好转化为更好的执行结果。'
    return '当前结果包含收益与代价的权衡，尚不能证明检测后果标签整体更好。'


def summarize(run):
    run = Path(run)
    p = protocol(run)
    manifest = Manifest(run)
    for stage in ('LABEL_AUDIT', 'TRAIN', 'CALIBRATION', 'PROBE', 'REPLAY', 'SOURCE_S3'):
        if not manifest.complete(stage):
            raise RuntimeError('Final control report requires completed stage: ' + stage)
    if read_json(run / 'SOURCE_S3_CHECK.json')['status'] != 'matched':
        raise RuntimeError('Original S3 reference comparison must pass before FINAL')
    audit = read_json(run / 'LABEL_AUDIT.json')
    probe = read_json(run / 'PROBE_RESULTS.json')
    replay = read_json(run / 'REPLAY_RESULTS.json')
    adverse = ('fog', 'rain', 'snow')
    conditions = replay['conditions']
    fit_quality = {t: audit['probe_fit_composition'][t]['focal_only_fraction_of_original_pairs'] for t in ('cls', 'reg')}
    policy_findings, aggregates = {}, {}
    for policy in ('Proposal-Task-Greedy', 'Proposal-Task-CommonConservative'):
        counts = {}
        for group in GROUPS:
            counts[group] = {k: sum(conditions[w][group + '-' + policy]['counts'].get(k, 0) for w in adverse)
                             for k in ('recovered', 'lost', 'new_fp', 'changed_proposals', 'proposals')}
        gain = sum(replay['paired_delta_ap_pp_new_minus_original'][w][policy]['ap70'] for w in adverse) / 3
        policy_findings[policy] = classify_execution(counts[GROUPS[0]], counts[GROUPS[1]], gain,
                                                   p['config']['almost_all_keep_modify_fraction'])
        aggregates[policy] = {'adverse_counts': counts, 'paired_mean_ap70_pp': gain}
    gap = {g: {w: 100 * (conditions[w][g + '-Assisted-Task-Greedy']['ap70']
                              - conditions[w][g + '-Proposal-Task-Greedy']['ap70']) for w in WEATHERS} for g in GROUPS}
    offline = {}
    for task in ('cls', 'reg'):
        offline[task] = {key: sum(probe['conditions'][task][w][GROUPS[1]]['Greedy']['all']['DetectionOutcomeCriterion'][key]
                                 - probe['conditions'][task][w][GROUPS[0]]['Greedy']['all']['DetectionOutcomeCriterion'][key]
                                 for w in adverse) / 3 for key in ('mean_detection_outcome_regret', 'same_detection_modify_ratio', 'modify_ratio')}
    md = '# Action Ranking Label Control\n\n## 一句话结论\n\n' + policy_findings['Proposal-Task-Greedy'] + '\n\n'
    md += '两组只改变训练排序标签，原特征、标准化、场景、单层结构及执行规则保持相同；共同保守阈值分别在原 train-calibration 上按同一检测后果标准确定。\n\n'
    md += '| Method | Clean AP70 | Fog AP70 | Rain AP70 | Snow AP70 | 恶劣天气平均 ΔAP70 vs Shared pp |\n|---|---:|---:|---:|---:|---:|\n'
    for method in conditions['clean']:
        scores = ' | '.join(f"{conditions[w][method]['ap70']:.6f}" for w in WEATHERS)
        gain = sum(conditions[w][method]['delta_ap_pp_vs_Shared']['ap70'] for w in adverse) / 3
        md += f'| {method} | {scores} | {gain:+.4f} |\n'
    md += '\n| Policy | Label | recovered | lost | newFP | changed | 修改比例 | 新减原 mean AP70 pp |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for policy, values in aggregates.items():
        for group, c in values['adverse_counts'].items():
            md += (f"| {policy} | {group} | {c['recovered']} | {c['lost']} | {c['new_fp']} | {c['changed_proposals']} | "
                   f"{c['changed_proposals']/max(1,c['proposals']):.2%} | {values['paired_mean_ap70_pp']:+.4f} |\n")
    md += '\n以上后果按 Fog/Rain/Snow 累计；Clean 和单天气细节见 REPLAY_RESULTS.md。\n\n'
    md += '1. **原训练偏好是否大量来自分数/定位平局打破？**\n\n'
    for task, fraction in fit_quality.items():
        large = fraction >= p['config']['large_quality_tiebreak_fraction']
        line = p['config']['large_quality_tiebreak_fraction']
        md += f"{task} 的原 probe-fit 非平局训练 pair 中，仅由 focal score/IoU 打破平局的比例为 {fraction:.2%}，{'达到' if large else '未达到'}预注册的 {line:.0%} 描述线。分数改善仍可能改善 AP。\n\n"
    md += '2. **是否改善共同检测后果评价？** 新减原，负 regret 差表示更接近最佳检测后果；不跨标签评价标准相减。\n\n'
    md += '| Task | Greedy共同Regret差 | 检测同效仍修改比例差 | 修改比例差 |\n|---|---:|---:|---:|\n'
    for task, values in offline.items():
        md += f"| {task} | {values['mean_detection_outcome_regret']:+.4f} | {values['same_detection_modify_ratio']:+.2%} | {values['modify_ratio']:+.2%} |\n"
    md += '\n3. **是否减少无检测收益修改和误伤？** 检测同效仍修改比例差见上表；Greedy 与共同 Conservative 的 lost/newFP 及修改量同时列出。检测同效只限定保存的检测后果，不等于动作对 AP 无效。\n\n'
    md += '4. **是否转化为 AP？**\n\n'
    for policy, finding in policy_findings.items():
        md += f"- {policy}：新减原三天气平均 AP70 为 {aggregates[policy]['paired_mean_ap70_pp']:+.4f} pp。{finding}\n"
    md += '\n5. **是否仅表现为动作变少？**\n\n' + ' '.join(policy_findings.values()) + '\n\n'
    md += '6. **Assisted 与 Proposal 的差距是否改变？** 相同 Greedy 策略下 Assisted−Proposal，单位 pp：\n\n| Label | Clean | Fog | Rain | Snow |\n|---|---:|---:|---:|---:|\n'
    for group, values in gap.items():
        md += '| ' + group + ' | ' + ' | '.join(f'{values[w]:+.4f}' for w in WEATHERS) + ' |\n'
    md += '\n## 目前能得出什么结论\n\n'
    md += '\n'.join('- ' + finding for finding in policy_findings.values()) + '\n'
    fallback = [g + '/' + t for g, tasks in probe['training'].items() for t, value in tasks.items() if value.get('fallback')]
    if fallback:
        md += '\n缺少可训练偏好的任务：' + ', '.join(fallback) + '。其 KEEP_ALL 回退不能视为实验成功。\n'
    md += '\nShared 与原 baseline 以及原标签 Greedy 与原 S3 的 AP30/AP50/AP70 均通过 1e-6 核对；逐帧 Shared 哈希和推理特征一致。没有访问正式 test。\n'
    md += '\n## 还不能得出什么结论\n\n'
    md += ('检测后果标签仍不是 AP 标签；同效 score/IoU 改善可能改变排序及 AP。单动作最佳不能保证整帧多动作或分类/回归联合最优。'
           '原/new pair 数、optimizer 更新次数不同，已逐轮报告；这属于标签处理的一部分。独立动作后果可能重复计数，整帧后果以回放为准。'
           '这是重复使用开发场景的对照，没有独立 test、bootstrap 或显著性检验。不能把竞争、任务交互或信息不足宣布为唯一根因，也不能由新标签无收益否定动作决策方向。\n')
    md += '\n## 下一步应该验证什么\n\n'
    md += ('保留本次标签、阈值和训练预算；结合逐帧动作检查收益和误伤落在哪些候选，区分检测后果同效的排序机会与真正恢复/丢失。'
           '只有在新场景复验后再讨论机制升级；不要在看过本次 validation 后自动调参、改变标签或重跑挑结果。\n')
    value = {'conclusions': policy_findings, 'aggregates': aggregates, 'probe_fit_focal_only_pair_fraction': fit_quality,
             'common_offline_differences': offline, 'assisted_minus_proposal_ap70_pp': gap,
             'label_audit': audit, 'probe_results': probe, 'replay_results': replay,
             'source_s3_check': read_json(run / 'SOURCE_S3_CHECK.json'), 'test_data_used': False,
             'inference_gt_fields': 0, 'no_trainable_preference_fallbacks': fallback}
    atomic_json(run / 'final_results.json', value)
    atomic_text(run / 'FINAL_RESULTS.md', md)


if __name__ == '__main__':
    summarize(cli(__doc__).parse_args().run)
