"""Scientific interpretation of the four preregistered development stages."""
from pathlib import Path
from .common import WEATHERS, Manifest, atomic_json, atomic_text, parser, read_json, verify_protocol_snapshot


def interpret(s1, s2, assisted_gain, proposal_gain, threshold):
    findings = []
    if s1 == 'S1_WEAK':
        findings.append(('A', '当前推理可见的分类输出特征不足以可靠预测分类来源效用。',
                         '优先验证更丰富的 source semantic evidence（来源对目标语义的证据）。'))
    if s2 == 'S2_WEAK':
        findings.append(('B', '当前预测几何信息不足以可靠判断回归来源效用。',
                         '优先验证 source-specific geometry evidence / uncertainty（每个来源的几何证据及不确定程度）。'))
    if s1 == 'S1_LEARNABLE' and s2 == 'S2_LEARNABLE':
        if assisted_gain >= threshold and proposal_gain <= 0:
            findings.append(('C', '来源/任务效用具有可学习信号；主要损失发生在从目标辅助选择转向全 proposal 执行的过程中。',
                             '优先验证 competition-aware collaborative action selection（把互相影响的候选一起决定）。'))
        elif assisted_gain >= threshold and proposal_gain > 0:
            findings.append(('D', '当前简单来源动作选择器已经兑现一部分 Oracle 空间。',
                             '在新场景上确认稳定性，再将机制升级为正式方法。'))
        elif assisted_gain <= 0:
            findings.append(('E', '单独学习的分类/回归偏好没有在联合 Task 回放中形成收益，存在任务交互或标签与 AP 不一致的证据。',
                             '优先验证 joint action utility（分类和回归同时改动会有什么后果），并检查单任务 Assisted 回放。'))
        else:
            findings.append(('UNRESOLVED', '两任务可学习，但 Assisted 收益小于预注册的明显收益线；当前证据尚不能定位主要瓶颈。',
                             '先核查样本覆盖、单任务回放和同效来源，保持原阈值，在新场景复验。'))
    return findings


def summarize(run):
    run = Path(run)
    protocol = verify_protocol_snapshot(run)
    manifest = Manifest(run)
    for stage in ('baseline', 'S0', 'S1', 'S2', 'S3'):
        if not manifest.complete(stage):
            raise RuntimeError('Final report requires completed stage: ' + stage)
    spec = protocol['config']
    s0, s1, s2, s3 = [read_json(run / (stage + '_RESULTS.json')) for stage in ('S0', 'S1', 'S2', 'S3')]
    adverse = ('fog', 'rain', 'snow')
    conditions = s3['conditions']
    mean_gain = lambda method: sum(conditions[w][method]['delta_ap_pp_vs_Shared']['ap70'] for w in adverse) / 3
    assisted_gain = mean_gain('Assisted-Task')
    proposal_greedy_gain = mean_gain('Proposal-Task-Greedy')
    proposal_gain = mean_gain('Proposal-Task-Conservative')
    # A–E compares identical source policies; Conservative has a separate gate.
    findings = interpret(s1['decision']['verdict'], s2['decision']['verdict'], assisted_gain,
                         proposal_greedy_gain, spec['assisted_meaningful_gain_pp'])
    disagreement = sum(s0['conditions']['validation'][w]['strata']['associated']['task_source_disagreement_rate'] for w in adverse) / 3
    verdicts = {stage: result['decision']['verdict'] for stage, result in (('S1', s1), ('S2', s2), ('S3', s3))}
    sentence = ' '.join(row[1] for row in findings)
    md = '# Direction B Action Utility Audit\n\n## 一句话结论\n\n' + sentence + '\n\n'
    md += (f"S0：恶劣天气关联候选的最佳分类/回归来源不同率为 **{disagreement:.2%}**，"
           f"{'达到' if disagreement >= spec['task_source_disagreement_gate'] else '未达到'}预注册的明显差异线 {spec['task_source_disagreement_gate']:.0%}。\n\n"
           f"S1：{verdicts['S1']}，检验分类来源效用能否从推理输出学会。\n\n"
           f"S2：{verdicts['S2']}，检验回归来源效用能否从预测几何学会。\n\n"
           f"S3：{verdicts['S3']}，Conservative 三天气平均 AP70 改变 **{proposal_gain:+.4f} pp**；"
           f"Assisted-Task 为 **{assisted_gain:+.4f} pp**，Proposal-Task-Greedy 为 **{proposal_greedy_gain:+.4f} pp**。\n\n")
    md += '| Method | Clean AP70 | Fog AP70 | Rain AP70 | Snow AP70 | adverse mean ΔAP70 |\n|---|---:|---:|---:|---:|---:|\n'
    for method in conditions['clean']:
        values = ' | '.join(f"{conditions[w][method]['ap70']:.6f}" for w in WEATHERS)
        md += f'| {method} | {values} | {mean_gain(method):+.4f} pp |\n'
    md += '\n以下后果按 Clean/Fog/Rain/Snow 四条件累计；S3 recovered>lost 门槛单独使用三种恶劣天气。\n\n'
    md += '| Method | recovered | lost | new FP | changed |\n|---|---:|---:|---:|---:|\n'
    for method in conditions['clean']:
        count = {key: sum(conditions[w][method]['counts'].get(key, 0) for w in WEATHERS)
                 for key in ('recovered', 'lost', 'new_fp', 'changed_proposals')}
        md += f"| {method} | {count['recovered']} | {count['lost']} | {count['new_fp']} | {count['changed_proposals']} |\n"
    md += '\n' + '\n'.join(f'{stage} verdict：**{value}**\n' for stage, value in verdicts.items())
    md += '\n竞争特征消融（有竞争特征减无竞争特征）：\n\n| Task | Weather | pairwise accuracy Δ | regret Δ |\n|---|---|---:|---:|\n'
    for task, result in (('classification', s1), ('regression', s2)):
        for weather, effect in result['competition_feature_effect'].items():
            md += f"| {task} | {weather} | {effect['pairwise_accuracy_difference']:+.4f} | {effect['regret_difference']:+.4f} |\n"
    md += '\n## 目前能得出什么结论\n\n'
    md += '情况 C/D 比较 Assisted-Task 与 Proposal-Task-Greedy，使用相同学习动作和 Greedy 门槛；Conservative 单独用于 S3_PASS 判定，避免把保守阈值造成的少修改归因给候选竞争。\n\n'
    for code, conclusion, _ in findings:
        md += f'- 情况 {code}：{conclusion}\n'
    md += '\nS1/S2 所有标准化、权重和 margin 门槛均来自固定 train 场景。S3-B 推理决策中的 GT 字段为 0；没有访问正式 test。原 Shared 完整 validation 索引的 AP30/50/70 均已通过 1e-6 复现检查。\n'
    md += '\n## 还不能得出什么结论\n\n'
    md += ('这是多次用于开发的 B0 小样本 validation，不能外推为独立测试性能或最终论文结论。'
           '最佳来源不同率含同效来源的选择影响，不能当作任务分离 AP 增量。'
           '单动作独立标签、整帧多动作、联合分类/回归和帧顺序 AP 之间仍有差异；'
           'Assisted 与 Proposal 的差不能单独证明 NMS 是唯一根因。'
           '线性探针弱只限定当前特征与线性模型；不能证明所有推理信息都没有可学习信号。\n')
    md += '\n## 下一步应该验证什么\n\n'
    for _, _, next_step in findings:
        md += '- ' + next_step + '\n'
    md += '\n保持本次预注册门槛；不要在看到这些 validation 结果后自动调阈值或重跑挑结果。完整 AP30/AP50、背景指标和重叠冲突计数见 S0–S3_RESULTS.md/json。\n'
    report = {'verdicts': verdicts, 'interpretations': [{'case': code, 'conclusion': conclusion, 'next_step': next_step}
                                                       for code, conclusion, next_step in findings],
              'task_source_disagreement_rate': disagreement, 'assisted_task_adverse_gain_pp': assisted_gain,
              'proposal_greedy_adverse_gain_pp': proposal_greedy_gain,
              'proposal_conservative_adverse_gain_pp': proposal_gain,
              'interpretation_comparison': ['Assisted-Task', 'Proposal-Task-Greedy'],
              'conditions': conditions, 'stage_results': {'S0': s0, 'S1': s1, 'S2': s2, 'S3': s3},
              'test_data_used': False, 'inference_gt_fields': 0}
    atomic_json(run / 'final_results.json', report)
    atomic_text(run / 'FINAL_RESULTS.md', md)


def main():
    args = parser(__doc__).parse_args()
    summarize(args.run)


if __name__ == '__main__':
    main()
