"""Pre-registered ablation tables and restrained six-case research conclusion."""
from pathlib import Path
import numpy as np
from .common import WEATHERS, contract, read_json, report, atomic_json


def ablations(run):
    contract(run)
    tasks = {task: read_json(Path(run)/(name+'_RESULTS.json')) for task, name in (('GEO', 'S4_GEO'), ('SEM', 'S4_SEM'))}
    value = {'primary_group': 'all_evidence', 'tasks': tasks, 'validation_model_selection': False,
             'loo_comparison': ['all_evidence', 'all_evidence_without_loo'],
             'gspr_comparison': [['output_plus_gspr', 'output_only'], ['all_evidence', 'all_evidence_without_gspr']],
             'comparisons': {}}
    md = '# S4 机制消融\n\n主配置运行前固定为 all_evidence；所有消融均沿用旧标签、场景和线性训练方法。\n\n'
    md += '| Task | Group | adverse pairwise | adverse regret | adverse action rate | Evidence decision |\n|---|---|---:|---:|---:|---|\n'
    def means(result):
        rows = [result['conditions'][w]['linear'] for w in ('fog', 'rain', 'snow')]
        accuracy = [r['pairwise_ranking_accuracy'] for r in rows]
        return (None if any(a is None for a in accuracy) else float(np.mean(accuracy)),
                float(np.mean([r['mean_action_regret'] for r in rows])),
                float(np.mean([r['action_rate'] for r in rows])))
    def fmt(x):
        return 'NA' if x is None else f'{x:.6f}'
    for task, task_result in tasks.items():
        for group, result in task_result['groups'].items():
            md += f"| {task} | {group} | {' | '.join(fmt(x) for x in means(result))} | {result['decision']['verdict']} |\n"
    for key, heading in (('loo', '移除 LOO：来源边际贡献'), ('gspr', '移除 GSPR：可靠性证据')):
        md += f'\n## {heading}\n\n| Task | Feature configuration | adverse pairwise | adverse regret | adverse action rate |\n|---|---|---:|---:|---:|\n'
        for task, task_result in tasks.items():
            comparison_groups = ('all_evidence', 'all_evidence_without_'+key) if key == 'loo' else ('output_only', 'output_plus_gspr', 'all_evidence', 'all_evidence_without_gspr')
            for group in comparison_groups:
                md += f"| {task} | {group} | {' | '.join(fmt(x) for x in means(task_result['groups'][group]))} |\n"
            comparisons = [('all_evidence', 'all_evidence_without_'+key)]
            if key == 'gspr':
                comparisons.append(('output_plus_gspr', 'output_only'))
            for with_group, without_group in comparisons:
                a, b = means(task_result['groups'][with_group]), means(task_result['groups'][without_group])
                value['comparisons'][f'{task}/{with_group}-vs-{without_group}'] = {
                    'pairwise_gain': None if a[0] is None or b[0] is None else a[0]-b[0],
                    'regret_difference': a[1]-b[1], 'relative_regret_reduction': (b[1]-a[1])/b[1] if b[1] > 0 else None,
                    'action_rate_difference': a[2]-b[2]}
    md += '\n每种任务/天气/方法的完整指标、背景与关联分层和权重归因保存在 JSON。对比为预注册机制检查，不能通过 validation 选择主方法。\n'
    report(run, 'S4_ABLATIONS', value, md)


def case_decision(geo_found, sem_found, assisted_mean, meaningful_gate, proposal_pass):
    if geo_found and not sem_found:
        return 1, '几何证据能改善回归来源判断；分类仍需更多语义或交互诊断。'
    if sem_found and not geo_found:
        return 2, '语义证据能改善分类来源判断；当前几何证据仍不足。'
    if geo_found and sem_found and proposal_pass:
        return 4, '两类证据均达到门槛且 Proposal 回放通过，可进入正式 evidence-to-utility 方法设计。'
    if geo_found and sem_found and assisted_mean >= meaningful_gate:
        return 3, '两类证据可学习且 Assisted 有收益，Proposal 未通过；下一步检查候选竞争和 collateral harm。'
    if not geo_found and not sem_found:
        return 5, '本轮冻结证据与线性模型均未达到门槛；不能据此宣称来源效用不可学习。下一步检查非线性交互与联合动作。'
    if assisted_mean <= 0:
        return 6, '独立动作排序改善，但 Assisted AP 没有提高；下一步研究 joint utility。'
    return 0, '两类排序均达到门槛，Assisted 有小幅正增益但未达到预设的明显收益门槛；保留为边界情况，不能强行归入 Case 3 或 Case 6。'


def final(run):
    protocol = contract(run)
    geo = read_json(Path(run)/'S4_GEO_RESULTS.json')
    sem = read_json(Path(run)/'S4_SEM_RESULTS.json')
    replay = read_json(Path(run)/'S4_REPLAY_RESULTS.json')
    assisted = float(np.mean([replay['conditions'][w]['Evidence-Assisted-Task']['delta_ap_pp_vs_Shared']['ap70'] for w in ('fog', 'rain', 'snow')]))
    case, interpretation = case_decision(geo['decision']['verdict'] == 'GEO_EVIDENCE_FOUND',
                                         sem['decision']['verdict'] == 'SEM_EVIDENCE_FOUND', assisted,
                                         protocol['base_protocol']['config']['assisted_meaningful_gain_pp'],
                                         replay['decision']['verdict'] == 'S3_PASS')
    value = {'schema': 1, 'primary_group': 'all_evidence', 'case': case, 'interpretation': interpretation,
             'geo': geo['decision'], 'sem': sem['decision'], 'replay': replay['decision'],
             'assisted_adverse_mean_ap70_gain_pp': assisted, 'identity': protocol['identity'],
             'inference_gt_fields': 0, 'test_data_used': False,
             'raw_point_evidence': 'Retained voxel slots only; unretained raw points are not reconstructed.',
             'limitations': ['Linear probe failure is not proof of unlearnability.',
                            'Large removal sensitivity is not evidence of correct contribution.',
                            'Independent action consequences do not guarantee joint AP.',
                            'Weight attribution is descriptive, not causal.',
                            'Online weather and selected development frames follow B0; no formal test claim.',
                            'Ego/query-self removal is undefined and masked, not silently assigned a measured zero.']}
    ranking_improved = any(result['decision']['checks']['pairwise_gain_vs_output'] and result['decision']['checks']['regret_reduction_vs_output'] for result in (geo, sem))
    value['case6_also_applicable'] = bool(ranking_improved and assisted <= 0)
    geo_found = geo['decision']['verdict'] == 'GEO_EVIDENCE_FOUND'
    sem_found = sem['decision']['verdict'] == 'SEM_EVIDENCE_FOUND'
    value['replay_verdict'] = 'REPLAY_PASS' if replay['decision']['verdict'] == 'S3_PASS' else 'REPLAY_FAIL'
    atomic_json(Path(run)/'final_results.json', value)
    md = '# Direction B S4 Source Evidence Audit\n\n## 一句话结论\n\n'
    md += f"Geometry evidence {'找到了' if geo_found else '本轮没找到足够稳定的信号'}；Semantic evidence {'找到了' if sem_found else '本轮没找到足够稳定的信号'}。**Case {case}**：{interpretation}\n\n"
    md += f"GEO：{geo['decision']['verdict']}；SEM：{sem['decision']['verdict']}；Proposal：{value['replay_verdict']}。Assisted-Task 恶劣天气平均 ΔAP70 = {assisted:+.4f} pp。\n\n"
    def fmt(x):
        return 'NA' if x is None else f'{x:.6f}'
    for heading, task_result in (('S4-GEO', geo), ('S4-SEM', sem)):
        md += f'## {heading}\n\n| Features | Fog Pair Acc | Rain | Snow | Mean | Regret | Verdict |\n|---|---:|---:|---:|---:|---:|---|\n'
        for group, result in task_result['groups'].items():
            rows = [result['conditions'][w]['linear'] for w in ('fog', 'rain', 'snow')]
            acc = [r['pairwise_ranking_accuracy'] for r in rows]
            mean = None if any(x is None for x in acc) else float(np.mean(acc))
            regret = float(np.mean([r['mean_action_regret'] for r in rows]))
            md += '| '+ ' | '.join([group]+[fmt(x) for x in acc+[mean, regret]]+[result['decision']['verdict']])+' |\n'
        md += '\n'
    md += '## Evidence ablation\n\n主配置固定 all_evidence。output_only、+geometry、+semantic、+gspr、+geometry+semantic、all-no-LOO 和 all-no-GSPR 均采用同一训练配方。\n\n'
    ablation = read_json(Path(run)/'S4_ABLATIONS.json')
    md += '| Task/comparison | Pairwise gain | Regret difference | Relative regret reduction |\n|---|---:|---:|---:|\n'
    for comparison, row in ablation['comparisons'].items():
        md += f"| {comparison} | {fmt(row['pairwise_gain'])} | {fmt(row['regret_difference'])} | {fmt(row['relative_regret_reduction'])} |\n"
    md += '\n## Replay\n\n| Method | Clean AP70 | Fog | Rain | Snow | adverse ΔAP70 pp | recovered/lost/newFP | changed |\n|---|---:|---:|---:|---:|---:|---|---:|\n'
    for method in replay['conditions']['clean']:
        rows = [replay['conditions'][w][method] for w in WEATHERS]
        gains = [replay['conditions'][w][method]['delta_ap_pp_vs_Shared']['ap70'] for w in ('fog', 'rain', 'snow')]
        counts = {k: sum(replay['conditions'][w][method]['counts'].get(k, 0) for w in ('fog', 'rain', 'snow')) for k in ('recovered', 'lost', 'new_fp', 'changed_proposals')}
        md += '| '+ ' | '.join([method]+[fmt(row['ap70']) for row in rows]+[fmt(float(np.mean(gains))), f"{counts['recovered']}/{counts['lost']}/{counts['new_fp']}", str(counts['changed_proposals'])])+' |\n'
    md += '\n计数汇总 Fog/Rain/Snow；Clean 单独受掉点门槛约束。全部天气 AP30/50/70、动作数、冲突和 cell 数见 S4_REPLAY_RESULTS.md/json。\n\n'
    md += '## 机制判断\n\n'
    md += f"{geo['decision']['verdict']} / {sem['decision']['verdict']} / {value['replay_verdict']}。主方法没有按 validation 重新选择。标准化权重只表示模型依赖或关联，不能解释为因果。\n\n"
    md += '| Task | output | geometry | gspr | semantic | leave-one-out | single-query | competition |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for label, task_result in (('GEO', geo), ('SEM', sem)):
        weights = task_result['groups']['all_evidence']['attribution']['sum_absolute_standardized_weights']
        md += '| '+ ' | '.join([label]+[fmt(weights[k]) for k in ('output', 'geometry', 'gspr', 'semantic', 'loo', 'singlequery', 'competition')])+' |\n'
    md += '\n## 目前能得出什么结论\n\n'+interpretation+'\n\n'
    if geo_found:
        md += '回归来源效用在本轮固定线性比较中出现稳定可学习信号；旧 output-level geometry 没有保留全部有用信息。具体是哪类证据提供增量，需要结合预注册消融判断。\n\n'
    if sem_found:
        md += '局部语义与边际证据提供了分类来源判断的额外信息；最终 score 无法概括全部局部证据。具体作用以消融结果为界。\n\n'
    md += '## 还不能得出什么结论\n\n不能把线性失败解释为不可学习，不能把删除后的变化幅度解释为正确贡献，不能从权重大小推出因果。不能把单动作排序提升直接等同于联合 AP 收益，也没有正式 test 结论。\n\n'
    md += '## 下一步应该验证什么\n\n'+interpretation+'\n\n'
    md += '完整 ranking、LOO/GSPR 消融和整帧 AP 分别见 S4_GEO_RESULTS.md、S4_SEM_RESULTS.md、S4_ABLATIONS.md、S4_REPLAY_RESULTS.md。\n\n'
    md += '本轮只增加来源证据。旧 S0 标签、候选、动作、fit/calibration/validation 场景、线性层、30 个 epoch 和训练超参数保持一致；R0 先复现旧 S1/S2。归一化只看 fit，阈值只看 train calibration。\n\n'
    md += 'GT 用于旧标签、Assisted 候选关联和最终评价；inference feature GT=0。S4 不访问正式 test。使用在线天气，不能当作正式固定天气 benchmark。\n\n'
    md += '几何统计使用 Shared 预测框中的保留点和 pillar；没有重建体素化丢弃的原始点。空 ROI、零向量 cosine、无其他来源和不可定义的移除均有 validity 标记。B0 learned Shared 的 LOO 与原始 AttFuse LOO 分开记录。\n\n'
    from .common import atomic_text
    atomic_text(Path(run)/'FINAL_RESULTS.md', md)


if __name__ == '__main__':
    from .common import parser
    cli = parser(__doc__)
    cli.add_argument('--mode', choices=('ablations', 'final'), required=True)
    args = cli.parse_args()
    (ablations if args.mode == 'ablations' else final)(args.run)
