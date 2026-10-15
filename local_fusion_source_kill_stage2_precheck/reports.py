"""Plain-language reports; unavailable values and proxy counts remain explicit."""
from .common import SPLITS, WEATHERS


def fmt(value, percent=False):
    if value is None:
        return 'N/A'
    if percent:
        return f'{value:.2%}'
    return f'{value:.2f}' if isinstance(value, float) else str(value)


def source_report(result):
    text = '# P0-A Source Redundancy Audit\n\n## 一句话结论\n\n' + result['conclusion'] + '\n\n'
    if not result['available']:
        return text + ('汇总 JSON 缺少每个 group 的完整 discrete6/action/source 集合；'
            'outcome8 的 top tie 数无法代替 discrete6 最佳集。需在服务器 CPU 读取原 S0 cache 后补齐。'
            'unique-source、cross-source、family redundancy 与 N2 先验均暂不下结论。\n')
    text += ('每个 source ID 为当前帧内 CAV 索引；query:i 仍使用全部来源，只以 i 为 query。'
             '相同最佳 discrete6 不保证框身份逐位相同或物理机制相同。\n\n'
             '| Scope | Beneficial | unique source | ≥2 sources | best actions 1/2/≥3 | best sources 1/2/≥3 | '
             'action mean/median/p90 | single/query/both | same-source both | cross-family only-one-source | scenes/max share |\n'
             '|---|---:|---:|---:|---|---|---|---|---:|---:|---|\n')
    for key, row in sorted(result['cells'].items()):
        if 'adverse equal-weight' in key:
            continue
        actions, sources = row['equivalent_best_actions'], row['unique_sources']
        family = row['family_counts']
        coverage = row['scene_coverage']
        text += (f"| {key} | {row['beneficial_discrete_groups']} | {fmt(row['unique_source_best_fraction'], True)} | "
                 f"{fmt(row['multi_source_best_fraction'], True)} | "
                 + '/'.join(str(actions[k]) for k in ('exactly_1', 'exactly_2', 'at_least_3')) + ' | '
                 + '/'.join(str(sources[k]) for k in ('exactly_1', 'exactly_2', 'at_least_3')) + ' | '
                 + '/'.join(fmt(actions[k]) for k in ('mean', 'median', 'p90')) + ' | '
                 + '/'.join(str(family[k]) for k in ('single_only', 'query_only', 'single_and_query'))
                 + f" | {family['same_source_cross_family']} | {family['same_source_cross_family_only']} | "
                 + f"{coverage['scene_count']}/{fmt(coverage['max_scene_share'], True)} |\n")
    text += ('\n## adverse weather equal-weight\n\n'
             '| Split | Task | unique-source fraction | multi-source fraction |\n|---|---|---:|---:|\n')
    for split in SPLITS:
        for task in ('cls', 'reg'):
            row = result['cells'][f'{split}/adverse equal-weight/{task}/associated']
            text += f"| {split} | {task} | {fmt(row['unique_source_best_fraction'], True)} | {fmt(row['multi_source_best_fraction'], True)} |\n"
    text += '\n## N2 先验与 family 冗余\n\n'
    for task, row in result['N2_prior_by_task'].items():
        text += (f"- {task}：{row['interpretation']} train/validation 的 multi-source fraction 为 "
                 f"{fmt(row['train_multi_source_fraction'], True)}/{fmt(row['validation_multi_source_fraction'], True)}；"
                 '只有同一 CAV 的 single+query 都最佳的 beneficial group 占比为 '
                 f"{fmt(row['unique_source_cross_family_only_fraction'], True)}。\n")
    text += ('\n每 scene 计数、所有类别 scene 覆盖与集中度、action/source 完整分布见 JSON。'
             'family 的 single/query/both 为互斥三类；same-source cross-family 可与 cross-source 同时存在。'
             '`MODIFY_STABLE_SOURCE_AMBIGUOUS` 在本报告只表示缓存状态下离散优于 KEEP 且最佳集跨 CAV，'
             '尚未执行扰动稳定性检验。它的数量等于 multi-source beneficial groups。\n\n'
             '对 N2 的先验：跨 source 等价最佳越常见，唯一正确 CAV 的主张越弱，N2 更容易获得类似结果；'
             '单 source 最佳较多时可增加 N2 PASS 的先验倾向，但随机混合仍须实测，不能提前判断 K1。\n')
    return text


def adequacy_report(value):
    text = '# P0-B Data Adequacy Audit\n\n## 一句话结论\n\n' + value['conclusion'] + '\n\n'
    text += ('## H1：原 S0 单任务 beneficial discrete groups\n\n'
             'associated 为主结果，background 在各自行单列；cls/reg 不相加。scene ID 带 split 前缀。\n\n'
             '| Scope | Total groups | Positives | Fraction | Unique-source | Multi-source | Scenes | Max scene share |\n'
             '|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for key, row in sorted(value['H1'].items()):
        coverage = row['scene_coverage']
        if 'adverse equal-weight' in key:
            coverage = coverage['pooled_coverage_for_context']
        text += (f"| {key} | {fmt(row['total_groups'])} | {fmt(row['beneficial_discrete_groups'])} | "
                 f"{fmt(row['positive_fraction'], True)} | {fmt(row['unique_source_beneficial_groups'])} | "
                 f"{fmt(row['multi_source_equivalent_beneficial_groups'])} | {coverage['scene_count']} | "
                 f"{fmt(coverage['max_scene_share'], True)} |\n")
    text += ('\nN/A 表示缺少原 cache/record、train Oracle 未生成或无正例导致比例不可定义，具体原因见 JSON。'
             'equal-weight 行的 n 是逐天气均值，scene 列给 pooled 覆盖作为背景；不当作新增样本。'
             '完整每 scene 计数保存在 JSON。跨天气重复候选的去重数只有原 cache 可提供。\n\n'
             '## H2：真实 Oracle 统计与可用性\n\n'
             'Task−Same recovered 为两条独立路径的净数量差，无法提供 Task-only GT 身份或同状态正例数。'
             '“selected separated”包含同 CAV 不同 family 和 KEEP/来源分开，不能直接当成跨 CAV 正例。\n\n'
             '| Variant/scope | Same recovered | Task recovered | Extra recovered net | Selected separated | '
             'Same selected A | Task selected A | Same-state true task-positive | Joint Same positive targets | Scenes/max share |\n'
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n')
    for variant, cells in value['H2'].items():
        for key, row in sorted(cells.items()):
            coverage = row.get('scene_coverage')
            if coverage and 'pooled_coverage_for_context' in coverage:
                coverage = coverage['pooled_coverage_for_context']
            scene = (str(coverage['scene_count']) + '/' + fmt(coverage['max_scene_share'], True)) if coverage else '待 record / train 未生成'
            fields = ('same_recovered', 'task_recovered', 'task_extra_recovered_targets',
                      'task_separated_selected_actions', 'same_selected_A', 'task_selected_A',
                      'true_task_improvement_targets', 'joint_same_positive_target_groups')
            text += '| ' + variant + '/' + key + ' | ' + ' | '.join(fmt(row.get(k)) for k in fields) + ' | ' + scene + ' |\n'
    text += ('\nSame-state true task-positive 由保存 Task current-state shortlist 的最佳 Task discrete6 '
             '严格优于 KEEP+Same 候选最佳 discrete6 定义；仅在原记录可用时计算。'
             '这是 shortlist 条件下的真实 outcome 比较，不能冒充穷举最优。'
             '联合 Same 表的统计单位是 GT-target ROI/current-state，不能冒充 S0 Shared candidate 的联合反事实。'
             '原 S0 只有 cls-only/reg-only outcome，joint Same candidate-level 标签未生成，本轮不运行 detector 来补造。\n\n'
             '## Task-separation proxy（辅助，不进入真实正例）\n\n')
    if value['task_separation_proxy'] is None:
        text += '待 S0 cache；原汇总的来源不同率不作为 true task-positive。\n'
    else:
        text += ('两任务都分别离散优于 KEEP，且最佳集合不相交；分别按 action name 和 CAV ID 计数。'
                 '这只描述单任务最优集合冲突，没有实际执行联合动作。\n\n'
                 '| Scope | Both individually beneficial | Disjoint best actions | Disjoint best CAVs |\n'
                 '|---|---:|---:|---:|\n')
        for key, row in sorted(value['task_separation_proxy'].items()):
            text += f"| {key} | {fmt(row['both_individually_beneficial'])} | {fmt(row['proxy_disjoint_best_action_sets'])} | {fmt(row['proxy_disjoint_best_cav_sets'])} |\n"
    text += ('\n## 能否据此进入 K2/K3\n\n'
             '当前仅盘点样本；没有最终 adequacy PASS/FAIL。Same 与 Task 应分别讨论 minimum-n 和 scene 约束。'
             'train Task 真标签缺失时，Task 学习样本量保持未知；train+validation true Task pooled 也保持未知。'
             '单列 observed_validation_* 只表示已知验证部分。'
             '建议候选门槛见 DATA_ADEQUACY_RECOMMENDATION.md，须人工冻结后才可实现下一阶段。\n')
    return text


def final_report(result):
    text = '# Source Utility Stage-2 Precheck\n\n## 一句话结论\n\n' + result['conclusion'] + '\n\n'
    text += (f"状态：{result['status']}。K1-0 = A-dominant descriptive decomposition；"
             '本轮没有 K1-0 PASS/FAIL 判定。K3 与所有新模型开发继续冻结。\n\n'
             'P0-A：' + result['P0_A']['conclusion'] + '\n\n'
             'P0-B：' + result['P0_B']['conclusion'] + '\n\n'
             '## 当前能回答与仍缺失的证据\n\n')
    for item in result['limitations']:
        text += '- ' + item + '\n'
    text += ('\n## 下一步需人工决定\n\n'
             '请根据 redundancy、真实同状态 Task 正例、scene 集中度选择 K2-Same/Task minimum-n，'
             '并分别讨论未来 K3-Same/Task 训练正例门槛。候选值见 DATA_ADEQUACY_RECOMMENDATION.md。'
             '本轮到此停止，没有训练、GPU 推理、K1、K2 perturbation 或 K3。\n')
    return text
