"""Human-review candidates only: no frozen rule and no automatic gate verdict."""
from .common import SPLITS


def propose(adequacy):
    candidates = {
        'K2-Same': {'minimum_positive_n_options': [50, 100, 200],
                    'preferred_starting_point': 100,
                    'count_unit': 'beneficial discrete candidate/focal groups, cls/reg separately; '
                                  'report weather appearances and deduplicated physical IDs',
                    'validation_positive_scene_options': [5, 7],
                    'max_single_scene_share_options': [.30, .40]},
        'K2-Task': {'minimum_positive_n_options': [20, 30, 50],
                    'preferred_starting_point': None,
                    'count_unit': 'true same-state discrete Task improvement target IDs; '
                                  'not Task-Same recovered difference, selected-separated count or proxy',
                    'validation_positive_scene_options': [5, 7],
                    'max_single_scene_share_options': [.30, .40]},
        'K3-Same': {'minimum_train_positive_n_options': [100, 200, 500],
                    'preferred_starting_point': 200,
                    'count_unit': 'deduplicated train beneficial candidate groups, cls/reg separate; '
                                  'count fit/calibration positives separately',
                    'train_positive_scene_options': [10, 15],
                    'max_single_scene_share_options': [.30, .40]},
        'K3-Task': {'minimum_train_positive_n_options': [50, 100, 200],
                    'preferred_starting_point': None,
                    'count_unit': 'true train joint Task-vs-Same positive targets; no proxy substitution',
                    'train_positive_scene_options': [5, 10],
                    'max_single_scene_share_options': [.30, .40]},
    }
    observations = {}
    for split in SPLITS:
        observations[split] = {}
        for task in ('cls', 'reg'):
            row = adequacy['H1'][f'{split}/adverse pooled/{task}/associated']
            observations[split][task] = {
                'positive_weather_appearances': row['beneficial_discrete_groups'],
                'unique_physical_positive_groups': row.get('unique_physical_positive_groups'),
                'scene_coverage': row['scene_coverage'],
                'count_options_support': {str(n): row['beneficial_discrete_groups'] >= n for n in (50, 100, 200, 500)},
                'not_a_gate_verdict': True}
    h2 = adequacy['H2']['A_ONLY']['validation/adverse pooled']
    task_n = h2['true_task_improvement_targets']
    if task_n is not None:
        available = [n for n in (20, 30, 50) if task_n >= n]
        candidates['K2-Task']['preferred_starting_point'] = max(available) if available else None
    return dict(status='CANDIDATES_ONLY_REQUIRES_HUMAN_FREEZE', frozen=False,
                candidates=candidates, observations=observations,
                task_true_validation_n=task_n,
                task_true_train_n=adequacy['H2']['A_ONLY']['train/adverse pooled']['true_task_improvement_targets'],
                split_K2_Same_Task_recommended=True, independent_K3_data_gates_recommended=True,
                rationale=('Same/H1 and Task/H2 use different action spaces and positive frequencies; '
                           'weather-100 is not inherited. Counts do not establish learnability or independent generalization.'),
                caveat=('Options are planning sensitivity checks, not a power calculation. '
                        'Repeated frames/weather are correlated. Sample n, scene coverage, concentration, '
                        'minimum per-weather descriptive coverage and treatment of inconclusive remain human decisions.'))


def render(value, adequacy):
    text = ('# Data Adequacy Recommendation\n\n## 一句话结论\n\n'
            '建议把 Same 与 Task 分开设样本量规则；本轮仅列候选值，所有门槛待人工冻结，K3 继续冻结。\n\n'
            '| 用途 | pooled 正例 n 候选 | scene 候选 | 单 scene 最大占比候选 |\n'
            '|---|---|---|---|\n'
            '| K2-Same | 50 / 100 / 200；优先讨论 100 | validation ≥5 / ≥7 | ≤30% / ≤40% |\n'
            '| K2-Task | 20 / 30 / 50；须先取得同状态真实正例数 | validation ≥5 / ≥7 | ≤30% / ≤40% |\n'
            '| K3-Same | train 100 / 200 / 500；优先讨论 200 | train ≥10 / ≥15 | ≤30% / ≤40% |\n'
            '| K3-Task | train 50 / 100 / 200；当前没有真实 train Task labels | train ≥5 / ≥10 | ≤30% / ≤40% |\n'
            '\n这些数字用于人工比较资源与可解释程度，不是已冻结规则，也不是统计功效保证。'
            'K2-Same 的 cls/reg 应分开计数；K3 还需单独检查 fit/calibration 分配。\n\n'
            '## 现有正例与集中度\n\n'
            '| Split | Task | adverse positive appearances | 去重物理 group | scenes | max scene share |\n'
            '|---|---|---:|---:|---:|---:|\n')
    for split, tasks in value['observations'].items():
        for task, row in tasks.items():
            coverage = row['scene_coverage']
            share = coverage['max_scene_share']
            text += (f"| {split} | {task} | {row['positive_weather_appearances']} | "
                     f"{row['unique_physical_positive_groups'] if row['unique_physical_positive_groups'] is not None else '待 cache'} | "
                     f"{coverage['scene_count']} | " + (f'{share:.2%}' if share is not None else 'N/A') + ' |\n')
    text += ('\n## 为什么不沿用每天气 100\n\n'
             'K0 的 beneficial group、Same Oracle 的 selected A、Task Oracle 的 selected A、'
             '以及同状态 Task 相对 Same 的真实额外正例是不同统计单位。不能用其中一个数量替代另一个。\n\n')
    for variant in ('FULL', 'A_ONLY'):
        for weather in ('fog', 'rain', 'snow'):
            row = adequacy['H2'][variant]['validation/' + weather]
            text += (f"- {variant} {weather}：Same selected A={row['same_selected_A']}，"
                     f"Task selected A={row['task_selected_A']}，Task−Same recovered 净差={row['task_extra_recovered_targets']}。\n")
    text += ('\n建议优先讨论 adverse pooled 的 stability 样本数，同时保留 train/validation 和逐天气表。'
             'pooled 不能冒充独立泛化证据，跨天气同一 frame/target 须另报去重数。'
             '旧 100/weather 可能筛掉现有关键动作，但本轮不因此降低或冻结任何最终门槛。\n\n'
             '## 仍待人工决定\n\n'
             '1. K2-Same 的对象选单任务 S0 candidate，还是联合 Same 的 GT-target/current-state；两者不能混计。\n'
             '2. pooled n 使用天气出现次数还是去重物理 group；是否允许 train+validation 合并作稳定性诊断。\n'
             '3. K2-Task 的 minimum-n、scene 覆盖与集中度上限，待真实同状态正例统计后冻结。\n'
             '4. K3-Same/Task 各自的 train/fit/calibration 正例要求；当前没有 train joint Task 真标签，Task 学习 adequacy 仍未知。\n'
             '5. 样本不足如何处理，以及稳定性成功率/效应量/置信度要求；本轮没有预设最终 kill rule。\n')
    return text
