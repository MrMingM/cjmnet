"""Human-readable group audit; primary and action-level denominators stay separate."""
from .common import WEATHERS


def primary_tables(result):
    conditions = result['conditions']
    text = ('## K0′ 主表：validation associated\n\n'
            '| Weather | Task | Groups | Discrete margin | Continuous-only | Exact tie | Beneficial-set |\n'
            '|---|---|---:|---:|---:|---:|---:|\n')
    for weather in WEATHERS:
        for task in ('cls', 'reg'):
            row = conditions[f'validation/{weather}/{task}/associated']
            text += (f"| {weather} | {task} | {row['group_count']} | {row['discrete_margin_rate']:.2%} | "
                     f"{row['continuous_only_margin_rate']:.2%} | {row['exact_outcome_tie_rate']:.2%} | "
                     f"{row['discrete_beneficial_set_rate']:.2%} |\n")
    for task, row in result['primary_adverse_macro'].items():
        text += (f"| adverse mean | {task} | — | {row['discrete_margin_rate']:.2%} | "
                 f"{row['continuous_only_margin_rate']:.2%} | {row['exact_outcome_tie_rate']:.2%} | "
                 f"{row['discrete_beneficial_set_rate']:.2%} |\n")
    text += ('\n## ACTION-LEVEL CONTEXT：action-level net TP count change\n\n'
             '仅统计 non-KEEP action；TP increase 是能直接带来净 TP 增长的 action 比例。\n\n'
             '| Weather | Task | TP increase actions | TP decrease actions | TP unchanged | Ratio decrease:increase |\n'
             '|---|---|---:|---:|---:|---:|\n')
    for weather in WEATHERS:
        for task in ('cls', 'reg'):
            risk = conditions[f'validation/{weather}/{task}/associated']['action_level_net_tp_count_change']
            ratio = risk['ratio_decrease_increase']
            text += (f"| {weather} | {task} | {risk['tp_increase']}/{risk['action_count']} "
                     f"({risk['tp_increase_rate']:.2%}) | {risk['tp_decrease']} "
                     f"({risk['tp_decrease_rate']:.2%}) | {risk['tp_unchanged_rate']:.2%} | "
                     + (f'{ratio:.2f}' if ratio is not None else 'undefined') + ' |\n')
    return text


def conclusion(result):
    macros = result['primary_adverse_macro']
    discrete = sum(r['discrete_margin_rate'] for r in macros.values()) / 2
    continuous = sum(r['continuous_only_margin_rate'] for r in macros.values()) / 2
    tie = sum(r['exact_outcome_tie_rate'] for r in macros.values()) / 2
    dominant = max(((discrete, '离散检测差异较明确'), (continuous, '连续分数/IoU 排序主导'),
                    (tie, '大量最佳动作并列')))[1]
    return f'K0′ 的恶劣天气主范围主要表现为{dominant}。'


def render(result):
    text = '# K0′ Group-level Label Margin Audit\n\n' + conclusion(result) + '\n\n'
    text += primary_tables(result)
    text += ('\n## 全 split/weather/task/stratum 指标\n\n'
             'adverse mean 为三天气等权平均；train 只检查趋势，background 单列。\n\n'
             '| Scope | Groups | Discrete | Continuous | Tie | Multi-top | Mean/median tie | '
             'Beneficial set | Continuous benefit | Harmful exists | KEEP-only |\n'
             '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|\n')
    for key, row in sorted(result['conditions'].items()):
        text += (f"| {key} | {row['group_count']} | {row['discrete_margin_rate']:.2%} | "
                 f"{row['continuous_only_margin_rate']:.2%} | {row['exact_outcome_tie_rate']:.2%} | "
                 f"{row['top_equivalent_multi_action_rate']:.2%} | {row['mean_top_tie_count']:.3f}/"
                 f"{row['median_top_tie_count']:.1f} | {row['discrete_beneficial_set_rate']:.2%} | "
                 f"{row['continuous_only_benefit_rate']:.2%} | {row['harmful_discrete_exists_rate']:.2%} | "
                 f"{row['keep_only_count']} |\n")
    text += ('\n## Scene 覆盖\n\n'
             '| Scope | All scenes/max share | Discrete scenes/max share | Beneficial scenes/max share |\n'
             '|---|---|---|---|\n')
    def coverage(row):
        share = row['max_single_scene_fraction']
        return str(row['scene_count']) + '/' + (f'{share:.2%}' if share is not None else 'N/A')
    for key, row in sorted(result['conditions'].items()):
        text += (f"| {key} | {len(row['scenes'])}/{row['single_scene_max_fraction']:.2%} | "
                 + coverage(row['discrete_margin_scene_coverage']) + ' | '
                 + coverage(row['beneficial_set_scene_coverage']) + ' |\n')
    text += (f"\n描述性信号：{result['signal']}；不构成最终 STOP。KEEP-only 按无竞争 outcome tie 计入，"
             '同时单列；并列来源完整集合见 k0_group_records.jsonl。旧 action-level 风险表已逐项复现。'
             '本审计不检验重复采样稳定性；K2 尚待人工授权。K3 继续冻结。\n')
    return text
