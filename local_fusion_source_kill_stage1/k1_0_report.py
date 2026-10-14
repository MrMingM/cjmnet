"""Absolute pp first; independent trajectories never imply additive attribution."""
from statistics import mean
from .common import WEATHERS, ADVERSE, settings

NAMES = ('Shared', 'Full-Same', 'Full-Task', 'AOnly-Same', 'AOnly-Task', 'BOnly-Same', 'BOnly-Task')
PREFIX = {'FULL': 'Full', 'A_ONLY': 'AOnly', 'B_ONLY': 'BOnly'}


def deltas(row, metric='ap70'):
    result = {}
    for prefix in ('Full', 'AOnly', 'BOnly'):
        result[prefix + '_Task-Shared_pp'] = 100 * (row[prefix + '-Task'][metric] - row['Shared'][metric])
        result[prefix + '_Task-Same_pp'] = 100 * (row[prefix + '-Task'][metric] - row[prefix + '-Same'][metric])
    return result


def retention(full, restricted):
    if abs(full) < settings()['unstable_ratio_denominator_pp']:
        return {'status': 'UNSTABLE_RATIO', 'value': None, 'denominator_pp': full}
    return {'status': 'stable', 'value': restricted / full, 'denominator_pp': full}


def describe(delta):
    spec = settings()
    full, a, b = (delta[p + '_Task-Shared_pp'] for p in ('Full', 'AOnly', 'BOnly'))
    ar, br = retention(full, a), retention(full, b)
    material = spec['narrative_material_gain_pp']
    if full >= material and ar['value'] is not None and ar['value'] >= spec['narrative_major_retention'] and br['value'] <= spec['narrative_weak_retention']:
        return ('AOnly 保留 Full 大部分收益，BOnly 较弱：Oracle headroom 主要来自改变离散检测后果的动作。',
                '下一阶段 K1 尤其需要 N2 random source mixing，检查收益是否依赖正确来源。')
    if full >= material and br['value'] is not None and br['value'] >= spec['narrative_major_retention'] and ar['value'] <= spec['narrative_weak_retention']:
        return ('BOnly 保留 Full 大部分收益，AOnly 大幅下降：§33 Oracle 收益主要来自 score/IoU/ranking-only 操作；source-specific headroom 的解释减弱。',
                '下一阶段 K1 尤其需要 N1 matched output perturbation 与 score-only calibration。')
    if a >= material and b >= material:
        return ('A/B 两条受限路径都有明显收益，两种机制同时存在。',
                '下一阶段仍需完整验证 N1 matched output perturbation、N2 random source mixing 与 score-only calibration。')
    return ('受限路径未形成清楚的单一主导机制；应依据绝对 pp 保留混合或弱证据判断。',
            '下一阶段完整验证 N1、N2、score-only；K2 独立检查标签稳定性。')


def summarize(conditions):
    ap = {}
    for weather in WEATHERS:
        ap[weather] = {'Shared': conditions[f'{weather}_same_FULL']['Shared']}
        for variant, prefix in PREFIX.items():
            for mode in ('same', 'task'):
                ap[weather][prefix + '-' + mode.title()] = conditions[f'{weather}_{mode}_{variant}']['AP']
    ap['adverse mean'] = {name: {metric: mean(ap[w][name][metric] for w in ADVERSE)
                                for metric in ('ap30', 'ap50', 'ap70')} for name in NAMES}
    differences = {weather: deltas(row) for weather, row in ap.items()}
    ratios = {weather: {p + '_' + comparison: retention(
        delta['Full_' + comparison + '_pp'], delta[p + '_' + comparison + '_pp'])
                        for p in ('AOnly', 'BOnly') for comparison in ('Task-Shared', 'Task-Same')}
              for weather, delta in differences.items()}
    interpretation, next_stage = describe(differences['adverse mean'])
    return dict(AP=ap, deltas_pp=differences, retention=ratios, conditions=conditions,
                interpretation=interpretation, next_stage_focus=next_stage,
                boundary='decomposition 是 restricted-policy attribution，不是线性可加分解。',
                k3_frozen=True, next_stage_auto_start=False, test_data_used=False)


def main_tables(result):
    text = ('## K1-0 主表：AP70（0–1）\n\n| Weather | ' + ' | '.join(NAMES) + ' |\n'
            '|---|' + '---:|' * len(NAMES) + '\n')
    for weather, row in result['AP'].items():
        text += '| ' + weather + ' | ' + ' | '.join(f"{row[name]['ap70']:.6f}" for name in NAMES) + ' |\n'
    text += ('\n## K1-0 收益（pp）\n\n'
             '| Weather | Full Task−Shared | A Task−Shared | B Task−Shared | Full Task−Same | A Task−Same | B Task−Same |\n'
             '|---|---:|---:|---:|---:|---:|---:|\n')
    for weather, row in result['deltas_pp'].items():
        keys = [p + '_' + comparison + '_pp' for comparison in ('Task-Shared', 'Task-Same')
                for p in ('Full', 'AOnly', 'BOnly')]
        text += '| ' + weather + ' | ' + ' | '.join(f'{row[key]:+.4f}' for key in keys) + ' |\n'
    text += '\n' + result['boundary'] + ' AOnly/BOnly 的收益不相加解释为 Full。\n'
    return text


def render(result):
    text = '# K1-0 Oracle Outcome-space Decomposition\n\n' + result['interpretation'] + '\n\n'
    text += main_tables(result)
    text += '\n## 全部 AP\n\n| Weather | Variant | AP30 | AP50 | AP70 |\n|---|---|---:|---:|---:|\n'
    for weather, row in result['AP'].items():
        for name, ap in row.items():
            text += f"| {weather} | {name} | {ap['ap30']:.9f} | {ap['ap50']:.9f} | {ap['ap70']:.9f} |\n"
    text += ('\n## 动作统计\n\nrecovered/lost/new_fp 为终态相对 Shared；step_* 为逐步后果（JSON）。'
             'A/B/E 为每个 target 的当前 state 相对 KEEP 的 key 分组。\n\n'
             '| Weather/mode/variant | targets | changed | shortlisted A/B/E | selected A/B/E/KEEP | '
             'task separated | recovered/lost/new_fp | availability A/B/E | E fraction |\n'
             '|---|---:|---:|---|---|---:|---|---|---:|\n')
    def fmt(value):
        return f'{value:.2%}' if value is not None else 'N/A'
    for name, row in result['conditions'].items():
        c = row['counts']
        shortlisted = '/'.join(str(c['shortlisted_' + x]) for x in ('A', 'B', 'E'))
        selected = '/'.join(str(c['selected_' + x]) for x in ('A', 'B', 'E', 'KEEP'))
        outcome = '/'.join(str(c[x]) for x in ('recovered', 'lost', 'new_fp'))
        available = '/'.join(fmt(row['availability_rates'][x]) for x in ('A', 'B', 'E'))
        text += (f"| {name} | {c['targets']} | {c['changed']} | {shortlisted} | {selected} | "
                 f"{c['task_separated']} | {outcome} | {available} | {fmt(row['shortlisted_group_rates']['E'])} |\n")
    text += '\n## Retention（辅助描述）\n\n| Weather | Comparison | Retention |\n|---|---|---|\n'
    for weather, rows in result['retention'].items():
        for name, row in rows.items():
            text += f"| {weather} | {name} | " + (fmt(row['value']) if row['status'] == 'stable' else 'UNSTABLE_RATIO') + ' |\n'
    text += ('\nFULL 的四天气 AP30/50/70 已全部通过原 §33 结果 ≤1e-6 门禁。'
             'B 是 key 定义下的 continuous-only：未保证每个非 focal 框、score 或物理身份逐位不变。'
             'AP 使用原 frame-order 口径，score/IoU/NMS 竞争均可能影响它。'
             '不据此证明来源特异性或可学习性。\n\n' + result['next_stage_focus'] + '\nK3 继续冻结，等待人工判断。\n')
    return text
