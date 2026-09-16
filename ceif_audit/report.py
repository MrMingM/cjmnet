"""Report measured effects without turning hindsight into deployable performance."""
import argparse
import json
from pathlib import Path

LABELS = {
    'baseline': '冻结基线', 'full_reference': '全通信参照',
    'generic_fp_oracle70': '真值删除全部 FP70（宽松机会参照）',
    'evidence_fp_oracle70': '只删除证据标记中的真实 FP70',
    'evidence_rule_feature': '证据规则选择特征干预（不使用 GT）',
    'pool_random_feature': '同一候选池随机特征干预（不使用 GT）',
    'evidence_hindsight_feature': '证据候选内事后选择特征干预（使用 GT）',
    'pool_hindsight_feature': '全部候选内事后选择特征干预（使用 GT）',
}


def write_weather(summary, path):
    p, totals, results = summary['protocol'], summary['totals'], summary['results']
    lines = ['# CEIF 训练前可行性审查', '',
        f"阶段：{p['phase']}；天气：{p['weather']}；帧数：{totals['frames']}；基线：{p['baseline']}。",
        f"数据：`{p['data_root']}`；在线天气：{p['online_weather']}；AP：非全局排序、平面 IoU。", '',
        '**这不是训练后的 CEIF 性能。事后选择使用 GT，而且只是有限局部特征替换，不是新模型的理论上限。**', '',
        '| 方法 | AP30 | AP50 | AP70 | ΔAP70 / 百分点 | TP70 | FP70 | FN70 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in results.items():
        lines.append(f"| {LABELS[name]} | {row['ap30']:.6f} | {row['ap50']:.6f} | {row['ap70']:.6f} | "
                     f"{100*(row['ap70']-results['baseline']['ap70']):+.4f} | {row['tp70']} | {row['fp70']} | {row['fn70']} |")
    lines += ['', '## 证据是否真的指向错误', '',
        f"证据标记 {totals['evidence_flagged_boxes']} 个基线框，其中 FP70={totals['flagged_false_positives']}，"
        f"TP70={totals['flagged_true_positives']}。不能只报告前者而隐去误伤风险。",
        f"实际筛查基线框 {totals['screened_base_boxes']} 个；有合格证据干预的帧 {totals['frames_with_eligible_action']}/{totals['frames']}。"
        f"平均保留端点 {totals['received_endpoints']/totals['frames']:.1f}，有效可信射线 {totals['valid_rays']/totals['frames']:.1f}。",
        f"执行 {totals['evaluated_actions']} 个有限候选干预，证据合格 {totals['eligible_actions']} 个；"
        f"合格候选中有益 {totals['useful_eligible_actions']} 个，有害 {totals['harmful_eligible_actions']} 个。",
        '有益定义：保留基线全部 GT 匹配、不增加 FP，并补回目标或减少 FP；有害定义：丢失原匹配或增加 FP。均按 IoU .7。', '',
        '## 如何读结论', '',
        '1. 全部 FP 删除涨幅大，只说明模型有误检，不能支持 CEIF。',
        '2. 证据约束的 FP 删除及特征事后干预也有收益，说明该证据覆盖了可修正机会。',
        '3. 不用 GT 的规则也有收益，是可识别性更强的信号；若规则下降，说明收益还不能被当前证据规则稳定兑现。',
        '4. 事后候选选择按匹配/FP 支配条件，不直接最大化 AP；必须看重新累计的实际 AP，不能用成功次数替代。',
        '5. 有益候选少或结果不涨，应先检查证据覆盖及候选空间；本实验不能否定尚未实现的特征重建能力。', '',
        '## 通信与覆盖边界', '',
        f"平均原特征消息总字节：{totals['feature_bytes']/totals['frames']:.1f}；"
        f"额外理想几何证据字节：{totals['geometry_extra_bytes']/totals['frames']:.1f}。",
        '几何载荷估算为每个已选块内 retained 端点 25 字节（xyz、三元意见及发送端计算的有效射线标记），每车非空证书加 64 字节位姿；'
        '未实现压缩，未包含全部协议开销，也未从原特征预算扣除，不可宣称同总带宽下收益。',
        '无点不作自由证据；不发送的源块不能提供端点、查询证据或替换特征。'
        '射线必须来自有对应关系的可信实际首回波；不会把 GT 框或 clean 配对云作为推理输入。', '',
        '## 分场景 ΔAP70 / 百分点', '',
        '| Scene | 无 GT 规则 | 证据候选事后选择 | 全候选事后选择 |', '|---|---:|---:|---:|']
    for scene, rows in summary['scene_results'].items():
        if rows['baseline']['ap70'] is not None:
            values = [100*(rows[m]['ap70']-rows['baseline']['ap70']) for m in
                      ('evidence_rule_feature', 'evidence_hindsight_feature', 'pool_hindsight_feature')]
            lines.append(f'| {scene} | {values[0]:+.4f} | {values[1]:+.4f} | {values[2]:+.4f} |')
    if p['smoke']:
        lines += ['', '**SMOKE：仅连通检查，禁止据此判断方法有效。**']
    if p['phase'] == 'benchmark-audit':
        lines += ['', '本次使用正式数据做探索性审查，含 GT 事后干预，不能报告成已训练方法的正式 test 性能；'
                  '不得依据该表继续调阈值/选择模型。后续泛化结论需要保留未调参数据。']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.run)
    summaries = {w: json.loads((root/w/'summary.json').read_text(encoding='utf-8'))
                 for w in ('clean', 'fog', 'rain', 'snow')}
    contracts = [dict(phase=s['protocol']['phase'], baseline=s['protocol']['baseline'],
        budget=s['protocol']['budget_bytes'], settings=s['protocol']['settings'], smoke=s['protocol']['smoke'],
        frontend=s['protocol']['comparison_contract'], sources=s['protocol']['audit_sources']) for s in summaries.values()]
    if any(c != contracts[0] for c in contracts):
        raise ValueError('Cannot merge different audit contracts')
    lines = ['# CEIF 四条件可行性审查', '',
        '每个天气详细解释见对应 audit.md。GT 事后干预不是已实现模型的性能。', '',
        '| Weather | Baseline AP70 | 证据 FP 清理 Δpp | 无 GT 特征规则 Δpp | 证据特征事后选择 Δpp | 全候选事后选择 Δpp |',
        '|---|---:|---:|---:|---:|---:|']
    for weather, summary in summaries.items():
        r = summary['results']; base = r['baseline']['ap70']
        deltas = [100*(r[m]['ap70']-base) for m in ('evidence_fp_oracle70', 'evidence_rule_feature',
                    'evidence_hindsight_feature', 'pool_hindsight_feature')]
        lines.append(f'| {weather} | {base:.6f} | '+ ' | '.join(f'{v:+.4f}' for v in deltas)+' |')
    lines += ['', '评判顺序：机会是否被证据覆盖 → 真正改特征是否获益 → 无 GT 规则能否识别 → '
              '是否损伤 Clean/其他天气。不能仅凭宽松 oracle 上涨宣布方法可行。',
              f"阶段：{contracts[0]['phase']}；smoke={contracts[0]['smoke']}。完整原始协议与候选日志已保存。"]
    (root/'feasibility.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(root/'feasibility.md')


if __name__ == '__main__':
    main()
