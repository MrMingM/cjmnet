"""Descriptive protocol differences for identical cached predictions."""
from support import CONDITIONS, THRESHOLDS


def difference(left, right):
    return {k: dict(AP=left[k] - right[k], pp=100 * (left[k] - right[k])) for k in left}


def summarize(results, contract, evidence):
    effects = {}
    for condition in CONDITIONS:
        ap = results[condition]['AP']
        row = dict(NoneSkipEffect=difference(ap['O'], ap['O*']), GTFilterEffect=difference(ap['O*'], ap['F']))
        if condition != 'clean':
            clean = results['clean']['AP']
            official = {k: clean['O'][k] - ap['O'][k] for k in ap['O']}
            fixed = {k: clean['F'][k] - ap['F'][k] for k in ap['F']}
            row['OfficialWeatherGap'] = {k: dict(AP=v, pp=100*v) for k, v in official.items()}
            row['FixedWeatherGap'] = {k: dict(AP=v, pp=100*v) for k, v in fixed.items()}
            row['GapMasking'] = difference(fixed, official)
            row['GapMasking_fraction_of_fixed_gap'] = {k: row['GapMasking'][k]['AP'] / fixed[k] if fixed[k] > 0 else None for k in fixed}
            clean_skip = difference(clean['O'], clean['O*'])
            for k in official:
                decomposed = row['GTFilterEffect'][k]['AP'] + row['NoneSkipEffect'][k]['AP'] - clean_skip[k]['AP']
                if abs(row['GapMasking'][k]['AP'] - decomposed) > 1e-12:
                    raise ValueError('GapMasking algebraic decomposition does not close')
        effects[condition] = row
    full = len(contract['frame_keys']) == 829
    reference = evidence['b']['d1a_summary']
    clean_delta = difference(results['clean']['AP']['O'], reference['AP']) if full else None
    aligned = all(abs(v['AP']) <= 1e-6 for v in clean_delta.values()) if full else None
    return dict(status='complete', stage='D-1C', full_validation=full,
        validation_frame_count_per_condition=len(contract['frame_keys']),
        contract_identity=contract['identity'], results=results, effects=effects,
        validation=dict(official_O_replay_matches_original_statistics_and_AP=True,
            same_prediction_cache_used_by_all_protocols=True, clean_Ostar_equals_F=True,
            O_vs_Ostar_matching_flags_identical=True, removed_and_retained_partition_F_TP=True,
            clean_O_vs_d1a_delta=clean_delta, clean_O_matches_d1a_within_1e_6=aligned,
            clean_prediction_count_vs_d1a=results['clean']['counts']['prediction_count'] - reference['prediction_total'] if full else None,
            clean_none_frame_count_vs_d1a=results['clean']['counts']['none_frames'] - reference['official_skipped_frame_count'] if full else None),
        interpretation='Descriptive differences of evaluation protocols applied to identical predictions; not a causal statistical estimate',
        scientific_status='SMOKE_ONLY' if not full else 'READY_FOR_INTERPRETATION' if aligned else 'CLEAN_O_DIFFERS_FROM_D1A_CHECK_BEFORE_INTERPRETING')


def write_results(output, summary):
    results, effects = summary['results'], summary['effects']
    lines = ['# D−1C：同一预测下的官方评测协议差异', '',
        f'范围：每条件 {summary["validation_frame_count_per_condition"]}/829 帧；**{summary["scientific_status"]}**。',
        '同一个官方 V2X-R LiDAR AttFuse checkpoint；每条件每帧只生成一份最终预测，O/O*/F共用其缓存。',
        '官方 dataset/model/native ROI/postprocess/eval 源码保持原样。没有训练或阈值调整。', '',
        '| Condition | Official GT | Fixed GT | Removed by weather | None frames | Official GT in None frames | Fixed GT in None frames |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for condition, result in results.items():
        c = result['counts']
        lines.append(f'| {condition} | {c["official_gt"]} | {c["fixed_gt"]} | {c["removed_by_weather"]} | {c["none_frames"]} | {c["official_gt_in_none_frames"]} | {c["fixed_gt_in_none_frames"]} |')
    lines += ['', '| Condition | Protocol | AP30 | AP50 | AP70 | Counted GT |', '|---|---|---:|---:|---:|---:|']
    for condition, result in results.items():
        for protocol, ap in result['AP'].items():
            lines.append(f'| {condition} | {protocol} | {ap["ap30"]:.8f} | {ap["ap50"]:.8f} | {ap["ap70"]:.8f} | {result["counted_gt"][protocol]} |')
    lines += ['', '下表每格为 **AP差 / 百分点pp**，保留正负号。', '',
        '| Condition | Metric | AP30 / pp | AP50 / pp | AP70 / pp |', '|---|---|---:|---:|---:|']
    for condition, row in effects.items():
        for name in ('NoneSkipEffect', 'GTFilterEffect', 'OfficialWeatherGap', 'FixedWeatherGap', 'GapMasking'):
            if name not in row:
                continue
            values = [f'{row[name][k]["AP"]:+.8f} / {row[name][k]["pp"]:+.4f}' for k in ('ap30', 'ap50', 'ap70')]
            lines.append(f'| {condition} | {name} | ' + ' | '.join(values) + ' |')
    lines += ['', '删除/保留目标的检出率：都取同一次 Protocol F 全部 GT 的官方一对一匹配；同一预测不会在两组重复使用。', '',
        '| Condition | GT group | Total | IoU≥0.3 detected / recall | IoU≥0.5 detected / recall | IoU≥0.7 detected / recall |',
        '|---|---|---:|---:|---:|---:|']
    for condition in ('fog', 'snow'):
        c = results[condition]['counts']
        for group in ('removed', 'retained'):
            values = [f'{c[group+"_detected"][str(t)]} / {100*c[group+"_recall"][str(t)]:.4f}%' if c[group+"_recall"][str(t)] is not None else '0 / N/A' for t in THRESHOLDS]
            lines.append(f'| {condition} | {group} | {c[group+"_gt"]} | ' + ' | '.join(values) + ' |')
    lines += ['', '协议精确定义：',
        '- O / OFFICIAL_EXACT：使用该条件原版最终GT，预测None时整帧跳过；非None空数组正常计分。',
        '- O* / OFFICIAL_GT_NO_SKIP：使用与O逐帧同一GT，None视为0预测，GT正常进入分母。',
        '- F / OFFICIAL_CLEAN_FIXED_GT：固定同帧官方Clean已选择的目标ID和几何，所有天气使用同一GT；None计0预测且保留GT分母。',
        '三者均直接调用固定版本eval_utils_opv2v，BEV IoU、VOC2010、global_sort=False、原帧顺序和逐帧分数排序。', '',
        'NoneSkipEffect = O − O*；GTFilterEffect = O* − F。',
        'OfficialWeatherGap = Clean_O − Weather_O；FixedWeatherGap = Clean_F − Weather_F。',
        'GapMasking = FixedWeatherGap − OfficialWeatherGap。',
        '因此 GapMasking = Weather_GTFilterEffect + Weather_NoneSkipEffect − Clean_NoneSkipEffect。',
        'GapMasking同时包含两种协议差异，不能全部归因于GT过滤；GT过滤的直接对照是O*−F。', '',
        '结果理解：']
    for condition in ('fog', 'snow'):
        row = effects[condition]
        lines.append(f'- {condition} AP70：None跳帧差 {row["NoneSkipEffect"]["ap70"]["pp"]:+.4f} pp；'
                     f'GT集合差 {row["GTFilterEffect"]["ap70"]["pp"]:+.4f} pp；'
                     f'天气下降被遮蔽量 {row["GapMasking"]["ap70"]["pp"]:+.4f} pp。')
    lines += ['正且较大的GTFilterEffect和GapMasking支持官方天气评测明显低估在固定Clean目标上的退化。',
        '若三档GTFilterEffect与GapMasking均接近0，则即使删除目标很多，当前预测下的AP实际协议差异仍小；需结合removed/retained recall与FP变化解释。',
        '删除目标检出率明显大于0，说明当前预测中部分零框内点目标仍被检出；本实验无法据此确定具体依赖了上下文还是协同来源。',
        '删除目标几乎全漏检也不能推出AP影响小：这些漏检是否进入分母仍可能显著影响AP。',
        '不预设效应正负，不以删除GT比例代替AP效应，也不把描述性差异称为因果统计估计。', '',
        '完整性检查：原版推理循环O与缓存回放O的TP/FP/GT/分数序列及AP一致；Clean O*=F；所有GT含None帧。',
        'D−1A full Clean重复对齐容差预先固定1e-6；若未对齐，先检查运行证据，不更改GT定义/阈值或重生成已缓存预测。']
    if summary['full_validation']:
        lines += [f'D−1A Clean差异（当前O−历史D−1A）：{summary["validation"]["clean_O_vs_d1a_delta"]}。']
    lines += ['每帧共同目标的几何容差1e-4m；数值差与逐位一致状态见summary.results.*.counts。',
        '保留GT原始顺序；匹配分数与最大IoU并列均遵循官方实现，不另设ID优先规则。',
        '逐目标匹配与预测文件哈希见per_frame_metrics.jsonl、removed_gt_metrics.jsonl和prediction_manifest.json。', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
