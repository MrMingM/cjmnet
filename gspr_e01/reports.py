"""Reports from committed machine-readable evidence; no scoring implementation."""
from pathlib import Path
from e01_common import METRICS, PROTOCOLS, SETTINGS, WEATHERS, atomic, frozen_identity, jsonl, read, reference_results, sha


def delta(a, b):
    return {w: {p: {m: 100*(a[w]['AP'][p][m]-b[w]['AP'][p][m]) for m in METRICS}
        for p in PROTOCOLS} for w in WEATHERS}


def comparisons(results):
    pairs = [('T1-D', 'T1-C'), ('T2-D', 'T2-C'), ('T0-D', 'T0-C')]
    pairs += [(t+'-'+v, 'T0-'+v) for t in ('T1', 'T2') for v in ('C', 'D')]
    out = {a+' minus '+b: delta(results[a], results[b]) for a, b in pairs}
    interactions = {t: {w: {p: {m: out[t+'-D minus '+t+'-C'][w][p][m]-out['T0-D minus T0-C'][w][p][m]
        for m in METRICS} for p in PROTOCOLS} for w in WEATHERS} for t in ('T1', 'T2')}
    return dict(differences_pp=out, gain_change_pp=interactions,
        unit='percentage points = 100 * AP difference',
        limitation='Single-seed descriptive gain changes; no established interaction causality')


def screen(results):
    gates = SETTINGS['screening_pp']
    rows = {}
    for t in ('T1', 'T2'):
        change = delta(results[t+'-D'], results['T0-D'])
        paired = delta(results[t+'-D'], results[t+'-C'])
        values = dict(snow_vs_T0_D_min=change['snow']['F']['ap70'],
            snow_D_minus_C_min=paired['snow']['F']['ap70'],
            clean_vs_T0_D_min=change['clean']['F']['ap70'], fog_vs_T0_D_min=change['fog']['F']['ap70'])
        rows[t] = dict(measured_pp=values, gates_pp=gates,
            passed={k: values[k] >= gates[k] for k in gates}, all_passed=all(values[k] >= gates[k] for k in gates))
    return dict(status='complete', recipes=rows, interpretation='Pre-registered exploratory engineering screen, not statistical significance')


def validate_summary(path, count):
    row = read(path)
    if row['status'] != 'complete' or row['frame_count'] != count or row['conditions'] != list(WEATHERS) or row['full_validation'] != (count == 829):
        raise ValueError('Incomplete three-weather evaluation: '+str(path))
    for name, digest in row['output_hashes'].items():
        if sha(Path(path).parent / name) != digest:
            raise ValueError('Frozen evaluator output changed: '+name)
    predictions = read(Path(path).parent / 'prediction_manifest.json')['files']
    if len(predictions) != 3*count or {(r['condition'], r['index']) for r in predictions} != {
            (w, i) for w in WEATHERS for i in range(count)}:
        raise ValueError('Prediction manifest incomplete')
    for record in predictions:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Prediction cache bytes changed')
    gt = read(Path(path).parent / 'gt_manifest.json')
    if len(gt['frames']) != 3*count or len({(f['condition'], f['index']) for f in gt['frames']}) != 3*count:
        raise ValueError('GT manifest incomplete')
    if count == 829 and any(row['results'][w]['counts']['fixed_gt'] != 6224
            or row['results'][w]['counts']['official_gt'] != n for w, n in zip(WEATHERS, (6224, 3673, 5854))):
        raise ValueError('Full validation GT totals differ')
    return row


def target_changes(left, right, count):
    a = {(r['condition'], r['index']): r for r in jsonl(Path(left) / 'per_frame_metrics.jsonl')}
    b = {(r['condition'], r['index']): r for r in jsonl(Path(right) / 'per_frame_metrics.jsonl')}
    if set(a) != set(b) or len(a) != 3*count:
        raise ValueError('Paired target frames differ')
    out = {}
    for w in WEATHERS:
        out[w] = {}
        for t in ('0.3', '0.5', '0.7'):
            gained = lost = both = 0
            improved = degraded = tied = 0
            for i in range(count):
                x, y = a[(w, i)], b[(w, i)]
                if x['frame_key'] != y['frame_key'] or x['fixed_gt_count'] != y['fixed_gt_count']:
                    raise ValueError('Paired GT/frame differs')
                sx = {str(m['object_id']) for m in x['protocols']['F']['matched_pairs'][t]}
                sy = {str(m['object_id']) for m in y['protocols']['F']['matched_pairs'][t]}
                gained += len(sx-sy)
                lost += len(sy-sx)
                both += len(sx & sy)
                diff = len(sx)-len(sy)
                improved += diff > 0
                degraded += diff < 0
                tied += diff == 0
            out[w][t] = dict(newly_detected=gained, lost=lost, matched_by_both=both,
                net_TP=gained-lost, frames_with_more_TP=improved, frames_with_fewer_TP=degraded,
                frames_with_equal_TP=tied)
    return out


def protocol_match(a, b):
    keys = ('official_source_hashes', 'official_config_sha256', 'input_manifest_identity',
        'frame_keys', 'protocols', 'inference_seed', 'PYTHONHASHSEED', 'loader_workers',
        'versions', 'HIP', 'torch_backend', 'frozen_d1c_gt_identity')
    missing = [k for k in keys if k not in a or k not in b]
    different = [k for k in keys if a.get(k) != b.get(k)]
    if missing or different:
        raise ValueError('Evaluation protocol differs: missing='+repr(missing)+' different='+repr(different))
    return list(keys)


def table(results, differences=False):
    lines = ['| Model / comparison | Weather | Protocol | AP30'+(' Δpp' if differences else '')+' | AP50 | AP70 |',
        '|---|---|---|---:|---:|---:|']
    for name, result in results.items():
        for w in WEATHERS:
            for p in PROTOCOLS:
                values = result[w][p] if differences else result[w]['AP'][p]
                lines.append('| '+name+' | '+w+' | '+p+' | '+' | '.join(f'{values[m]:+.6f}' if differences else f'{values[m]:.10f}' for m in METRICS)+' |')
    return lines


def e1_report(root, original, smoke=False):
    root, original = Path(root), Path(original)
    count = 12 if smoke else 829
    summaries, model_roots, histories, audit_rows, target_rows = {}, {}, {}, {}, {}
    reference, reference_source = reference_results()
    base_protocol = None
    for t in ('T1', 'T2'):
        for v in ('C', 'D'):
            key = t+'-'+v
            model = root / t / v
            selected = read(model / ('smoke_selected.json' if smoke else 'selected.json'))
            contract = read(model / 'contract.json')
            if contract['smoke_only'] != smoke or contract['identity'] != frozen_identity({k: x for k, x in contract.items() if k != 'identity'}):
                raise ValueError('Training contract identity differs')
            if (contract['variant'] != v or contract['epochs'] != 20
                    or contract['config_sha256'] != sha(model / 'config.yaml')
                    or contract['settings']['e01']['recipe'] != t):
                raise ValueError('Training variant/configuration/recipe differs')
            history = [read(model / 'epochs' / f'{e:02d}' / 'complete.json') for e in range(1, 3 if smoke else 21)]
            if len(history) != (2 if smoke else 20):
                raise ValueError('Missing committed training epochs')
            for i, epoch in enumerate(history, 1):
                if epoch['epoch'] != i or epoch['contract'] != contract['identity'] or epoch['train_updates'] != (4 if smoke else 1010):
                    raise ValueError('Training epoch or update count differs')
                for artifact in ('checkpoint', 'resume_state', 'trace', 'clean_summary'):
                    if sha(epoch[artifact]) != epoch[artifact+'_sha256']:
                        raise ValueError('Committed epoch artifact differs')
                clean = read(epoch['clean_summary'])
                if (clean['status'] != 'complete' or clean['conditions'] != ['clean'] or clean['frame_count'] != count
                        or clean['checkpoint_sha256'] != epoch['checkpoint_sha256']
                        or clean['results']['clean']['AP']['O']['ap70'] != epoch['AP70']):
                    raise ValueError('Clean selection candidate evidence differs')
                if v == 'D':
                    pairing = read(model / 'pairing' / f'{i:02d}.json')
                    if not pairing['exact'] or pairing['D_trace_sha256'] != epoch['trace_sha256']:
                        raise ValueError('C/D epoch pairing not validated')
                    peer = histories[t+'-C'][i-1]
                    if pairing['C_trace_sha256'] != peer['trace_sha256'] or pairing['consumed_frames'] != (32 if smoke else 8080):
                        raise ValueError('Paired C trace / consumed frame count differs')
            best = min(history, key=lambda r: (-r['AP70'], r['epoch']))
            if selected['checkpoint_sha256'] != best['checkpoint_sha256'] or selected['epoch'] != best['epoch']:
                raise ValueError('Clean-only selection differs')
            result = validate_summary(model / 'final_evaluation/summary.json', count)
            if result['checkpoint_sha256'] != selected['checkpoint_sha256'] or sha(selected['checkpoint']) != selected['checkpoint_sha256']:
                raise ValueError('Selected / evaluated checkpoint differs')
            protocol = read(model / 'final_evaluation/protocol.json')
            if base_protocol is not None:
                protocol_match(base_protocol, protocol)
            else:
                base_protocol = protocol
            if v == 'D' and not read(model / 'initialization_audit.json')['exact']:
                raise ValueError('Shared initialization not validated')
            for i in (0, 1, 2) if smoke else (0,):
                if not read(model / 'restore_audit' / f'{i:02d}.json')['exact_model_Adam_scheduler_criterion_RNG']:
                    raise ValueError('Full state restoration not validated')
            summaries[key], model_roots[key], histories[key] = result, str(model), history
            audit_rows[key] = dict(contract_sha256=sha(model / 'contract.json'),
                selected_sha256=sha(model / ('smoke_selected.json' if smoke else 'selected.json')),
                checkpoint_sha256=selected['checkpoint_sha256'], epochs=len(history),
                sampling=[dict(epoch=r['epoch'], counts=r['sampling_counts'], fractions=r['weather_fractions'],
                    effective_GT=r['gt_total'], empty_GT_resampling_attempts=r['official_resampling_attempts'],
                    trace_sha256=r['trace_sha256']) for r in history],
                shared_initialization=read(model / 'initialization_audit.json') if v == 'D' else None,
                pairing_exact=True, pairing_role='reference C trace' if v == 'C' else 'compared against C',
                full_state_restore=True)
            target_rows[key] = read(model / 'target_analysis.json')
    # The smoke has a different frame set and cannot be compared with full T0 AP.
    if not smoke:
        for v, folder in (('C', 'L0_C'), ('D', 'L1_D')):
            directory = original / folder / 'final_evaluation'
            summary = validate_summary(directory / 'summary.json', 829)
            if summary != reference['results'][v]:
                raise ValueError('T0 differs from the full precision archived result')
            protocol_match(base_protocol, read(directory / 'protocol.json'))
            summaries['T0-'+v], model_roots['T0-'+v] = summary, str(original / folder)
    values = {k: v['results'] for k, v in summaries.items()}
    compared = comparisons(values) if not smoke else dict(status='smoke_only', T0_comparison='ineligible: different frame count')
    tradeoff = screen(values) if not smoke else dict(status='smoke_only', screening='not eligible')
    changes = {t+' D-C': target_changes(root / t / 'D/final_evaluation', root / t / 'C/final_evaluation', count)
        for t in ('T1', 'T2')}
    if not smoke:
        changes.update({t+'-'+v+' minus T0-'+v: target_changes(root / t / v / 'final_evaluation',
            original / ('L0_C' if v == 'C' else 'L1_D') / 'final_evaluation', count)
            for t in ('T1', 'T2') for v in ('C', 'D')})
    full = dict(status='complete', smoke_only=smoke, eligible_for_scientific_conclusions=not smoke,
        single_seed=SETTINGS['seed'], results=values, models=model_roots, T0_reference=reference_source,
        checkpoint_sha256={k: v['checkpoint_sha256'] for k, v in summaries.items()}, frames_per_weather=count)
    training_audit = dict(status='complete', models=audit_rows, C_D_exact_pairing=True,
        training_changes='per-CAV weather distribution only; original native GT filtering and empty resampling',
        ROCm='Shared initialization/data are exact; bitwise training determinism is not established')
    targets = dict(status='complete', models=target_rows, fixed_GT_target_changes=changes)
    for name, value in (('E1_RESULTS', full), ('E1_TRAINING_AUDIT', training_audit),
            ('E1_MATCHED_COMPARISON', compared), ('E1_WEATHER_TRADEOFF', tradeoff), ('E1_TARGET_ANALYSIS', targets)):
        atomic(root / (name+'.json'), value)
    lines = ['# E1 Snow 混合训练结果', '',
        'Smoke：仅验证运行链，不能作为科研结果。' if smoke else '四组模型均完成20轮；主结果由每轮完整Clean官方O AP70选出，精确相同时选最早轮。',
        '', *table(values)]
    if not smoke:
        lines += ['', '## 配对差值（百分点）', '', *table(compared['differences_pp'], True)]
        lines += ['', '## GSPR 增益随训练比例的变化（百分点）', '', *table(compared['gain_change_pp'], True)]
    lines += ['', '目标级一对一匹配、TP/FP/FN、按距离/体积/真实点数的召回见 E1_TARGET_ANALYSIS.json。',
        '天气真实读取次数、重采样和GT分布见 E1_TRAINING_AUDIT.json。AP变化不能直接换算为恢复的具体目标数。',
        '单种子；ROCm算子不保证逐位确定；validation同时用于选模和研究筛选。工程门槛不代表统计显著性。']
    write_text(root / 'E1_RESULTS.md', lines)
    conclusions = ['# E1 结论', '', '本报告先给结果，再限定解释范围。']
    if not smoke:
        for t, row in tradeoff['recipes'].items():
            x = row['measured_pp']
            conclusions += ['', f'{t}：D的Snow F AP70相对T0-D变化 {x["snow_vs_T0_D_min"]:+.6f} pp；',
                f'同方案D−C为 {x["snow_D_minus_C_min"]:+.6f} pp；Clean/Fog F变化分别为 {x["clean_vs_T0_D_min"]:+.6f}/{x["fog_vs_T0_D_min"]:+.6f} pp。',
                '预设工程筛选全部通过。' if row['all_passed'] else '预设工程筛选未全部通过；完整结果仍保留，T2不由T1的结果决定。']
    else:
        conclusions += ['', '少量真实数据链路通过；不输出天气能力、机制或改进效果结论。']
    conclusions += ['', '增加Snow改变了输入和经过原生过滤后的GT分布。将D−C与同结构相对T0的变化分开解释；',
        '单次种子无法证明严格交互因果关系。未读正式test；后续论文主张需独立评测和多种子验证。']
    write_text(root / 'E1_CONCLUSIONS.md', conclusions)


def write_text(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    temp.replace(path)


def synthesis(root):
    root = Path(root)
    e0, e1 = read(root / 'E0/E0_RESULTS.json'), read(root / 'E1/E1_RESULTS.json')
    if e0['status'] != 'complete' or e1['status'] != 'complete' or not e1['eligible_for_scientific_conclusions']:
        raise ValueError('Both full experiments must complete before synthesis')
    interventions = read(root / 'E0/E0_INTERVENTIONS.json')
    tradeoff = read(root / 'E1/E1_WEATHER_TRADEOFF.json')
    point = read(root / 'E0/E0_POINT_DEGRADATION.json')
    errors = read(root / 'E0/E0_TARGET_ANALYSIS.json')
    lines = ['# E0 + E1 综合结论', '', '完整证据来自同一冻结validation协议；E0复用独立诊断包，E1完成四组20轮训练。',
        '', '## Fog 的收益是否依赖点的空间权重', '']
    changes = interventions['delta_from_R0_pp']
    for arm in ('R1', 'R2', 'R3'):
        lines.append(f'{arm} 相对R0：Fog F AP70 {changes[arm]["fog"]["F"]["ap70"]:+.6f} pp；Snow F AP70 {changes[arm]["snow"]["F"]["ap70"]:+.6f} pp。')
    spatial = changes['R2']['fog']['F']['ap70'] < -.5 and changes['R3']['fog']['F']['ap70'] < -.5
    lines += ['', '两种空间对应干预均降低Fog表现，支持当前空间加权具有作用；仍需结合幅值、分布和目标证据解释。' if spatial
        else '当前干预尚未同时给出清晰的空间对应证据；保留E0中的分布、目标及幅值解释，不能直接认定噪声识别已校准。',
        '', '## Snow 训练数据与天气取舍', '']
    for t, row in tradeoff['recipes'].items():
        x = row['measured_pp']
        lines.append(f'{t}：D Snow F相对T0 {x["snow_vs_T0_D_min"]:+.6f} pp；匹配D−C {x["snow_D_minus_C_min"]:+.6f} pp；Clean/Fog变化 {x["clean_vs_T0_D_min"]:+.6f}/{x["fog_vs_T0_D_min"]:+.6f} pp。')
        c_change = delta(e1['results'][t+'-C'], e1['results']['T0-C'])['snow']['F']['ap70']
        lines.append(f'同结构普通AttFuse加入Snow后的变化为 {c_change:+.6f} pp；将这项数据分布作用与上面的匹配D−C分开解释。')
    missing = point['aggregate']['snow']
    old = errors['weather']['snow']['counts']['0.7']
    r1_snow = changes['R1']['snow']['F']['ap70']
    priority = ('先验证Snow中被抑制目标与幅值因素，再做配对天气可靠性监督的小型检查；证据是R1的Snow F变化 '+f'{r1_snow:+.6f} pp。'
        if r1_snow > .5 else '先验证覆盖度建模的可用性；证据是下面记录的真实零点目标与Clean覆盖缺失。单纯过滤现有点无法补回未观测的返回。'
        if missing['zero_retained_point_targets'] > 0 else
        '现有干预和零点目标统计尚不足以确定前端改进方向；先检查E0逐目标关联，再选择结构或监督方案。')
    lines += ['', f'Snow中固定目标的零真实点观测次数为 {missing["zero_retained_point_targets"]}/{missing["fixed_targets"]}；',
        f'共同ego坐标覆盖单元缺失为 {missing["missing_clean_cells"]}/{missing["clean_occupied_cells"]}。这是观测缺失代理，无法测量全部有效信息。',
        f'历史C/D在Snow IoU≥0.7下：新增 {old["recovered"]}、丢失 {old["lost"]}，TP净变化 {old["D_tp"]-old["C_tp"]:+d}。',
        'AP同时受误报和最终分数排序影响；E0的固定最终框换分数回放及定位分析提供可观察证据，无法据此断言NMS内部的完整因果关系。',
        '', 'AP30/AP50、逐目标召回与各类误差完整见 ../E1/E1_RESULTS.md 和 ../E1/E1_TARGET_ANALYSIS.json；不能只用AP70判断整体能力。',
        '', '## 下一步顺序与最小验证', '',
        '1. 先依据E0目标证据区分低覆盖与权重抑制，再验证E1中普通AttFuse和GSPR各自的Snow变化。E1改变输入与GT分布，不能单独证明结构机制。',
        '2. E2推荐：'+priority,
        '   最小检查固定原模型，按E0挽救/丢失目标分组比较可靠性、真实覆盖与误差；先核验绿色通道语义，再决定是否加入点监督。该检查仍不能证明r是已校准的真实噪声概率。',
        '3. E3最值得先验证“车辆的真实目标覆盖是否能预测融合收益”。用冻结特征与目标证据做离线检验；',
        '   仅在可靠性/不确定性具有额外判别信息后，再考虑让覆盖与质量影响融合权重。当前r/u未经概率校准，直接输入attention的收益无法确定。',
        '4. 暂不重复本轮排除的BN校准和FiLM，不据单次validation结果堆叠分支；本轮没有实现E2或E3。',
        '', '当前可确定项、仍无法确定项的逐项证据见 ../E0/E0_CONCLUSIONS.md 和 ../E1/E1_CONCLUSIONS.md。',
        '所有主要结果仍来自参与选模/筛选的validation；论文最终主张须预留独立评测并进行多随机种子验证。']
    write_text(root / 'reports/E01_FINAL_CONCLUSIONS.md', lines)
    atomic(root / 'reports/E01_EVIDENCE.json', dict(status='complete',
        sources={str(p.relative_to(root)): sha(p) for p in (
            root / 'E0/E0_RESULTS.json', root / 'E0/E0_INTERVENTIONS.json',
            root / 'E0/E0_POINT_DEGRADATION.json', root / 'E0/E0_TARGET_ANALYSIS.json', root / 'E1/E1_RESULTS.json',
            root / 'E1/E1_TRAINING_AUDIT.json', root / 'E1/E1_TARGET_ANALYSIS.json')},
        limitations=SETTINGS['limits'], E2_E3_implemented=False))
