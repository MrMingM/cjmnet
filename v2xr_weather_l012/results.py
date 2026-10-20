"""Reports contain computed server values only; partial runs leave absent models absent."""
from auditlib import CONFIG, atomic_json, complete, read_json, sha256

VARIANTS = dict(C='L0_C', D='L1_D', C_BN='L2_C_BN', D_BN='L2_D_BN',
                C_FiLM='L2_C_FiLM', D_FiLM='L2_D_FiLM')
PARENTS = dict(D='C', C_BN='C', D_BN='D', C_FiLM='C', D_FiLM='D')


def write_evaluation(output, summary):
    lines = ['# V2X-R weather experiment: official O with O*/F audit', '',
        f'Frames per condition: {summary["frame_count"]}; full validation: {summary["full_validation"]}.',
        'O uses current-weather official GT and skips None frames. O* counts those frames. F fixes Clean GT.',
        'AP, GT filtering, NMS and matching are the unchanged frozen official functions.', '',
        '| Weather | Protocol | AP30 | AP50 | AP70 | Counted GT |', '|---|---|---:|---:|---:|---:|']
    for weather, row in summary['results'].items():
        for protocol, ap in row['AP'].items():
            lines.append(f'| {weather} | {protocol} | {ap["ap30"]:.10f} | {ap["ap50"]:.10f} | {ap["ap70"]:.10f} | {row["counted_gt"][protocol]} |')
        lines += ['', f'{weather} full counts: `{row["counts"]}`', '']
    lines += ['Checkpoint SHA256: `' + summary['checkpoint_sha256'] + '`.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def result(root, variant):
    folder = root / VARIANTS[variant]
    summary = complete(folder / 'final_evaluation/summary.json')
    if (not summary['full_validation'] or summary['frame_count'] != 829 or
            summary['conditions'] != ['clean', 'fog', 'snow'] or summary['model_variant'] != variant):
        raise ValueError('Invalid/incomplete final evaluation: ' + variant)
    if variant.endswith('_BN'):
        artifact = complete(folder / 'artifact.json')
        if read_json(folder / 'contract.json')['smoke_only']:
            raise ValueError('Smoke BN cannot enter final matrix')
    else:
        from evaluation import selection
        artifact = selection(folder)
    if (artifact['checkpoint_sha256'] != summary['checkpoint_sha256'] or
            sha256(artifact['checkpoint']) != summary['checkpoint_sha256']):
        raise ValueError('Final evaluation/checkpoint identity mismatch: ' + variant)
    # Verify every frame cache too when skipping/completing the overall workflow.
    predictions = read_json(folder / 'final_evaluation/prediction_manifest.json')
    for row in predictions['files']:
        if sha256(row['path']) != row['sha256']:
            raise ValueError('Final prediction bytes changed')
    return summary


def ap70(summary):
    return {c: summary['results'][c]['AP']['O']['ap70'] for c in ('clean', 'fog', 'snow')}


def final(root, partial=False):
    reports = root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    found = {v: result(root, v) for v, directory in VARIANTS.items()
             if (root / directory / 'final_evaluation/summary.json').exists()}
    if not partial and set(found) != set(VARIANTS):
        raise ValueError('Six complete variants are required for final report')
    if len({s['frozen_d1c_gt_identity'] for s in found.values()}) > 1:
        raise ValueError('Final matrix GT identities differ')
    differences = {}
    for variant, parent in PARENTS.items():
        if variant in found and parent in found:
            differences[variant + '-' + parent] = {
                c: {p: {k: 100 * (found[variant]['results'][c]['AP'][p][k] - found[parent]['results'][c]['AP'][p][k])
                       for k in ('ap30', 'ap50', 'ap70')} for p in ('O', 'O*', 'F')}
                for c in ('clean', 'fog', 'snow')}
    baseline_differences = {v: {c: 100 * (ap70(s)[c] - CONFIG['clean_only_O_AP70'][c])
                              for c in ('clean', 'fog', 'snow')} for v, s in found.items()}
    gates = {name: read_json(reports / (name + '_GATE.json')) for name in ('L0', 'L1')
             if (reports / (name + '_GATE.json')).exists()}
    audit = dict(status='partial' if partial else 'complete',
        main_protocol='O', no_test=True, settings=CONFIG, gates=gates,
        sources=read_json(root / 'run_identity.json'),
        models={v: dict(summary_sha256=sha256(root / VARIANTS[v] / 'final_evaluation/summary.json'),
            checkpoint_sha256=s['checkpoint_sha256'], gt_identity=s['gt_identity'],
            prediction_cache_identity=s['prediction_cache_identity'],
            selection=read_json(root / VARIANTS[v] / ('artifact.json' if v.endswith('_BN') else 'selected.json')),
            history=read_json(root / VARIANTS[v] / 'history.json') if (root / VARIANTS[v] / 'history.json').exists() else None,
            adaptation=read_json(root / VARIANTS[v] / 'adaptation.json') if (root / VARIANTS[v] / 'adaptation.json').exists() else None)
            for v, s in found.items()},
        pairing=pairing(root) if 'D' in found else None,
        failure_history=[dict(path=str(p), sha256=sha256(p), record=read_json(p))
            for p in (root / 'stages').glob('*/failure.json')],
        limitations=['single seed, validation-selected, no test results',
            'official per-CAV Fog probability 3/7; cooperative frames can contain mixed weather',
            'official GT depends on sampled point clouds; empty training GT causes official resampling',
            'official IntermediateFusionDataset does not invoke configured world flip/rotation/scaling; actual calls audited',
            'original empty-voxel synthetic point fallback remains; missing Fog files never fall back',
            'BN cumulative per-forward batch statistics, not globally pooled point statistics',
            'BN inference preprocessing uses official max_voxel_test=70000, no augmentation',
            'kernel nondeterminism may limit exact numerical replay despite full RNG restoration'])
    data = dict(status=audit['status'], results=found, matched_deltas_pp=differences,
        clean_only_reference_delta_O_AP70_pp=baseline_differences, gates=gates)
    atomic_json(reports / 'FINAL_RESULTS.json', data)
    atomic_json(reports / 'EXPERIMENT_AUDIT.json', audit)
    lines = ['# L0/L1/L2 actual validation results', '', '状态：' + ('L0 门槛停止，后续正式模型未运行；独立 smoke 已完成。' if partial else '六组流程完成。'), '',
        '| Model | Clean O AP70 | Fog O AP70 | Snow O AP70 |', '|---|---:|---:|---:|']
    for variant, summary in found.items():
        ap = ap70(summary)
        lines.append(f'| {variant} | {ap["clean"]:.10f} | {ap["fog"]:.10f} | {ap["snow"]:.10f} |')
    lines += ['', '| Matched comparison | Clean pp | Fog pp | Snow pp |', '|---|---:|---:|---:|']
    for label, row in differences.items():
        lines.append('| ' + label + ' | ' + ' | '.join(f'{row[c]["O"]["ap70"]:+.6f}' for c in ('clean', 'fog', 'snow')) + ' |')
    for name, gate in gates.items():
        lines += ['', name + ' gate: `' + str(gate) + '`']
    for variant, summary in found.items():
        lines += ['', '## ' + variant, '', '| Weather | Protocol | AP30 | AP50 | AP70 | GT denominator | None frames |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for c, row in summary['results'].items():
            for p, ap in row['AP'].items():
                lines.append(f'| {c} | {p} | {ap["ap30"]:.10f} | {ap["ap50"]:.10f} | {ap["ap70"]:.10f} | {row["counted_gt"][p]} | {row["counts"]["none_frames"]} |')
        lines += ['', '完整 GT/None/removed/retained 计数：`' + str({c: r['counts'] for c, r in summary['results'].items()}) + '`']
        selection_record = audit['models'][variant]['selection']
        if 'epoch' in selection_record:
            lines += ['', f'按 Clean O 选中 epoch {selection_record["epoch"]}；checkpoint SHA256：`{summary["checkpoint_sha256"]}`。']
        adaptation = audit['models'][variant]['adaptation']
        if adaptation:
            lines += ['', '适配生效检查：`' + str({k: v for k, v in adaptation.items()
                if k in ('only_allowed_buffers_changed', 'output_change_max', 'calibration_frames',
                    'film_parameters', 'total_parameters', 'ever_nonidentity', 'frozen_backbone_exact', 'frozen_bn_exact',
                    'observed_input_variation_between_cavs')}) + '`']
    lines += ['', '原始 Clean-only 参照的三天气变化（pp）：`' + str(baseline_differences) + '`', '',
        '源码、数据、模型、训练曲线、适配生效检查、配对身份及限制见 EXPERIMENT_AUDIT.json；逐帧证据保存在各模型 final_evaluation。']
    lines += ['', '限制与边界：'] + ['- ' + text for text in audit['limitations']]
    (reports / 'FINAL_RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    conclusion = ['# 结论与下一步', '', '以下数值均由本次服务器产物计算。', '']
    if partial:
        conclusion += ['L0 未达到预先固定的 Fog +5 pp 门槛，流程按协议停止。该结果本身不能证明数据管线错误。',
                       '先查看 L0_DIAGNOSIS.md 和采样/GT/损失曲线，再决定是否开展独立的新实验。']
    else:
        conclusion += ['匹配对照 O AP70 差值（pp）：`' + str({k: {c: r[c]['O']['ap70'] for c in r} for k, r in differences.items()}) + '`',
            '结合三天气绝对 AP、O*/F 及 GT/None 分母判断收益，不能只看 Fog 或 weather gap。',
            '当前仅一个种子且按 validation 选模；后续若继续研究，应先冻结候选，再采用独立预注册的多种子验证。']
    (reports / 'CONCLUSIONS.md').write_text('\n'.join(conclusion) + '\n', encoding='utf-8', newline='\n')
    return data


def pairing(root):
    rows = {}
    for variant in ('C', 'D'):
        rows[variant] = read_json(root / VARIANTS[variant] / 'history.json')['epochs']
    return dict(epoch_trace_equal={str(c['epoch']): c['trace_sha256'] == d['trace_sha256']
        for c, d in zip(rows['C'], rows['D'])},
        interpretation='Trace includes physical paths, worker Python/NumPy RNG, actual reads including retries, GT and augmented input hashes. Exact equality establishes those observations only; no bitwise GPU determinism claim.')
