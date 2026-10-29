"""Report measured AP and matched targets; never gate later models on AP gains."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from bridge import load
from auditlib import atomic_json, complete, read_json, sha256

VARIANTS = {v: v for v in ('A1', 'A2', 'A3')}
write_evaluation = load('repe_results', 'gspr_repe/results.py').write_evaluation


def stage1(root):
    from auditlib import bound_json
    tests = complete(root / 'self_test/summary.json')
    records = {}
    for variant in VARIANTS:
        folder = root / 'smoke' / variant
        checkpoint = complete(folder / 'smoke_checkpoint.json')
        history = read_json(folder / 'history.json')['epochs']
        summary = complete(folder / 'evaluation_smoke12/summary.json')
        if (len(history) != 1 or history[0]['train_updates'] != 2 or len(history[0]['repe']) != 2
                or summary['frame_count'] != 12 or summary['conditions'] != ['clean', 'fog', 'snow']
                or summary['checkpoint_sha256'] != checkpoint['checkpoint_sha256']
                or sha256(checkpoint['checkpoint']) != checkpoint['checkpoint_sha256']):
            raise ValueError('Incomplete real smoke training/diagnostics/inference: ' + variant)
        if not any(any(x > 0 for x in row['update_l2'].values()) for row in history[0]['repe']):
            raise ValueError('No actual GGCA parameter update in smoke: ' + variant)
        records[variant] = dict(checkpoint=checkpoint, epoch=history[0], evaluation=summary)
    bound_json(root / 'stage1/acceptance.json', dict(status='complete', tests=tests,
        smoke=records, stage2_allowed=True, A0_training_executed=False))
    (root / 'stage1/STAGE1_REPORT.md').write_text('# 第一阶段验收\n\n结构、完整状态恢复、三组真实小训练和三天气推理完成。A0没有训练。低AP不作为技术失败。\n', encoding='utf-8')


def evaluation_folders(root):
    ref = complete(root / 'A0/evaluation_reference.json')
    folder = Path(ref['root'])
    if sha256(folder / 'summary.json') != ref['summary_sha256']:
        raise ValueError('A0 evaluation reference changed')
    from lineage import training_root
    return {'A0': folder, **{v: training_root(root, v) / 'final_evaluation' for v in VARIANTS}}


def stratified(root, folders):
    run = read_json(root / 'run_identity.json')
    evidence = [json.loads(s) for s in (Path(run['roots']['gt_audit_root']) / 'per_frame.jsonl').read_text(encoding='utf-8').splitlines()]
    strata = defaultdict(lambda: dict(gt=0, tp=0))
    matched, frame_counts, localization = {}, {}, defaultdict(list)
    for variant, folder in folders.items():
        for line in (folder / 'per_frame_metrics.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line); weather, index = row['condition'], row['index']
            frame = evidence[index]
            if frame['frame_key'] != row['frame_key']:
                raise ValueError('Target comparison frame identity mismatch')
            pairs = row['protocols']['F']['matched_pairs']['0.7']
            ids = {str(p['object_id']) for p in pairs}
            matched[(variant, weather, index)] = {str(p['object_id']): p['iou'] for p in pairs}
            localization[(variant, weather)].extend(p['iou'] for p in pairs)
            frame_counts[(variant, weather, index)] = row['protocols']['F']['thresholds']['0.7']
            for gt in frame['conditions']['clean']['final_gt']:
                oid = str(gt['object_id']); corners = gt['corners']
                center = [sum(p[k] for p in corners)/len(corners) for k in (0, 1)]
                distance = math.hypot(*center)
                cav_counts = [c['official_points_in_box'][c['before_point_filter_ids'].index(oid)]
                              for c in frame['conditions'][weather]['cavs'] if oid in c['before_point_filter_ids']]
                n = sum(cav_counts)
                for group, label in (('distance_ego_m', '0-30' if distance < 30 else '30-60' if distance < 60 else '60+'),
                                     ('native_points_sum_cavs', '0' if n == 0 else '1-5' if n <= 5 else '6-20' if n <= 20 else '21+')):
                    target = strata[(variant, weather, group, label)]
                    target['gt'] += 1; target['tp'] += oid in ids
    changes = []
    for left, right in (('A1', 'A0'), ('A2', 'A1'), ('A3', 'A2'), ('A3', 'A0')):
        for weather in ('clean', 'fog', 'snow'):
            for index in range(829):
                a, b = matched[(left, weather, index)], matched[(right, weather, index)]
                common = set(a) & set(b)
                ca, cb = frame_counts[(left, weather, index)], frame_counts[(right, weather, index)]
                changes.append(dict(contrast=left+'-'+right, condition=weather, index=index,
                    frame_key=evidence[index]['frame_key'], recovered=sorted(set(a)-set(b)), lost=sorted(set(b)-set(a)),
                    matched_iou_delta={k: a[k]-b[k] for k in common}, fp_delta=ca['fp']-cb['fp'], fn_delta=ca['fn']-cb['fn']))
    return dict(strata=[dict(model=k[0], condition=k[1], group=k[2], bin=k[3], **v,
                            recall=v['tp']/v['gt']) for k, v in strata.items()],
        target_changes=changes, localization=[dict(model=k[0], condition=k[1], matched_iou_mean=sum(v)/len(v) if v else None,
                            matched_count=len(v)) for k, v in localization.items()],
        interpretation='Fixed Clean GT and original matcher at IoU .7. Localization on matched targets is selection-biased; common-target deltas are also retained.')


def final(root):
    from lineage import assert_fair, training_root
    (root / 'reports').mkdir(exist_ok=True)
    version_fairness = assert_fair(root)
    folders = evaluation_folders(root)
    summaries = {v: complete(folder / 'summary.json') for v, folder in folders.items()}
    ref = complete(root / 'baseline_reference.json')
    for variant, summary in summaries.items():
        if not summary['full_validation'] or summary['frame_count'] != 829 or summary['conditions'] != ['clean', 'fog', 'snow']:
            raise ValueError('Incomplete formal evaluation: ' + variant)
        if summary['frozen_d1c_gt_identity'] != ref['original_training_contract']['frozen_gt']:
            raise ValueError('Incompatible evaluation GT identity')
    contrasts = {}
    for left, right in (('A1', 'A0'), ('A2', 'A1'), ('A3', 'A2'), ('A3', 'A0')):
        contrasts[left+'-'+right] = {weather: {protocol: {key: 100*(summaries[left]['results'][weather]['AP'][protocol][key]
            - summaries[right]['results'][weather]['AP'][protocol][key]) for key in ('ap30', 'ap50', 'ap70')}
            for protocol in ('O', 'O*', 'F')} for weather in ('clean', 'fog', 'snow')}
    details = stratified(root, folders)
    histories = {'A0': read_json(Path(ref['a0_root']) / 'history.json')}
    histories.update({v: read_json(training_root(root, v) / 'history.json') for v in VARIANTS})
    traces = {v: [e['trace_sha256'] for e in h['epochs']] for v, h in histories.items()}
    mechanism = complete(root / 'mechanism/summary.json')
    for path, digest in mechanism['record_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Mechanism diagnostic artifact changed')
    observed = { (r['model'], r['condition'], r['index']): r
                 for r in (complete(Path(p)) for p in mechanism['record_sha256']) }
    for weather in ('clean', 'fog', 'snow'):
        for index in range(12):
            if len({observed[(v, weather, index)]['input_sha256'] for v in folders}) != 1:
                raise ValueError('Mechanism forwards do not use matched inputs')
    mechanism_links = []
    for change in details['target_changes']:
        if change['contrast'] != 'A3-A2' or change['index'] >= 12:
            continue
        row = observed[('A3', change['condition'], change['index'])]
        recovered, lost = set(change['recovered']), set(change['lost'])
        for target in row['targets']:
            mechanism_links.append(dict(condition=change['condition'], index=change['index'],
                **target, recovered=target['object_id'] in recovered, lost=target['object_id'] in lost))
    overhead = []
    for variant in folders:
        for weather in ('clean', 'fog', 'snow'):
            observations = [observed[(variant, weather, i)]['observation'] for i in range(12)]
            overhead.append(dict(model=variant, condition=weather,
                matched_input_frames=12, synchronized_forward_mean_seconds=sum(r['seconds'] for r in observations)/12,
                parameters=observations[0]['parameters'], peak_memory_bytes=max(r['peak_memory_bytes'] for r in observations)))
    report = root / 'reports'; report.mkdir(exist_ok=True)
    data = dict(status='complete', version_fairness=version_fairness, baseline_reference=ref, results=summaries, AP_delta_pp=contrasts,
        target_analysis=details, history=histories, sampling_trace_equal=all(v == traces['A0'] for v in traces.values()),
        mechanism=mechanism, sparse_protection_target_links=mechanism_links, matched_inference_overhead=overhead,
        limitations=['single training seed; no statistical significance claim',
            'validation used for selection; no test evaluation',
            'geometric support is a local proxy, not a PointDSC registration constraint or noise ground truth',
            'target-linked module diagnosis covers first 12 validation frames per weather; full uncached inference distributions are in each repe_inference.json',
            'A0 historical inference timing/VRAM may have a different measurement scope; no invented A0 overhead baseline'])
    atomic_json(report / 'FINAL_RESULTS.json', data)
    rows = [dict(model=v, condition=w, protocol=p, **ap) for v, s in summaries.items()
            for w, wr in s['results'].items() for p, ap in wr['AP'].items()]
    with (report / 'ABLATION_TABLE.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'condition', 'protocol', 'ap30', 'ap50', 'ap70'])
        writer.writeheader(); writer.writerows(rows)
    lines = ['# GGCA正式消融结果', '', 'A0复用指定旧B0，未重新训练。下表单位为百分点；负结果保留。', '',
             '| 对照 | Clean F AP70 | Fog F AP70 | Snow F AP70 |', '|---|---:|---:|---:|']
    for name, values in contrasts.items():
        lines.append('| '+name+' | '+' | '.join(f'{values[w]["F"]["ap70"]:+.6f}' for w in ('clean', 'fog', 'snow'))+' |')
    lines += ['', '逐轮实际抽样轨迹完全一致：'+str(data['sampling_trace_equal']), '',
              'AP不能单独证明机制。逐目标恢复/丢失、共同检出目标的定位变化及分层召回见FINAL_RESULTS.json；模块诊断见mechanism/。',
              'A3−A2的Snow差值必须结合被保护目标的恢复和丢失检查；没有因果干预或点噪声真值时，只能报告关联。', '',
              *data['limitations']]
    (report / 'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    worst = sorted(details['target_changes'], key=lambda r: len(r['lost'])+max(r['fp_delta'], 0), reverse=True)[:100]
    (report / 'FAILURE_CASES.md').write_text('# 高漏检/新增误检案例\n\n```json\n'+json.dumps(worst, ensure_ascii=False, indent=2)+'\n```\n', encoding='utf-8')
