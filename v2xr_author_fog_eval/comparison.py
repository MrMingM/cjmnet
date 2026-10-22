"""Full-precision comparison and authenticated original-matcher diagnostics; no scorer."""
import json
import math
from pathlib import Path
import statistics

from v2xr_author_fog_eval.core import (CHECKPOINT_SHA, CONDITIONS, FROZEN_GT, HERE, METRICS,
    PROTOCOLS, TOTALS, completed, identity, no_test, read, sha256, text_file, write)

AUTHOR = 'Author_Fog_AttFuse'
PAIRS = ((AUTHOR, 'C'), ('D', AUTHOR), ('D', 'C'))


def validate_summary(summary, name):
    if (summary.get('status') != 'complete' or summary.get('frame_count') != 829
            or summary.get('full_validation') is not True or summary.get('conditions') != list(CONDITIONS)
            or summary.get('model_variant') != name or summary.get('frozen_d1c_gt_identity') != FROZEN_GT
            or not summary.get('output_hashes') or not summary.get('gt_identity')
            or not summary.get('prediction_cache_identity')):
        raise ValueError('Incomplete/different GT or model identity in comparison: ' + name)
    if name == AUTHOR and summary['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Wrong author checkpoint')
    for weather in CONDITIONS:
        row = summary['results'][weather]
        counts, counted = row['counts'], row['counted_gt']
        if (counts['fixed_gt'] != 6224 or counts['official_gt'] != TOTALS[weather]
                or counts['removed_gt'] != 6224 - TOTALS[weather]
                or counts['removed_by_weather'] != counts['removed_gt']
                or counted['F'] != 6224 or counted['O*'] != TOTALS[weather]
                or counted['O'] != TOTALS[weather] - counts['official_gt_in_none_frames']
                or not 0 <= counts['none_frames'] <= 829):
            raise ValueError('GT/None protocol denominator differs: ' + name + '/' + weather)
        for protocol in PROTOCOLS:
            for key in METRICS:
                value = row['AP'][protocol][key]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid AP in comparison')
        if weather == 'clean' and row['AP']['O*'] != row['AP']['F']:
            raise ValueError('Clean O*/F definitions differ')


def differences(a, b):
    return {w: {p: {m: 100 * (a['results'][w]['AP'][p][m] - b['results'][w]['AP'][p][m])
                    for m in METRICS} for p in PROTOCOLS} for w in CONDITIONS}


def fog_counts(summary):
    n = summary['results']['fog']['counts']
    tp = n['retained_detected']['0.7'] + n['removed_detected']['0.7']
    if tp > n['prediction_count'] or tp > n['fixed_gt']:
        raise ValueError('Invalid frozen one-to-one match totals')
    return dict(predictions=n['prediction_count'], none_frames=n['none_frames'],
        empty_non_none_frames=n['empty_non_none_frames'], fixed_gt=n['fixed_gt'], official_gt=n['official_gt'],
        F_tp70=tp, F_fp70=n['prediction_count'] - tp, F_fn70=n['fixed_gt'] - tp,
        retained_gt=n['retained_gt'], retained_tp70=n['retained_detected']['0.7'],
        removed_gt=n['removed_gt'], removed_tp70=n['removed_detected']['0.7'],
        retained_fn70=n['retained_gt'] - n['retained_detected']['0.7'],
        removed_fn70=n['removed_gt'] - n['removed_detected']['0.7'],
        source='Frozen replay F original-matcher one-to-one assignment; retained/removed partition the same F matches')


def fog_protocol_counts(summary, frames):
    fixed = fog_counts(summary)
    fallback = dict(F=dict(status='complete', tp=fixed['F_tp70'], fp=fixed['F_fp70'],
                          fn=fixed['F_fn70'], counted_gt=6224, source='authenticated archived summary'))
    if frames is None:
        return dict(**fallback, **{p: dict(status='unavailable', reason='Historical per-frame original matcher records missing')
                                  for p in ('O', 'O*')})
    result = {}
    for protocol in PROTOCOLS:
        rows = [frames[('fog', i)]['protocols'][protocol] for i in range(829)]
        totals = {k: sum(r['thresholds']['0.7'][k] for r in rows) for k in ('tp', 'fp', 'fn')}
        counted = sum(r['counted_gt'] for r in rows)
        if counted != summary['results']['fog']['counted_gt'][protocol] or totals['tp'] + totals['fn'] != counted:
            raise ValueError('Fog per-frame protocol TP/FN denominator differs')
        if totals['tp'] + totals['fp'] != fixed['predictions']:
            raise ValueError('Fog per-frame protocol prediction count differs')
        result[protocol] = dict(status='complete', **totals, counted_gt=counted,
                                source='authenticated per_frame_metrics.jsonl original matcher flags')
    if result['F']['tp'] != fixed['F_tp70']:
        raise ValueError('Fog fixed-GT TP totals differ')
    return result


def effects(models):
    result = {}
    for left, right in PAIRS:
        rows = {}
        for weather in CONDITIONS:
            metrics = {}
            for key in METRICS:
                a, b = (models[n]['results'][weather]['AP'] for n in (left, right))
                metrics[key] = dict(
                    O_delta_pp=100 * (a['O'][key] - b['O'][key]),
                    F_delta_pp=100 * (a['F'][key] - b['F'][key]),
                    None_skip_component_pp=100 * ((a['O'][key] - a['O*'][key]) - (b['O'][key] - b['O*'][key])),
                    GT_filter_component_pp=100 * ((a['O*'][key] - a['F'][key]) - (b['O*'][key] - b['F'][key])))
            rows[weather] = metrics
        result[left + '-' + right] = rows
    return result


def evaluation_markdown(summary):
    lines = ['# Author_Fog_AttFuse：固定作者权重 validation 评测', '',
        f"每条件 {summary['frame_count']}/829 帧；完整 validation：{summary['full_validation']}。本次无训练、无选模。", '',
        'O：天气 GT，None 整帧跳过。O*：同天气 GT，None 作为零预测计入。F：固定 Clean GT，None 计入。',
        '三协议共享一份预测，调用冻结官方 BEV IoU / VOC2010 AP，global_sort=False。', '',
        '| 条件 | 协议 | AP30 | AP50 | AP70 | GT 分母 |', '|---|---|---:|---:|---:|---:|']
    for w in CONDITIONS:
        row = summary['results'][w]
        for p in PROTOCOLS:
            lines.append(f"| {w} | {p} | " + ' | '.join(f'{row["AP"][p][m]:.10f}' for m in METRICS)
                         + f" | {row['counted_gt'][p]} |")
    lines += ['', '| 条件 | 官方 GT | 固定 GT | 天气移除 | None 帧 | 预测框 |', '|---|---:|---:|---:|---:|---:|']
    for w in CONDITIONS:
        c = summary['results'][w]['counts']
        lines.append(f"| {w} | {c['official_gt']} | {c['fixed_gt']} | {c['removed_gt']} | {c['none_frames']} | {c['prediction_count']} |")
    lines += ['', '保留/移除目标在 IoU≥0.3/0.5/0.7 的一对一匹配计数完整记录于 summary.json，来自原版匹配器。', '',
              'Checkpoint SHA256：`' + summary['checkpoint_sha256'] + '`。',
              '冻结 GT 身份：`' + summary['frozen_d1c_gt_identity'] + '`。',
              '作者官方来源：' + summary['official_checkpoint_source']]
    return '\n'.join(lines)


def read_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def authenticated_frames(directory, summary, definitions, frozen_gt):
    """Raw historical evidence is optional; an existing but changed record fails closed."""
    names = ('protocol.json', 'gt_manifest.json', 'per_frame_metrics.jsonl', 'removed_gt_metrics.jsonl')
    missing = [n for n in names if not (directory / n).is_file()]
    for name in names:
        if (directory / name).is_file() and sha256(directory / name) != summary['output_hashes'][name]:
            raise ValueError('Historical audit bytes differ: ' + str(directory / name))
    if missing:
        return None, dict(status='unavailable', directory=str(directory), missing=missing,
                          reason='No per-frame or concrete target recovery claims without authenticated raw records')
    protocol, gt = read(directory / 'protocol.json'), read(directory / 'gt_manifest.json')
    if (protocol['protocols'] != definitions or gt['definition'] != definitions
            or gt['identity'] != summary['gt_identity']
            or protocol['prediction_cache_identity'] != summary['prediction_cache_identity']):
        raise ValueError('Historical protocol/cache/GT chain differs')
    if identity({k: v for k, v in gt.items() if k != 'identity'}) != gt['identity']:
        raise ValueError('GT manifest identity invalid')
    fields = ('frame_key', 'official_gt_identity', 'fixed_gt_identity', 'official_ids',
              'fixed_ids', 'official_count', 'fixed_count', 'removed_ids')
    expected = {(r['condition'], r['index']): r for r in frozen_gt['frames']}
    keys = [(r['condition'], r['index']) for r in gt['frames']]
    if len(gt['frames']) != 3 * 829 or len(set(keys)) != 2487 or set(keys) != set(expected):
        raise ValueError('Incomplete historical GT frames')
    for row in gt['frames']:
        original = expected[(row['condition'], row['index'])]
        if any(row[f] != original[f] for f in fields):
            raise ValueError('Historical per-frame GT identity/order/geometry differs')
    frames = read_lines(directory / 'per_frame_metrics.jsonl')
    result = {(r['condition'], r['index']): r for r in frames}
    if len(frames) != 2487 or result.keys() != expected.keys():
        raise ValueError('Incomplete/duplicate historical metric frames')
    for weather in CONDITIONS:
        subset = [result[(weather, i)] for i in range(829)]
        counts = summary['results'][weather]['counts']
        if sum(r['prediction_count'] for r in subset) != counts['prediction_count']:
            raise ValueError('Per-frame prediction count differs from summary')
        tp = sum(r['protocols']['F']['thresholds']['0.7']['tp'] for r in subset)
        if tp != counts['retained_detected']['0.7'] + counts['removed_detected']['0.7']:
            raise ValueError('Per-frame original matching count differs from summary')
        for row in subset:
            gtrow = expected[(weather, row['index'])]
            if (row['frame_key'] != gtrow['frame_key'] or row['fixed_gt_count'] != gtrow['fixed_count']
                    or row['official_gt_count'] != gtrow['official_count']
                    or row['removed_ids'] != gtrow['removed_ids']):
                raise ValueError('Per-frame index/GT differs')
            pairs = row['protocols']['F']['matched_pairs']['0.7']
            targets = [p['object_id'] for p in pairs]
            if (len(targets) != len(set(targets)) or len({p['prediction_index'] for p in pairs}) != len(pairs)
                    or len(pairs) != row['protocols']['F']['thresholds']['0.7']['tp']):
                raise ValueError('Non-one-to-one or inconsistent original matching evidence')
            if not set(targets) <= set(gtrow['fixed_ids']):
                raise ValueError('Matched target absent from frozen fixed GT')
    removed_rows = read_lines(directory / 'removed_gt_metrics.jsonl')
    removed_expected = {(r['condition'], r['index'], object_id)
                        for r in gt['frames'] for object_id in r['removed_ids']}
    removed_keys = [(r['condition'], r['index'], r['object_id']) for r in removed_rows]
    if set(removed_keys) != removed_expected or len(removed_keys) != len(removed_expected):
        raise ValueError('Removed-target audit is incomplete/duplicated')
    for row in removed_rows:
        frame = result[(row['condition'], row['index'])]
        pairs = {p['object_id']: p for p in frame['protocols']['F']['matched_pairs']['0.7']}
        expected_pair = pairs.get(row['object_id'])
        detection = row['detection']['0.7']
        if (row['frame_key'] != frame['frame_key']
                or row['prediction_cache_sha256'] != frame['prediction_cache_sha256']
                or detection['detected'] != (expected_pair is not None) or detection['match'] != expected_pair):
            raise ValueError('Removed-target trace differs from original fixed-GT matches')
    return result, dict(status='complete', directory=str(directory),
        hashes={n: sha256(directory / n) for n in names}, frames_per_condition=829)


def frame_gains(left, right, weather):
    rows = []
    for i in range(829):
        a, b = left[(weather, i)], right[(weather, i)]
        if a['frame_key'] != b['frame_key'] or a['removed_ids'] != b['removed_ids']:
            raise ValueError('Unpaired frames in gain analysis')
        pairs = [r['protocols']['F']['matched_pairs']['0.7'] for r in (a, b)]
        detected_a, detected_b = ({p['object_id'] for p in ps} for ps in pairs)
        recovered, lost = detected_a - detected_b, detected_b - detected_a
        removed = set(a['removed_ids'])
        row = dict(index=i, frame_key=a['frame_key'], condition=weather,
            prediction_delta=a['prediction_count'] - b['prediction_count'],
            F_tp70_delta=len(detected_a) - len(detected_b),
            newly_detected_fixed_ids=sorted(recovered), no_longer_detected_fixed_ids=sorted(lost),
            newly_detected_removed_ids=sorted(recovered & removed),
            no_longer_detected_removed_ids=sorted(lost & removed),
            protocol_counts={p: {k: a['protocols'][p]['thresholds']['0.7'][k] - b['protocols'][p]['thresholds']['0.7'][k]
                                for k in ('tp', 'fp', 'fn')} for p in PROTOCOLS})
        rows.append(row)
    delta = [r['F_tp70_delta'] for r in rows]
    distribution = dict(status='complete', frames=829,
        benefited_frames=sum(x > 0 for x in delta), degraded_frames=sum(x < 0 for x in delta),
        equal_tp_frames=sum(x == 0 for x in delta), net_tp70=sum(delta),
        min_tp_delta=min(delta), median_tp_delta=statistics.median(delta), max_tp_delta=max(delta),
        newly_detected_fixed_targets=sum(len(r['newly_detected_fixed_ids']) for r in rows),
        no_longer_detected_fixed_targets=sum(len(r['no_longer_detected_fixed_ids']) for r in rows),
        interpretation='Observed paired original-matcher detection transitions on fixed GT; not causal recovery claims or AP-derived target counts')
    return rows, distribution


def report(root, request):
    root = Path(root)
    preflight = read(root / 'preflight.json')
    source = preflight['cd_reference']
    if sha256(source['path']) != source['sha256']:
        raise ValueError('C/D comparison source changed')
    archive = read(source['path'])
    author = completed(root / 'author_evaluation/summary.json')
    models = {AUTHOR: author, **{v: archive['results'][v] for v in ('C', 'D')}}
    for name, summary in models.items():
        validate_summary(summary, name)
    frozen_path = no_test(request['arguments']['d1c_root']) / 'gt_manifest.json'
    if sha256(frozen_path) != preflight['d1c']['gt_manifest.json']:
        raise ValueError('Frozen GT artifact changed before comparison')
    frozen_gt = read(frozen_path)
    raw, audit_sources = {}, {}
    settings_checks = {}
    author_protocol = read(root / 'author_evaluation/protocol.json')
    for name, summary in models.items():
        folder = root / 'author_evaluation' if name == AUTHOR else (
            no_test(request['arguments']['cd_root']) / ('L0_C' if name == 'C' else 'L1_D') / 'final_evaluation')
        raw[name], audit_sources[name] = authenticated_frames(folder, summary, preflight['protocols'], frozen_gt)
        protocol_path = folder / 'protocol.json'
        if protocol_path.is_file():
            candidate = read(protocol_path)
            fields = ('protocols', 'inference_seed', 'PYTHONHASHSEED', 'loader_workers',
                      'versions', 'HIP', 'official_source_hashes', 'official_config_sha256',
                      'input_manifest_identity', 'frame_keys', 'torch_backend')
            changed = [f for f in fields if candidate.get(f) != author_protocol.get(f)]
            if changed:
                raise ValueError('Historical runtime/protocol settings differ: ' + name + ': ' + repr(changed))
            settings_checks[name] = dict(status='direct_artifact_verified', fields=list(fields))
        else:
            settings_checks[name] = dict(status='archive_and_source_verified_only',
                reason='Full-precision archived summary binds complete frozen GT/protocol result; direct runtime protocol file was not available')
    reports = root / 'reports'
    reports.mkdir(exist_ok=True)
    distributions = {}
    for left, right in PAIRS:
        label = left + '-' + right
        distributions[label] = {}
        for weather in CONDITIONS:
            if raw[left] is None or raw[right] is None:
                distributions[label][weather] = dict(status='unavailable', reason='Historical raw audit records missing; see audit_sources')
                continue
            rows, dist = frame_gains(raw[left], raw[right], weather)
            filename = 'frame_gains_' + label + '_' + weather + '.jsonl'
            text_file(reports / filename, '\n'.join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows))
            dist['rows'] = filename
            distributions[label][weather] = dist
    paper = read(HERE / 'paper_reference.json')
    data = dict(status='complete', run_identity=request['identity'],
        comparison_scope='Same 829 paired validation frames, frozen GT identities and O/O*/F evaluator; different checkpoints/training procedures; no causal or test-generalization claim',
        absolute_AP={n: {w: s['results'][w]['AP'] for w in CONDITIONS} for n, s in models.items()},
        checkpoint_sha256={n: s['checkpoint_sha256'] for n, s in models.items()},
        deltas_pp={a + '-' + b: differences(models[a], models[b]) for a, b in PAIRS},
        protocol_components_pp=effects(models),
        fog_matching={n: dict(**fog_counts(s), by_protocol=fog_protocol_counts(s, raw[n])) for n, s in models.items()},
        per_frame_gain_distribution=distributions, audit_sources=audit_sources,
        protocol_settings_checks=settings_checks,
        reference_source=source, frozen_gt_identity=FROZEN_GT, protocols=preflight['protocols'],
        paper_reference=paper,
        paper_differences={w: dict(paper_AP70=paper['AP70'][w], validation_O_AP70=author['results'][w]['AP']['O']['ap70'],
                                  validation_minus_paper_pp=100 * (author['results'][w]['AP']['O']['ap70'] - paper['AP70'][w]))
                           for w in CONDITIONS})
    write(reports / 'MATCHED_COMPARISON.json', data)
    text_file(reports / 'MATCHED_COMPARISON.md', comparison_markdown(data))
    text_file(reports / 'CONCLUSIONS.md', conclusions(data))
    outputs = ['MATCHED_COMPARISON.json', 'MATCHED_COMPARISON.md', 'CONCLUSIONS.md']
    outputs += [v['rows'] for weather_rows in distributions.values() for v in weather_rows.values() if 'rows' in v]
    write(reports / 'completion.json', dict(status='complete', run_identity=request['identity'],
        output_hashes={n: sha256(reports / n) for n in outputs}))
    return data


def comparison_markdown(data):
    lines = ['# 同集评测对照', '', '全部为完整 829 帧 validation；差值单位为百分点，100×(左模型AP−右模型AP)。', '',
        '| 模型 | 条件 | 协议 | AP30 | AP50 | AP70 |', '|---|---|---|---:|---:|---:|']
    for name, weather_rows in data['absolute_AP'].items():
        for weather, protocols in weather_rows.items():
            for protocol, ap in protocols.items():
                lines.append(f'| {name} | {weather} | {protocol} | ' + ' | '.join(f'{ap[m]:.10f}' for m in METRICS) + ' |')
    lines += ['', '| 左−右 | 条件 | 协议 | ΔAP30 pp | ΔAP50 pp | ΔAP70 pp |', '|---|---|---|---:|---:|---:|']
    for label, weather_rows in data['deltas_pp'].items():
        for weather, protocols in weather_rows.items():
            for protocol, ap in protocols.items():
                lines.append(f'| {label} | {weather} | {protocol} | ' + ' | '.join(f'{ap[m]:+.6f}' for m in METRICS) + ' |')
    lines += ['', '| Fog 模型 | 预测框 | None帧 | F TP@.7 | F FP@.7 | F 漏检 | 保留目标TP | 移除目标TP |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, c in data['fog_matching'].items():
        lines.append(f"| {name} | {c['predictions']} | {c['none_frames']} | {c['F_tp70']} | {c['F_fp70']} | {c['F_fn70']} | {c['retained_tp70']} | {c['removed_tp70']} |")
    lines += ['', '保留/移除目标 TP 是同一次 F 一对一匹配的分组；不能把它当作单独用天气 GT 匹配得到的 O TP。', '',
              '| Fog 模型 | 协议 | TP@.7 | FP@.7 | 漏检 | GT分母 |', '|---|---|---:|---:|---:|---:|']
    for name, counts in data['fog_matching'].items():
        for protocol in PROTOCOLS:
            row = counts['by_protocol'][protocol]
            if row['status'] == 'complete':
                lines.append(f"| {name} | {protocol} | {row['tp']} | {row['fp']} | {row['fn']} | {row['counted_gt']} |")
            else:
                lines.append(f'| {name} | {protocol} | 缺少原始逐帧匹配记录 | — | — | — |')
    lines += ['',
              '| 对照 | 条件 | F ΔAP70 | None跳帧差异 | GT筛选差异 | 合计 O ΔAP70 |', '|---|---|---:|---:|---:|---:|']
    for label, weather_rows in data['protocol_components_pp'].items():
        for weather, metrics in weather_rows.items():
            e = metrics['ap70']
            lines.append(f"| {label} | {weather} | {e['F_delta_pp']:+.6f} | {e['None_skip_component_pp']:+.6f} | {e['GT_filter_component_pp']:+.6f} | {e['O_delta_pp']:+.6f} |")
    lines += ['', '上表是同预测下的算术分解，便于判断更换计分目标和 None 处理后领先是否保留；不能据此证明训练机制的因果作用。', '',
              '| 对照 | 条件 | TP增加帧 | TP减少帧 | TP相同帧 | 目标新增检出 | 目标不再检出 |', '|---|---|---:|---:|---:|---:|---:|']
    for label, weather_rows in data['per_frame_gain_distribution'].items():
        for weather, row in weather_rows.items():
            if row['status'] != 'complete':
                lines.append(f'| {label} | {weather} | 缺少历史逐帧证据 | — | — | — | — |')
            else:
                lines.append(f"| {label} | {weather} | {row['benefited_frames']} | {row['degraded_frames']} | {row['equal_tp_frames']} | {row['newly_detected_fixed_targets']} | {row['no_longer_detected_fixed_targets']} |")
    lines += ['', '逐帧指标及目标 ID 的差异见 frame_gains_*.jsonl。缺少 C/D 原始审计文件时，仅报告已认证 summary 的整体计数，不生成目标身份结论。', '',
        '## 论文公开数字与本次 validation', '',
        '**这张表不属于同集公平对照。Table 5 未明确标注 split；相关 Table 7/8 明确使用 testing。本次仅 validation 829 帧，不读取 test。**', '',
        '| 条件 | 论文纯LiDAR AP70 | 本次作者权重 validation O AP70 | 本次−论文 pp |', '|---|---:|---:|---:|']
    for weather, row in data['paper_differences'].items():
        lines.append(f"| {weather} | {row['paper_AP70']:.4f} | {row['validation_O_AP70']:.10f} | {row['validation_minus_paper_pp']:+.6f} |")
    lines += ['', '论文把指标称为 3D mAP；本次保留冻结实现的 BEV IoU/AP。不得据这张表调整阈值、GT 或宣称同设置复现差额。',
        '来源：[论文 v5 Table 5](https://arxiv.org/html/2411.08402v5#S5.T5)、[官方仓库天气结果](https://github.com/ylwhxht/V2X-R#multi-modal-diffusion-denoising-results-attfuse-train-with-fog-simulation)。',
        '论文 LiDAR+4D Radar+MDD 涉及不同输入模态，不列入本次纯 LiDAR 公平对照。', '',
        'C/D 来源：`' + data['reference_source']['path'] + '`；SHA256：`' + data['reference_source']['sha256'] + '`。',
        '完整精度、审计来源、运行设置的直接核验范围和缺失文件列表见 MATCHED_COMPARISON.json。']
    return '\n'.join(lines)


def conclusions(data):
    lines = ['# 结论', '', '以下数值仅在服务器完整评测成功后生成。', '']
    for weather in CONDITIONS:
        e = data['protocol_components_pp']['D-' + AUTHOR][weather]['ap70']
        prefix = f"{weather}：D−作者 O AP70 {e['O_delta_pp']:+.6f} pp；固定 GT F 为 {e['F_delta_pp']:+.6f} pp。"
        if e['O_delta_pp'] > 0 and e['F_delta_pp'] > 0:
            finding = '在固定 Clean 目标并计入 None 帧后，D 仍领先。'
        elif e['O_delta_pp'] > 0:
            finding = 'D 在 O 下领先，在固定 GT 下没有领先；不能据 O 宣称固定目标检出更强。'
        elif e['F_delta_pp'] > 0:
            finding = 'D 在固定 GT 下领先，O 下没有领先。'
        else:
            finding = '本次 D 在 O 和 F 下均未领先作者。'
        lines += [prefix + finding,
            f"None跳帧差异项 {e['None_skip_component_pp']:+.6f} pp；GT筛选差异项 {e['GT_filter_component_pp']:+.6f} pp。", '']
        extra = e['None_skip_component_pp'] + e['GT_filter_component_pp']
        if e['O_delta_pp'] > 0 and e['F_delta_pp'] > 0 and extra > e['F_delta_pp']:
            lines += ['按算术分解，协议相关的额外差值大于固定 GT 下的差值；固定 GT 下的领先仍存在。此分解不证明训练机制的因果作用。', '']
    lines += ['Fog 的预测框、正确匹配和漏检来自原版匹配器计数，见 MATCHED_COMPARISON.md。',
        '逐帧增益仅在历史原始审计文件通过哈希和 GT 身份检查后输出；缺少时报告明确的缺失来源。', '',
        '公平范围：同一 validation、同一冻结 GT 和评分协议的固定模型性能。C/D 在 validation 上选模；作者 checkpoint 的训练细节和选模过程未由本次验证。',
        '论文 Table 5 的 split 未明确；相关消融表使用 testing，不能把论文公开数字视作同一 validation。',
        '单次评测不能证明统计显著性、训练机制因果关系或正式 test 泛化；本次无训练、无阈值调整、无 test 访问。']
    return '\n'.join(lines)
