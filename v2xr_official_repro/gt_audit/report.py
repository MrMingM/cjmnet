"""ID-based paired comparisons and reports; standard library only."""
from collections import Counter
import math

TOLERANCE = dict(center_m=1e-4, dimensions_m=1e-4, yaw_rad=1e-5, corners_m=1e-4)


def geometry(a, b):
    if len(a['box']) != 7 or len(b['box']) != 7:
        raise ValueError('Expected official hwl seven-component boxes')
    x, y = a['box'], b['box']
    delta = dict(center_m=max(abs(x[i] - y[i]) for i in range(3)),
        dimensions_m=max(abs(x[i] - y[i]) for i in range(3, 6)),
        yaw_rad=abs(math.atan2(math.sin(x[6] - y[6]), math.cos(x[6] - y[6]))))
    if 'corners' in a:
        delta['corners_m'] = max(abs(v - w) for p, q in zip(a['corners'], b['corners']) for v, w in zip(p, q))
    if not all(math.isfinite(v) for v in delta.values()):
        raise ValueError('Nonfinite GT geometry')
    return dict(changed=any(v > TOLERANCE[k] for k, v in delta.items()), **delta)


def evidence(condition, object_id):
    rows = []
    for cav in condition['cavs']:
        if object_id in cav['before_point_filter_ids']:
            i = cav['before_point_filter_ids'].index(object_id)
            rows.append(dict(cav=cav['cav'], ego=cav['ego'],
                points=cav['official_points_in_box'][i],
                physical_post_fov_points=cav['physical_post_fov_points_in_box'][i],
                fallback=cav['official_empty_voxel_fallback']))
    best = max((r['points'] for r in rows), default=None)
    physical_best = max((r['physical_post_fov_points'] for r in rows), default=None)
    ego = next((r['points'] for r in rows if r['ego']), None)
    return dict(per_cav=rows, best_cav_points=best, physical_best_cav_points=physical_best,
        best_cavs=[r['cav'] for r in rows if r['points'] == best], ego_points=ego,
        any_fallback=any(r['fallback'] for r in rows))


def compare_frame(frame):
    clean = frame['conditions']['clean']
    comparisons, removals = {}, []
    c = {r['object_id']: r for r in clean['final_gt']}
    for weather in ('fog', 'snow'):
        target = frame['conditions'][weather]
        w = {r['object_id']: r for r in target['final_gt']}
        if (clean['raw_yaml'] != target['raw_yaml']
                or clean['context']['participating_cavs'] != target['context']['participating_cavs']):
            raise ValueError('Raw YAML or participating CAV context differs across weather')
        for left, right in zip(clean['context']['cavs'], target['context']['cavs']):
            if {k: v for k, v in left.items() if k != 'lidar_file'} != {k: v for k, v in right.items() if k != 'lidar_file'}:
                raise ValueError('Ego/CAV pose, transform, time delay or YAML differs across weather')
        if len(clean['cavs']) != len(target['cavs']):
            raise ValueError('Participating CAV count changed')
        for left, right in zip(clean['cavs'], target['cavs']):
            if left['cav'] != right['cav'] or left['before_point_filter_ids'] != right['before_point_filter_ids']:
                raise ValueError('Pre-point-filter candidate IDs changed across weather')
            for a, b in zip(left['before_point_filter_boxes'], right['before_point_filter_boxes']):
                if geometry({'box': a}, {'box': b})['changed']:
                    raise ValueError('Pre-point-filter candidate geometry changed across weather')
        shared, removed, added = sorted(c.keys() & w.keys()), sorted(c.keys() - w.keys()), sorted(w.keys() - c.keys())
        details = [dict(object_id=k, **geometry(c[k], w[k])) for k in shared]
        changed = [d for d in details if d['changed']]
        additions = []
        for k in added:
            ce, we = evidence(clean, k), evidence(target, k)
            reason = ('point_evidence_became_positive' if ce['best_cav_points'] == 0 and we['best_cav_points'] > 0
                      else 'evaluation_range_or_cav_representative_change')
            additions.append(dict(object_id=k, reason=reason, clean_evidence=ce, weather_evidence=we))
        for k in removed:
            ce, we = evidence(clean, k), evidence(target, k)
            point = bool(we['per_cav']) and we['best_cav_points'] == 0
            split = ('ego_and_peer' if (ce['ego_points'] or 0) > 0 and any(r['points'] > 0 and not r['ego'] for r in ce['per_cav'])
                     else 'ego_only' if (ce['ego_points'] or 0) > 0 else 'peer_only')
            removals.append(dict(index=frame['index'], frame_key=frame['frame_key'],
                ego=clean['context']['ego'], weather=weather, object_id=k, clean_box=c[k],
                clean_distance_ego_xy_m=math.hypot(*c[k]['box'][:2]),
                reason='all_cav_zero_points' if point else 'evaluation_range_or_cav_representative_change',
                clean_evidence=ce, weather_evidence=we, clean_evidence_split=split,
                fallback_involved=ce['any_fallback'] or we['any_fallback']))
        comparisons[weather] = dict(retained=len(shared), removed=len(removed), added=len(added),
            retained_ids=shared, removed_ids=removed, added_ids=added, added_details=additions,
            geometry_changed_objects=changed,
            geometry_max_delta={key: max((d[key] for d in details), default=0) for key in TOLERANCE},
            gt_count_delta=len(w) - len(c), ids_equal=set(c) == set(w),
            geometry_equal=not changed, gt_equal=not removed and not added and not changed)
    frame['comparisons'] = comparisons
    fog = {r['object_id']: r for r in frame['conditions']['fog']['final_gt']}
    snow = {r['object_id']: r for r in frame['conditions']['snow']['final_gt']}
    details = [dict(object_id=k, **geometry(fog[k], snow[k])) for k in sorted(fog.keys() & snow.keys())]
    frame['fog_to_snow'] = dict(ids_equal=set(fog) == set(snow),
        geometry_changed_objects=[r for r in details if r['changed']],
        common_ids=len(details), geometry_max_delta={k: max((r[k] for r in details), default=0) for k in TOLERANCE})
    return removals


def distribution(values, edges):
    counts = Counter()
    missing = 0
    for value in values:
        if value is None:
            missing += 1
            continue
        for low, high in zip(edges, edges[1:] + [None]):
            if value >= low and (high is None or value < high):
                counts[f'[{low},{high if high is not None else "inf"})'] += 1
                break
        else:
            raise ValueError('Value outside histogram bins')
    return dict(bins={f'[{lo},{hi if hi is not None else "inf"})': counts[f'[{lo},{hi if hi is not None else "inf"})']
                      for lo, hi in zip(edges, edges[1:] + [None])}, missing=missing)


def summarize(frames, removed):
    full = len(frames) == 829
    totals = {c.title() + '_GT_total': sum(f['conditions'][c]['final_gt_count'] for f in frames)
              for c in ('clean', 'fog', 'snow')}
    summary = dict(status='complete', stage='D-1B', full_validation=full,
        validation_frame_count=len(frames), detector_used=False, skipped_frames=0, **totals,
        geometry_tolerance=TOLERANCE, comparisons={})
    for weather in ('fog', 'snow'):
        pairs = [f['comparisons'][weather] for f in frames]
        losses = [r for r in removed if r['weather'] == weather]
        counts = {k: sum(r[k] for r in pairs) for k in ('retained', 'removed', 'added')}
        net = counts['removed'] - counts['added']
        denominator = totals['Clean_GT_total']
        result = dict(**counts, net_fewer=net,
            removed_percent_clean=100 * counts['removed'] / denominator if denominator else None,
            net_fewer_percent_clean=100 * net / denominator if denominator else None,
            GT_changed_frames=sum(not r['gt_equal'] for r in pairs),
            GT_unchanged_frames=sum(r['gt_equal'] for r in pairs),
            equal_count_but_changed_frames=sum(r['gt_count_delta'] == 0 and not r['gt_equal'] for r in pairs),
            id_changed_frames=sum(not r['ids_equal'] for r in pairs),
            geometry_changed_frames=sum(not r['geometry_equal'] for r in pairs),
            geometry_changed_objects=sum(len(r['geometry_changed_objects']) for r in pairs),
            geometry_max_delta={k: max((r['geometry_max_delta'][k] for r in pairs), default=0) for k in TOLERANCE},
            per_frame_gt_count_delta_distribution=dict(sorted(Counter(r['gt_count_delta'] for r in pairs).items())),
            removed_reason_counts=dict(Counter(r['reason'] for r in losses)),
            added_reason_counts=dict(Counter(a['reason'] for r in pairs for a in r['added_details'])),
            removed_distance_ego_xy_m=distribution([r['clean_distance_ego_xy_m'] for r in losses], [0, 10, 20, 30, 40, 50, 70, 100]),
            removed_clean_best_cav_points=distribution([r['clean_evidence']['best_cav_points'] for r in losses], [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]),
            removed_clean_physical_best_cav_points=distribution([r['clean_evidence']['physical_best_cav_points'] for r in losses], [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]),
            removed_clean_ego_points=distribution([r['clean_evidence']['ego_points'] for r in losses], [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]),
            removed_clean_evidence_split=dict(Counter(r['clean_evidence_split'] for r in losses)),
            removed_best_cav_ids=dict(Counter(cav for r in losses for cav in r['clean_evidence']['best_cavs'])),
            removed_with_fallback_involved=sum(r['fallback_involved'] for r in losses))
        point = result['removed_reason_counts'].get('all_cav_zero_points', 0)
        result['point_filter_share_of_removed'] = point / counts['removed'] if counts['removed'] else None
        result['mostly_point_dependent_removals'] = point > counts['removed'] / 2 if counts['removed'] else None
        if (totals['Clean_GT_total'] != counts['retained'] + counts['removed']
                or totals[weather.title() + '_GT_total'] != counts['retained'] + counts['added']):
            raise ValueError('Paired object accounting does not close')
        summary['comparisons'][weather] = result
    secondary = [f['fog_to_snow'] for f in frames]
    summary['fog_to_snow'] = dict(id_changed_frames=sum(not r['ids_equal'] for r in secondary),
        common_id_pairs=sum(r['common_ids'] for r in secondary),
        geometry_changed_objects=sum(len(r['geometry_changed_objects']) for r in secondary),
        geometry_max_delta={k: max((r['geometry_max_delta'][k] for r in secondary), default=0) for k in TOLERANCE})
    summary['all_conditions_GT_identical'] = (all(r['GT_changed_frames'] == 0 for r in summary['comparisons'].values())
        and not summary['fog_to_snow']['id_changed_frames'] and not summary['fog_to_snow']['geometry_changed_objects'])
    summary['official_weather_AP_directly_comparable_to_paired_fixed_GT_AP'] = False
    summary['fallback'] = {c: dict(cav_occurrences=sum(r['official_empty_voxel_fallback'] for f in frames for r in f['conditions'][c]['cavs']),
        affected_frames=sum(any(r['official_empty_voxel_fallback'] for r in f['conditions'][c]['cavs']) for f in frames)) for c in ('clean', 'fog', 'snow')}
    return summary


def write_results(output, summary):
    full, same = summary['full_validation'], summary['all_conditions_GT_identical']
    lines = ['# D−1B：官方天气 evaluation GT 审计', '',
        f'范围：{summary["validation_frame_count"]}/829 帧；' + ('完整 validation。' if full else '**SMOKE ONLY，不能代表全量结论。**'),
        'GT 独立于 detector；没有 prediction=None 跳帧。', '',
        f'A. 在已审计帧和约定数值容差内，Clean/Fog/Snow 最终 GT **{"完全一致" if same else "不一致"}**。', '',
        '| 条件 | 最终 GT 总数 |', '|---|---:|']
    lines += [f'| {c} | {summary[c + "_GT_total"]} |' for c in ('Clean', 'Fog', 'Snow')]
    lines += ['', 'B. 以下目标按同一物理帧的 object ID 配对；removed 与净减少分别报告。', '',
        '| Clean→天气 | retained | removed | added | 净少 | 净少/Clean | 变化帧 | 几何变化目标 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for weather, r in summary['comparisons'].items():
        percent = r['net_fewer_percent_clean']
        text = f'{percent:.4f}%' if percent is not None else 'N/A（Clean GT=0）'
        lines += [f'| {weather} | {r["retained"]} | {r["removed"]} | {r["added"]} | {r["net_fewer"]} | {text} | {r["GT_changed_frames"]} | {r["geometry_changed_objects"]} |']
    lines += ['', 'C. 每个候选框的点数直接取自官方 native ROI 调用；同时核对天气间候选 ID、几何和 CAV context 一致。']
    for weather, r in summary['comparisons'].items():
        point = r['removed_reason_counts'].get('all_cav_zero_points', 0)
        causal = ('没有删除目标，删除原因占比无定义' if not r['removed'] else
                  f'{point}/{r["removed"]} 个删除目标在天气下所有候选 CAV 框内均为零点；'
                  f'占 {100 * r["point_filter_share_of_removed"]:.2f}%，'
                  f'主要由点依赖过滤导致：{r["mostly_point_dependent_removals"]}')
        lines += [f'- {weather}：{causal}。fallback 涉及 {r["removed_with_fallback_involved"]} 个删除目标。',
                  f'  新增原因计数：{r["added_reason_counts"]}。新增可来自天气点进入框内、临时点，或去重代表框改变后的范围筛选。']
    lines += ['', 'D. 官方天气 AP 与 paired fixed-GT AP 不能直接相减解释为纯模型天气鲁棒性。',
        'GT 有变化时，计分目标集合也随天气改变，难目标可能退出计分，新增点也可能使目标进入计分。',
        '后续应分开报告官方协议与固定 GT 协议，并在相同权重、输入、预测后处理和 AP 实现下做配对控制。',
        '即使本次 GT 完全一致，也只排除 GT 集合差异；官方 prediction=None 跳帧与其他评测差异仍须对齐。',
        '本审计没有计算 AP，也没有修改过滤行为；无法据此量化 AP 偏差。', '',
        '几何容差：center/dimensions/corners=1e-4 m；yaw=1e-5 rad，角度按 2π 周期比较。',
        '点数是各 CAV 经官方裁点/FOV/空体素处理后、送入 GT 过滤的点数。best-CAV 为最大单 CAV 点数；跨 CAV 不相加。',
        'physical_post_fov 点数在临时点生成之前，由相同 native 算子旁路计算；fallback 单独列出。',
        '完整输入、源码、native 扩展、运行配置、逐帧恢复契约见 protocol.json / input_manifest.json / contract.json。', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
