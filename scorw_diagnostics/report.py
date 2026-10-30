"""Scene-stratified arithmetic on committed diagnostics; no causal performance claims."""
from pathlib import Path
import os
import numpy as np
from .core import read, write, text, digest
from .voxel_trace import REASONS


def add(a, b):
    for k, v in b.items():
        a[k] = a.get(k, 0) + v


def fraction(a, b):
    return a / b if b else None


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run(c, out, bdir, adir, smoke):
    total = dict(clouds=0, frames=0, original=0, emitted=0, active_CAV_clouds=0, inactive_CAV_clouds=0,
                 clean_s4=0, fog_s4=0)
    clean = {}; fog = {}; origins = {}; bins = {}; scenes = {}; targets = []; examples = []; shards = {}
    for p in sorted((bdir / 'frames').glob('*/*/RESULT.json')):
        r = read(p); s = r['scene']; total['frames'] += 1
        scene = scenes.setdefault(s, dict(frames=0, clouds=0, original=0, clean={}, fog={}, targets=0,
                    detected_clean=0, detected_fog=0, original_target_points=0, clean_target_slots=0, fog_hard_target_slots=0))
        scene['frames'] += 1; per_target = {}
        for cloud in r['clouds']:
            total['clouds'] += 1; total['original'] += cloud['original']; total['emitted'] += cloud['emitted']
            total['active_CAV_clouds' if cloud['active'] else 'inactive_CAV_clouds'] += 1
            total['clean_s4'] += cloud['clean_s4']; total['fog_s4'] += cloud['fog_s4']
            scene['clouds'] += 1; scene['original'] += cloud['original']
            add(clean, cloud['clean']); add(fog, cloud['fog']); add(scene['clean'], cloud['clean']); add(scene['fog'], cloud['fog'])
            for k, v in cloud['by_origin'].items():
                a = origins.setdefault(k, dict(total=0, near_fog_count=0, near_clean_count=0, clean={}, fog={}, near_fog={}, near_clean={}))
                for name in ('total', 'near_fog_count', 'near_clean_count'):
                    a[name] += v[name]
                for name in ('clean', 'fog', 'near_fog', 'near_clean'):
                    add(a[name], v[name])
            for v in cloud['by_original_distance']:
                key = '%g-%g' % (v['low'], v['high']); a = bins.setdefault(key, dict(original=0, clean={}, fog={}))
                a['original'] += v['original']; add(a['clean'], v['clean']); add(a['fog'], v['fog'])
        for obj in r['objects']:
            a = per_target.setdefault(obj['object_id'], dict(scene=s, timestamp=r['timestamp'], object_id=obj['object_id'],
                distance_ego_m=obj['distance_ego_m'], clean_slots=0, fog_hard_slots=0, replaced=0, raw_unique=0,
                detected_clean=obj['detected_clean'], detected_fog=obj['detected_fog'], active_CAVs=0))
            if obj['active_CAV']:
                a['active_CAVs'] += 1; a['clean_slots'] += obj['clean']['PillarVFE_slot']
                a['fog_hard_slots'] += obj['hard_network']; a['raw_unique'] += obj['unique_box_points']; a['replaced'] += obj['weather_replaced']
        for a in per_target.values():
            targets.append(a); scene['targets'] += 1; scene['detected_clean'] += a['detected_clean']; scene['detected_fog'] += a['detected_fog']
            scene['original_target_points'] += a['raw_unique']; scene['clean_target_slots'] += a['clean_slots']; scene['fog_hard_target_slots'] += a['fog_hard_slots']
        examples.extend(r['examples'][:max(0, c['example_count']-len(examples))])
        shards[str(p)] = digest(p)
    expected_frames = c['smoke_frames'] if smoke else c['expected_frames']
    if total['frames'] != expected_frames or (not smoke and total['clouds'] != c['expected_pcds']):
        raise ValueError('Incomplete full Validate coverage')
    if sum(clean.values()) != total['original'] or sum(fog.values()) != total['original']:
        raise ValueError('Final point count conservation failed')
    for k, a in origins.items():
        a['near_fog_fractions'] = {name: fraction(v, a['near_fog_count']) for name, v in a['near_fog'].items()}
    object_bins = {}
    for lo, hi in zip(c['distance_edges_m'], c['distance_edges_m'][1:]):
        a = [r for r in targets if lo <= r['distance_ego_m'] < hi]
        object_bins['%g-%g' % (lo, hi)] = dict(target_instances=len(a),
            **{k: sum(r[k] for r in a) for k in ('raw_unique', 'clean_slots', 'fog_hard_slots', 'replaced')})
    summary = dict(status='passed', mode='smoke' if smoke else 'full', totals=total, clean=clean, fog=fog,
        scenes=scenes, by_origin=origins, source_association_failed=0,
        S3_is_not_network_input=True, model_forward_recomputed=False,
        network_definition='4D points in valid native-collated PillarVFE input slots; not downstream maxpool contribution',
        target_association='original Clean position in frozen D6 3D GT; overlapping points excluded from unique totals')
    write(out / 'POINT_FLOW_SUMMARY.json', summary)
    write(out / 'POINT_FLOW_BY_DISTANCE.json', dict(by_original_sensor_distance=bins, by_original_target_ego_distance=object_bins,
          scope='sensor-range point cohorts and ego-range target cohorts are separate'))
    write(out / 'OBJECT_POINT_FLOW.json', dict(per_target=targets, per_CAV_frame_shards=shards,
          ambiguous_points='per-object and per-cloud counts in immutable frame shards; excluded from unique assignment',
          continuous_frames_independent=False))
    write(out / 'NEAR_FIELD_WEATHER_FLOW.json', dict(radius_m=c['near_field_m'], frame='each CAV original local sensor',
          by_origin=origins, inactive_CAV_separate=True, fractions_denominator='all emitted near-field points of the same origin'))
    paired = dict(per_scene=scenes, target_instances=len(targets),
          descriptive_target_correlation=correlation([a['fog_hard_slots']-a['clean_slots'] for a in targets],
                                                     [int(a['detected_fog'])-int(a['detected_clean']) for a in targets]),
          scene_correlation=correlation([v['fog_hard_target_slots']/max(1,v['clean_target_slots']) for v in scenes.values()],
                                       [(v['detected_fog']-v['detected_clean'])/max(1,v['targets']) for v in scenes.values()]),
          causal_claim=False, target_observations_not_independent_scenes=True)
    write(out / 'OBSERVATION_DETECTION_ASSOCIATION.json', paired)
    lines = ['# 固定规则目标案例', '', '按scene、timestamp、CAV、object_id遍历，取最先出现的40–60m有效目标，最多6例。',
             '距离为目标到ego的三维距离。只用原始Clean位置关联；重叠框点单列。']
    for e in examples:
        a = e['summary']; lines += ['', '## %s / %s / CAV %s / 目标 %s' % (a['scene'],a['timestamp'],a['cav'],a['object_id']),
            '距离 %.3fm；框内原始%d点，唯一归属%d点，重叠归属%d点。' %
            (a['distance_ego_m'],a['raw_box_points'],a['unique_box_points'],a['ambiguous_points']),
            '唯一归属点中：硬回波%d，天气替换%d，来源不确定%d，无返回%d；硬回波进入有效槽位%d。' %
            (a['original_hard'],a['weather_replaced'],a['unknown'],a['lost'],a['hard_network']),
            '天气替换点去向：'+str(a['weather_flow'])]
    text(out / 'EXAMPLE_TARGETS.md', '\n'.join(lines)+'\n'); write(out / 'EXAMPLE_POINT_CASES.json', examples)
    audit = read(adir.parent / 'A1/AP_PROTOCOL_AUDIT.json'); factors = read(adir / 'FACTOR_CONTRIBUTIONS.json')
    lines = ['# SCOR-W诊断结果', '', '范围：%s，%d帧、%d份CAV点云、%d个scene。' %
             ('SMOKE，仅工程检查' if smoke else '完整Validate',total['frames'],total['clouds'],len(scenes)),
             '', '## 问题一：AP差异来自哪里', '',
             '原D6缓存全局排序AP已复现。固定同一预测、GT及匹配结果改变排序后的结果：',
             '', '|天气|指标|全局排序|按帧累计|全局减按帧|', '|---|---|---:|---:|---:|']
    for mode, ts in audit['cache_replay'].items():
        for t, v in ts.items():
            lines.append('|%s|AP%d|%.10f|%.10f|%+.10f|' % (mode,round(float(t)*100),v['True'],v['False'],v['True']-v['False']))
    lines += ['', 'GT和匹配的条件贡献见下表；顺序固定为排序→GT→匹配，完整交叉结果见AP_PROTOCOL_COMPARISON.md。',
              '', '|天气|指标|排序变化|GT变化|匹配变化|', '|---|---|---:|---:|---:|']
    for mode, ts in factors['values'].items():
        for t, v in ts.items():
            lines.append('|%s|AP%d|%+.10f|%+.10f|%+.10f|' % (mode,round(float(t)*100),v['sorting'],v['GT_given_frame_order'],v['matcher_given_native_GT']))
    clean_ap = audit['cache_replay']['clean']
    lines += ['', 'D6 Clean全局排序的AP30−AP70为%.10f，按帧累计为%.10f；两者使用同一份匹配记录。' %
              (clean_ap['0.3']['True']-clean_ap['0.7']['True'],clean_ap['0.3']['False']-clean_ap['0.7']['False'])]
    conditional = factors['values']['clean']['0.7']
    labels = {'sorting':'排序','GT_given_frame_order':'GT定义','matcher_given_native_GT':'匹配规则'}
    largest = max(labels,key=lambda k:abs(conditional[k]))
    lines += ['在预定条件分解中，Clean AP70变化幅度最大的一项为%s（%+.10f）；这只解释固定D6预测下的变化。' %
              (labels[largest],conditional[largest])]
    if factors['historical_residual']:
        lines += ['与历史原生AP的剩余差值（固定D6预测，已改用原生GT、原生匹配和按帧累计）：'+
                  '；'.join('AP%d %+.10f'%(round(float(t)*100),v['unexplained_AP_difference'])
                           for t,v in factors['historical_residual'].items())+'。']
    lines += ['', '已证实部分为固定缓存上的上述变化。历史原生推理的预测框尚未逐帧核实，剩余差异保留未解释。',
              'AP30/50/70间距同时受排序、GT和匹配影响，仅凭间距无法判断模型定位精度。',
              '', '## 问题二：目标信息与近场天气点', '',
              '相同目标唯一归属点中，Clean进入有效槽位%d条，Fog保留目标原位置且进入有效槽位%d条。' %
              (sum(a['clean_slots'] for a in targets),sum(a['fog_hard_slots'] for a in targets)),
              '统计仅覆盖冻结GT中的几何关联点；来源0表示模拟器硬回波，框内点没有逐点车辆语义真值。']
    near = origins.get('1', {}).get('near_fog', {})
    lines += ['近场天气来源点总数%d；自车过滤%d、范围过滤%d、体素边界/容量丢弃%d、进入有效槽位%d、未激活CAV%d。' %
        (origins.get('1',{}).get('near_fog_count',0),near.get('ego_filter',0),near.get('range_filter',0),
         sum(near.get(k,0) for k in ('voxel_boundary','voxel_total_limit','voxel_point_limit')),near.get('PillarVFE_slot',0),near.get('inactive_CAV',0)),
        '被过滤与进入网络两种情况的数量和比例见NEAR_FIELD_WEATHER_FLOW.json，不预设哪一种占主导。',
        '分距离和逐scene结果分别见POINT_FLOW_BY_DISTANCE.json与POINT_FLOW_SUMMARY.json。',
        '观测变化与检测变化的描述性关联见OBSERVATION_DETECTION_ASSOCIATION.json；没有反事实干预，不能证明AP下降的唯一原因。',
        '真实雾天校验仍缺失。本次不训练、不生成天气、不改天气参数、不运行GPU推理。']
    text(out / 'FINAL_DIAGNOSIS.md', '\n'.join(lines)+'\n')
