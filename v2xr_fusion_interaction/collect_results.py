"""Produce the full comparison and paired mechanism reports after three workers finish."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from v2xr_fusion_interaction.common.audit import OUT,WEATHERS,atomic_json,identity,read_json,sha256
from diagnostics import fixed_metrics,verify_gt
from support import atomic_jsonl


def markdown(path,lines):
    tmp=path.with_suffix('.tmp')
    tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    tmp.replace(path)


def points_bin(n):
    return '0' if n==0 else '1-5' if n<=5 else '6-20' if n<=20 else '21+'


def compare_targets(group,root):
    parent,control={'F2':(None,None),'F3':('F2','uniform2'),'F4':('F3','uniform3')}[group]
    paths={'P0':OUT/'BASELINE/final_evaluation',group:root/'final_evaluation'}
    previous={'F3':'F2','F4':'F3'}.get(group)
    if previous:
        paths[previous+'_trained']=OUT/previous/'final_evaluation'
    if parent:
        paths.update({parent:root/'controls'/parent,control:root/'controls'/control})
    metrics={}
    for name,path in paths.items():
        verify_gt(path)
        metrics[name]=fixed_metrics(path)
    changes,records={},[]
    for weather in WEATHERS:
        current=defaultdict(lambda:defaultdict(Counter))
        with (root/f'mechanism_{weather}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                frame=json.loads(line)
                key=f'{weather}/{frame["index"]}'
                matched={name:{str(x['object_id']) for x in data[key]['matched_pairs']['0.7']}
                         for name,data in metrics.items()}
                for oid,row in frame['targets'].items():
                    labels=['all',row['source'],'distance/'+row['distance'],
                           'ego_points/'+points_bin(row['ego_lidar']+row['ego_radar']),
                           'neighbor_points/'+points_bin(row['neighbor_lidar']+row['neighbor_radar'])]
                    if row['cohort']:
                        labels.append('historical_lost')
                    for scale,v in row['scales'].items():
                        if v.get('location_counts'):
                            largest=max(v['location_counts'],key=v['location_counts'].get)
                            labels.append(f'scale{scale}/{largest}')
                    present={name:oid in ids for name,ids in matched.items()}
                    for mode,value in present.items():
                        if mode==group:
                            continue
                        for label in labels:
                            counts=current[mode][label]
                            a,b=value,present[group]
                            counts['recovered']+=not a and b
                            counts['lost']+=a and not b
                            counts['both_detected']+=a and b
                            counts['both_missed']+=not a and not b
                    records.append(dict(weather=weather,index=frame['index'],frame=frame['frame'],object_id=oid,
                                        detected=present,**row))
        changes[weather]=current
    if len(records)!=3*6235 or sum(r['cohort'] for r in records)!=327:
        raise ValueError('Incomplete fixed target reconciliation')
    atomic_jsonl(root/'TARGET_COMPARISONS.jsonl',records)
    return changes


def ap70(evaluation,weather):
    return evaluation['results'][weather]['AP']['F']['ap70']


def tables(all_eval):
    lines=['| 组别或同权重控制 | 天气 | 协议 | AP30 | AP50 | AP70 | 匹配 | 误检 | 漏检 | 帧/秒 | 峰值显存 GiB |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group,data in all_eval.items():
        for weather in WEATHERS:
            result=data['results'][weather]
            perf=data['performance'][weather]
            for protocol in ('O','O*','F'):
                ap=result['AP'][protocol]
                counts=result['matches'][protocol]['0.7']
                lines.append(f'| {group} | {weather} | {protocol} | {ap["ap30"]:.6f} | {ap["ap50"]:.6f} | '
                             f'{ap["ap70"]:.6f} | {counts["tp"]} | {counts["fp"]} | {counts["fn"]} | '
                             f'{perf["inference_fps"]:.2f} | {perf["peak_gpu_allocated_bytes"]/2**30:.3f} |')
    return lines


def mechanism_report(group,data,mechanism,changes):
    lines=[f'# {group} 细尺度交互诊断','',
           '| 天气 | 尺度 | 观测分组 | 格数 | 实际修正 RMS | 修正 / 原融合 | 非零比例 | 同权重 F2−F3 差幅 | F3−F4 差幅 |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    pairs=[]
    for weather in WEATHERS:
        for scale in range(2):
            row=mechanism['weather'][weather][f'{scale}/all']
            for label in ('empty','ego_unique','neighbor_unique','multiple'):
                n=row.get(label+'/cells',0)
                if not n:
                    continue
                def value(field):
                    return (row[label+'/'+field+'_square_sum']/n)**.5
                active,base=value('active'),value('base')
                ratio=active/base if base else 0.
                lines.append(f'| {weather} | {scale} | {label} | {n} | {active:.6g} | {ratio:.6g} | '
                             f'{row[label+"/nonzero_delta"]/n:.2%} | {value("F2_F3"):.6g} | {value("F3_F4"):.6g} |')
            for i in range(3):
                prefix=f'pair_{i}'
                n=row.get(prefix+'/cells',0)
                if n:
                    total=sum(row.get(f'pair_{j}/cells',0) for j in range(3))
                    a=(row[prefix+'/raw_square_sum']/n)**.5
                    b=(row[prefix+'/gated_square_sum']/n)**.5
                    pairs.append(f'| {weather} | {scale} | {i} | {n/total:.2%} | {a:.6g} | {b:.6g} |')
    lines+=['','| 天气 | 尺度 | 车对有观测方数 | 格子比例 | 约束前车对 RMS | 约束后车对 RMS |',
            '|---|---:|---:|---:|---:|---:|',*pairs,
            '','| 天气 | 对照 | 历史目标恢复 | 仅邻车有点恢复 / 丢失 | 双方有点恢复 / 丢失 | 全部恢复 / 丢失 |',
            '|---|---|---:|---:|---:|---:|']
    for w,entries in changes.items():
        for mode,groups in entries.items():
            def trans(label):
                row=groups.get(label,{})
                return f'{row.get("recovered",0)} / {row.get("lost",0)}'
            lines.append(f'| {w} | {mode} | {groups.get("historical_lost",{}).get("recovered",0)} | '
                         f'{trans("neighbor_only")} | {trans("both")} | {trans("all")} |')
    if group=='F4':
        lines+=['','| 天气 | 尺度 | 独有格 | 同权重 F3 非零 | F4 置零比例 |',
                '|---|---:|---|---:|---:|']
        for w in WEATHERS:
            for scale in range(2):
                row=mechanism['weather'][w][f'{scale}/all']
                for label in ('ego_unique','neighbor_unique'):
                    prior=row.get(label+'/F3_nonzero',0)
                    after=row.get(label+'/F4_nonzero',0)
                    fraction=f'{(prior-after)/prior:.2%}' if prior else '—'
                    lines.append(f'| {w} | {scale} | {label} | {prior} | {fraction} |')
    lines+=['','按车辆数分层的统计在 MECHANISM.json。逐目标距离、点数、各尺度区域以及同权重控制结果在 TARGET_COMPARISONS.jsonl。']
    return lines


def assess(groups):
    lines=['# 方法判断','','## 正式成绩与同权重控制','']
    verdict={}
    base=groups['P0']['evaluation']
    for group in ('F2','F3','F4'):
        data=groups[group]
        full=data['evaluation']
        d={w:ap70(full,w)-ap70(base,w) for w in WEATHERS}
        text='、'.join(f'{w} {100*v:+.3f} 个百分点' for w,v in d.items())
        lines.append(f'- **{group} 对本次 P0 复测**。{text}。')
        same={}
        for mode,control in data['controls'].items():
            same[mode]={w:ap70(full,w)-ap70(control,w) for w in WEATHERS}
            lines.append(f'  同权重对 {mode}，'+ '、'.join(f'{w} {100*v:+.3f} 个百分点' for w,v in same[mode].items())+'。')
        weather=['fog','snow']
        verdict[group]=dict(delta_vs_p0=d,delta_vs_same_weight=same,
            adverse_weather_pattern='两种天气均提升' if all(d[w]>0 for w in weather)
                 else '结果具有天气依赖性' if d['fog']*d['snow']<0 else '两种天气未同时提升')
        previous={'F3':'F2','F4':'F3'}.get(group)
        if previous:
            control='uniform2' if group=='F3' else 'uniform3'
            improvement=all(ap70(full,w)>ap70(groups[previous]['evaluation'],w) for w in weather)
            beats_controls=all(same[control][w]>0 and same[previous][w]>0 for w in weather)
            supports=improvement and beats_controls
            if group=='F4':
                def net(w,name):
                    item=data['target_changes'][w][name].get('neighbor_only',{})
                    return item.get('recovered',0)-item.get('lost',0)
                supports &= all(net(w,previous)>=0 and net(w,control)>=0 for w in weather)
                supports &= any(net(w,control)>0 for w in weather)
            verdict[group]['mechanism']='有支持证据' if supports else '机制证据不足'
            lines.append(f'  {verdict[group]["mechanism"]}。联合核对独立训练对照、同权重原公式和等幅度对照。')
        else:
            mechanism=read_json(OUT/group/'MECHANISM.json')['weather']
            relative=[]
            for w in weather:
                for scale in range(2):
                    row=mechanism[w][f'{scale}/all']
                    base_energy=sum(v for k,v in row.items() if k.endswith('/base_square_sum'))
                    effect=sum(v for k,v in row.items() if k.endswith('/active_square_sum'))
                    relative.append((effect/base_energy)**.5 if base_energy else 0.)
            verdict[group]['relative_residual_rms_fog_then_snow']=relative
            lines.append('  Fog/Snow 两尺度的修正与原输出 RMS 比分别为 '+ ' / '.join(f'{v:.4%}' for v in relative)+'。')
    lines+=['','本验证集已多次用于研究方向筛选。以上为单种子验证集结果，论文结论需由独立测试集或留出场景确认。']
    return lines,verdict


def main():
    contract=identity(read_json(OUT/'CONTRACT.json'))
    groups={'P0':read_json(OUT/'BASELINE/FINAL_RESULTS.json')}
    if groups['P0']['status']!='complete' or groups['P0']['contract']!=contract:
        raise ValueError('Canonical P0 evaluation differs')
    for group in ('F2','F3','F4'):
        root=OUT/group
        data=read_json(root/'EVALUATED.json')
        if (data['status']!='complete' or data['contract']!=contract
                or data['mechanism_sha256']!=sha256(root/'MECHANISM.json')
                or any(sha256(root/name)!=digest for name,digest in data['mechanism_artifacts'].items())):
            raise ValueError(f'{group} evidence incomplete')
        targets=compare_targets(group,root)
        mechanism=read_json(root/'MECHANISM.json')
        report=root/'MECHANISM_ANALYSIS.md'
        markdown(report,mechanism_report(group,data,mechanism,targets))
        data.update(target_changes=targets,mechanism_report_sha256=sha256(report),
                    target_comparisons_sha256=sha256(root/'TARGET_COMPARISONS.jsonl'))
        atomic_json(root/'FINAL_RESULTS.json',data)
        groups[group]=data
    all_eval={group:data['evaluation'] for group,data in groups.items()}
    for group,data in groups.items():
        all_eval.update({group+'/'+k:v for k,v in data.get('controls',{}).items()})
    metrics=tables(all_eval)
    lines=['# F2、F3、F4 正式对照','',
           '各天气 829 帧。增益统一相对本次 P0 复测。F 采用固定的 6,235 个双模态 Clean 目标。','',
           '| 组别 | 物理 GPU | 总参数 | 新增可训练参数 | 选中轮次 |',
           '|---|---:|---:|---:|---:|']
    for group,row in groups.items():
        lines.append(f'| {group} | {row["device"]["physical_gpu"]} | {row["parameters"]:,} | '
                     f'{row["trainable_parameters"]:,} | {row.get("selected",{}).get("epoch",18)} |')
    lines+=['',*metrics]
    markdown(OUT/'SUMMARY.md',lines)
    assessment,verdict=assess(groups)
    markdown(OUT/'METHOD_ASSESSMENT.md',assessment)
    atomic_json(OUT/'FINAL_RESULTS.json',dict(status='complete',contract=contract,groups=groups,verdict=verdict))
    entry=['## 细尺度跨车交互 F2/F3/F4 正式结果','',
           '三组均从 P0 第 18 轮起步，冻结原网络并各训练 20 轮。使用卡 2、3、4 并行。'
           '使用固定 Clean 验证 O AP70 选模。正式目录为 `'+str(OUT)+'`。','',
           *lines[4:], '', *assessment[1:],
           '', '完整逐目标结果、机制统计及日志均见正式目录。结果 SHA256 `'+sha256(OUT/'FINAL_RESULTS.json')+'`。']
    markdown(OUT/'AI_CONTEXT_APPEND.md',entry)
    context=HERE.parent/'AI_CONTEXT.md'
    marker=f'<!-- v2xr-fusion-interaction-v1 {contract} -->'
    if context.is_file() and marker.encode() not in context.read_bytes():
        with context.open('ab') as stream:
            stream.write(('\n\n'+marker+'\n'+'\n'.join(entry)+'\n').encode('utf-8'))
    print('ALL THREE INTERACTION EXPERIMENTS COMPLETE',flush=True)


if __name__=='__main__':
    main()
