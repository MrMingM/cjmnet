"""B0/B2 cached F comparison. No point-cloud IO, model, Torch or inference."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
import cache_core as core


def relabel(value):
    """The unchanged B1 helper uses B1/b1 as internal comparison labels only."""
    if isinstance(value, dict):
        return {('B2' if k=='B1' else 'b2_'+k[3:] if k.startswith('b1_') else k): relabel(v) for k,v in value.items()}
    if isinstance(value, list):
        return [relabel(v) for v in value]
    return value


def center(gt):
    return tuple(sum(c[k] for c in gt['corners'])/len(gt['corners']) for k in (0,1))


def neighborhood_proxy(oid, geometries, metadata):
    """GT-center neighborhood is explicitly NOT occupied BEV eight-neighbor cells."""
    origin=center(geometries[oid])
    nearby=[]
    for other,gt in geometries.items():
        if other == oid:
            continue
        xy=center(gt)
        distance=math.hypot(xy[0]-origin[0],xy[1]-origin[1])
        count=metadata[other]['points_sum_across_cavs']
        nearby.append((distance,other,count))
    observed=sorted((x for x in nearby if x[2] is not None and x[2]>0),key=lambda x:(x[0],str(x[1])))
    result=dict(actual_bev_eight_neighbor_status='NOT_CACHED',
                definition='Other fixed-Clean GT centers at 2/5/10m, with positive current-weather native box point counts; not spatial BEV occupancy.',
                nearest_observed_other_gt=None if not observed else dict(distance_m=observed[0][0],object_id=observed[0][1],points_sum_across_cavs=observed[0][2]),
                radii={})
    for radius in (2,5,10):
        candidates=[x for x in nearby if x[0]<=radius]
        result['radii'][str(radius)]=dict(known_observed_other_targets=sum(x[2] is not None and x[2]>0 for x in candidates),
                                         unknown_point_count_targets=sum(x[2] is None for x in candidates))
    # Metadata missing for the target itself makes this comparison unavailable,
    # even if no other reference centers happen to be within the radius.
    valid=metadata[oid]['points_sum_across_cavs'] is not None and result['radii']['5']['unknown_point_count_targets']==0
    number=result['radii']['5']['known_observed_other_targets']
    result['proxy_5m_bin']='missing' if not valid else '0' if number==0 else '1' if number==1 else '2+'
    return result


def enrich(models, evidence, targets):
    for weather in ('fog','snow'):
        result=targets['results'][weather]
        frames={v:{f['index']:f for f in models[v]['frames'] if f['condition']==weather} for v in ('B0','B2')}
        changed={(tuple(t['frame_key']),t['object_id']):t for t in result['target_changes']}
        strata=defaultdict(lambda:dict(gt=0,b0_tp=0,b2_tp=0,lost=0,recovered=0))
        for i,key in enumerate(models['B0']['evaluation_contract']['frame_keys']):
            gts={g['object_id']:g for g in models['B0']['cached'][('clean',i)]['official_gt']}
            current=evidence[i]['conditions'][weather] if evidence is not None else None
            metadata={oid:core.target_metadata(gt,current) for oid,gt in gts.items()}
            matchsets={v:{p['object_id'] for p in frames[v][i]['protocols']['F']['matched_pairs']['0.7']} for v in models}
            for oid in gts:
                proxy=neighborhood_proxy(oid,gts,metadata)
                m=metadata[oid]
                # Joint distance/points prevents attributing a distance trend to
                # density without showing their co-occurrence.
                groups=(('other_observed_gt_within_5m_proxy',proxy['proxy_5m_bin']),
                        ('distance_x_points',m['distance_bin']+' / '+m['point_bin']))
                for group,label in groups:
                    row=strata[(group,label)]
                    a,b=oid in matchsets['B0'],oid in matchsets['B2']
                    row['gt']+=1;row['b0_tp']+=a;row['b2_tp']+=b
                    row['lost']+=a and not b;row['recovered']+=b and not a
                target=changed.get((tuple(key),oid))
                if target is not None:
                    target['neighborhood']=proxy
                    target['point_observation_per_cav']=None if current is None or metadata[oid]['points_sum_across_cavs'] is None else [
                        dict(cav=c.get('cav'),points=c['official_points_in_box'][c['before_point_filter_ids'].index(oid)])
                        for c in current.get('cavs',[]) if oid in c.get('before_point_filter_ids',[]) and 'official_points_in_box' in c]
                    target['prediction_evidence']={}
                    for v in ('B0','B2'):
                        target['prediction_evidence'][v]={}
                        for threshold,pair in target['matches'][v].items():
                            if pair is not None:
                                cache=models[v]['cached'][(weather,i)]
                                target['prediction_evidence'][v][threshold]=dict(**pair,
                                    predicted_corners=cache['prediction_boxes'][pair['prediction_index']])
        extra=[]
        for (group,label),row in sorted(strata.items()):
            missed=row['gt']-row['b0_tp']
            extra.append(dict(group=group,bin=label,**row,b0_recall=row['b0_tp']/row['gt'],b2_recall=row['b2_tp']/row['gt'],
                loss_rate_among_b0_correct=row['lost']/row['b0_tp'] if row['b0_tp'] else None,
                recovery_rate_among_b0_missed=row['recovered']/missed if missed else None))
        result['strata'].extend(extra)
        result['examples']={}
        for status in ('lost','recovered'):
            candidates=[t for t in result['target_changes'] if t['status']==status]
            # Deterministic inspection sample, not a representative random sample.
            result['examples'][status]=sorted(candidates,key=lambda t:(-t['distance_m'],str(t['frame_key']),str(t['object_id'])))[:6]
        result['observation_metadata_missing_targets']=sum(r['gt'] for r in result['strata'] if r['group']=='point_bin' and r['bin']=='missing')
    targets['neighborhood_limit']='Actual B2 BEV occupancy, neighboring features, similarity weights and residual activations are absent from detection caches. Other-target GT-center proximity is a labeled-data proxy, not mechanism evidence.'
    return targets


def save(output,name,value):
    path=output/name;temporary=path.with_suffix(path.suffix+'.tmp')
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temporary.write_text(text,encoding='utf-8',newline='\n');os.replace(temporary,path)


def csv_targets(output,targets):
    fields=['weather','scene','frame','object_id','status','distance_m','points_sum_across_cavs','observing_cavs',
            'other_observed_gt_within_5m_proxy','nearest_observed_other_gt_distance_m',
            'b0_iou07','b0_score07','b2_iou07','b2_score07','b0_iou05','b2_iou05']
    with (output/'TARGET_CHANGES.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for weather in ('fog','snow'):
            for t in targets['results'][weather]['target_changes']:
                near=t['neighborhood']['nearest_observed_other_gt']
                row=dict(weather=weather,scene=t['frame_key'][0],frame=t['frame_key'][1],object_id=t['object_id'],status=t['status'],
                    distance_m=t['distance_m'],points_sum_across_cavs=t['points_sum_across_cavs'],observing_cavs=t['observing_cavs'],
                    other_observed_gt_within_5m_proxy=t['neighborhood']['proxy_5m_bin'],
                    nearest_observed_other_gt_distance_m=None if near is None else near['distance_m'])
                for v in ('B0','B2'):
                    for threshold in ('0.7','0.5'):
                        pair=t['matches'][v][threshold]
                        row[v.lower()+'_iou'+threshold.replace('.','')]=None if pair is None else pair['iou']
                        if threshold=='0.7':row[v.lower()+'_score07']=None if pair is None else pair['score']
                writer.writerow(row)


def markdown(targets,models,warnings):
    lines=['# B0/B2：已有缓存的目标恢复与丢失对照','',
        '只读分析；无训练、无推理、无点云加载、无新AP/IoU计算。使用完整829帧固定Clean标注的官方F协议匹配，主要阈值0.7。目标以frame_key+object_id对齐，同一目标跨帧分别计数。',
        '', '| 天气 | B0 AP70 | B2 AP70 | 差/百分点 | B0正确 | B2正确 | 恢复 | 丢失 | 正确净变化 | 误检净变化 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for w in ('clean','fog','snow'):
        r=targets['results'][w];c=r['counts']
        lines.append(f"| {w} | {100*r['AP']['B0']['ap70']:.6f}% | {100*r['AP']['B2']['ap70']:.6f}% | {r['ap70_delta_pp']:+.6f} | {c['b0_tp']} | {c['b2_tp']} | {c['recovered']} | {c['lost']} | {c['tp_net']:+} | {c['fp_net']:+} |")
    lines+=['', 'AP还受到帧内分数顺序及累计评测序列中正确检测分布影响，计数不能直接换算AP贡献。0.7失配不代表目标完全没有预测；JSON同时列出原缓存0.3、0.5匹配及分数、框角点，用于检查可能的定位退化。',
        '', '## 邻域证据能说明什么','',
        '现有预测缓存没有B2实际使用的八邻域网格点占用、特征、相似度和增强幅度，这部分明确为NOT_CACHED。本脚本不读原点云补算。',
        '可用替代描述：原标注审计中的目标框内点数、有点车辆数；以及2/5/10米内其他固定Clean目标中，当前天气框内点数大于0的目标数和最近距离。这些是目标级观测代理，不能当作0.4米俯视网格八邻域，也不能涵盖路面或背景点。',
        '点数为多车原生计数之和，可能重复观测同一表面。有点车辆数不是空间覆盖率。目标邻近半径只用于诊断分层，不用于调检测阈值或选择权重。']
    for w in ('fog','snow'):
        r=targets['results'][w]
        lines+=['', '## '+w+'：距离、点数及观测分层','',
            '| 分组 | 区间 | 总目标 | B0正确 | B2正确 | 恢复 | 丢失 | B0漏检中恢复率 | B0正确中丢失率 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in r['strata']:
            def rate(key):return '无分母' if row[key] is None else f'{row[key]:.2%}'
            lines.append(f"| {row['group']} | {row['bin']} | {row['gt']} | {row['b0_tp']} | {row['b2_tp']} | {row['recovered']} | {row['lost']} | {rate('recovery_rate_among_b0_missed')} | {rate('loss_rate_among_b0_correct')} |")
        lines+=['', f"点数元数据缺失目标帧数：{r['observation_metadata_missing_targets']}。missing不会按0点处理。距离、点数、覆盖互相关联；分层不是控制其他因素后的因果效应，样本少的区间也不能推广。"]
        for status,label in (('recovered','恢复'),('lost','丢失')):
            lines+=['', '### '+label+'案例（按距离降序选取，不代表随机样本）','']
            for t in r['examples'][status]:
                a=t['matches']['B0']['0.7'];b=t['matches']['B2']['0.7']
                lines.append(f"- 场景/帧 `{t['frame_key']}`，ID `{t['object_id']}`，距离{t['distance_m']:.2f}米，点数{t['points_sum_across_cavs']}，有点车辆{t['observing_cavs']}；B0/B2的0.7匹配：`{json.dumps([a,b],ensure_ascii=False)}`；5米邻近有观测目标分组：{t['neighborhood']['proxy_5m_bin']}。")
    lines+=['', '## 权重及证据','']
    for v in ('B0','B2'):
        s=models[v]['selected'];lines.append(f"- {v}：选中第{s['epoch']}轮；`{s['checkpoint']}`；SHA256 `{s['checkpoint_sha256']}`。")
    lines+=['', '完整恢复/丢失列表见TARGET_CHANGES.csv和TARGET_CHANGES.json；后者保留分层、匹配分数与框、逐帧变化和元数据。输入文件哈希见INPUT_AUDIT.json。',
        '', '## 证据边界','',
        '已证实事实由上述缓存计数、匹配和审计给出。远距离/少点/邻近目标观测与检测变化的共同出现只构成描述性关联。未保存的网格特征使本次无法确认B2是否从具体邻居获得有效信息，也不能据此给出邻域增强的因果机制结论。']
    lines += ['- '+w for w in warnings+targets['limits']]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4')
    args=parser.parse_args()
    core.require(sys.platform=='linux','Real cache diagnosis runs only on server')
    # Stage journals store their original absolute paths. Preserve /data spelling
    # even when the dataset disk has been migrated behind a directory symlink.
    root=Path(args.root).absolute();core.require(root.is_dir(),'Experiment root missing')
    output=root/'b2_cache_diagnosis'
    core.require(not output.is_symlink(),'Refuse output symlink');output.mkdir(exist_ok=True)
    core.require(not any((output/n).exists() for n in ('B0_B2_DIAGNOSIS.md','COMPLETE.json','FAILURE.json')),'Existing diagnosis preserved; refusing repeat overwrite')
    core.require(not any(p.is_symlink() for p in output.rglob('*')),'Output contains symlink')
    import fcntl
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reader=core.Reader()
        try:
            folder=Path(__file__).resolve().parent
            for name,sha in reader.json(folder/'TOOL_MANIFEST.json')['files'].items():
                core.require(Path(name).name==name,'Unsafe manifest member');reader.track(folder/name,sha)
            print('VERIFY B0/B2 selected weights, evaluation identity and full cached F matches',flush=True)
            models={v:core.load_model_evidence(root,v,reader) for v in ('B0','B2')}
            aliases={'B0':models['B0'],'B1':models['B2']}
            integrity=relabel(core.verify_pair(aliases))
            integrity['models']=['B0','B2']
            evidence,warning=core.optional_geometry_evidence(root,aliases,reader)
            print('COMPARE saved matches; analyze distance, points and observation proxies',flush=True)
            targets=enrich(models,evidence,relabel(core.analyze_targets(aliases,evidence)))
            warnings=[] if warning is None else [warning]
            reader.unchanged()
            save(output,'TARGET_CHANGES.json',targets)
            csv_targets(output,targets)
            save(output,'INPUT_AUDIT.json',dict(**integrity,files=reader.files,unchanged_after_read=True))
            save(output,'B0_B2_DIAGNOSIS.md',markdown(targets,models,warnings))
            save(output,'COMPLETE.json',dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),
                outputs={n:core.digest(output/n) for n in ('TARGET_CHANGES.json','TARGET_CHANGES.csv','INPUT_AUDIT.json','B0_B2_DIAGNOSIS.md')}))
            print('COMPLETE',str(output),flush=True)
        except Exception as error:
            save(output,'FAILURE.json',dict(status='blocked',reason=str(error),traceback=traceback.format_exc()))
            save(output,'B0_B2_DIAGNOSIS.md','# 诊断未完成\n\n'+str(error)+'\n\n未启动推理或补算，未修改原实验。\n')
            raise


if __name__=='__main__':main()
