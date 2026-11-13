"""Scene-held-out candidate statistics and official detection replay."""
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json

from context import ARCHIVE,F0,OUT,WEATHER,read,setup,write

run=setup()
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluate import official_match_trace,statistics
from opencood.utils.eval_utils_opv2v import calculate_ap
from support import corner_array
from collect import iou_matrix

CONTROLS=('score','distance')
RADAR=('radar','radar_offset','radar_std','radar_unique')
GSPR=('gspr_points','reliability','uncertainty')
FEATURES=dict(score=('score',),radar_count=('radar',),radar_geometry=RADAR,gspr=GSPR,
              joint=RADAR+GSPR+('lidar','lidar_radar_overlap','lidar_unique'),
              controls=CONTROLS,controls_radar=CONTROLS+RADAR,controls_gspr=CONTROLS+GSPR,
              controls_joint=CONTROLS+RADAR+GSPR+('lidar','lidar_radar_overlap','lidar_unique'))
BUDGET=.10


def cache(split,weathers):
    for weather in weathers:
        paths=sorted((OUT/split/weather).glob('*.json.gz'))
        if not paths or split=='validate' and len(paths)!=829:
            raise ValueError(f'Incomplete collection {split}/{weather}')
        for path in paths:
            with gzip.open(path,'rt') as stream:
                yield json.load(stream)


def stat_rows(rows,frame,t=.7):
    boxes=torch.tensor([r['corners'] for r in rows],dtype=torch.float32).reshape(-1,8,3)
    scores=torch.tensor([r['ranking'] if 'ranking' in r else r['score'] for r in rows],dtype=torch.float32)
    gt=frame['gt']
    stats=statistics()
    matches,_=official_match_trace(boxes if len(rows) else None,scores,torch.from_numpy(corner_array(gt)),
                                   [str(x['object_id']) for x in gt],stats,t)
    return stats[t],matches


def combine(items,threshold=.7):
    scores={k:[] for k in ('tp','fp','score')}
    scores['gt']=0
    for s in items:
        scores['gt']+=s['gt']
        for k in ('tp','fp','score'):
            scores[k].extend(s[k])
    counts=dict(tp=sum(scores['tp']),fp=sum(scores['fp']),gt=scores['gt'])
    counts['fn']=counts['gt']-counts['tp']
    ap=calculate_ap({threshold:deepcopy(scores)},threshold,False)[0] if counts['gt'] else None
    ranked=calculate_ap({threshold:scores},threshold,True)[0] if counts['gt'] else None
    return dict(ap=ap,globally_ranked_ap=ranked,**counts)


def group_ci(rows,fields):
    scenes={}
    for scene,data in rows:
        if scene not in scenes:
            scenes[scene]=np.zeros(len(fields))
        scenes[scene]+=np.asarray([data[k] for k in fields])
    if len(scenes)<2:
        return None
    a=np.stack(list(scenes.values()))
    rng=np.random.default_rng(20261113)
    sampled=a[rng.integers(len(a),size=(2000,len(a)))].sum(1)
    return {k:np.quantile(sampled[:,i],[.025,.975]).tolist() for i,k in enumerate(fields)}


def damaged(frame,oid):
    a=frame['clean_counts'][oid]['lidar']
    b=frame['observed'][oid]
    return a>0 and b['lidar']<=.7*a and b['radar']>0


def label(rows,frame):
    _,match=stat_rows(rows,frame)
    return {m['prediction_index']:str(m['object_id']) for m in match}


def remaining(frame):
    base=frame['final']
    base_hits=set(label(base,frame).values())
    known={p['anchor'] for p in base}
    candidates=[p for p in frame['low'] if p['anchor'] not in known]
    # Retain existing detections and do not reinsert boxes suppressed by them.
    overlap=iou_matrix([r['corners'] for r in candidates],[r['corners'] for r in base])
    keep=~(overlap>frame['nms_threshold']).any(1)
    candidates=[r for r,k in zip(candidates,keep) if k]
    target=dict(frame,gt=[g for g in frame['gt'] if str(g['object_id']) not in base_hits])
    hits=label(candidates,target)
    return candidates,hits,base_hits


def candidate_rows(frames,low=False):
    for frame in frames:
        if low:
            rows,hits,_=remaining(frame)
        else:
            rows=frame['final']
            hits=label(rows,frame)
        gt=[g['corners'] for g in frame['gt']]
        overlaps=iou_matrix([r['corners'] for r in rows],gt)
        for i,p in enumerate(rows):
            oid=hits.get(i)
            closest=oid
            if closest is None and overlaps.shape[1] and overlaps[i].max()>=.1:
                closest=str(frame['gt'][overlaps[i].argmax()]['object_id'])
            yield dict(p,scene=frame['key'][0],frame=frame['index'],weather=frame['weather'],
                       y=int(oid is not None),damage=None if closest is None else damaged(frame,closest))


def matrix(rows,fields):
    result=[]
    for row in rows:
        one=[]
        for k in fields:
            v=row[k]
            one.extend(v if isinstance(v,list) else [v])
        result.append(one)
    return np.asarray(result,dtype=float)


def model():
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
                         LogisticRegression(max_iter=1000,C=1.))


def assess(y,score,limit):
    order=np.argsort(-score,kind='stable')
    yy=y[order]
    keep=np.cumsum(1-yy)<=limit
    return dict(n=len(y),positive=int(y.sum()),roc_auc=float(roc_auc_score(y,score)) if len(set(y))==2 else None,
        pr_auc=float(average_precision_score(y,score)) if y.sum() else None,
        tp_at_fp_budget=int(yy[keep].sum()),fp_used=int((1-yy[keep]).sum()),fp_budget=limit)


def scene_auc_ci(y,score,scenes):
    scene_ids=np.unique(scenes)
    if len(scene_ids)<2:
        return None
    rng=np.random.default_rng(20261113)
    groups=[np.flatnonzero(scenes==s) for s in scene_ids]
    draws=[]
    for choices in rng.integers(len(groups),size=(300,len(groups))):
        sel=np.concatenate([groups[i] for i in choices])
        if len(set(y[sel]))==2:
            draws.append(roc_auc_score(y[sel],score[sel]))
    return np.quantile(draws,[.025,.975]).tolist() if draws else None


def run_a(dev,val,low=False):
    train=list(candidate_rows(dev,low))
    test=list(candidate_rows(val,low))
    yy=np.asarray([r['y'] for r in train])
    y=np.asarray([r['y'] for r in test])
    groups=np.asarray([r['scene'] for r in train])
    scene=np.asarray([r['scene'] for r in test])
    if set(groups)&set(scene):
        raise ValueError('Train development and formal validation scenes overlap')
    budget=int((1-y).sum()*.10)
    slices={'all':np.ones(len(test),bool)}
    for weather in WEATHER:
        slices[weather]=np.asarray([r['weather']==weather for r in test])
    for a,b in ((0,30),(30,60),(60,1e9)):
        slices[f'distance_{a}_{b}']=np.asarray([a<=r['distance']<b for r in test])
    for flag in (True,False,None):
        slices[f'damage_{flag}']=np.asarray([r['damage'] is flag for r in test])
    result={}
    fitted={}
    for name,fields in FEATURES.items():
        x,z=matrix(train,fields),matrix(test,fields)
        if name in ('score','radar_count'):
            oof=x[:,0]
            score=z[:,0]
        else:
            oof=np.empty(len(train))
            for tr,te in GroupKFold(n_splits=5).split(x,yy,groups):
                oof[te]=model().fit(x[tr],yy[tr]).predict_proba(x[te])[:,1]
            fitted[name]=model().fit(x,yy)
            score=fitted[name].predict_proba(z)[:,1]
        result[name]=dict(dev_oof=assess(yy,oof,int((1-yy).sum()*.1)),
            evaluation={k:assess(y[m],score[m],int((1-y[m]).sum()*.1)) for k,m in slices.items()},
            roc_auc_scene_ci=scene_auc_ci(y,score,scene))
    dist={}
    for label in (0,1):
        group=[r for r in test if r['y']==label]
        dist[str(label)]={}
        for field in ('score','distance','radar','lidar','reliability','uncertainty','radar_std','radar_offset','lidar_radar_overlap'):
            a=matrix(group,[field])
            dist[str(label)][field]=[np.quantile(col[np.isfinite(col)],[.1,.5,.9]).tolist()
                if np.isfinite(col).any() else None for col in a.T]
    return dict(metrics=result,distributions=dist,dev_scenes=len(set(groups)),val_scenes=len(set(scene)),
                dev_candidates=len(train),val_candidates=len(test),budget=budget),fitted


def coverage(frames):
    out=defaultdict(Counter)
    for frame in frames:
        hits=set(label(frame['final'],frame).values())
        c=frame['coverage']
        d=out[frame['weather']]
        d.update(dict(zip(('raw_candidates','above_threshold','nms_input','nms_output','final_candidates','lower_candidates'),c['counts'])))
        for i,g in enumerate(frame['gt']):
            oid=str(g['object_id'])
            if oid in hits:
                continue
            stages=('raw','above','nms_in','nms_out','final')
            names=('no_raw','score_removed','geometry_removed','nms_removed','range_removed')
            reason=next((k for k,s in zip(names,stages) if not c[s][i]),'has_final_overlap')
            d[reason]+=1
            d['missed']+=1
            if damaged(frame,oid):
                d['damaged_'+reason]+=1
    return out


def ranking(rows,name,fitted):
    if name=='lower_score':
        return np.asarray([r['score'] for r in rows])
    if name=='radar_count':
        return np.asarray([r['radar'] for r in rows])
    return fitted['controls_joint'].predict_proba(matrix(rows,FEATURES['controls_joint']))[:,1] if rows else np.empty(0)


def added(frame,rows,rank,threshold):
    order=np.argsort(-rank,kind='stable')
    kept=[rows[i] for i in order if rank[i]>=threshold]
    # Keep all official detections ahead of optional additions during official matching.
    return frame['final']+[dict(r,ranking=.049*(1-k/max(1,len(kept)))) for k,r in enumerate(kept)]


def p1(dev,val,fitted):
    variants=('lower_score','radar_count','radar_gspr')
    # Set each cutoff from training-only development with a common added FP cap.
    thresholds={}
    dev_rows=[(f,*remaining(f)[:2]) for f in dev]
    for name in variants:
        scores=[]; y=[]
        for f,r,h in dev_rows:
            scores.extend(ranking(r,name,fitted))
            y.extend(i in h for i in range(len(r)))
        scores=np.asarray(scores); y=np.asarray(y,dtype=bool)
        candidates=np.unique(scores)[::-1]
        neg=np.sort(scores[~y])
        fp=len(neg)-np.searchsorted(neg,candidates,side='left')
        keep=np.flatnonzero(fp<=int(BUDGET*len(dev)))
        thresholds[name]=float(candidates[keep[-1]]) if len(keep) else None

    results={}
    for weather in WEATHER:
        frames=[f for f in val if f['weather']==weather]
        all_stats=defaultdict(list)
        deltas=defaultdict(list)
        pools=defaultdict(list)
        for f in frames:
            rows,_,base_hits=remaining(f)
            base,_=stat_rows(f['final'],f)
            all_stats['F0'].append(base)
            lower,_=stat_rows(f['low'],f)
            all_stats['lower_threshold_0.05'].append(lower)
            for name in variants:
                rank=ranking(rows,name,fitted)
                thresh=thresholds[name] if thresholds[name] is not None else float('inf')
                s,h=stat_rows(added(f,rows,rank,thresh),f)
                all_stats[name].append(s)
                recovered={m['object_id'] for m in h}-base_hits
                deltas[name].append((f['key'][0],dict(tp=s['tp'].count(1)-sum(base['tp']),
                    fp=s['fp'].count(1)-sum(base['fp']),damaged_recovered=sum(damaged(f,o) for o in recovered))))
                order=np.argsort(-rank,kind='stable')
                ranked=[rows[i] for i in order]
                target=dict(f,gt=[g for g in f['gt'] if str(g['object_id']) not in base_hits])
                match=label([dict(r,ranking=1-k/(len(ranked)+1)) for k,r in enumerate(ranked)],target)
                pools[name].extend((float(rank[i]),f['index'],k,k in match) for k,i in enumerate(order))

            # GT oracle: greedily choose unique missing GT matches from the same low pool.
            target=dict(f,gt=[g for g in f['gt'] if str(g['object_id']) not in base_hits])
            matches=label(rows,target)
            chosen=[r for i,r in enumerate(rows) if i in matches]
            oracle,_=stat_rows(added(f,chosen,np.asarray([r['score'] for r in chosen]),0),f)
            all_stats['gt_oracle'].append(oracle)
        frontier={}
        max_fp=int(BUDGET*len(frames))
        for name,pool in pools.items():
            ordered=sorted(pool,key=lambda r:(-r[0],r[1],r[2]))
            fp=tp=0
            for _,_,_,hit in ordered:
                if not hit and fp==max_fp:
                    break
                fp+=not hit
                tp+=hit
            frontier[name]=dict(recovered=tp,fp=fp,budget=max_fp,
                                note='Offline ranked frontier; deployment cutoff comes from training scenes')
        paired=[]
        for (scene,a),(_,b) in zip(deltas['lower_score'],deltas['radar_gspr']):
            paired.append((scene,{k:b[k]-a[k] for k in a}))
        results[weather]=dict(joint_minus_score_scene_ci=group_ci(paired,('tp','fp','damaged_recovered')),
            metrics={k:combine(v) for k,v in all_stats.items()},
            budget_frontier=frontier,deltas={k:dict(total={m:sum(x[m] for _,x in v) for m in ('tp','fp','damaged_recovered')},
             scene_ci=group_ci(v,('tp','fp','damaged_recovered'))) for k,v in deltas.items()})
    return dict(training_cutoffs=thresholds,fp_budget_per_frame=BUDGET,weather=results,coverage=coverage(val))



def run_b():
    from scipy.stats import spearmanr
    records=[read(OUT/'B'/f'{i:06d}.json') for i in range(120)]
    modes=('full','radar_off','shuffle','cross_frame')
    report={}
    for name in modes:
        total={str(t):combine([r['outputs'][name]['scoring'][str(t)] for r in records],t) for t in (.3,.5,.7)}
        scores=[]
        changes=defaultdict(Counter)
        blocks=[]
        archives=Counter()
        for i,r in enumerate(records):
            a,b=r['outputs']['full'],r['outputs'][name]
            hits=lambda x:{str(m['object_id']) for m in x['scoring']['0.7']['matches']}
            base,new=hits(a),hits(b)
            damage=set(r['damage'])
            for tag,first,second in (('damaged_radar',base&damage,new&damage),('remaining',base-damage,new-damage)):
                changes[tag].update(dict(base=len(first),current=len(second),recovered=len(second-first),lost=len(first-second)))
            overlap=iou_matrix(a['boxes'],b['boxes'])
            pairs=[]
            ia,ib=set(),set()
            for x,y in sorted(zip(*np.where(overlap>=.5)),key=lambda k:(-overlap[k],*k)):
                if x not in ia and y not in ib:
                    ia.add(x)
                    ib.add(y)
                    pairs.append((a['scores'][x],b['scores'][y]))
            old_tp={p['prediction_index'] for p in a['scoring']['0.7']['matches']}
            new_tp={p['prediction_index'] for p in b['scoring']['0.7']['matches']}
            changes['boxes'].update(dict(paired=len(pairs),new_tp=len(new_tp-ib),
                new_fp=len(b['boxes'])-len(ib)-len(new_tp-ib),lost_tp=len(old_tp-ia),
                lost_fp=len(a['boxes'])-len(ia)-len(old_tp-ia)))
            scores.extend(b['scores'])
            if len(pairs)>1:
                aa,bb=np.asarray(pairs).T
                if aa.std()>0 and bb.std()>0:
                    changes['boxes']['spearman_sum']+=float(spearmanr(aa,bb).statistic)
                    changes['boxes']['spearman_frames']+=1
            if 'archived_matches' in b:
                archives.update(dict(predictions=b['archived_prediction_count'],tp=len(b['archived_matches']),
                    different_target_frames=int(hits(b)!={str(m['object_id']) for m in b['archived_matches']})))
            blocks.append((f'{r["frame"][0]}:{i//16}',dict(tp=len(new)-len(base),
                fp=sum(b['scoring']['0.7']['fp'])-sum(a['scoring']['0.7']['fp']),
                damaged=len(new&damage)-len(base&damage))))
        feat={k:float(np.mean([r['outputs'][name][k]['rms'] for r in records]))
              for k in ('lidar_features','radar_features','head_scores')}
        report[name]=dict(metrics=total,changes_vs_full=changes,scene_block_ci=group_ci(blocks,('tp','fp','damaged')),
                          score_quantiles=np.quantile(scores,[.1,.5,.9]).tolist() if scores else None,
                          features=feat,archive_comparison=archives)
    return dict(frames=120,scenes=len({r['frame'][0] for r in records}),modes=report,
                checkpoint=read(OUT/'B_COMPLETE.json')['weight'],
                repeated_native_max_abs=records[0]['head']['repeat_max_abs'])


def markdown(name,lines):
    (OUT/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def paired_a():
    counts=Counter()
    rows=[]
    frames=[read(OUT/'A_pairs'/f'{i:06d}.json') for i in range(120)]
    for f in frames:
        on,off=f['on'],f['off']
        recovered=set(f['recovered'])
        counts.update(recovered=len(recovered),lost=len(f['lost']),paired=len(f['pairs']),
            new_true=sum(on[i]['target'] is not None for i in f['new']),
            new_false=sum(on[i]['target'] is None for i in f['new']),
            disappeared_true=sum(off[i]['target'] is not None for i in f['disappeared']),
            disappeared_false=sum(off[i]['target'] is None for i in f['disappeared']))
        for i,r in enumerate(on):
            if r['target'] in recovered or (r['target'] is None and i in f['new']):
                rows.append(dict(r,y=int(r['target'] in recovered)))
    y=np.asarray([r['y'] for r in rows])
    result=dict(counts=counts,scenes=len({f['frame'][0] for f in frames}),samples=len(rows))
    if len(set(y))==2:
        result['positive_negative']=dict(positive=int(y.sum()),negative=int((1-y).sum()))
        result['radar_point_count']=assess(y,np.asarray([r['radar'] for r in rows]),int((1-y).sum()*.1))
        result['score']=assess(y,np.asarray([r['score'] for r in rows]),int((1-y).sum()*.1))
    return result


def report_all():
    d0=read(OUT/'D0.json')
    markdown('D0_RADAR_FIELD_AUDIT.md',[
        '# Radar 第四通道核验','',
        f'抽查 {len(d0["samples"])} 份雷达文件，场景覆盖 {d0["sampled_scenes"]}。复用逐包字段清单。',
        '发布 PCD 字段为 x、y、z 和打包的 rgb。Open3D 将 RGB 的三个 8 位分量除以 255，实际模型只收到红色分量。',
        f'原始位解包与 Open3D 颜色最大差 {d0["decoder_max"]:.3g}，加载器红通道最大差 {d0["loader_max"]:.3g}。',
        f'各颜色通道分位点为 {d0["rgb_quantiles"]}，离散值数量为 {d0["rgb_unique"]}。三通道相同的点占 {d0["rgb_same_fraction"]:.3%}。',
        '论文补充材料确认模拟器原始阵列包含速度。现有公开仓库与样本未建立导出到 RGB 的映射和量纲，物理含义标为未知。',
        '绿色与蓝色分量未进入模型。颜色分量不同时，红色投影不能无损保留完整 RGB。缺少编码映射，无法判断丢失了哪些物理量。',
        'A 使用位置、几何、点数和 GSPR。B 操纵冻结模型实际收到的雷达张量，结论适用于当前输入表征。',
        '', '证据见 D0.json、原始包清单和加载器哈希。',f'[官方补充材料]({d0["supplementary"]})'])
    b=run_b()
    write(OUT/'B_RESULTS.json',b)
    bl=['# B 雷达内容敏感性','',
        f'同一 F3 第 10 轮权重。{b["frames"]} 个 Fog 帧、{b["scenes"]} 个场景。重复前向最大绝对差 {b["repeated_native_max_abs"]:.3g}。',
        '', '| 干预 | F AP30 | F AP50 | F AP70 | TP / FP / FN | 受损目标匹配 |',
        '|---|---:|---:|---:|---:|---:|']
    for n,r in b['modes'].items():
        a=r['metrics']; c=a['0.7']
        bl.append(f'| {n} | {a["0.3"]["ap"]:.6f} | {a["0.5"]["ap"]:.6f} | {c["ap"]:.6f} | {c["tp"]} / {c["fp"]} / {c["fn"]} | {r["changes_vs_full"]["damaged_radar"]["current"]} |')
    bl+=['','shuffle 保持柱坐标，按车移动柱内点及其内容，再重新编码。跨帧替换使用同场景相隔约半段的帧，其雷达仍在供体主车坐标中。',
         '全组在同一批重新加载的输入上运行。历史预测文件逐帧核对哈希，并作为辅助对照。',
         '受损和普通目标、预测增减、置信度、排序和幅度见 B_RESULTS.json。区间按 16 帧场景块配对重采样。',
         '对照测量模型对雷达内容及配准的敏感性。幅度、错位和分布变化都可影响结果，当前证据无法定位具体网络层。']
    markdown('B_RADAR_SENSITIVITY_REPORT.md',bl)
    dev=list(cache('train_dev',('clean','fog')))
    val=list(cache('validate',WEATHER))
    a,_=run_a(dev,val)
    low,_=run_a(dev,val,low=True)
    paired=paired_a()
    write(OUT/'A_RESULTS.json',dict(final=a,low=low,paired_f0=paired))
    al=['# A 有效雷达证据的可分性','',
        f'训练开发集 {a["dev_scenes"]} 个场景、{a["dev_candidates"]} 个候选。正式验证 {a["val_scenes"]} 个场景、{a["val_candidates"]} 个候选。',
        '标签来自固定 Clean 真值的官方一对一匹配。候选缓存保留单车与多车证据。无 GSPR 点时统计为空，统计拟合在训练集内插补并保留缺失指示。',
        '', '| 特征 | 验证 ROC-AUC | 验证 PR-AUC | 10% FP 排序预算保留 TP |',
        '|---|---:|---:|---:|']
    for name,r in a['metrics'].items():
        m=r['evaluation']['all']
        al.append(f'| {name} | {m["roc_auc"]:.4f} | {m["pr_auc"]:.4f} | {m["tp_at_fp_budget"]} |')
    al+=['',f'F0 同输入开关诊断覆盖 120 个 Fog 帧及 {paired["scenes"]} 个场景。逐目标恢复和丢失为 {dict(paired["counts"])}。',
         f'恢复目标与开雷达新增误检的对照统计为 {dict(paired)}。',
         'A_RESULTS.json 保存低分池、距离、天气、点损分层、特征分位数、场景重采样区间和训练场景五折结果。',
         '候选正确性的可分性与雷达开关的因果贡献是两个指标，固定权重恢复结果见 B。']
    markdown('A_SEPARABILITY_REPORT.md',al)

    scenes=sorted({f['key'][0] for f in dev})
    calibration=set(scenes[::4])
    train=[f for f in dev if f['key'][0] not in calibration]
    held=[f for f in dev if f['key'][0] in calibration]
    rows=list(candidate_rows(train,low=True))
    fit={'controls_joint':model().fit(matrix(rows,FEATURES['controls_joint']),[r['y'] for r in rows])}
    p=p1(held,val,fit)
    p.update(training_scenes=len(set(scenes)-calibration),calibration_scenes=sorted(calibration))
    write(OUT/'P1_RESULTS.json',p)
    pl=['# P1 低分候选恢复可行性','',
        '检测头全部解码。原后处理依次做分数筛选、几何过滤、去重和范围过滤。低分池采用固定 0.05 下限，使用原几何与去重，排除与正式框重叠的追加项。',
        '统计规则拟合和阈值校准使用不同的训练场景。验证集等预算排序是离线度量。真值选取为当前候选池的诊断参照。',
        '', '| 天气 | 模式 | F AP70 | TP / FP / FN |',
        '|---|---|---:|---:|']
    for weather,r in p['weather'].items():
        for name,m in r['metrics'].items():
            pl.append(f'| {weather} | {name} | {m["ap"]:.6f} | {m["tp"]} / {m["fp"]} / {m["fn"]} |')
        pl.extend(['',f'{weather} 漏检阶段计数 {dict(p["coverage"][weather])}',
                   f'{weather} 相同误检预算的候选恢复 {r["budget_frontier"]}',''])
    pl+=['完整中间统计、场景区间、全局排序 AP 和开发阈值见 P1_RESULTS.json。']
    markdown('P1_FEASIBILITY_REPORT.md',pl)
    fog=p['weather']['fog']
    frontier=fog['budget_frontier']
    gap=frontier['radar_gspr']['recovered']-frontier['lower_score']['recovered']
    oracle=fog['metrics']['gt_oracle']['tp']-fog['metrics']['F0']['tp']
    joint=fog['deltas']['radar_gspr']['total']
    score=fog['deltas']['lower_score']['total']
    decision='INSUFFICIENT_EVIDENCE'
    if (gap>0 and joint['tp']>score['tp'] and joint['fp']<=score['fp']
            and fog['metrics']['radar_gspr']['ap']>fog['metrics']['F0']['ap']
            and fog['joint_minus_score_scene_ci']['tp'][0]>0):
        decision='GO_P1'
    markdown('FINAL_RESEARCH_DECISION.md',[
        '# 研究决策', '', f'**{decision}**','',
        f'1. 同权重 F3 的真实雷达条件匹配 {b["modes"]["full"]["changes_vs_full"]["damaged_radar"]["current"]} 个受损目标，关雷达匹配 {b["modes"]["radar_off"]["changes_vs_full"]["damaged_radar"]["current"]} 个。证据覆盖历史 120 帧 Fog。',
        f'2. F0 正常候选仅检测分数的 ROC-AUC 为 {a["metrics"]["score"]["evaluation"]["all"]["roc_auc"]:.4f}，加入雷达、GSPR 和距离后为 {a["metrics"]["controls_joint"]["evaluation"]["all"]["roc_auc"]:.4f}。',
        f'3. 当前 F0 Fog 低分池的真值选取多恢复 {oracle} 个目标。特征干预和候选恢复测量不同条件下的收益，现有实验未估计两条路径各自的贡献份额。',
        f'4. Fog 相同误检预算下，雷达与 GSPR 相比分数排序的恢复差为 {gap:+d} 个。训练阈值下，联合规则增加 {joint["tp"]} TP、{joint["fp"]} FP，分数规则增加 {score["tp"]} TP、{score["fp"]} FP。',
        '5. 下一步优先核验同误检预算下的候选规则是否跨场景稳定，以及是否保留 B 中依赖真实雷达内容的目标。',
        '', '判断来自固定权重和验证场景评估。B 为历史诊断切片。雷达第四通道物理映射仍未知。'])
    write(OUT/'FINAL_RESULTS.json',dict(decision=decision,field='D0.json',separability='A_RESULTS.json',
        sensitivity=b,p1=p,identities={k:read(OUT/k/'IDENTITY.json') for k in ('train_dev','validate','B')}))
    print('COMPLETE REPORTS',flush=True)


if __name__=='__main__':
    report_all()
