"""Fit held-out scene probes; compare matched out-of-scene predictions."""
from collections import Counter
from pathlib import Path
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(os.environ['OBS_OUT'])
WEATHER=('clean','fog','snow')
LAYERS=('lidar','radar','concat','trunk','fused')
KEYS=('base','radar_info','gspr','points','states','score','ray','encoded','error','xy',*LAYERS)


def load(root,weather,kind):
    rows={k:[] for k in KEYS}
    scene=[]
    for path in sorted((root/'cache').glob('*.npz')):
        with np.load(path,allow_pickle=False) as f:
            for key in KEYS:
                rows[key].append(f[f'{weather}_{kind}_{key}'])
            scene.extend([f['frame'][0]]*len(rows['error'][-1]))
    out={k:np.concatenate(v) for k,v in rows.items()}
    out['scene']=np.asarray(scene)
    return out


def fit_predict(x,y,test):
    classes=np.unique(y)
    if len(classes)!=2:
        return np.full(len(test),np.nan)
    rng=np.random.default_rng(261116)
    ids=np.concatenate([rng.choice(i,min(12000,len(i)),replace=False)
                        for cls in classes for i in [np.flatnonzero(y==cls)]])
    pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=350,class_weight='balanced',C=.25))
    return pipe.fit(x[ids],y[ids]).predict_proba(test)[:,1]


def metrics(y,p):
    valid=np.isfinite(p)
    y,p=y[valid],p[valid]
    if len(np.unique(y))!=2:
        return dict(n=len(y),positives=int(y.sum()),roc_auc=None,pr_auc=None,macro_f1=None,
                    confusion=confusion_matrix(y,p>=.5,labels=[0,1]).tolist())
    return dict(n=len(y),positives=int(y.sum()),roc_auc=float(roc_auc_score(y,p)),
                pr_auc=float(average_precision_score(y,p)),macro_f1=float(f1_score(y,p>=.5,average='macro')),
                confusion=confusion_matrix(y,p>=.5,labels=[0,1]).tolist())


def held_out(train_x, train_y, train_scene, test_x, test_scene):
    pred=np.full(len(test_scene),np.nan)
    for s in sorted(set(test_scene)):
        keep=train_scene!=s
        use=test_scene==s
        pred[use]=fit_predict(train_x[keep],train_y[keep],test_x[use])
    return pred


def comparisons(y,scene,probs):
    results={}
    for name,p in probs.items():
        results[name]=dict(pooled=metrics(y,p),scenes={s:metrics(y[scene==s],p[scene==s]) for s in sorted(set(scene))})
    return results


def delta(y,scenes,base,added):
    groups=[]
    for s in sorted(set(scenes)):
        mask=(scenes==s)&np.isfinite(base)&np.isfinite(added)
        a=metrics(y[mask],base[mask])
        b=metrics(y[mask],added[mask])
        if a['pr_auc'] is not None:
            groups.append([b[k]-a[k] for k in ('roc_auc','pr_auc')])
    if not groups:
        return dict(n_scenes=0,mean=None,ci95=None)
    t=np.asarray(groups)
    ids=np.random.default_rng(19).integers(len(t),size=(2000,len(t)))
    return dict(n_scenes=len(t),mean=t.mean(0).tolist(),
                ci95=np.quantile(t[ids].mean(1),[.025,.975],axis=0).T.tolist() if len(t)>1 else None)


def probe(data):
    base=np.column_stack((data['base'],data['radar_info']))
    out={}
    for j,code in enumerate('ABCD'):
        y=data['states'][:,j]
        keep=y>=0
        quality=Counter(y.tolist())
        scene=data['scene'][keep]
        scores={}
        if keep.any():
            subsets={'count_distance':base, **{k:data[k] for k in LAYERS}}
            for name,x in subsets.items():
                pred=held_out(x[keep],y[keep],scene,x[keep],scene)
                scores[name]=dict(pooled=metrics(y[keep],pred),scenes={s:metrics(y[keep][scene==s],pred[scene==s])
                                                                        for s in sorted(set(scene))})
        out[code]=dict(labelled=int(keep.sum()),unknown=int((~keep).sum()),
                       positives=int(quality[1]),negatives=int(quality[0]),scores=scores)
    return out


def design(data):
    base=data['base']
    radar=data['radar_info']
    gspr=data['gspr']
    # Only present-frame ray categories enter non-oracle groups.
    live=np.column_stack((data['states'][:,0]==1,data['states'][:,2]==1,
                          data['states'][:,0]<0,data['states'][:,3]==1))
    oracle=np.column_stack((data['states'][:,1]==1,data['states'][:,1]<0))
    c=np.column_stack((base,radar))
    cg=np.column_stack((c,gspr))
    strong=np.column_stack((cg,data['fused'],data['trunk'],data['score']))
    return {'A':base,'B':np.column_stack((base,gspr)),'C':c,'C_GSPR':cg,
            'D':np.column_stack((c,live)),'E':np.column_stack((cg,live)),
            'strong':strong,'strong_live':np.column_stack((strong,live)),
            'oracle_C':np.column_stack((c,live,oracle)),
            'oracle_E':np.column_stack((cg,live,oracle)),
            'oracle_strong':np.column_stack((strong,live,oracle))}


def stratify(data,pred):
    l,r,_=data['points'].T
    y=data['error']
    dist=np.linalg.norm(data['xy'],axis=1)
    s=data['states']
    masks={'lidar0':l==0,'lidar1_5':(l>=1)&(l<=5),'loss_B':s[:,1]==1,'radar_only_D':s[:,3]==1,
           'r0_l0':(r==0)&(l==0),'rpos_l0':(r>0)&(l==0),'r0_l1_5':(r==0)&(l>=1)&(l<=5),
           'rpos_l1_5':(r>0)&(l>=1)&(l<=5),'distance0_30':dist<30,'distance30_60':(dist>=30)&(dist<60),
           'distance60_plus':dist>=60}
    return {k:dict(n=int(m.sum()),errors=int(y[m].sum()),scenes=len(set(data['scene'][m])),
                   C=metrics(y[m],pred['C'][m]),D=metrics(y[m],pred['D'][m]),
                   delta=delta(y[m],data['scene'][m],pred['C'][m],pred['D'][m])) for k,m in masks.items()}


def errors(train,test,within):
    xtr,xt=design(train),design(test)
    y=test['error']
    scenes=test['scene']
    pred={k:held_out(xtr[k],train['error'],train['scene'],x,scenes) for k,x in xt.items()}
    pair=[('C','D'),('C_GSPR','E'),('strong','strong_live'),('D','oracle_C'),
          ('E','oracle_E'),('strong_live','oracle_strong')]
    states={}
    for j,k in enumerate('ABCD'):
        states[k]={str(z):dict(n=int((test['states'][:,j]==z).sum()),
                               errors=int(y[test['states'][:,j]==z].sum())) for z in (-1,0,1)}
    return dict(n=len(y),errors=int(y.sum()),scenes=len(set(scenes)),coverage=states,
                models=comparisons(y,scenes,pred),
                delta={f'{b}-{a}':delta(y,scenes,pred[a],pred[b]) for a,b in pair},
                strata=stratify(test,pred) if within else {})


def display(v):
    return '—' if v is None else f'{v:.4f}'


def report(root,probes,errors_result,receipt):
    lines=['# F0 观测状态诊断','',
           '固定 F0 在三天气推理，冻结权重和批归一化状态。两项分析复用同一缓存。误检、漏检由固定 Clean 真值的官方一对一匹配定义，框重叠阈值 0.7。','',
           '## 标注质量与 D2-1','',
           'A 为局部近水平射线穿越的空证据；C 为近水平射线受限的遮挡代理；其余缺少可判证据的区域标 Unknown。B 为同车配对，Clean 邻域至少 3 点且天气零点；D 为 LiDAR 零点且 Radar 有点。四种状态独立，Radar 点仅代表回波。','',
           '| 天气 | 状态 | 阳性 | 阴性 | Unknown | 特征 | ROC-AUC | PR-AUC | Macro-F1 | 混淆矩阵 |',
           '|---|---|---:|---:|---:|---|---:|---:|---:|---|']
    for w,cols in probes.items():
        for state,r in cols.items():
            for layer,item in r['scores'].items():
                a=item['pooled']
                lines.append(f"| {w} | {state} | {r['positives']} | {r['negatives']} | {r['unknown']} | {layer} | {display(a['roc_auc'])} | {display(a['pr_auc'])} | {display(a['macro_f1'])} | {a['confusion']} |")
    lines+=['','## D2-2 额外解释价值','',
            '预测框组使用正式输出。目标组在离线已知位置抽取固定窗口，因此目标组是位置已知的诊断。它只使用真值中心定位窗口，真值类别、尺寸、匹配结果及 Clean 信息均未进入部署侧特征。Clean 损失只进入 Oracle 组。','',
            '所有模型按场景留出。跨天气同时移除与测试相同的训练场景。区间为 2,000 次场景重采样，最多六个场景，区间可能不稳定。后续复核可从 JSON 查看逐场景和稀疏分层。','',
            '| 对象 | 天气对照 | 样本 | 错误 | 场景 | 特征组 | ROC-AUC | PR-AUC |',
            '|---|---|---:|---:|---:|---|---:|---:|']
    for kind,cols in errors_result.items():
        for weather,r in cols.items():
            for name,item in r['models'].items():
                m=item['pooled']
                lines.append(f"| {kind} | {weather} | {r['n']} | {r['errors']} | {r['scenes']} | {name} | {display(m['roc_auc'])} | {display(m['pr_auc'])} |")
    lines+=['','| 对象 | 天气对照 | 比较 | 有效场景 | ROC/PR 差值 | 95% 场景区间 |','|---|---|---|---:|---|---|']
    for kind,cols in errors_result.items():
        for weather,r in cols.items():
            for comp,d in r['delta'].items():
                lines.append(f"| {kind} | {weather} | {comp} | {d['n_scenes']} | {d['mean']} | {d['ci95']} |")
    tests=[errors_result[k][w]['delta']['strong_live-strong'] for k in ('prediction','target')
           for w in ('fog','snow','fog_to_snow','snow_to_fog')]
    robust=lambda t: t['n_scenes']>=4 and t['ci95'] is not None and t['ci95'][1][0]>0
    meaningful=any(robust(t) for t in tests)
    verdict='值得继续' if all(robust(t) for t in tests) else '证据不足' if meaningful else '阶段终止'
    if root.name=='smoke':
        verdict='冒烟通过，正式判断待全量'
    lines+=['','## 研究判断','',f'**{verdict}**。比较以强基线加入当帧状态后的场景平均 PR-AUC 增量为准。'
            '本报告仅验证当前定义和当前验证集。线性探针难以分辨、或状态能预测错误，都无法单独确定融合造成了错误。'
            '历史 D0/D1 包含模型内部干预，本轮使用冻结 F0 和离线相关分析，证据口径不同。',
            '',f"与归档 F0 的逐帧匹配差异核对 `{receipt['archive_drift']}`。"]
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(mode):
    root=ROOT/mode
    receipt=json.loads((root/'CACHE_MANIFEST.json').read_text())
    import hashlib
    for filename,digest in receipt['files'].items():
        if hashlib.sha256((root/'cache'/filename).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Cached frame changed: {filename}')
    probes={}
    for w in WEATHER:
        probes[w]=probe(load(root,w,'grid'))
        print('D2-1',w,flush=True)
    result={}
    for kind in ('target','prediction'):
        datasets={w:load(root,w,kind) for w in WEATHER}
        result[kind]={}
        for a,b in [(w,w) for w in WEATHER]+[('fog','snow'),('snow','fog')]:
            result[kind][b if a==b else f'{a}_to_{b}']=errors(datasets[a],datasets[b],a==b)
            print('D2-2',kind,a,b,flush=True)
    for name,obj in [('D2-1',probes),('D2-2',result)]:
        (root/(name+'.json')).write_text(json.dumps(obj,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    report(root,probes,result,receipt)
    names=('D2-1.json','D2-2.json','FINAL_REPORT.md','CACHE_MANIFEST.json','CONTRACT.json')
    (root/'RESULT_HASHES.json').write_text(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                                                       for n in names},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main(sys.argv[1])
