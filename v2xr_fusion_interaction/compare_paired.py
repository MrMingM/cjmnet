"""Paired F AP70 uncertainty with chronological, within-scene block resampling."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from v2xr_fusion_interaction.common.audit import OUT,WEATHERS,atomic_json,read_json,sha256

BLOCK_SIZE=16
DRAWS=2000


def frames(path):
    records=defaultdict(list)
    with (path/'per_frame_metrics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            f=r['protocols']['F']['thresholds']['0.7']
            records[r['condition']].append((r['index'],tuple(r['frame_key']),r['protocols']['F']['counted_gt'],
                                           f['tp_flags'],f['fp_flags']))
    return {w:sorted(records[w]) for w in WEATHERS}


def ap(tp,fp,gt):
    positives=np.cumsum(tp,dtype=np.float64)
    denominator=positives+np.cumsum(fp,dtype=np.float64)
    precision=positives/denominator
    envelope=np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.dot(tp,envelope)/gt)


def block_slices(reference):
    scenes=defaultdict(list)
    for i,row in enumerate(reference):
        scenes[row[1][0]].append(i)
    for indices in scenes.values():
        if indices!=list(range(indices[0],indices[-1]+1)):
            raise ValueError('Expected contiguous scenes in official order')
    return [[indices[i:i+BLOCK_SIZE] for i in range(0,len(indices),BLOCK_SIZE)]
            for indices in scenes.values()]


def scores(data,blocks,draws):
    packed=[]
    for indices in blocks:
        tp=np.array([x for i in indices for x in data[i][3]],dtype=np.uint8)
        fp=np.array([x for i in indices for x in data[i][4]],dtype=np.uint8)
        packed.append((tp,fp,sum(data[i][2] for i in indices)))
    def evaluate(indices):
        tp,fp,gt=zip(*(packed[i] for i in indices))
        return ap(np.concatenate(tp),np.concatenate(fp),sum(gt))
    return evaluate(range(len(blocks))),np.array([evaluate(d) for d in draws])


def main():
    final=read_json(OUT/'FINAL_RESULTS.json')
    paths={'P0':OUT/'BASELINE/final_evaluation'}
    pairs=[('P0','F2'),('P0','F3'),('P0','F4'),('F2','F3'),('F3','F4')]
    evaluations={}
    for group,data in final['groups'].items():
        evaluations[group]=data['evaluation']
        if group=='P0':
            continue
        paths[group]=OUT/group/'final_evaluation'
        for mode,record in data['controls'].items():
            label=f'{group}/{mode}'
            paths[label]=OUT/group/'controls'/mode
            evaluations[label]=record
            pairs.append((label,group))
    inputs={name:frames(path) for name,path in paths.items()}
    out={}
    for w in WEATHERS:
        reference=inputs['P0'][w]
        assert len(reference)==829 and sum(x[2] for x in reference)==6235
        strata=block_slices(reference)
        blocks=[x for scene in strata for x in scene]
        boundaries=np.cumsum([0]+[len(scene) for scene in strata])
        rng=np.random.default_rng(20261112)
        draws=np.concatenate([np.sort(rng.integers(a,b,size=(DRAWS,b-a)),axis=1)
                              for a,b in zip(boundaries[:-1],boundaries[1:])],axis=1)
        readings={}
        for name,data in inputs.items():
            if [x[:3] for x in data[w]]!=[x[:3] for x in reference]:
                raise ValueError(f'Paired frame/target denominators differ: {name}/{w}')
            point,samples=scores(data[w],blocks,draws)
            expected=evaluations[name]['results'][w]['AP']['F']['ap70']
            if not np.isclose(point,expected,atol=1e-12,rtol=0):
                raise ValueError(f'AP reassembly differs: {name}/{w}/{point}/{expected}')
            readings[name]=(point,samples)
        out[w]={}
        for base,test in pairs:
            p,a=readings[base]
            q,b=readings[test]
            lo,hi=np.quantile(100*(b-a),[.025,.975])
            out[w][f'{test} vs {base}']=dict(delta_pp=100*(q-p),interval95_pp=[float(lo),float(hi)],
                                                interval_excludes_zero=bool(lo>0 or hi<0))
        print('PAIRED COMPLETE',w,len(blocks),flush=True)
    result=dict(status='complete',contract=final['contract'],draws=DRAWS,block_frames=BLOCK_SIZE,
                source_sha256={g:sha256(p/'per_frame_metrics.jsonl') for g,p in paths.items()},
                weather=out)
    atomic_json(OUT/'PAIRED_RESULTS.json',result)
    lines=['# 固定真值 AP70 配对比较','','配对重采样以场景内相邻 16 帧为一组，同一组抽样用于全部模型。'
           '每次按原始块次序拼接，保留官方 AP 的帧顺序。区间反映本验证集的样本波动。'
           '模型各只有一个训练种子，区间不覆盖跨种子波动。','',
           '| 天气 | 比较 | 差值（百分点） | 95% 配对区间 |','|---|---|---:|---|']
    for w,comparisons in out.items():
        for name,r in comparisons.items():
            lo,hi=r['interval95_pp']
            lines.append(f'| {w} | {name} | {r["delta_pp"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    (OUT/'PAIRED_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
