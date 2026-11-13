"""Same-input F0 Radar on/off pairs on the historical Fog slice."""
from collections import Counter

from collect import (as_boxes, evidence, feature_context, inspect_head, iou_matrix,
                     pack, run, train_utils)
from context import F0, OUT, bind, frozen_clean, load, sha, write
from checkpoint import seed
from evaluate import official_match_trace, statistics
from metrics import valid_nms_candidates
from support import corner_array
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch


def labels(boxes,scores,gt):
    stat=statistics()
    pairs,_=official_match_trace(boxes,scores,torch.from_numpy(corner_array(gt)),
                                 [str(g['object_id']) for g in gt],stat,.7)
    return {p['prediction_index']:str(p['object_id']) for p in pairs}


def main():
    model,config,digest=load(run,'F0')
    root=OUT/'A_pairs'
    bind(root/'IDENTITY.json',dict(weight=digest,config=sha(config),frames=120,code=sha(__file__)))
    data,_=run.build(config,condition='fog',visualize=True)
    pp=data.native.post_processor
    seed(20261016)
    loader=DataLoader(Subset(data,range(120)),batch_size=1,num_workers=4,
                      collate_fn=data.native.collate_batch_test)
    for i,batch in enumerate(loader):
        if (root/f'{i:06d}.json').exists():
            continue
        device=train_utils.to_device(batch,torch.device('cuda:0'))
        gt=frozen_clean(i,data.keys[i])
        output=[]
        ctx=None
        for mode in ('normal','radar_off'):
            model.mode=mode
            with torch.inference_mode():
                out=model(device['ego'])
                if ctx is None:
                    ctx=feature_context(batch['ego'],model)
                p,d,c=(x.cpu().numpy() for x in inspect_head(device['ego'],out,pp))
                with valid_nms_candidates(Counter()):
                    boxes,scores=pp.post_process(device,{'ego':out})
                rows=pack(boxes,scores,c,d,p,ctx)
                hits=labels(boxes,scores,gt)
                for k,row in enumerate(rows):
                    row['target']=hits.get(k)
                output.append(rows)
        a,b=output
        overlap=iou_matrix([r['corners'] for r in a],[r['corners'] for r in b])
        pairs=[]
        used_a,used_b=set(),set()
        for x,y in sorted(zip(*np.where(overlap>=.5)),key=lambda k:(-overlap[k],*k)):
            if x not in used_a and y not in used_b:
                used_a.add(x);used_b.add(y)
                pairs.append(dict(on=int(x),off=int(y),iou=float(overlap[x,y]),
                    same_target=a[x]['target']==b[y]['target'],score_delta=a[x]['score']-b[y]['score']))
        on={r['target'] for r in a if r['target'] is not None}
        off={r['target'] for r in b if r['target'] is not None}
        write(root/f'{i:06d}.json',dict(frame=data.keys[i],pairs=pairs,on=a,off=b,
            new=[k for k in range(len(a)) if k not in used_a],
            disappeared=[k for k in range(len(b)) if k not in used_b],
            recovered=sorted(on-off),lost=sorted(off-on)))
        if (i+1)%20==0:
            print('A_PAIR_FRAMES',i+1,flush=True)
    write(OUT/'A_PAIRS_COMPLETE.json',dict(frames=120,weight=digest))
    print('COMPLETE A_PAIRS',flush=True)


if __name__=='__main__':
    main()

