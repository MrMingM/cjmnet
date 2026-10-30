"""Fixed Clean GT F and actual native inference, frame-order AP, distance recall."""
from pathlib import Path
import random
import copy
import numpy as np
import torch
from scorw_diagnostics.core import stage,write,read,identity
from scorw_diagnostics.ap_audit import ap,native_gt
from scorw_r3.metrics import iou_matrix,match_matrix
from scorw_repair.iou_audit import IoUAudit
from .model import state_hash
from .iou import exact_iou


def evaluate(c,h,model,split,out,parent,limit=0,supplement=True):
    from .data import Data,seed
    from opencood.utils import common_utils,box_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools.train_utils import to_device
    data=Data(c,h,split);model.eval();initial=state_hash(model.state_dict());records=[]
    natives={}
    if supplement:
        for weather in ('clean','fog'):
            cfg=copy.deepcopy(h);cfg['data_augment']=[];cfg['validate_dir']=str(Path(c['dataset'])/weather/split)
            natives[weather]=build_dataset(cfg,visualize=False,train=False)
            expected=[(s['scene'],t) for s,t in data.keys];actual=[]
            for scene in natives[weather].scenario_database.values():
                first=next(iter(scene.values()));actual.extend((Path(f['lidar']).parent.parent.name,t) for t,f in first.items() if t!='ego')
            if actual!=expected:raise ValueError('Native frame order mismatch')
    n=limit or len(data.keys)
    for i in range(n):
        s,t=data.keys[i];folder=out/'frames'/s['scene']/t
        def work():
            row=dict(scene=s['scene'],timestamp=t,conditions={})
            gt_hashes={}
            for weather in ('clean','fog'):
                f=data.frame(i,weather);input,_=data.batch([f]);gt=f['gt'];ids=f['ids']
                if split=='validate':
                    old=read(Path(c['dataset'])/'stages/D6/frames'/s['scene']/t/'RESULT.json')
                    if f['input_hash']!=old['predictions'][weather]['input_hash'] or identity(gt.tolist())!=old['clean_GT_sha256']:raise ValueError('F data/GT does not match audited D6')
                paths={'F':(input,gt,ids)}
                if supplement:
                    np.random.seed(seed(c,'native',split,i));random.seed(seed(c,'native',split,i))
                    ng,ni,nc,sample=native_gt(natives[weather],i)
                    native=natives[weather].collate_batch_test([sample])['ego']
                    paths['native']=(to_device(native,torch.device('cuda:0')),ng,ni)
                row['conditions'][weather]={}
                for protocol,(batch,g,object_ids) in paths.items():
                    audit=IoUAudit(None)
                    with torch.inference_mode():
                        output=model(batch)
                        if not all(torch.isfinite(output[k]).all() for k in ('psm','rm')):raise ValueError('Nonfinite model head')
                        with exact_iou(common_utils,audit):
                            boxes,scores=data.post.post_process({'ego':batch},{'ego':output})
                    boxes=np.empty((0,8,3),np.float32) if boxes is None else boxes.cpu().numpy()
                    scores=np.empty(0,np.float32) if scores is None else scores.cpu().numpy()
                    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():raise ValueError('Nonfinite predictions')
                    mat=iou_matrix(boxes,g,audit);matches={str(q):match_matrix(mat,scores,q) for q in (.3,.5,.7)}
                    gt_hashes[(weather,protocol)]=identity(g.tolist())
                    row['conditions'][weather][protocol]=dict(matches=matches,gt_ids=object_ids,gt_sha256=identity(g.tolist()),
                        target_distance=np.linalg.norm(g.mean(1),axis=1).tolist(),boxes=boxes.tolist(),scores=scores.tolist(),iou_audit=audit.report)
            if gt_hashes[('clean','F')]!=gt_hashes[('fog','F')]:raise ValueError('Weather-dependent F GT')
            if supplement and gt_hashes[('clean','native')]!=gt_hashes[('fog','native')]:raise ValueError('Native Clean/Fog GT changed')
            write(folder/'RESULT.json',row)
        stage(folder,dict(parent=parent,frame=i,weights=initial,split=split,supplement=supplement),work)
        records.append(read(folder/'RESULT.json'))
        if (i+1)%20==0 or i+1==n:print('EVAL %s %d/%d'%(split,i+1,n),flush=True)
    values={};gt_identity={}
    for protocol in (('F','native') if supplement else ('F',)):
        values[protocol]={}
        for weather in ('clean','fog'):
            rr=[r['conditions'][weather][protocol] for r in records];metrics={}
            gt_identity[protocol]=identity([r['conditions']['clean'][protocol]['gt_sha256'] for r in records])
            for q in (.3,.5,.7):
                matches=[r['matches'][str(q)] for r in rr]
                counts={k:sum(m[k] for m in matches) for k in ('TP','FP','FN','gt')}
                distance=[]
                for lo,hi in zip(c['distance_edges'],c['distance_edges'][1:]):
                    num=den=0
                    for r in rr:
                        mask=(np.asarray(r['target_distance'])>=lo)&(np.asarray(r['target_distance'])<hi)
                        den+=int(mask.sum());num+=sum(bool(mask[k]) for k in r['matches'][str(q)]['matched_gt'])
                    distance.append(dict(low=lo,high=hi,TP=num,GT=den,recall=num/den if den else None))
                metrics['AP%d'%round(q*100)]=dict(AP=ap(matches,False),**counts,by_distance=distance)
            values[protocol][weather]=metrics
    if initial!=state_hash(model.state_dict()):raise ValueError('Evaluation changed model state')
    result=dict(status='passed',split=split,frames=n,full_scope=n==len(data.keys),metrics=values,GT_identity=gt_identity,
                checkpoint_state_sha256=initial,global_sort=False,main_protocol='F',supplement='actual native inference; native Clean metadata GT, XYZ reconstructed yaw boxes then XY final range; cap before communication',
                F_protocol='D6 fixed Clean corners; communication before cap; 120m any active CAV; XY GT_RANGE; unchanged point order',
                REAL_WORLD_VALIDATION_MISSING=True)
    write(out/'RESULT.json',result);return result
