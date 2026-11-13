"""Replay archived F3 interventions and add a same-scene wrong-frame control."""
from collections import Counter
import json

from context import ARCHIVE, OUT, bind, frozen_clean, load, read, setup, sha, write

run=setup()
import numpy as np
import torch

from checkpoint import seed
from diagnostics import input_context
from evaluate import official_match_trace, statistics
from metrics import valid_nms_candidates
from opencood.tools import train_utils
from support import corner_array
from v2xr_weather_l012.data import tensor_digest


def describe(tensor):
    a=tensor.detach().float()
    if not a.numel():
        return dict(n=0,rms=0.,mean=0.,std=0.,abs_q=None)
    sample=a.flatten()[::max(1,a.numel()//16384)].abs()
    return dict(n=a.numel(),rms=float(a.square().mean().sqrt()),mean=float(a.mean()),
        std=float(a.std(unbiased=False)),abs_q=torch.quantile(sample,sample.new_tensor([.1,.5,.9])).tolist())


def official(boxes,scores,gt):
    record={}
    for threshold in (.3,.5,.7):
        stat=statistics()
        pairs,_=official_match_trace(boxes,scores,torch.from_numpy(corner_array(gt)),
                                    [str(r['object_id']) for r in gt],stat,threshold)
        record[str(threshold)]=dict(matches=pairs,**stat[threshold])
    return record


def main():
    model,config,digest=load(run,'F3')
    base=OUT/'B'
    bind(base/'IDENTITY.json',dict(weight=digest,code=sha(__file__),frames=120,config=sha(config)))
    before=tensor_digest(model.state_dict())
    seed(20261016)
    data,_=run.build(config,condition='fog',visualize=True)
    current={}
    def radar_hook(_, args):
        d=args[0]
        current['effective_radar']=tensor_digest({k:d[k] for k in ('voxel_features','voxel_coords','voxel_num_points')})
        current['effective_coords']=tensor_digest(d['voxel_coords'])
    def backbone_hook(_,args):
        x=args[0]['spatial_features']
        current['lidar_features']=describe(x[:,:64])
        current['radar_features']=describe(x[:,64:])
    model.radar_pillar_vfe.register_forward_pre_hook(radar_hook)
    model.backbone.register_forward_pre_hook(backbone_hook)
    archive={k:read(ARCHIVE/k/'DIAGNOSTIC.json') for k in ('full','radar_off','shuffle')}
    for name,a in archive.items():
        if a['config']['sha256']!=digest or a['config']['frames']!=120:
            raise ValueError(f'Archived identity differs: {name}')
    raw_gt={n:[] for n in archive}
    for name in archive:
        with (ARCHIVE/name/'per_frame_metrics.jsonl').open() as stream:
            raw_gt[name]=[r for line in stream if (r:=json.loads(line))['condition']=='fog']
    eligible=[i for i in range(120)]
    device=torch.device('cuda:0')
    collate=lambda x:data.native.collate_batch_test([data[x]])
    clean_ref=__import__('context').LR0/'final_evaluation/predictions/clean'
    from torch.utils.data import DataLoader,Subset
    ids={}
    for i,key in enumerate(data.keys):
        ids.setdefault(key[0],[]).append(i)
    loader=DataLoader(Subset(data,eligible),batch_size=1,num_workers=4,collate_fn=data.native.collate_batch_test)
    for i,host in enumerate(loader):
        path=base/f'{i:06d}.json'
        if path.exists():
            continue
        key=list(data.keys[i])
        gt=frozen_clean(i,key)
        real=train_utils.to_device(host,device)
        group=ids[key[0]]
        donor_index=group[(group.index(i)+max(1,len(group)//2))%len(group)]
        if donor_index==i:
            raise ValueError('Cross-frame donor equals recipient')
        donor=collate(donor_index)
        rec=donor['ego']['processed_radar']
        n=int(host['ego']['record_len'].sum())
        source_n=int(donor['ego']['record_len'].sum())
        # Same scene and agent slot, independently projected to donor ego coordinates.
        # Drop donor-only slots, leave unmatched recipient slots empty.
        keep=rec['voxel_coords'][:,0]<n
        replaced={k:v[keep].to(device) for k,v in rec.items()}
        old=real['ego']['processed_radar']
        outputs={}
        head={}
        for mode,name in (('normal','full'),('radar_off','radar_off'),('radar_shuffle','shuffle'),('normal','cross_frame')):
            ego=dict(real['ego'],processed_radar=replaced if name=='cross_frame' else old)
            model.mode=mode
            current.clear()
            with torch.inference_mode():
                out=model(ego)
                if i==0 and name=='full':
                    repeat=model(ego)
                    head['repeat_max_abs']=max(float((v-repeat[k]).abs().max()) for k,v in out.items())
                    if any(not torch.allclose(v,repeat[k],atol=1e-4,rtol=1e-4) for k,v in out.items()):
                        raise ValueError('Normal repeated inference exceeds tolerance')
                if name=='shuffle' and current['effective_coords']!=tensor_digest(old['voxel_coords']):
                    raise ValueError('Shuffle changed Radar occupied coordinates')
                if name=='radar_off' and current['effective_coords']!=tensor_digest(old['voxel_coords'][:0]):
                    raise ValueError('Radar-off occupancy is nonempty')
                output=dict(current,head_scores=describe(out['psm'].sigmoid()))
                with valid_nms_candidates(Counter()):
                    boxes,scores=data.native.post_processor.post_process(real,{'ego':out})
                output.update(scoring=official(boxes,scores,gt),boxes=[] if boxes is None else boxes.cpu().tolist(),
                              scores=[] if scores is None else scores.cpu().tolist())
                if name in archive:
                    source=ARCHIVE/name/'predictions/fog'/f'{i:06d}.json'
                    detail=raw_gt[name][i]
                    cached=read(source)
                    if key!=cached['frame_key'] or sha(source)!=detail['prediction_cache_sha256']:
                        raise ValueError('Archive frame or prediction digest differs')
                    output['archived_prediction_count']=cached['prediction_count']
                    output['archived_matches']=detail['protocols']['F'].get('matched_pairs',{}).get('0.7',[])
                outputs[name]=output
        strata=archive['full']['diagnostics']['fixed_target_rows']
        damage=[str(r['object_id']) for r in gt if 'damaged|radar:True' in strata[f'fog/{i}/{r["object_id"]}']['bins']]
        write(path,dict(frame=key,index=i,gt=len(gt),damage=damage,outputs=outputs,head=head,
            radar_before=tensor_digest(old),lidar_before=tensor_digest(real['ego']['processed_lidar']),
            donor=list(data.keys[donor_index]),donor_agents=source_n,recipient_agents=n))
        if (i+1)%20==0:
            print('B_FRAMES',i+1,flush=True)
    if before!=tensor_digest(model.state_dict()):
        raise ValueError('Model state changed during B')
    write(OUT/'B_COMPLETE.json',dict(weight=digest,frames=len(eligible),model_state=before))
    print('COMPLETE B',flush=True)


if __name__=='__main__':
    main()

