"""Fixed selected weights: S adaptive/uniform/shuffled/off or C on/off; GT after forward only."""
import copy
import json
from pathlib import Path
import types
from auditlib import METHOD,atomic_json,bound_json,complete,identity,read_json,sha256,sources

def run(args):
    import torch
    from torch.utils.data import DataLoader,Subset
    from opencood.hypes_yaml import yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.utils import eval_utils_opv2v as ev
    from models import create,OBSERVATIONS
    from evaluation import selection
    from evaluate import official_match_trace
    from checkpoint import seed
    from data import tensor_digest
    root=Path(args.output_root);run_root=Path(args.experiment_root);v=args.variant
    chosen=(complete(run_root/'smoke'/v/'smoke_checkpoint.json') if args.smoke else selection(run_root/v))
    indices=[0,828] if args.smoke else METHOD['mechanism_indices']
    modes=('adaptive','uniform','shuffled','off') if v=='E1' else ('on','off')
    contract=dict(variant=v,sources=sources(),checkpoint=chosen['checkpoint'],checkpoint_sha256=chosen['checkpoint_sha256'],
        indices=indices,modes=modes,smoke_only=args.smoke,
        scope=str(len(indices))+' fixed validation frames/weather; F protocol; not model selection',
        S_uniform='same point-mean alpha among valid points with neighbors, all other S/A1 operations fixed',
        S_shuffle='one seeded permutation of alpha within pillar-count bins, over eligible point slots',
        fixed_prefix='replay identical first GSPR/PFN inputs to GGCA for every switch; no repeated-prefix numerical confound')
    contract['identity']=identity(contract);bound_json(root/'contract.json',contract)
    gt=[json.loads(s) for s in (Path(args.gt_audit_root)/'per_frame.jsonl').read_text().splitlines()]
    frames=[];report={};buckets={};propagation=[]
    h0=yaml_utils.load_yaml(str(Path(args.runtime_root)/'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    model=create(h0,v).cuda().eval()
    model.load_state_dict(torch.load(chosen['checkpoint'],map_location='cpu',weights_only=False),strict=True)
    original_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
    net=model.ggca;saved=net.forward
    cache={};active=[modes[0]]
    def replay(self,*values):
        if 'args' not in cache:
            cache['args']=tuple(x.detach().clone() if isinstance(x,torch.Tensor) else x for x in values)
        args0=cache['args']
        if v=='E2' and 'value' in cache:
            return cache['value']
        if v=='E1':
            self.capture=True
            self.control=active[0]
            if active[0]=='uniform':self.control_constant=cache['uniform']
            elif active[0]=='shuffled':self.control_alpha=cache['shuffled']
        result=saved(*args0)
        if v=='E2':cache['value']=result.detach().clone()
        if v=='E1' and active[0]=='adaptive':
            alpha=torch.cat([s['alpha'] for s in self.captures]);valid=torch.cat([s['valid'] for s in self.captures])
            eligible=torch.cat([s['has_neighbor'] for s in self.captures])&valid
            cache['uniform']=float(alpha[eligible].mean()) if eligible.any() else 0.
            shuffled=alpha.clone();counts=valid.sum(1)
            bins=torch.bucketize(counts,torch.tensor([1,2,5,10,20],device=counts.device))
            generator=torch.Generator(device=alpha.device).manual_seed(20261101+cache['index'])
            for b in range(6):
                mask=eligible&(bins[:,None]==b)
                vals=alpha[mask]
                if vals.numel():shuffled[mask]=vals[torch.randperm(len(vals),generator=generator,device=alpha.device)]
            cache['shuffled']=shuffled
        return result
    net.forward=types.MethodType(replay,net)
    with torch.inference_mode():
        for weather,suffix in (('clean',''),('fog','_fog_0.060'),('snow','_snow_2.5_2.0')):
            seed(20261016)
            h=copy.deepcopy(h0)
            h.update(validate_dir=args.prepared[1]['input_manifest']['overlay'],train_sim=False,
                eval_sim=weather!='clean',sim_weather=suffix)
            dataset=build_dataset(h,visualize=False,train=False)
            if len(dataset)!=829:raise ValueError('Mechanism validation split differs')
            keys=[]
            for scene in dataset.scenario_database.values():
                ego=next(a for a in scene.values() if a['ego'])
                keys.extend([[Path(a['yaml']).parent.parent.name,t] for t,a in ego.items() if t!='ego'])
            if keys!=args.prepared[1]['b']['selected_frame_keys']:raise ValueError('Mechanism frame/CAV order differs')
            loader=DataLoader(Subset(dataset,indices),batch_size=1,num_workers=0,shuffle=False,collate_fn=dataset.collate_batch_test)
            stats={m:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.3,.5,.7)} for m in modes}
            recall={m:{label:dict(gt=0,tp=0) for label in ('0','1-5','6-20','21+')} for m in modes}
            for index,batch in zip(indices,loader):
                cache.clear();cache['index']=index
                # replay checks args key, not cache emptiness: index is metadata only.
                data=train_utils.to_device(batch,torch.device('cuda:0'))
                raw_digest=tensor_digest(data['ego']['processed_lidar'])
                targets=gt[index]['conditions']['clean']['final_gt']
                corners=torch.tensor([a['corners'] for a in targets],dtype=torch.float32).reshape(-1,8,3)
                ids=[str(a['object_id']) for a in targets]
                target_counts={}
                for oid in ids:
                    count=0
                    for cav in gt[index]['conditions'][weather]['cavs']:
                        before=[str(s) for s in cav['before_point_filter_ids']]
                        if oid in before:count+=cav['official_points_in_box'][before.index(oid)]
                    target_counts[oid]=count
                for mode in modes:
                    active[0]=mode
                    if v=='E2':model.c_enabled=mode=='on'
                    current=copy.deepcopy(data)
                    if tensor_digest(current['ego']['processed_lidar'])!=raw_digest:raise ValueError('Switch raw inputs differ')
                    pred=model(current['ego'])
                    boxes,scores,_=dataset.post_process(current,{'ego':pred})
                    pairs,_=official_match_trace(boxes,scores,corners,ids,stats[mode],.7)
                    for t in (.3,.5):ev.caluclate_tp_fp(boxes,scores,corners,stats[mode],t)
                    matched={str(p['object_id']) for p in pairs}
                    for oid,count in target_counts.items():
                        label='0' if count==0 else '1-5' if count<=5 else '6-20' if count<=20 else '21+'
                        recall[mode][label]['gt']+=1;recall[mode][label]['tp']+=oid in matched
                    row=dict(model=v,condition=weather,index=index,frame_key=keys[index],mode=mode,
                        input_sha256=raw_digest,matched_ids=sorted(matched),target_native_counts=target_counts,
                        original_pfn_checks=model.weighting_checks)
                    if v=='E1':
                        d=net.pillar_rows
                        row['S']=net.last['self_retention'];row['residual_rms_mean']=float(d[:,7].mean())
                        for n in range(1,33):
                            mask=d[:,0].long()==n
                            if not mask.any():continue
                            key=weather+'|'+mode+'|'+str(n)
                            total=buckets.setdefault(key,dict(pillars=0,alpha_sum=0.,eta_sum=0.,residual_rms_sum=0.))
                            total['pillars']+=int(mask.sum());total['alpha_sum']+=float(d[mask,12].sum())
                            total['eta_sum']+=float(d[mask,4].sum());total['residual_rms_sum']+=float(d[mask,7].sum())
                    elif mode=='on':
                        row['C']=model.c_last
                        propagation.append(dict(condition=weather,index=index,**model.c_last))
                    frames.append(row);OBSERVATIONS.clear()
                print('MECHANISM',v,weather,index,flush=True)
            report[weather]={m:dict(AP={f'ap{int(t*100)}':float(ev.calculate_ap(copy.deepcopy(stats[m]),t,False)[0])
                for t in (.3,.5,.7)},tp=sum(stats[m][.7]['tp']),fp=sum(stats[m][.7]['fp']),
                fn=stats[m][.7]['gt']-sum(stats[m][.7]['tp']),target_point_recall=recall[m]) for m in modes}
    if any(not torch.equal(t.detach().cpu(),original_state[k]) for k,t in model.state_dict().items()):
        raise ValueError('Mechanism changed weights or BN state')
    if sha256(chosen['checkpoint'])!=chosen['checkpoint_sha256']:raise ValueError('Selected weight changed')
    atomic_json(root/'summary.json',dict(status='complete',contract=contract,results=report,
        pillar_statistics=buckets,propagation=propagation,frames=frames,optimization_steps=0,
        model_and_BN_unchanged=True,GT_used_only_for_scoring=True,
        distinction='pillar point count groups differ from target native point-count recall groups'))
