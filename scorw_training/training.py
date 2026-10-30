"""Independent fresh runs, deterministic sample schedule and full optimizer/RNG resumes."""
from pathlib import Path
import os,random
import numpy as np
import torch
from scorw_diagnostics.core import write,read,digest,identity
from .data import Data,schedule,seed
from .model import create,criterion,state_hash
from .evaluate import evaluate


def save_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def checkpoint(out,model,opt,scheduler,epoch,cursor,history,trace,losses):
    value=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        epoch=epoch,cursor=cursor,history=history,trace=trace,losses=losses,
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    tmp=out/'resume/new.pth';save_torch(tmp,value);sha=digest(tmp);target=tmp.with_name(sha+'.pth');os.replace(tmp,target)
    old=read(out/'RESUME.json') if (out/'RESUME.json').exists() else None
    write(out/'RESUME.json',dict(path=str(target),sha256=sha,epoch=epoch,cursor=cursor))
    if old and old['path']!=str(target):Path(old['path']).unlink()


def train(c,h,variant,out,base,parent,smoke):
    out.mkdir(parents=True,exist_ok=True);model=create(c,h,variant,base).cuda();lossfn=criterion(c,h,variant).cuda()
    write(out/'TRAINING_CONFIG.json',dict(group=variant,protocol=c,hypes=h,point_supervision=False,
          original_gspr=variant=='S2',initial_backbone_identity=state_hash(base)))
    opt=torch.optim.Adam(model.parameters(),**c['optimizer']);sched=torch.optim.lr_scheduler.MultiStepLR(opt,**c['scheduler'])
    epoch=cursor=0;history=[];trace=[];losses=[]
    if (out/'RESUME.json').exists():
        ref=read(out/'RESUME.json')
        if digest(ref['path'])!=ref['sha256']:raise ValueError('Resume identity changed')
        state=load(ref['path']);model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        for group in opt.state.values():
            for k,v in group.items():
                if torch.is_tensor(v):group[k]=v.cuda()
        epoch=state['epoch'];cursor=state['cursor'];history=state['history'];trace=state['trace'];losses=state['losses']
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        checkpoint(out,model,opt,sched,0,0,history,trace,losses)
    data=Data(c,h,'train',True);epochs=1 if smoke else c['epochs'];n=len(data.keys)
    while epoch<epochs:
        order,weather=schedule(c,epoch,n)
        if smoke:order=order[:c['smoke_steps']*c['batch_size']]
        while cursor<len(order):
            stop=min(len(order),cursor+c['batch_size']);slots=order[cursor:stop]
            frames=[data.frame(i,'fog' if variant!='S0' and weather[i] else 'clean',epoch) for i in slots]
            batch,labels=data.batch(frames);model.train();opt.zero_grad(set_to_none=True)
            output=model(batch);loss=lossfn(output,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();gspr_nonzero=0;grad_norm=0.
            before={k:v.detach().clone() for k,v in model.named_parameters() if k.startswith('gspr.')}
            for name,p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite gradient '+name)
                    if name.startswith('gspr.'):
                        norm=float(p.grad.norm());grad_norm+=norm;gspr_nonzero+=norm>0
            if variant=='S2' and not gspr_nonzero:raise ValueError('GSPR has no active gradient')
            opt.step()
            if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):raise ValueError('Nonfinite state after optimizer step')
            changed=sum(not torch.equal(v,dict(model.named_parameters())[k].detach()) for k,v in before.items())
            if variant=='S2' and not changed:raise ValueError('GSPR weights did not update')
            trace.append(dict(indices=slots,frames=[[f['scene'],f['timestamp'],f['active'],f['weather'],f['input_hash']] for f in frames],
                              loss=float(loss),gspr_nonzero_gradients=gspr_nonzero,gspr_gradient_norm=grad_norm,gspr_updated_parameters=changed))
            losses.append(float(loss));cursor=stop
            if cursor%(c['batch_size']*c['checkpoint_steps'])==0 or cursor==len(order):checkpoint(out,model,opt,sched,epoch,cursor,history,trace,losses)
            if len(trace)%20==0 or cursor==len(order):print('%s epoch %d/%d samples %d/%d loss %.6f'%(variant,epoch+1,epochs,cursor,len(order),float(loss)),flush=True)
        folder=out/'epochs'/('%02d'%(epoch+1));folder.mkdir(parents=True,exist_ok=True)
        model_path=folder/'model.pth'
        if model_path.exists():
            if state_hash(load(model_path))!=state_hash(model.state_dict()):raise ValueError('Epoch checkpoint differs from resume state')
        else:save_torch(model_path,{k:v.detach().cpu() for k,v in model.state_dict().items()})
        result=evaluate(c,h,model,'validate',folder/'validation',identity([parent,variant,epoch,digest(model_path)]),c['smoke_eval_frames'] if smoke else 0,False)
        score=sum(result['metrics']['F'][w]['AP70']['AP'] for w in ('clean','fog'))/2
        row=dict(epoch=epoch+1,score=score,mean_training_loss=float(np.mean(losses)),checkpoint=str(model_path),sha256=digest(model_path),
                 sampling_identity=identity([r['frames'] for r in trace]),sample_count=sum(len(r['indices']) for r in trace))
        write(folder/'SAMPLING.json',dict(rows=trace,order=order,weather=weather,point_supervision=False))
        write(folder/'COMPLETE.json',dict(**row,sampling_sha256=digest(folder/'SAMPLING.json'),evaluation_sha256=digest(folder/'validation/RESULT.json')))
        history.append(row);write(out/'HISTORY.json',history)
        best=max(history,key=lambda r:(r['score'],-r['epoch']));write(out/'SELECTED.json',best)
        sched.step();epoch+=1;cursor=0;trace=[];losses=[]
        checkpoint(out,model,opt,sched,epoch,cursor,history,trace,losses)
    model.cpu();del model,opt,lossfn;torch.cuda.empty_cache()
    return read(out/'SELECTED.json')
