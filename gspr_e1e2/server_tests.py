"""Server-only behavioral checks, exact A1 recovery and full-state next-step resume."""
import copy
from contextlib import contextmanager
from pathlib import Path
import torch
from auditlib import METHOD,ROOT,atomic_json,load_frozen
from checkpoint import seed,save,state,restore

@contextmanager
def deterministic_resume_test():
    # Diagnostic histograms have no gradient and are never model inputs. HIP
    # histc is unsupported in strict deterministic mode: count them on CPU here.
    # This scope affects only synthetic resume tests, never formal training.
    old_hist=torch.histc
    old_mode=torch.are_deterministic_algorithms_enabled()
    def cpu_hist(x,*args,**kwargs):
        return old_hist(x.detach().cpu(),*args,**kwargs).to(x.device)
    torch.histc=cpu_hist
    torch.use_deterministic_algorithms(True)
    try:yield
    finally:
        torch.histc=old_hist
        torch.use_deterministic_algorithms(old_mode)

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value

def exact(a,b):
    import numpy as np
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('State keys differ')
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('State lengths differ')
        for x,y in zip(a,b):exact(x,y)
    elif isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise ValueError('Strict tensor equality failed')
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('Strict numpy state equality failed')
    elif a!=b:raise ValueError('State value differs')

def bn(model):
    return {k:v.clone() for k,v in model.state_dict().items() if any(t in k for t in ('running_mean','running_var','num_batches_tracked'))}

def test_s():
    from modules import SelfRetention,GGCA
    net=SelfRetention(METHOD['S']).cuda()
    prior=torch.tensor([[.9,.9,.1]],device='cuda:0')
    u=torch.tensor([[.1,.1,.5]],device='cuda:0')
    attention=torch.full((1,3,4),.25,device='cuda:0')
    neighbors=torch.tensor([[[.1]*4,[.9]*4,[.1]*4]],device='cuda:0')
    mask=torch.ones_like(neighbors,dtype=torch.bool)
    alpha,_,_=net(prior,u,attention,neighbors,mask)
    assert alpha[0,0]>alpha[0,1]>alpha[0,2] and (alpha>=0).all() and (alpha<=.5*prior).all()
    assert torch.equal(net(prior,u,attention,neighbors,~mask)[0],torch.zeros_like(prior))
    before=net.logits.detach().clone();opt=torch.optim.Adam(net.parameters(),lr=.002)
    alpha.sum().backward();assert net.logits.grad.abs().sum()>0
    opt.step();assert not torch.equal(before,net.logits)
    rows={}
    for counts in ([0,0],[1,1],[2,2],[3,5],[8,8]):
        n=torch.tensor(counts,device='cuda:0');valid=torch.arange(8,device='cuda:0')[None]<n[:,None]
        f=torch.randn(2,8,64,device='cuda:0',requires_grad=True)
        b=torch.randn(2,64,device='cuda:0',requires_grad=True)
        x=torch.randn(2,8,3,device='cuda:0')
        r=torch.full((2,8),.8,device='cuda:0');unc=torch.full_like(r,.2);ranges=torch.ones_like(r)*40
        g=GGCA(METHOD).cuda();g.enable_s();g=g.cuda()
        with torch.no_grad():g.project.weight.normal_(0,.02)
        y=g(f,b,x,valid,r,unc,ranges,1)
        dirty=f.detach().masked_fill(~valid[...,None],float('nan'))
        z=g(dirty,b,x.masked_fill(~valid[...,None],float('nan')),valid,r,unc,ranges,1)
        exact(y,z)
        y.square().mean().backward()
        assert torch.isfinite(y).all() and torch.isfinite(f.grad).all()
        rows[str(counts)]=g.last
    return dict(alpha_behavior=alpha.detach().cpu().tolist(),updated=True,degenerate=rows)

def test_c():
    from modules import Enhancement
    c=Enhancement().cuda()
    x=torch.zeros(2,64,5,5,device='cuda:0');x[0,:,0,0]=1
    occupied=torch.zeros(2,1,5,5,dtype=torch.bool,device='cuda:0');occupied[0,0,0,0]=True
    y=c(x,{},occupied,constrained=False,state_gate=False)
    assert y[0,:,0,1].abs().sum()>0 # propagation into an originally empty target
    assert torch.equal(y[1],x[1]) and torch.equal(y[0,:,4,4],x[0,:,4,4])
    exact(c(x,{},torch.zeros_like(occupied),constrained=False,state_gate=False),x)
    dirty=x.clone();dirty[0,:,4,4]=1000
    z=c(dirty,{},occupied,constrained=False,state_gate=False)
    exact(z[0,:,3,4],dirty[0,:,3,4]) # unobserved grid never becomes a source
    return dict(empty_target_receives=True,unobserved_not_source=True,no_wrap_or_cross_cav=True)

def run(args):
    from models import Model,criterion
    from opencood.hypes_yaml import yaml_utils
    from baseline import initialize_model
    root=Path(args.output_root)
    legacy=load_frozen('full_state_fixtures','server_tests.py',ROOT/'v2xr_weather_l012')
    legacy.test_full_state(root)
    s,c=test_s(),test_c()
    h=yaml_utils.load_yaml(str(Path(args.runtime_root)/'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    batch=legacy.batch();batch.update(lidar_pose=[[[0.,0.,0.,0.,0.,0.]]],ego_flag=[[True]])
    seed();reference=Model(h['model']['args'],'E0').cuda().train()
    # Nonzero A1 residual: equality tests must not be hidden by project initialization.
    with torch.no_grad():reference.ggca.project.weight.normal_(0,.02)
    shared=copy.deepcopy(reference.state_dict())
    with torch.no_grad():expected=reference(copy.deepcopy(batch))
    expected_bn=bn(reference)
    rows={}
    for variant in ('E1','E2'):
        seed();model=Model(h['model']['args'],variant).cuda().train()
        whole=copy.deepcopy(model.state_dict());whole.update(shared);model.load_state_dict(whole,strict=True)
        if variant=='E1':model.ggca.s_enabled=False
        else:model.c_enabled=False
        with torch.no_grad():actual=model(copy.deepcopy(batch))
        for k in ('psm','rm'):exact(actual[k],expected[k])
        exact(bn(model),expected_bn)
        model.load_state_dict(whole,strict=True)
        if variant=='E1':model.ggca.s_enabled=True;model.ggca.control='off'
        else:model.c_enabled=True;model.c_force_zero=True
        with torch.no_grad():zero=model(copy.deepcopy(batch))
        for k in ('psm','rm'):exact(zero[k],expected[k])
        exact(bn(model),expected_bn)
        if variant=='E1':model.ggca.control='adaptive'
        else:model.c_force_zero=False
        with deterministic_resume_test():
            opt=torch.optim.Adam(model.parameters(),lr=.002,eps=1e-10,weight_decay=1e-4)
            scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[10,15],gamma=.1)
            loss_fn=criterion(h,variant).cuda()
            before={k:v.clone() for k,v in model.state_dict().items()}
            seen=set()
            def step():
                opt.zero_grad();p=model(copy.deepcopy(batch));loss=loss_fn(p,legacy.target(p))
                loss.backward();model.repe_check_gradients()
                for k,v in model.named_parameters():
                    if v.grad is not None:
                        assert torch.isfinite(v.grad).all()
                        if v.grad.abs().sum()>0:seen.add(k)
                opt.step()
                return model.repe_statistics(loss_fn)
            for _ in range(3):step()
            prefix='ggca.s.' if variant=='E1' else 'repe_c.'
            assert any(k.startswith(prefix) for k in seen)
            assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith(prefix))
            scheduler.step(1)
            path=root/(variant+'_full_state.pth');save(path,state(model,opt,scheduler,loss_fn,1,'extension_test'))
            saved=torch.load(path,map_location='cpu',weights_only=False)
            step();after=copy.deepcopy(model.state_dict());optimizer_after=copy.deepcopy(opt.state_dict())
            restore(path,model,opt,scheduler,loss_fn,'extension_test',1)
            exact(cpu(model.state_dict()),saved['model']);exact(cpu(opt.state_dict()),saved['optimizer'])
            exact(cpu(scheduler.state_dict()),saved['scheduler']);exact(cpu(loss_fn.state_dict()),saved['criterion'])
            from checkpoint import rng,ACCUMULATORS
            exact(cpu(rng()),saved['rng'])
            exact(cpu({k:getattr(loss_fn,k) for k in ACCUMULATORS if hasattr(loss_fn,k)}),saved['accumulators'])
            step();exact(model.state_dict(),after);exact(opt.state_dict(),optimizer_after)
        # Independently verify pristine shared initialization, not the synthetic modified state.
        init_args=copy.copy(args);init_args.variant=variant
        initialize_model(model,init_args)
        rows[variant]=dict(disabled_exact_A1=True,zero_extension_exact_A1=True,BN_exact=True,
            original_GSPR_weighting_checks=model.weighting_checks,gradient_parameters=sorted(seen),
            new_parameters_updated=True,full_saved_state_exact=True,full_model_optimizer_next_step_exact=True,
            resume_next_step_scope='strict deterministic synthetic test; diagnostic histogram on CPU only',initialization_exact=True)
        del model,opt,scheduler,actual,zero,loss_fn
        torch.cuda.empty_cache()
    atomic_json(root/'summary.json',dict(status='complete',S=s,C=c,models=rows,
        full_state_rng_resume=True,real_data_acceptance='separate two-update smoke and three-weather inference',E0_optimization_steps=0))
