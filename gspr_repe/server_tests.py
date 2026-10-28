"""Server acceptance: actual module math, full official detector, gradients and recovery."""
import copy
from pathlib import Path
import torch
from auditlib import ROOT, atomic_json, load_frozen
from modules import Aggregation, Observation, Enhancement, statistics


def finite_state(model, stage, variant):
    bad=[name for name,value in model.state_dict().items()
         if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all()]
    if bad:
        raise ValueError(f'{variant} {stage}: nonfinite parameters/buffers: {bad[:20]}')


def finite_gradients(model, variant):
    found=[(name,p.grad) for name,p in model.named_parameters() if p.grad is not None]
    bad=[name for name,grad in found if not torch.isfinite(grad).all()]
    if not found or bad:
        raise ValueError(f'{variant} before optimizer: missing/nonfinite full-model gradients: {bad[:20]}')
    return len(found)


def test_degenerate_statistics():
    """Server regression: singleton/coincident/untrusted points and zero upstream gradients."""
    device='cuda:0'
    points=torch.zeros(3,4,4,device=device,requires_grad=True)
    valid=torch.tensor([[True,False,False,False],[True,True,True,False],
                        [True,True,False,False]],device=device)
    reliability=torch.tensor([[.9,.9,.9,.9],[.9,.9,.9,.9],[.1,.1,.1,.1]],device=device,requires_grad=True)
    uncertainty=torch.zeros_like(reliability)
    result=statistics(points,valid,reliability,uncertainty)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[:,5],torch.zeros(3,device=device),rtol=0,atol=1e-7)
    # The old square root failed even if the next zero-initialized layer sent zero gradients.
    point_grad,r_grad=torch.autograd.grad((result*0).sum(),(points,reliability))
    assert torch.isfinite(point_grad).all() and torch.isfinite(r_grad).all()
    result=statistics(points,valid,reliability,uncertainty)
    point_grad,r_grad=torch.autograd.grad(result.sum(),(points,reliability))
    assert torch.isfinite(point_grad).all() and torch.isfinite(r_grad).all()
    # Also exercise the real observation head, whose final weights start at zero.
    result=statistics(points,valid,reliability,uncertainty)
    x=torch.randn(3,64,1,1,device=device,requires_grad=True)
    observation=Observation().to(device)(x,result[:,:,None,None])
    loss=observation['q'].sum()+observation['s'].sum()
    point_grad,r_grad=torch.autograd.grad(loss,(points,reliability))
    assert torch.isfinite(point_grad).all() and torch.isfinite(r_grad).all()
    return dict(singleton=True,coincident_points=True,no_trusted_points=True,
        zero_upstream_gradient=True,zero_initialized_observation_backward=True)


def unit():
    device = 'cuda:0'
    x = torch.randn(3,64,7,9,device=device,requires_grad=True)
    stats = torch.zeros(3,7,7,9,device=device)
    stats[:,:,2:5,2:5] = 1
    stats[:,0] *= 6
    obsnet, c = Observation().to(device), Enhancement().to(device)
    obs = obsnet(x,stats)
    out = c(x,obs,stats[:,:1]>0)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    empty = {k:torch.zeros_like(x[:,:1]) for k in ('q','s','unknown','candidate')}
    torch.testing.assert_close(c(x,empty,stats[:,:1]>0),x,rtol=0,atol=0)
    points=torch.randn(4,8,4,device=device)
    valid=torch.arange(8,device=device)[None]<torch.tensor([1,3,5,8],device=device)[:,None]
    features=torch.randn(4,8,10,device=device)
    r=torch.rand(4,8,device=device)*valid
    u=torch.rand_like(r)*valid
    a=Aggregation(10,64).to(device)
    original=torch.randn(4,64,device=device)
    expected=a(features,original,points,valid,r,u)
    features[~valid]=9999
    points[~valid]=-9999
    actual=a(features,original,points,valid,r,u)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.isfinite(statistics(points,valid,r,u)).all()
    expected.sum().backward()
    assert a.score[-1].weight.grad is not None
    # Demonstrate feature/source dependence rather than merely varying a gate.
    c(x.detach(),obs,stats[:,:1]>0)
    weights=c.last['source_mass'].clone()
    altered=dict(obs,q=torch.zeros_like(obs['q']))
    torch.testing.assert_close(c(x,altered,stats[:,:1]>0),x,rtol=0,atol=0)
    assert weights.max()>0
    return dict(padding=True,empty_sources=True,multiple_cavs=True,source_intervention=True)


def run(args):
    from opencood.hypes_yaml import yaml_utils
    from checkpoint import seed, save
    from models import Model, FLAGS
    root=Path(args.output_root); root.mkdir(parents=True,exist_ok=True)
    legacy=load_frozen('repe_server_fixture','server_tests.py',ROOT/'v2xr_weather_l012')
    legacy.test_full_state(root)
    checks=unit()
    checks['degenerate_statistics']=test_degenerate_statistics()
    recovery(root)
    h=yaml_utils.load_yaml(str(Path(args.runtime_root)/'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    integration=load_frozen('gspr_integration','integration.py',ROOT/'v2xr_gspr_clean')
    seed(); reference=integration.PointPillarGsprOfficial(h['model']['args']).cuda().eval()
    seed(); baseline=Model(h['model']['args'],'B0').cuda().eval()
    inputs=legacy.batch()
    with torch.no_grad():
        a,b=reference(copy.deepcopy(inputs)),baseline(copy.deepcopy(inputs))
        for key in ('psm','rm'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    del reference,baseline,a,b
    rows={}
    # Include isolated B and C switches as structural tests beyond formal B0--B4.
    for variant, flags in list(FLAGS.items())+[('B_only',(False,True,False)),('C_only',(False,False,True))]:
        atomic_json(root/'numerical_progress.json',dict(status='running',variant=variant,phase='initialization'))
        print('STRUCTURE TEST '+variant+': initialize/train/backward/update/eval/reload',flush=True)
        cfg=copy.deepcopy(h['model']['args']); cfg['repe_modules']=flags
        model=Model(cfg,variant).cuda().train()
        batch=copy.deepcopy(inputs)
        raw=batch['processed_lidar']
        raw['voxel_num_points']=torch.tensor([1,2,3,4],device='cuda:0',dtype=torch.int32)
        raw['voxel_coords'][:,0]=torch.tensor([0,0,1,2],device='cuda:0')
        batch['record_len']=torch.tensor([2,1],device='cuda:0')
        for i,n in enumerate((1,2,3,4)):
            raw['voxel_features'][i,n:]=0
        finite_state(model,'before training forward',variant)
        prediction=model(batch)
        finite_state(model,'after training forward',variant)
        before={k:v.detach().clone() for k,v in model.named_parameters() if k.startswith('repe_')}
        objective=prediction['psm'].square().mean()+prediction['rm'].square().mean()
        if 'repe_aux' in prediction:
            objective=objective+.01*sum(prediction['repe_aux'])
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        if not torch.isfinite(objective):
            raise ValueError(variant+' nonfinite structural objective')
        atomic_json(root/'numerical_progress.json',dict(status='running',variant=variant,phase='backward'))
        objective.backward()
        gradient_count=finite_gradients(model,variant)
        model.repe_check_gradients()
        finite_state(model,'before optimizer',variant)
        optimizer.step()
        finite_state(model,'after optimizer',variant)
        updated={name:any(not torch.equal(before[k],v) for k,v in model.named_parameters() if k.startswith(name+'.'))
                 for name in ('repe_a','repe_b','repe_c') if hasattr(model,name)}
        assert all(updated.values()), updated
        model.eval()
        atomic_json(root/'numerical_progress.json',dict(status='running',variant=variant,phase='updated_model_eval'))
        with torch.no_grad():
            expected=model(copy.deepcopy(batch))
            save(root/(variant+'.pth'),model.state_dict())
            restored=Model(cfg,variant).cuda().eval()
            restored.load_state_dict(torch.load(root/(variant+'.pth'),weights_only=False),strict=True)
            actual=restored(copy.deepcopy(batch))
            for key in ('psm','rm'):
                torch.testing.assert_close(expected[key],actual[key],rtol=1e-5,atol=1e-5)
        atomic_json(root/'numerical_progress.json',dict(status='complete',variant=variant,phase='eval_and_reload_passed'))
        rows[variant]=dict(updated=updated,finite_full_model_gradients=True,gradient_parameter_count=gradient_count,
            finite_parameters_and_buffers=True,parameters=sum(p.numel() for p in model.parameters()))
        del model,restored,prediction,expected,actual,objective,optimizer
        torch.cuda.empty_cache()
    assert rows['B3']['parameters']==rows['B4']['parameters']
    atomic_json(root/'summary.json',dict(status='complete',checks=checks,models=rows,
        disabled_exact=True,full_state_resume=True,wrong_contract_rejected=True))


def recovery(root):
    from argparse import Namespace
    from orchestrator import Pipeline
    from auditlib import read_json
    import tempfile
    test=Path(tempfile.mkdtemp(prefix='journal_test_',dir=root))
    (test/'logs').mkdir(exist_ok=True)
    output=test/'artifact'; output.mkdir(exist_ok=True)
    pipeline=Pipeline(Namespace(output_root=str(test)), {'identity':'recovery-fixture'})
    def fail():
        raise RuntimeError('injected interruption')
    try:
        pipeline.stage('fixture',output,callback=fail)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Failure was silently accepted')
    assert read_json(test/'stages/fixture/failure.json')['status']=='technical_failure'
    def succeed():
        (output/'value.txt').write_text('complete',encoding='utf-8')
    pipeline.stage('fixture',output,callback=succeed)
    resumed=Pipeline(Namespace(output_root=str(test)), {'identity':'recovery-fixture'})
    resumed.stage('fixture',output,callback=fail)
    (output/'value.txt').write_text('tampered',encoding='utf-8')
    try:
        Pipeline(Namespace(output_root=str(test)),{'identity':'recovery-fixture'}).stage('fixture',output)
    except ValueError:
        pass
    else:
        raise AssertionError('Tampered completion was accepted')
