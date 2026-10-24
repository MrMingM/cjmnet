"""Server acceptance: actual module math, full official detector, gradients and recovery."""
import copy
from pathlib import Path
import torch
from auditlib import ROOT, atomic_json, load_frozen
from modules import Aggregation, Observation, Enhancement, statistics


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
        cfg=copy.deepcopy(h['model']['args']); cfg['repe_modules']=flags
        model=Model(cfg,variant).cuda().train()
        batch=copy.deepcopy(inputs)
        raw=batch['processed_lidar']
        raw['voxel_num_points']=torch.tensor([1,2,3,4],device='cuda:0',dtype=torch.int32)
        raw['voxel_coords'][:,0]=torch.tensor([0,0,1,2],device='cuda:0')
        batch['record_len']=torch.tensor([2,1],device='cuda:0')
        for i,n in enumerate((1,2,3,4)):
            raw['voxel_features'][i,n:]=0
        prediction=model(batch)
        before={k:v.detach().clone() for k,v in model.named_parameters() if k.startswith('repe_')}
        objective=prediction['psm'].square().mean()+prediction['rm'].square().mean()
        if 'repe_aux' in prediction:
            objective=objective+.01*sum(prediction['repe_aux'])
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        objective.backward(); model.repe_check_gradients(); optimizer.step()
        updated={name:any(not torch.equal(before[k],v) for k,v in model.named_parameters() if k.startswith(name+'.'))
                 for name in ('repe_a','repe_b','repe_c') if hasattr(model,name)}
        assert all(updated.values()), updated
        model.eval()
        with torch.no_grad():
            expected=model(copy.deepcopy(batch))
            save(root/(variant+'.pth'),model.state_dict())
            restored=Model(cfg,variant).cuda().eval()
            restored.load_state_dict(torch.load(root/(variant+'.pth'),weights_only=False),strict=True)
            actual=restored(copy.deepcopy(batch))
            for key in ('psm','rm'):
                torch.testing.assert_close(expected[key],actual[key],rtol=1e-5,atol=1e-5)
        rows[variant]=dict(updated=updated,parameters=sum(p.numel() for p in model.parameters()))
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
