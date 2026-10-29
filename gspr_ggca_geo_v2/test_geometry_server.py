"""Behavioral geometry tests. Execute on Haiguang only, with real Torch."""
import copy
import importlib.util
from pathlib import Path
import torch
from auditlib import METHOD, atomic_json
from modules import GGCA, masked_softmax
from geometry import pair_geometry, geometric_messages


def dense(pair, values, t):
    return values.new_zeros(values.shape[0],t,t).scatter(-1,pair['indices'],values)


def inputs():
    x=torch.tensor([[0.,0.,0.],[.035,.012,0.],[.09,.005,.008],
        [.14,.02,.003],[.015,.018,.36],[.055,.003,.36],[.10,.025,.365],[.135,.01,.36]],device='cuda:0')[None]
    f=torch.arange(8*64,device='cuda:0',dtype=torch.float32).reshape(1,8,64).remainder(19)/19+.05
    valid=torch.ones(1,8,device='cuda:0',dtype=torch.bool)
    prior=torch.full((1,8),.7,device='cuda:0')
    ranges=torch.full_like(prior,30.)
    return x,f,valid,prior,ranges


def run(root, old_source):
    torch.manual_seed(20261017)
    x,f,valid,p,ranges=inputs()
    pair=pair_geometry(x,f,valid,ranges,METHOD)
    zero=torch.zeros_like(pair['compatibility'])
    msg,rel=geometric_messages(f,p,zero,pair,valid,masked_softmax)
    rows={}
    # Same p, changed structure. Inspect both relations and controlled nonzero
    # residuals so zero-initialized output projection cannot hide a failure.
    changed=x.clone();changed[:,1,2]+=.23
    pair2=pair_geometry(changed,f,valid,ranges,METHOD)
    msg2,rel2=geometric_messages(f,p,zero,pair2,valid,masked_softmax)
    assert (msg-msg2).abs().max()>1e-4
    net=GGCA(METHOD).cuda().eval()
    with torch.no_grad():
        net.query.weight.zero_();net.key.weight.zero_()
        for q in net.geometry.parameters():q.zero_()
        net.project.weight.copy_(torch.eye(64,device='cuda:0')*.4)
    base=torch.linspace(.1,1.1,64,device='cuda:0')[None]
    u=torch.full_like(p,.1);r=p/(1-u)
    out1=net(f,base,x,valid,r,u,ranges,2)
    out2=net(f,base,changed,valid,r,u,ranges,2)
    assert (out1-out2).abs().max()>1e-5
    rows['equal_prior_structural_change']=dict(message_max_delta=float((msg-msg2).abs().max()),
        nonzero_residual_output_delta=float((out1-out2).abs().max()))
    # Near-equal priors must not restore the old cancellation.
    close_p=p+torch.linspace(-1e-4,1e-4,8,device='cuda:0')
    a,_=geometric_messages(f,close_p,zero,pair,valid,masked_softmax)
    b,_=geometric_messages(f,close_p,zero,pair2,valid,masked_softmax)
    assert (a-b).abs().max()>1e-4
    weak={**pair,'compatibility':pair['compatibility']*.1}
    weak_msg,weak_rel=geometric_messages(f,p,zero,weak,valid,masked_softmax)
    torch.testing.assert_close(weak_rel['exchange_mass'],rel['exchange_mass']*.1)
    torch.testing.assert_close(weak_rel['neighbor_trust'],rel['neighbor_trust'])
    assert (weak_msg-f).norm() < (msg-f).norm()
    # Reliability changes trust but not geometry; uniform trust scaling leaves
    # normalized A1 attention unchanged and is explicitly not called support.
    _,low_rel=geometric_messages(f,p*.2,zero,pair,valid,masked_softmax)
    torch.testing.assert_close(low_rel['support'],rel['support'],rtol=0,atol=0)
    torch.testing.assert_close(low_rel['neighbor_trust'],rel['neighbor_trust']*.2)
    rows['absolute_support_separated_from_trust']=dict(strong_exchange=float(rel['exchange_mass'].mean()),
        weak_exchange=float(weak_rel['exchange_mass'].mean()),trust=float(rel['neighbor_trust'].mean()),
        lowered_trust=float(low_rel['neighbor_trust'].mean()))
    edges=dense(pair,rel['exchange'],8)
    assert edges[:,:4,4:].abs().max()==0 and edges[:,4:,:4].abs().max()==0
    assert edges[:,:4,:4].sum()>0 and edges[:,4:,4:].sum()>0
    rows['two_real_surfaces']=dict(cross_surface_exchange=0.,both_surfaces_retained=True)
    # High-p isolated conflict retains its own information, rather than
    # mistaking lack of a compatible neighbor for a noise label.
    isolated=x.clone();isolated[:,0,2]=1.3
    isolated_pair=pair_geometry(isolated,f,valid,ranges,METHOD)
    high=p.clone();high[:,0]=.99
    im,ir=geometric_messages(f,high,zero,isolated_pair,valid,masked_softmax)
    torch.testing.assert_close(im[:,0],f[:,0],rtol=0,atol=0)
    assert ir['self_mass'][0,0]==1 and not ir['trust_known'][0,0]
    low=p.clone();low[:,1]=.03
    lm,lr=geometric_messages(f,low,zero,pair,valid,masked_softmax)
    assert lr['support'][0,1]>0 and lr['neighbor_trust'][0,1]>.5
    assert (lm[:,1]-f[:,1]).abs().max()>1e-4
    rows['conflicting_high_and_supported_low']=dict(high_conflict_self_mass=float(ir['self_mass'][0,0]),
        low_supported_neighbor_trust=float(lr['neighbor_trust'][0,1]),
        low_supported_exchange=float(lr['exchange_mass'][0,1]))
    # A coherent spurious cluster with the same relative geometry/features and
    # priors is unidentifiable by this method. Record rather than conceal it.
    coherent=x.clone();coherent[:,4:]=x[:,:4]+torch.tensor([0.,0.,.7],device=x.device)
    cf=f.clone();cf[:,4:]=f[:,:4]
    cp=pair_geometry(coherent,cf,valid,ranges,METHOD)
    ce=dense(cp,cp['compatibility'],8)
    torch.testing.assert_close(ce[:,:4,:4],ce[:,4:,4:],rtol=1e-5,atol=2e-6)
    rows['coherent_anomaly_boundary']=dict(indistinguishable_from_matching_real_cluster=True,
        limitation='No point noise truth; locally coherent clutter can remain supported.')
    # Pair geometry is rigid-motion invariant; this deliberately does not claim
    # invariance of the inherited XYZ-conditioned A1/PFN or changed voxelization.
    angle=.73
    rotation=x.new_tensor([[torch.cos(torch.tensor(angle)),0.,torch.sin(torch.tensor(angle))],
                          [0.,1.,0.],[-torch.sin(torch.tensor(angle)),0.,torch.cos(torch.tensor(angle))]])
    rigid=pair_geometry(x@rotation+x.new_tensor([3.,-4.,.8]),f,valid,ranges,METHOD)
    torch.testing.assert_close(dense(rigid,rigid['compatibility'],8),dense(pair,pair['compatibility'],8),rtol=1e-4,atol=8e-6)
    rows['rigid_geometry_invariance']=dict(conditional_on_same_features_ranges_membership=True)
    perm=torch.tensor([5,1,7,0,4,2,6,3],device=x.device)
    inverse=torch.argsort(perm)
    pp=pair_geometry(x[:,perm],f[:,perm],valid[:,perm],ranges[:,perm],METHOD)
    restored=dense(pp,pp['compatibility'],8)[:,inverse][:,:,inverse]
    torch.testing.assert_close(restored,dense(pair,pair['compatibility'],8),rtol=1e-5,atol=1e-6)
    rows['point_order_equivariance']=True
    # A3 uses the exact same graph/parameters, changing only the final scalar
    # correction strength. This test uses the already nonzero projection.
    original_output=net(f,base,x,valid,r,u,ranges,2)
    protected_output=net(f,base,x,valid,r,u,ranges,3)
    evidence=net.pillar_rows[:,5]
    torch.testing.assert_close(protected_output-base,(original_output-base)*evidence[:,None],rtol=2e-4,atol=1e-7)
    assert ((evidence>=0)&(evidence<=1)).all()
    rows['A3_only_attenuates_A2_residual']=dict(sufficiency=float(evidence[0]),same_pair_geometry=True)
    # Equal but tiny priors keep N_eff high. Absolute evidence must nevertheless
    # switch off A3's correction, while A2's geometry remains active.
    low_r=torch.full_like(r,.01)/(1-u)
    low_a2=net(f,base,x,valid,low_r,u,ranges,2)
    low_a3=net(f,base,x,valid,low_r,u,ranges,3)
    torch.testing.assert_close(low_a3,base,rtol=0,atol=0)
    assert net.pillar_rows[0,15]>7.9 and net.pillar_rows[0,14]<.1
    assert net.pillar_rows[0,5]==0 and (low_a2-base).abs().max()>1e-4
    rows['A3_low_absolute_trust_with_high_effective_count']=dict(
        effective_points=float(net.pillar_rows[0,15]),absolute_prior_mass=float(net.pillar_rows[0,14]),
        sufficiency=float(net.pillar_rows[0,5]),exact_original_pillar_retained=True)
    # Actual old A1 implementation and new inheritance must match with a
    # nonzero branch, input gradients, parameter gradients and an Adam update.
    path=Path(old_source)/'modules.py'
    spec=importlib.util.spec_from_file_location('_ggca_original_A1_test',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    torch.manual_seed(71);a=old.GGCA(METHOD).cuda().train()
    torch.manual_seed(71);b=GGCA(METHOD).cuda().train()
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    with torch.no_grad():a.project.weight.normal_(0,.03)
    b.load_state_dict(a.state_dict(),strict=True)
    optimizers=[torch.optim.Adam(n.parameters(),lr=.002) for n in (a,b)]
    observed=[]
    for model,opt in zip((a,b),optimizers):
        feature=f.detach().clone().requires_grad_();xyz=x.detach().clone().requires_grad_()
        output=model(feature,base,xyz,valid,r,u,ranges,1)
        output.square().sum().backward()
        observed.append((output.detach(),feature.grad.clone(),xyz.grad.clone(),
            {k:q.grad.clone() for k,q in model.named_parameters() if q.grad is not None}))
        opt.step()
    for av,bv in zip(observed[0][:3],observed[1][:3]):torch.testing.assert_close(av,bv,rtol=0,atol=0)
    for k,v in observed[0][3].items():torch.testing.assert_close(v,observed[1][3][k],rtol=0,atol=0)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    rows['preserved_A1_nonzero_equivalence']=dict(initialization=True,output=True,input_gradients=True,
        parameter_gradients=True,Adam_step=True)
    atomic_json(Path(root)/'GEOMETRY_MECHANISM_TESTS.json',dict(status='complete',revision=2,cases=rows,
        zero_residual_tests='separate full-model server_tests',new_auxiliary_loss=False))
    return rows
