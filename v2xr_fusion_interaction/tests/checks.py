"""Test formulas and scope on the assigned ROCm device."""
import torch
from opencood.models.common_modules.self_attn import AttFusion
from v2xr_fusion_interaction.common.fusion import Interaction,rms
from v2xr_fusion_interaction.common.support import support_maps


@torch.no_grad()
def unit(cfg):
    device='cuda:0'
    counts=torch.tensor([3,1,2],device=device)
    layer=Interaction(AttFusion(8),8,cfg).to(device).eval()
    x=torch.randn(6,8,2,2,device=device)
    source=torch.zeros(6,2,2,2,dtype=torch.int32,device=device)
    source[0,0,0,0]=1
    source[1,0,1,1]=2
    source[0:3,1,0,0]=1
    source[4:6,:,:,0]=1
    layer.support=source
    base=layer.native(x,counts)
    assert torch.equal(layer(x,counts),base)
    layer.W.weight.normal_(std=.1)
    out={}
    for mode in ('F2','F3','F4','uniform2','uniform3','off'):
        layer.mode=mode
        out[mode]=layer(x,counts)
        assert torch.isfinite(out[mode]).all()
        assert torch.equal(out[mode][1],base[1])
    assert torch.equal(out['off'],base)
    assert torch.equal(out['F4'][0,:,0],base[0,:,0])
    assert torch.equal(out['F3'][0,:,1,1],base[0,:,1,1])
    assert torch.equal(out['F4'][0,:,1],out['F3'][0,:,1])
    a=x[:3].permute(0,2,3,1)
    n=torch.nn.functional.layer_norm(a,(8,))
    d=layer.W(layer.U(n[0])[None]*layer.V(n[1:]))
    multiplier=torch.tensor([[[.25,.25],[1.,0.]],[[.25,0.],[1.,0.]]],device=device)
    for mode,delta in (('F2',d.mean(0)),('F3',(d*multiplier[...,None]).mean(0))):
        expected=base[0]+.1*delta.tanh().permute(2,0,1)
        torch.testing.assert_close(out[mode][0],expected)
    for mode,parent in (('uniform2','F3'),('uniform3','F4')):
        torch.testing.assert_close(rms((out[mode]-base).permute(0,2,3,1)).mean((1,2)),
                                   rms((out[parent]-base).permute(0,2,3,1)).mean((1,2)),atol=1e-7,rtol=1e-5)
    perm=torch.tensor([0,2,1,3,4,5],device=device)
    for mode in ('F2','F3','F4'):
        layer.mode=mode
        layer.support=source[perm]
        torch.testing.assert_close(layer(x[perm],counts),out[mode])
    layer.mode='F2'
    changed=x.clone()
    changed[1:3]*=-1
    assert not torch.allclose(layer(changed,counts)[0]-layer.native(changed,counts)[0],out['F2'][0]-base[0])
    identical=x[:1].expand(3,-1,-1,-1).clone()
    assert torch.isfinite(layer(identical,counts[:1])).all()
    coordinates=torch.tensor([[0,0,1,1],[1,0,3,3],[1,0,2,2]],device=device)
    data=dict(processed_lidar=dict(voxel_coords=coordinates,voxel_num_points=counts.new_tensor([1,0,3])),
              processed_radar=dict(voxel_coords=coordinates[:0],voxel_num_points=counts[:0]))
    maps=support_maps(data,2,(2,2),2)
    assert maps[...,0].sum()==2 and not torch.count_nonzero(maps[...,1])
    assert maps[0,1,1,0]==1 and maps[1,1,1,0]==1
    return dict(formula=True,grouping=True,zero_single=True,neighbor_order=True,
                unique_and_empty=True,uniform_rms=True,uses_neighbor=True,raw_pillar_support=True)
