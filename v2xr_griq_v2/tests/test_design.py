"""Run on HaiGuang before opening formal jobs."""
import numpy as np
import torch
from v2xr_griq_v2.dataset import nested
from v2xr_griq_v2.multi_quality import D,loss,probs,Quality
from v2xr_griq_v2.local_evidence import coordinates,grid,extract_local,intervene,DL,DR
from v2xr_griq_v2.local_complement import LocalQuality


def main():
    assert set(range(32)) <= set(nested(list(range(250)),list(range(32))))
    p=probs(torch.randn(20,3),'ordinal')
    assert (p[:,:-1]>=p[:,1:]).all()
    for y,expected in [(torch.zeros(4,3),[True,False,False]),(torch.tensor([[1.,0,0]]*4),[True,True,False])]:
        z=torch.zeros(4,3,requires_grad=True)
        loss(z,y,'ordinal',torch.ones(3)).sum().backward()
        assert (z.grad.abs().sum(0)>0).tolist()==expected
    box=np.array([0,0,0,2,2,6,np.pi/2])
    pts=np.array([[0,-2,0],[0,2,0],[-1,0,0],[1,0,0]])
    assert grid(coordinates(pts,box),box).tolist()==[3,5,4,1]
    empty=np.empty((0,3))
    ctx=(([empty],[empty]),empty,np.empty(0),np.empty(0),np.empty(0))
    values=extract_local(np.array([box]),ctx,lambda a,b:np.zeros(len(a),bool))
    assert values.shape==(1,6,DL+DR) and values[0,:,5].all() and values[0,:,DL+5].all()
    assert extract_local(np.empty((0,7)),ctx,lambda a,b:None).shape[0]==0
    model=LocalQuality('cross')
    x=torch.randn(5,D*2)
    local=torch.from_numpy(values).expand(5,-1,-1)
    assert torch.equal(model(x,local),Quality.forward(model,x))
    model(x,local).sum().backward()
    assert torch.isfinite(model.alpha.grad)
    a=np.arange(6*(DL+DR),dtype=np.float32).reshape(1,6,-1)
    b=intervene(a,'permute',np.random.default_rng(7))
    assert np.array_equal(a[:,:,:DL],b[:,:,:DL]) and np.array_equal(a[:,:,DL:].sum(1),b[:,:,DL:].sum(1))
    print('DESIGN PASS',flush=True)


if __name__=='__main__':
    main()
