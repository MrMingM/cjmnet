"""Same-position low-rank ego-neighbor interactions on native AttFuse outputs."""
import torch
from torch import nn
from torch.nn import functional as F


def rms(tensor):
    return tensor.float().square().mean(-1).sqrt()


class Interaction(nn.Module):
    def __init__(self, native, channels, cfg):
        super().__init__()
        self.native=native
        self.mode=cfg['mode']
        self.beta=cfg['beta']
        self.single_support=cfg['single_support']
        self.U=nn.Linear(channels,cfg['rank'],bias=False)
        self.V=nn.Linear(channels,cfg['rank'],bias=False)
        self.W=nn.Linear(cfg['rank'],channels,bias=False)
        nn.init.zeros_(self.W.weight)
        self.support=None
        self.observe=False
        self.report=[]

    @property
    def needs_support(self):
        return self.observe or self.mode not in ('F2','off')

    def residual(self, interactions, gate=None):
        if gate is not None:
            interactions=interactions*gate[...,None]
        # W is linear and bias-free. Moving it after the mean avoids C channels per pair.
        return self.beta*self.W(interactions.mean(0)).tanh()

    def forward(self,x,record_len):
        baseline=self.native(x,record_len)
        self.report=[]
        if self.mode=='off':
            return baseline
        lengths=record_len.tolist()
        if sum(lengths)!=len(x) or min(lengths)<1:
            raise ValueError('CAV layout differs from record_len')
        if self.needs_support and (self.support is None or self.support.shape!=(*x.shape[:1],*x.shape[-2:],2)):
            raise ValueError('Observation layout differs from the fusion grid')
        frames=x.split(lengths)
        sources=self.support.split(lengths) if self.needs_support else [None]*len(lengths)
        outputs=[]
        for index,(features,source) in enumerate(zip(frames,sources,strict=True)):
            base=baseline[index].permute(1,2,0)
            if len(features)==1:
                # Retain a zero gradient connection when a training batch is all single-vehicle.
                zero=sum(p.reshape(-1)[0]*0 for p in (self.U.weight,self.V.weight,self.W.weight))
                outputs.append((base+zero).permute(2,0,1))
                if self.observe:
                    self.report.append(dict(cars=1))
                continue
            features=features.permute(0,2,3,1)
            channels=features.shape[-1:]
            u=self.U(F.layer_norm(features[0],channels))
            v=self.V(F.layer_norm(features[1:],channels))
            h=u.unsqueeze(0)*v
            masks=unique=None
            if source is not None:
                observed=source.gt(0).any(-1)
                pair_count=observed[:1].int()+observed[1:].int()
                gate=(pair_count==2).to(h.dtype)+(pair_count==1).to(h.dtype)*self.single_support
                masks=observed.sum(0)
                unique=masks==1
            d2=self.residual(h) if self.mode in ('F2','uniform2') or self.observe else None
            d3=self.residual(h,gate) if source is not None else None
            d4=d3.masked_fill(unique[...,None],0.) if d3 is not None else None
            if self.mode in ('uniform2','uniform3'):
                parent,child=(d2,d3) if self.mode=='uniform2' else (d3,d4)
                scale=rms(child).sum()/rms(parent).sum().clamp_min(1e-30)
                delta=parent*scale
                if not torch.isclose(rms(delta).mean(),rms(child).mean(),rtol=2e-5,atol=1e-9):
                    raise FloatingPointError('Uniform control could not match the child mean cell RMS')
            else:
                delta={'F2':d2,'F3':d3,'F4':d4}[self.mode]
            outputs.append((base+delta).permute(2,0,1))
            if self.observe:
                if self.mode not in ('F2','F3','F4'):
                    raise ValueError('Mechanism capture expects a trained group mode')
                location=torch.where(masks==0,0,torch.where(masks>1,3,torch.where(observed[0],1,2)))
                pair=self.W(h)
                self.report.append(dict(cars=len(features),base=rms(base),active=rms(delta),location=location,
                    F2=rms(d2),F3=rms(d3),F4=rms(d4),F2_F3=rms(d2-d3),F3_F4=rms(d3-d4),
                    pair_category=pair_count, pair_raw=rms(pair),pair_gated=rms(pair*gate[...,None]),
                    sensor_pillars=source.sum((1,2)),support_vehicle_count=masks))
        return torch.stack(outputs)
