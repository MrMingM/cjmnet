"""Local controls and same-cell cross-modal residual."""
import torch
from torch import nn
from v2xr_griq_v2.multi_quality import Quality, mlp, matched_widths, size
from v2xr_griq_v2.local_evidence import DL,DR


class LocalQuality(Quality):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.alpha = nn.Parameter(torch.zeros(()))
        if variant=='cross':
            self.cross = mlp(DL+DR+7,24,16,16)
            self.head = nn.Linear(32,3)
        else:
            if variant in ('lidar','independent'):
                self.l = mlp(DL,24,16,16)
            if variant in ('radar','independent'):
                self.r = mlp(DR,24,16,16)
            self.head = nn.Linear(64 if variant=='independent' else 32,3)

    def delta(self,x,local):
        # Explicit masks keep unknown reliability separate from low reliability.
        l,r = local.split([DL,DR],dim=2)
        def pool(v):
            return torch.cat([v.mean(1),v.amax(1)],1)
        if self.variant=='cross':
            lm,rm,gm = l[:,:,5],r[:,:,5],l[:,:,6]
            sparse = (l[:,:,0]<=torch.log1p(l.new_tensor(5.))).to(l.dtype)
            distance = (x[:,2:3]/100).expand_as(lm)
            joint = torch.stack([(1-lm)*(1-rm),lm*(1-rm),(1-lm)*rm,lm*rm,
                                 sparse*(1-rm),(1-gm)*(1-rm)*l[:,:,2],distance],2)
            return self.head(pool(self.cross(torch.cat([l,r,joint],2))))
        parts = []
        if self.variant in ('lidar','independent'):
            parts.append(pool(self.l(l)))
        if self.variant in ('radar','independent'):
            parts.append(pool(self.r(r)))
        return self.head(torch.cat(parts,1))

    def forward(self,x,local=None):
        base = super().forward(x)
        if local is None:
            return base
        return base+self.alpha*self.delta(x,local)


def model(variant, mode):
    if variant in ('cross','lidar','radar','independent'):
        return LocalQuality(variant)
    if variant=='joint_local_capacity':
        target = size(LocalQuality('cross'))
        return Quality(mode,'joint',matched_widths(target,mode,'joint'))
    if variant=='joint':
        return Quality(mode,'joint',matched_widths(size(Quality()),mode,'joint'))
    if variant=='strict_capacity':
        return Quality(mode,'additive',matched_widths(size(Quality()),mode,'additive'))
    return Quality(mode)
