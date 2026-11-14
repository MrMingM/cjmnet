"""Additive conditional quality probabilities and matched model controls."""
import torch
from torch import nn
from torch.nn import functional as F
from v2xr_griq_experiments.features import BASE, GSPR, RADAR

NB, NG, NR = map(len,(BASE,GSPR,RADAR))
D = NB+NG+NR
LEVELS = (.3,.5,.7)
LOSS_WEIGHTS = ((1,1,1),(1,1,2),(2,2,1))


def mlp(n,h,k,out):
    return nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Linear(h,k),nn.ReLU(),nn.Linear(k,out))


def probs(z, mode):
    p = z.sigmoid()
    return p.cumprod(1) if mode == 'ordinal' else p


def loss(z, y, mode, weights):
    if mode == 'strict':
        return F.binary_cross_entropy_with_logits(z[:,0],y[:,2])[None]
    masks = torch.stack([torch.ones_like(y[:,0]),y[:,0],y[:,1]],1) if mode=='ordinal' else torch.ones_like(y)
    errors = F.binary_cross_entropy_with_logits(z,y,reduction='none')
    return (errors*masks).sum(0)/masks.sum(0).clamp_min(1)*weights


class Quality(nn.Module):
    def __init__(self, mode='ordinal', architecture='additive', widths=None):
        super().__init__()
        self.mode = mode
        self.architecture = architecture
        n = 1 if mode=='strict' else 3
        if architecture=='additive':
            hl,kl,hr,kr = widths or (48,24,32,16)
            self.lidar = mlp(2*(NB+NG),hl,kl,n)
            self.radar = mlp(2*NR,hr,kr,n)
        else:
            h,k = widths or (48,24)
            self.joint = mlp(2*D,h,k,n)

    def forward(self,x,local=None):
        if self.architecture=='joint':
            return self.joint(x)
        v,m = x.split(D,dim=1)
        cut = NB+NG
        return self.lidar(torch.cat([v[:,:cut],m[:,:cut]],1))+self.radar(torch.cat([v[:,cut:],m[:,cut:]],1))


def size(net):
    return sum(p.numel() for p in net.parameters())


def matched_widths(target, mode, arch):
    n = 1 if mode=='strict' else 3
    def count(inp,h,k):
        return (inp+1)*h+(h+1)*k+(k+1)*n
    if arch=='joint':
        options = ((abs(count(2*D,h,k)-target),abs(h-48)+abs(k-24),(h,k))
                   for h in range(28,100) for k in range(12,55))
    else:
        options = ((abs(count(2*(NB+NG),h,k)+count(2*NR,r,16)-target),abs(h-48)+abs(k-24)+abs(r-32),(h,k,r,16))
                   for h in range(42,55) for k in range(18,31) for r in range(28,38))
    return min(options)[2]
