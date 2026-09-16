import math
import torch
from torch import nn
import torch.nn.functional as F


def stencil(xyz, shape, extent):
    """Bilinear sampling, border padding, align_corners=False; batch size one."""
    h, w = shape
    x = ((xyz[:, 0]-extent[0])/(extent[3]-extent[0])*w-.5).clamp(0, w-1)
    y = ((xyz[:, 1]-extent[1])/(extent[4]-extent[1])*h-.5).clamp(0, h-1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0+1).clamp_max(w-1), (y0+1).clamp_max(h-1)
    dx, dy = x-x0, y-y0
    indices = torch.stack((y0*w+x0,y0*w+x1,y1*w+x0,y1*w+x1),1)
    weights = torch.stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy),1)
    z = 2*(xyz[:,2]-extent[2])/(extent[5]-extent[2])-1
    coordinates = torch.stack((torch.ones_like(z),z,z.sin(),z.cos(),
                               (2*math.pi*x).sin(),(2*math.pi*x).cos(),
                               (2*math.pi*y).sin(),(2*math.pi*y).cos()),1)
    return indices, weights, coordinates


class QueryDecoder(nn.Module):
    """Coordinate-conditioned linear feature query; analytic correction direction."""
    def __init__(self, channels=384):
        super().__init__()
        self.channels = channels
        self.weight = nn.Linear(8,channels,bias=False)
        self.bias = nn.Linear(8,1,bias=False)
        nn.init.normal_(self.weight.weight,std=.02)
        nn.init.zeros_(self.bias.weight)

    def terms(self, feature, xyz, extent):
        indices, weights, coordinates = stencil(xyz,feature.shape[-2:],extent)
        vector = self.weight(coordinates)/math.sqrt(self.channels)
        sampled = (feature[0].flatten(1).T[indices]*weights[:,:,None]).sum(1)
        logits = (sampled*vector).sum(1)+self.bias(coordinates).squeeze(1)
        return logits, vector, indices, weights

    def forward(self, feature, xyz, extent):
        return self.terms(feature,xyz,extent)[0]


def interval_loss(logits, lower, upper):
    if not logits.numel():
        return logits.sum()*0
    probability = logits.sigmoid()
    return (F.relu(lower-probability).square()+F.relu(probability-upper).square()).mean()


def project_observations(feature, decoder, packet, extent, step):
    """One damped local affine projection. No autograd.grad/double-backward kernels.

    Simultaneous overlapping corrections are averaged, so not an exact global QP.
    Inconsistent intervals retain their gap; only a compromise correction is used.
    """
    xyz, lower, upper = packet['xyz'], packet['lower'], packet['upper']
    if not len(xyz):
        return feature
    logits, vector, indices, weights = decoder.terms(feature,xyz,extent)
    lo = torch.logit(lower.clamp(.001,.999))
    hi = torch.logit(upper.clamp(.001,.999))
    change = torch.where(lower > 0,F.relu(lo-logits),0)-torch.where(upper < 1,F.relu(logits-hi),0)
    conflict = lower > upper
    midpoint = torch.logit(((lower+upper)/2).clamp(.001,.999))
    change = torch.where(conflict,midpoint-logits,change)
    change = change.clamp(-2,2)/(1+F.relu(lower-upper))
    norm = vector.square().sum(1)*weights.square().sum(1)
    update = change[:,None]*vector/(norm[:,None]+.01)
    # Scatter only at the four neighboring feature locations, never a whole box.
    flat = feature.new_zeros((feature.shape[-2]*feature.shape[-1],feature.shape[1]))
    density = feature.new_zeros((len(flat),1))
    flat = flat.index_add(0,indices.flatten(),(update[:,None,:]*weights[:,:,None]).reshape(-1,feature.shape[1]))
    active = ((lower > 0)|(upper < 1)).to(feature.dtype)
    density = density.index_add(0,indices.flatten(),(weights*active[:,None]).reshape(-1,1))
    correction = (flat/density.clamp_min(1)).T.reshape_as(feature)
    return feature+step*correction


class MiniCEIF(nn.Module):
    def __init__(self,channels=384,hidden=32):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(channels,hidden,1),nn.SiLU(),
            nn.Conv2d(hidden,hidden,3,padding=1),nn.SiLU(),nn.Conv2d(hidden,channels,1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        # Same parameter exists in the auxiliary-only control, but is unused.
        self.step_logit = nn.Parameter(torch.tensor(-2.1972246))  # sigmoid = .1

    def forward(self,feature,decoder,packet,extent,project=True,return_prior=False):
        prior = feature+.1*torch.tanh(self.residual(feature))
        corrected = prior
        if project:
            corrected = project_observations(corrected,decoder,packet,extent,self.step_logit.sigmoid())
        return (corrected,prior) if return_prior else corrected
