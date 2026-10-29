"""Column-local pair compatibility; no registration or correspondence supervision.

Only the geometric compatibility is detached. A1's learned attention, point
features and GSPR priors retain their original gradients.
"""
import math
import torch
from torch.nn import functional as F


@torch.no_grad()
def pair_geometry(x, f, valid, ranges, cfg):
    """Return a sparse pair graph, conditional on fixed PFN features.

    PFN features themselves already depend on GSPR. Translation/rotation
    invariant conditional on unchanged point features and
sensor distances. The inherited A1/PFN itself is not rotation invariant.
    """
    m, t, _ = x.shape
    allowed = valid[:, :, None] & valid[:, None, :]
    allowed &= ~torch.eye(t, device=x.device, dtype=torch.bool)[None]
    d2 = (x[:, :, None] - x[:, None]).square().sum(-1)
    k = min(cfg['neighbors'], t)
    idx = d2.masked_fill(~allowed, float('inf')).topk(k, largest=False).indices
    candidate = torch.zeros_like(allowed).scatter(-1, idx, allowed.gather(-1, idx))
    # A fixed, sensor-distance-aware metric scale prevents cancellation under
    # uniform dilation. No single pillar center/plane or largest component.
    radius = (cfg['geometry_radius_min'] + cfg['geometry_range_scale'] * ranges).clamp(
        cfg['geometry_radius_min'], cfg['geometry_radius_max'])
    pair_radius = (radius[:, :, None] + radius[:, None]) / 2
    radial = (1-d2 / pair_radius.square().clamp_min(1e-6)).clamp(0, 1).square()
    graph = radial * candidate
    # Reciprocal union permits edges/unequal sampling; only candidate pairs are
    # eventually used for aggregation. Shared support cannot erase an edge.
    symmetric = torch.maximum(graph, graph.transpose(1, 2))
    common = torch.bmm(symmetric, symmetric.transpose(1, 2))
    degree = (symmetric > 0).sum(-1)
    denominator = (torch.minimum(degree[:, :, None], degree[:, None])-1).clamp_min(1)
    common = (common / denominator).clamp(0, 1)
    normed = F.normalize(f, dim=-1, eps=1e-6)
    feature = torch.bmm(normed, normed.transpose(1, 2)).clamp(0, 1).square()
    geometry = graph * (cfg['geometry_common_floor'] + (1-cfg['geometry_common_floor'])*common)
    geometry *= cfg['geometry_feature_floor'] + (1-cfg['geometry_feature_floor'])*feature
    return dict(indices=idx, mask=allowed.gather(-1, idx),
                compatibility=geometry.gather(-1, idx),
                spatial=radial.gather(-1, idx), common=common.gather(-1, idx))


def geometric_messages(f, prior, learned, pair, valid, masked_softmax):
    m = len(f)
    b = torch.arange(m, device=f.device)[:, None, None]
    idx, mask = pair['indices'], pair['mask']
    neighbors = f[b, idx]
    neighbor_prior = prior[b, idx]
    # These are EXACTLY the A1 weights. Geometry has no second reliability vote.
    attention = masked_softmax(learned + neighbor_prior.clamp_min(1e-6).log(), mask, -1)
    exchange = attention * pair['compatibility']
    exchange_mass = exchange.sum(-1)
    # No renormalization by sum(compatibility): absolute support is retained.
    messages = (1-exchange_mass)[..., None]*f + (exchange[..., None]*neighbors).sum(-2)
    count = mask.sum(-1).clamp_min(1)
    support = pair['compatibility'].sum(-1) / count
    total_support = pair['compatibility'].sum(-1)
    trust = (pair['compatibility']*neighbor_prior.detach()).sum(-1) / total_support.clamp_min(1e-6)
    # Trust is undefined without support. Keep validity separately; zero is not
    # an inferred noise label, and the message then retains the entire self.
    trust_known = total_support > 1e-6
    trust = torch.where(trust_known, trust, torch.zeros_like(trust))
    return messages, dict(attention=attention, exchange=exchange, self_mass=1-exchange_mass,
        support=support, neighbor_trust=trust, trust_known=trust_known,
        exchange_mass=exchange_mass)


def chunk(net, features, base, xyz, valid, r, u, ranges, level):
    from modules import masked_softmax
    f = features.masked_fill(~valid[..., None], 0)
    x = xyz.masked_fill(~valid[..., None], 0)
    r, u = r.masked_fill(~valid, 0), u.masked_fill(~valid, 0)
    ranges = ranges.masked_fill(~valid, 0)
    prior = (r*(1-u)).clamp_min(1e-6)*valid
    n = valid.sum(1).to(f.dtype)
    mass = prior.sum(1).clamp_min(1e-6)
    pair = pair_geometry(x.detach(), f.detach(), valid, ranges.detach(), net.cfg)
    idx = pair['indices']; k = idx.shape[-1]
    b = torch.arange(len(f), device=f.device)[:, None, None]
    q, key = net.query(f), net.key(f)
    rel = x[b, idx]-x[:, :, None]
    distance = (rel.square().sum(-1)+1e-6).sqrt()
    relative = torch.cat((rel/rel.new_tensor([.4,.4,4.]), distance[...,None],
        r[:,:,None,None].expand(-1,-1,k,-1), r[b,idx][...,None],
        u[:,:,None,None].expand(-1,-1,k,-1), u[b,idx][...,None]), -1)
    learned = (q[:,:,None]*key[b,idx]).sum(-1)/math.sqrt(q.shape[-1])
    learned = net.cfg['relation_logit_bound']*(learned+net.geometry(relative).squeeze(-1)).tanh()
    messages, relation = geometric_messages(f, prior, learned, pair, valid, masked_softmax)
    # A bounded positive support bonus lets nonuniform geometry affect pillar
    # pooling even with equal p and a symmetric conservative exchange graph.
    # No support gives the original prior, never a zero/noise verdict.
    query_prior = prior*(1+net.cfg['geometric_adjustment_bound']*relation['support'])
    weights = masked_softmax(query_prior.clamp_min(1e-6).log(), valid, 1)
    pooled = (weights[...,None]*messages).sum(1)
    mean_r, mean_u = r.sum(1)/n.clamp_min(1), u.sum(1)/n.clamp_min(1)
    variance_r = ((r-mean_r[:,None]).square()*valid).sum(1)/n.clamp_min(1)
    low = x.masked_fill(~valid[...,None], float('inf')).amin(1)
    high = x.masked_fill(~valid[...,None], -float('inf')).amax(1)
    low = torch.where((n>0)[:,None], low, torch.zeros_like(low))
    high = torch.where((n>0)[:,None], high, torch.zeros_like(high))
    coverage = ((high-low).square().sum(-1)+1e-6).sqrt()-.001
    mean_range = ranges.sum(1)/n.clamp_min(1)
    effective = mass.square()/prior.square().sum(1).clamp_min(1e-6)
    stats = torch.stack((n/32,mean_r,mean_u,variance_r,coverage.clamp_max(4)/4,
                        mean_range/150,effective/32,mass/n.clamp_min(1)), -1)
    strength = net.cfg['residual_bound']*net.strength(stats).sigmoid().squeeze(-1)
    support = relation['support']
    mean_support = (support*valid).sum(1)/n.clamp_min(1)
    known = relation['trust_known'] & valid
    mean_trust = (relation['neighbor_trust']*known).sum(1)/known.sum(1).clamp_min(1)
    # A3 alone adds this entire protection factor. A2 uses the identical graph.
    # N_eff measures concentration, not absolute trust: uniformly tiny p can
    # still give a large N_eff. Both concentration and absolute prior mass
    # must support a correction. Neither is a point-noise truth label.
    count_evidence = torch.minimum(((effective-2)/4).clamp(0,1), ((mass-2)/4).clamp(0,1))
    sufficient = count_evidence*(1-mean_u).clamp(0,1)
    sufficient *= coverage/(coverage+.05+.002*mean_range)*mean_support
    sufficient = torch.where(n<=2, torch.zeros_like(sufficient), sufficient)
    if level==3:
        strength = strength*sufficient
    strength = strength*(n>0)
    correction = net.project(net.residual_norm(pooled-base)).tanh()
    amplitude = (base.detach().square().mean(-1)+1e-6).sqrt()
    residual = strength[:,None]*amplitude[:,None]*correction
    result = base if net.force_zero else base+residual
    protected = (level==3) & (sufficient<.25) & (n>0)
    diag = torch.stack((n,mean_r,mean_u,mean_support,strength,sufficient,protected.to(f.dtype),
        (residual.detach().square().mean(-1)+1e-12).sqrt(),
        ((prior>=.5)&(support>=.5)&valid).sum(1),
        ((prior>=.5)&(support<.5)&valid).sum(1),
        ((prior<.5)&(support>=.5)&valid).sum(1),
        ((prior<.5)&(support<.5)&valid).sum(1),mean_trust,
        (relation['exchange_mass']*valid).sum(1)/n.clamp_min(1),
        torch.where(n>0,mass,torch.zeros_like(mass)),effective), -1).detach()
    if getattr(net, 'capture_relations', False):
        net.relations = {k:v.detach().clone() for k,v in dict(**pair,**relation,messages=messages).items()}
    return result, diag
