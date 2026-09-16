"""No GT access: construct local intervals from actual received observations."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from ceif_audit.core import Settings, point_blocks
from ceif_audit.run import build_sources


def stratified_indices(xyz, count, confidence, grid, extent, rng):
    if not len(xyz) or count <= 0:
        return np.empty(0,dtype=int)
    # Half the quota favors distinct high-confidence blocks, half explores.
    ids,_ = point_blocks(xyz,grid,extent)
    random_order = rng.permutation(len(xyz))
    _, first = np.unique(ids[random_order],return_index=True)
    representatives = random_order[first]
    priorities = confidence.ravel()[ids[representatives]]
    focused = representatives[np.argsort(-priorities,kind='stable')[:count//2]]
    rest = np.setdiff1d(np.arange(len(xyz)),focused,assume_unique=True)
    random = rng.choice(rest,min(count-len(focused),len(rest)),replace=False)
    return np.concatenate((focused,random)).astype(int)


def make_packet(model, inp, encoded, masks, seed, max_queries=1024):
    settings = Settings()
    sources = build_sources(model.engine.base,inp,masks,model.grid,model.lidar_range,settings)
    rng = np.random.default_rng(seed)
    queries, labels, label_weights = [], [], []
    confidence = encoded['obs'][:,12].detach().cpu().numpy()
    quota = max(1,max_queries//max(2*len(sources),1))
    for peer, source in enumerate(sources):
        ids = np.flatnonzero(source.belief >= settings.support_belief)
        selected = stratified_indices(source.xyz[ids],quota,confidence[peer],model.grid,model.lidar_range,rng)
        selected = ids[selected]
        queries.extend(source.xyz[selected]); labels.extend([1.]*len(selected))
        label_weights.extend(source.belief[selected])
        if len(source.ranges):
            endpoints = source.origin+source.directions*source.ranges[:,None]
            ids = stratified_indices(endpoints,quota,confidence[peer],model.grid,model.lidar_range,rng)
            distance = source.ranges[ids]*rng.uniform(.2,.8,len(ids))
            valid = (source.ranges[ids]-distance > settings.depth_margin)&(distance > 1.)
            q = source.origin+source.directions[ids[valid]]*distance[valid,None]
            block, inside = point_blocks(q,model.grid,model.lidar_range)
            inside &= source.received.ravel()[block]
            inside &= (q[:,2] >= model.lidar_range[2])&(q[:,2] < model.lidar_range[5])
            q = q[inside]
            queries.extend(q); labels.extend([0.]*len(q))
            label_weights.extend(source.ray_beliefs[ids[valid]][inside])
    xyz = np.asarray(queries,dtype=np.float32).reshape(-1,3)
    lower, upper = np.zeros(len(xyz),np.float32),np.ones(len(xyz),np.float32)
    for source in sources:
        if len(source.xyz) and len(xyz):
            # Endpoint neighborhood is deliberately much narrower than ray tolerance.
            neighbors = cKDTree(source.xyz).query_ball_point(xyz,.05)
            hit = np.array([source.belief[ids].max() if ids else 0. for ids in neighbors],np.float32)
            lower = np.maximum(lower,hit)
        upper = np.minimum(upper,1-source.free_at(xyz))
    target = masks.device
    packet = {k:torch.as_tensor(v,device=target,dtype=torch.float32) for k,v in
        dict(xyz=xyz,lower=lower,upper=upper,labels=labels,label_weights=label_weights).items()}
    packet['extra_bytes'] = sum(s.certificate_bytes for s in sources[1:])
    packet['rays'] = sum(len(s.ranges) for s in sources)
    return packet


def decoder_loss(logits,packet):
    """Clean sensor-derived supervision, not dense GT occupancy labels."""
    keep = (packet['label_weights'] >= .6)&(packet['lower'] <= packet['upper'])
    terms = []
    for label in (0,1):
        indices = keep & (packet['labels'] == label)
        if indices.any():
            terms.append(torch.nn.functional.binary_cross_entropy_with_logits(logits[indices],packet['labels'][indices]))
    return torch.stack(terms).mean() if terms else logits.sum()*0
