"""Exact original AttFuse/GSPR modules; common initialization and empty-CAV scatter."""
import copy
import numpy as np
import torch
from pathlib import Path
from scorw_diagnostics.core import identity,REPO


def hypes(c):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    h=load_yaml(str(REPO/c['source_config']))
    h['train_params']['epoches']=c['epochs'];h['train_params']['batch_size']=c['batch_size']
    for k in ('weather_augmentation','prediction_distillation','DG_params'):h.pop(k,None)
    return h


def state_hash(state):
    from scorw_r3.core import array_hash
    return identity({k:array_hash(v.detach().cpu().contiguous().numpy()) for k,v in sorted(state.items())})


def create(c,h,variant,base=None):
    from attfuse_code.point_pillar_intermediate import PointPillarIntermediate
    torch.manual_seed(c['seed'])
    if variant=='S2':
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from attfuse_gspr.point_pillar_gspr_attfuse import PointPillarGsprAttfuse
        args=copy.deepcopy(h['model']['args']);args['gspr']=load_yaml(str(REPO/c['gspr_config']))['model']['args']['gspr']
        model=PointPillarGsprAttfuse(args)
    else:model=PointPillarIntermediate(h['model']['args'])
    if base is not None:
        own=model.state_dict()
        if any(k not in own or own[k].shape!=v.shape for k,v in base.items()):raise ValueError('Common backbone schema mismatch')
        own.update(base);model.load_state_dict(own,strict=True)
        if any(not torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in base.items()):raise ValueError('Common initialization mismatch')
    original=model.scatter
    class EmptyAware(torch.nn.Module):
        def forward(self,b):
            features=b['pillar_features'].reshape(-1,original.num_bev_features);coords=b['voxel_coords']
            count=int(b['record_len'].sum().item())
            a=features.new_zeros(count,original.num_bev_features,original.nz*original.nx*original.ny)
            for k in range(count):
                keep=coords[:,0]==k;cc=coords[keep]
                index=(cc[:,1]+cc[:,2]*original.nx+cc[:,3]).long();a[k,:,index]=features[keep].t()
            b['spatial_features']=a.view(count,original.num_bev_features*original.nz,original.ny,original.nx)
            return b
    adapted=EmptyAware();fixture=dict(pillar_features=torch.arange(128,dtype=torch.float32).reshape(2,64),voxel_coords=torch.tensor([[0,0,0,0],[1,0,0,1]]),record_len=torch.tensor([2]))
    if not torch.equal(original(dict(fixture))['spatial_features'],adapted(dict(fixture))['spatial_features']):raise ValueError('Scatter equivalence failed')
    model.scatter=adapted
    if any(s in type(m).__module__.lower() for m in model.modules() for s in ('ggca','repe')):raise ValueError('Forbidden extra module')
    return model


def criterion(c,h,variant):
    if variant=='S2':
        from attfuse_gspr.loss import GsprPointPillarLoss
        return GsprPointPillarLoss(dict(h['loss']['args'],**c['gspr_loss']))
    from opencood.loss.point_pillar_loss import PointPillarLoss
    return PointPillarLoss(h['loss']['args'])
