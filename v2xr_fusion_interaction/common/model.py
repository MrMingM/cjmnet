"""Keep every LR0 module and attach two trainable interaction branches."""
import torch
from opencood.models.point_pillar_intermediate_lrf import PointPillarIntermediateLRF
from empty_input import preserve_empty_vehicle_slots
from .fusion import Interaction
from .support import support_maps
from .audit import lr

SHAPES=((128,100,352),(256,50,176),(512,25,88))


class Model(PointPillarIntermediateLRF):
    def __init__(self,args,cfg):
        super().__init__(args)
        backbone=args['base_bev_backbone']
        if (backbone['layer_strides']!=[2,2,2] or backbone['num_filters']!=[128,256,512]
                or backbone['compression'] or (self.scatter.ny,self.scatter.nx)!=(200,704)
                or tuple(args['voxel_size'][:2])!=(.4,.4)):
            raise ValueError('The verified P0 feature layout differs')
        self.load_state_dict(torch.load(lr.WEIGHT,map_location='cpu',weights_only=True),strict=True)
        self.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            for i in range(2):
                self.backbone.fuse_modules[i]=Interaction(self.backbone.fuse_modules[i],SHAPES[i][0],cfg)
        preserve_empty_vehicle_slots(self)
        for i,layer in enumerate(self.backbone.fuse_modules):
            layer.register_forward_pre_hook(self._check_shape(i))
        self.eval()

    @staticmethod
    def _check_shape(scale):
        def check(_,args):
            if tuple(args[0].shape[1:])!=SHAPES[scale]:
                raise ValueError(f'Runtime fusion scale {scale} differs from P0')
        return check

    @property
    def branches(self):
        return tuple(self.backbone.fuse_modules[:2])

    def forward(self,ego):
        if self.training or any(module.training for module in self.modules()):
            raise RuntimeError('Frozen P0 must stay in eval mode, including during updates')
        try:
            for i,branch in enumerate(self.branches):
                if branch.needs_support:
                    branch.support=support_maps(ego,int(ego['record_len'].sum()),SHAPES[i][1:],2**(i+1))
            return super().forward(ego)
        finally:
            for branch in self.branches:
                branch.support=None


def build_model(cfg,settings):
    return Model(cfg['model']['args'],settings).to('cuda:0')
