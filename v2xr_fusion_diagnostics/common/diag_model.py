"""Observe the existing model and replay its deblocks and heads."""
from collections import namedtuple
import torch

from runtime import create_model
from opencood.hypes_yaml import yaml_utils
from diag_bootstrap import pot

Capture = namedtuple('Capture', 'prediction spatial pre post record_len')


class FrozenP0:
    def __init__(self):
        self.cfg = yaml_utils.load_yaml(str(pot.CONFIG))
        self.model = create_model(self.cfg)
        self.model.load_state_dict(torch.load(pot.WEIGHT, map_location='cpu', weights_only=True), strict=True)
        self.model.eval().requires_grad_(False)
        self.backbone = self.model.backbone
        self.values = {}
        self.backbone.register_forward_pre_hook(self._backbone_input)
        for i, fusion in enumerate(self.backbone.fuse_modules):
            fusion.register_forward_hook(self._observe(i))

    def _backbone_input(self, _, args):
        self.values['spatial'] = args[0]['spatial_features']

    def _observe(self, scale):
        def hook(_, args, result):
            self.values[scale] = (args[0], result)
        return hook

    def capture(self, ego):
        assert not self.model.training and not torch.is_grad_enabled()
        self.values.clear()
        pred = self.model(ego)
        pre, post = zip(*(self.values[i] for i in range(3)))
        if [tuple(t.shape[1:]) for t in pre] != [(128,100,352),(256,50,176),(512,25,88)]:
            raise ValueError('Unexpected official fusion tensor shapes')
        return Capture(pred,self.values['spatial'],pre,post,ego['record_len'])

    def predict(self, fused):
        assert not torch.is_grad_enabled()
        features = torch.cat([layer(value) for layer,value in zip(self.backbone.deblocks,fused,strict=True)],dim=1)
        return dict(psm=self.model.cls_head(features),rm=self.model.reg_head(features))

    def ego_fusion(self, state):
        return tuple(layer(pre[:1],torch.ones_like(state.record_len))
                     for layer,pre in zip(self.backbone.fuse_modules,state.pre,strict=True))


def ego_only(ego):
    changes = {f'processed_{sensor}':{k:v[ego[f'processed_{sensor}']['voxel_coords'][:,0]==0]
                                     for k,v in ego[f'processed_{sensor}'].items()}
               for sensor in ('lidar','radar')}
    return dict(ego,record_len=torch.ones_like(ego['record_len']),**changes)


def radar_off(ego):
    return dict(ego,processed_radar={k:v[:0] for k,v in ego['processed_radar'].items()})


def tokens(features):
    return features.flatten(2).permute(2,0,1)


def attention(features, native):
    qkv = tokens(features)
    return (torch.bmm(qkv,qkv.transpose(1,2))/native.att.sqrt_dim).softmax(-1)


def fuse(weights, values):
    context = torch.bmm(weights,tokens(values))
    return context[:,0,:].T.reshape(1,*values.shape[1:])


def error(a,b):
    return max(float((a[k]-b[k]).abs().max()) for k in ('psm','rm'))
