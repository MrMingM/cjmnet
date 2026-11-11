"""Insert POT-CF at AttBEVBackbone.forward's third fuse_modules call."""
import math
import torch

from opencood.models.point_pillar_intermediate_lrf import PointPillarIntermediateLRF
from empty_input import preserve_empty_vehicle_slots
from fusion import CollaborativeFusion, support_maps


class Model(PointPillarIntermediateLRF):
    def __init__(self, args, cfg, solver, mode):
        super().__init__(args)
        backbone = args['base_bev_backbone']
        if (backbone['layer_strides'] != [2, 2, 2]
                or backbone['num_filters'] != [128, 256, 512] or backbone['compression']):
            raise ValueError('Unexpected LR0 backbone structure')
        self.stride = math.prod(backbone['layer_strides'])
        self.coarse_shape = ((self.scatter.ny + self.stride - 1) // self.stride,
                             (self.scatter.nx + self.stride - 1) // self.stride)
        self.radar_off = self.single_vehicle = False
        # All variants start with identical LR0 weights and identical data RNG.
        with torch.random.fork_rng(devices=[]):
            original = self.backbone.fuse_modules[2]
            self.backbone.fuse_modules[2] = CollaborativeFusion(original, 512, cfg, solver, mode)
        preserve_empty_vehicle_slots(self)

    @property
    def fusion(self):
        return self.backbone.fuse_modules[2]

    def forward(self, ego):
        if self.single_vehicle or self.radar_off:
            ego = self.restrict(ego)
        if self.fusion.mode != 'module_off':
            self.fusion.sources = support_maps(ego, int(ego['record_len'].sum()), self.coarse_shape, self.stride)
        try:
            return super().forward(ego)
        finally:
            self.fusion.sources = None

    def restrict(self, ego):
        """Interventions alter model input only. Native labels and postprocess still see full data."""
        result = dict(ego)
        if self.single_vehicle:
            record_len = ego['record_len']
            starts = record_len.cumsum(0) - record_len
            mapping = record_len.new_full((int(record_len.sum()),), -1)
            mapping[starts.long()] = torch.arange(len(starts), device=starts.device, dtype=mapping.dtype)
            result['record_len'] = torch.ones_like(record_len)
        for sensor in ('lidar', 'radar'):
            name = f'processed_{sensor}'
            data = ego[name]
            if self.radar_off and sensor == 'radar':
                result[name] = {key: value[:0] for key, value in data.items()}
            elif self.single_vehicle:
                ids = mapping[data['voxel_coords'][:, 0].long()]
                keep = ids >= 0
                selected = {key: value[keep] for key, value in data.items()}
                selected['voxel_coords'][:, 0] = ids[keep].to(selected['voxel_coords'].dtype)
                result[name] = selected
        return result
