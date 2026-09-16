import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from v2x_dgw_recoverability_audit.m8_ucq_route_student import UCQRouteStudent


class PointPillarUCQRoute(nn.Module):
    def __init__(self, args):
        super(PointPillarUCQRoute, self).__init__()
        self.max_cav = args['max_cav']
        
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args.get('compression', 0) > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        # UCQ Route Fusion Net
        ucq_args = args.get('ucq_route_fusion', {})
        self.fusion_net = UCQRouteStudent(
            feature_channels=ucq_args.get('feature_channels', [64, 128, 256]),
            actions=ucq_args.get('actions', ['send_full', 'send_compressed', 'send_none'])
        )

        # Output heads
        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        if args.get('backbone_fix', False):
            self.backbone_fix()

    def backbone_fix(self):
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

    def forward(self, data_dict):
        # 1. Encode Features (VFE -> Scatter -> Backbone)
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # Organize Multiscale features [N, C, H, W] -> [B, max_cav, C, H, W]
        # N = sum(record_len)
        B = len(record_len)
        
        # We need to pad to max_cav
        def pad_to_max_cav(tensor):
            C, H, W = tensor.shape[1:]
            out = torch.zeros((B, self.max_cav, C, H, W), device=tensor.device, dtype=tensor.dtype)
            idx = 0
            for i, l in enumerate(record_len):
                l_int = int(l.item())
                out[i, :l_int] = tensor[idx:idx+l_int]
                idx += l_int
            return out

        features_2x = pad_to_max_cav(batch_dict['spatial_features_2x'])
        features_4x = pad_to_max_cav(batch_dict['spatial_features_4x'])
        features_8x = pad_to_max_cav(batch_dict['spatial_features_8x'])
        
        multiscale_features = [features_2x, features_4x, features_8x]

        # Generate source_valid mask [B, max_cav]
        source_valid = torch.zeros((B, self.max_cav), dtype=torch.bool, device=features_2x.device)
        for i, l in enumerate(record_len):
            source_valid[i, :int(l.item())] = True

        # Generate density_heatmap [B, max_cav, H, W] (from point counts)
        # Using spatial_features from scatter as proxy for density (non-zero voxels)
        scatter_feat = pad_to_max_cav(batch_dict['spatial_features'])
        density_heatmap = (scatter_feat.sum(dim=2) > 0).float() # simplistic density map

        # Dummy baseline confidence and proposal heatmap (in actual use, provided by RPN)
        target_H, target_W = features_8x.shape[-2:]
        baseline_confidence = torch.ones((B, target_H, target_W), device=features_2x.device)
        proposal_heatmap = torch.ones((B, target_H, target_W), device=features_2x.device)

        # 2. UCQ-Route Fusion
        ucq_out = self.fusion_net(
            features=multiscale_features,
            source_valid=source_valid,
            attention_maps=None,
            proposal_heatmap=proposal_heatmap,
            density_heatmap=density_heatmap,
            baseline_confidence=baseline_confidence
        )

        # The output of UCQRouteStudent includes ego_clean_proxy which is ego's merged features
        ego_fused = ucq_out['ego_clean_proxy']  # [B, C, H, W]
        
        if self.shrink_flag:
            ego_fused = self.shrink_conv(ego_fused)
        if self.compression:
            ego_fused = self.naive_compressor(ego_fused)

        psm = self.cls_head(ego_fused)
        rm = self.reg_head(ego_fused)

        output_dict = {
            'psm': psm,
            'rm': rm,
            'ucq_out': ucq_out # Expose for the loss function!
        }
        return output_dict
