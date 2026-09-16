import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.weather_feature_adapter import \
    WeatherFeatureAdapter
from opencood.models.sub_modules.counterfactual_utility_restorer import \
    CounterfactualUtilityRestorer
from opencood.models.sub_modules.counterfactual_fusion_restorer import \
    CounterfactualFusionRestorer
from opencood.models.sub_modules.snow_voxel_denoiser import \
    SnowVoxelDenoiser


class PointPillarWhere2comm(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2comm, self).__init__()
        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        snow_denoiser_args = args.get('snow_voxel_denoiser', {})
        self.snow_voxel_denoiser_enabled = bool(
            snow_denoiser_args.get('enable', False))
        if self.snow_voxel_denoiser_enabled:
            self.snow_voxel_denoiser = SnowVoxelDenoiser(
                channels=self.pillar_vfe.get_output_feature_dim(),
                hidden_channels=int(snow_denoiser_args.get(
                    'hidden_channels', 64)),
                max_points=int(snow_denoiser_args.get('max_points', 32)),
                grid_size=args['point_pillar_scatter'].get(
                    'grid_size', [1, 1, 1]),
                gate_floor=float(snow_denoiser_args.get(
                    'gate_floor', 0.05)),
                activation_threshold=float(snow_denoiser_args.get(
                    'activation_threshold', 0.5)))

        adapter_args = args.get('weather_feature_adapter', {})
        self.weather_adapter_enabled = bool(
            adapter_args.get('enable', False))
        cure_args = args.get('counterfactual_utility_restorer', {})
        self.cure_pre_enabled = bool(cure_args.get('enable', False))
        self.cure_apply_fusion_gate = bool(
            cure_args.get('apply_fusion_gate', False))
        fusion_cure_args = args.get(
            'counterfactual_fusion_restorer', {})
        self.cure_post_enabled = bool(
            fusion_cure_args.get('enable', False))
        self.cure_enabled = (
            self.cure_pre_enabled or self.cure_post_enabled)
        if self.cure_pre_enabled and self.cure_post_enabled:
            raise ValueError(
                'pre-fusion and post-fusion CURE cannot both be enabled')
        if self.cure_enabled and self.weather_adapter_enabled:
            raise ValueError(
                'weather_feature_adapter and counterfactual utility '
                'restorer cannot be enabled together')
        if self.snow_voxel_denoiser_enabled and (
                self.cure_enabled or self.weather_adapter_enabled):
            raise ValueError(
                'snow_voxel_denoiser cannot be combined with a weather '
                'feature adapter or CURE restorer')
        if self.cure_pre_enabled:
            self.counterfactual_utility_restorer = \
                CounterfactualUtilityRestorer(
                    channels=64,
                    hidden_channels=int(cure_args.get(
                        'hidden_channels', 64)),
                    groups=int(cure_args.get('groups', 8)),
                    fusion_gate_floor=float(cure_args.get(
                        'fusion_gate_floor', 0.5)))
        if self.cure_post_enabled:
            self.counterfactual_fusion_restorer = \
                CounterfactualFusionRestorer(
                    channels=int(args['head_dim']),
                    hidden_channels=int(fusion_cure_args.get(
                        'hidden_channels', 64)),
                    groups=int(fusion_cure_args.get('groups', 8)))
        if self.weather_adapter_enabled:
            self.weather_feature_adapter = WeatherFeatureAdapter(
                channels=64,
                hidden_channels=int(adapter_args.get(
                    'hidden_channels', 64)),
                groups=int(adapter_args.get('groups', 8)))

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        where2comm_args = args['where2comm_fusion']
        where2comm_args.setdefault('lidar_range', args['lidar_range'])
        where2comm_args.setdefault('grid_size',
                                   args['point_pillar_scatter'].get('grid_size', None))
        self.fusion_net = Where2comm(where2comm_args)
        self.multi_scale = args['where2comm_fusion']['multi_scale']

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """

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

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def encode_features(self, data_dict):
        """Encode every CAV once so Oracle fusion passes can share the cache."""
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        snow_voxel_aux = None
        if self.snow_voxel_denoiser_enabled:
            batch_dict['pillar_features'], snow_voxel_aux = \
                self.snow_voxel_denoiser(
                    batch_dict['pillar_features'],
                    voxel_num_points,
                    voxel_coords)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        spatial_features_before_adapter = batch_dict['spatial_features']
        cure_aux = None
        if self.cure_pre_enabled:
            batch_dict['spatial_features'], cure_aux = \
                self.counterfactual_utility_restorer(
                    spatial_features_before_adapter, record_len)
        elif self.weather_adapter_enabled:
            batch_dict['spatial_features'] = self.weather_feature_adapter(
                spatial_features_before_adapter)
        batch_dict = self.backbone(batch_dict)

        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)

        spatial_features_2d_before_compression = spatial_features_2d
        # Compressor
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        return {
            'spatial_features': batch_dict['spatial_features'],
            'spatial_features_before_adapter':
                spatial_features_before_adapter,
            'spatial_features_2d': spatial_features_2d,
            'spatial_features_2d_before_compression':
                spatial_features_2d_before_compression,
            'psm_single': psm_single,
            'record_len': record_len,
            'pairwise_t_matrix': pairwise_t_matrix,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'cure_aux': cure_aux,
            'snow_voxel_aux': snow_voxel_aux
        }

    def forward_from_encoded(self, encoded, agent_mask=None, return_aux=False,
                             spatial_agent_weights=None):
        """Fuse a selected coalition without repeating PointPillar encoding."""
        psm_single = encoded['psm_single']
        spatial_features = encoded['spatial_features']
        spatial_features_2d = encoded['spatial_features_2d']
        record_len = encoded['record_len']
        pairwise_t_matrix = encoded['pairwise_t_matrix']
        voxel_coords = encoded['voxel_coords']
        voxel_num_points = encoded['voxel_num_points']
        cure_aux = encoded.get('cure_aux', None)
        snow_voxel_aux = encoded.get('snow_voxel_aux', None)
        if spatial_agent_weights is None and \
                self.cure_apply_fusion_gate and cure_aux is not None:
            spatial_agent_weights = cure_aux['fusion_weight']

        if self.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            fused_feature, communication_rates, weather_info = self.fusion_net(
                spatial_features,
                psm_single,
                record_len,
                pairwise_t_matrix,
                self.backbone,
                weather_inputs={
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points
                },
                agent_mask=agent_mask,
                spatial_agent_weights=spatial_agent_weights)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, weather_info = self.fusion_net(
                spatial_features_2d,
                psm_single,
                record_len,
                pairwise_t_matrix,
                weather_inputs={
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points
                },
                agent_mask=agent_mask,
                spatial_agent_weights=spatial_agent_weights)

        if self.cure_post_enabled:
            per_agent_feature = encoded[
                'spatial_features_2d_before_compression']
            ego_indices = []
            offset = 0
            for length_value in record_len.detach().cpu().tolist():
                length = int(length_value)
                if length <= 0:
                    raise ValueError(
                        'record_len entries must be positive')
                ego_indices.append(offset)
                offset += length
            if offset != per_agent_feature.shape[0]:
                raise ValueError(
                    'record_len and per-agent feature rows differ')
            ego_feature = per_agent_feature[ego_indices]
            fused_feature, cure_aux = \
                self.counterfactual_fusion_restorer(
                    fused_feature, ego_feature)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm, 'rm': rm, 'com': communication_rates}
        if return_aux:
            output_dict.update({
                'fused_feature': fused_feature,
                'single_confidence': psm_single
            })
            if cure_aux is not None:
                output_dict['cure_aux'] = cure_aux
            if snow_voxel_aux is not None:
                output_dict['snow_voxel_aux'] = snow_voxel_aux
        if weather_info is not None:
            output_dict.update({
                'com_before_weather': weather_info['comm_rate_before_weather'],
                'com_weather': weather_info['comm_rate_after_weather'],
                'weather_R_min': weather_info['R']['min'],
                'weather_R_max': weather_info['R']['max'],
                'weather_R_mean': weather_info['R']['mean'],
                'weather_R_density_mean': weather_info['R_density']['mean'],
                'weather_R_isolation_mean': weather_info['R_isolation']['mean'],
                'weather_R_isolation_low_frac': weather_info['R_isolation_low_frac']
            })
            if return_aux and 'R_map' in weather_info:
                output_dict['weather_R_map'] = weather_info['R_map']
        return output_dict

    def forward(self, data_dict):
        return_aux = bool(data_dict.get('return_aux', False))
        agent_mask = data_dict.get('agent_mask', None)
        spatial_agent_weights = data_dict.get('spatial_agent_weights', None)
        encoded = self.encode_features(data_dict)
        return self.forward_from_encoded(
            encoded, agent_mask=agent_mask, return_aux=return_aux,
            spatial_agent_weights=spatial_agent_weights)
