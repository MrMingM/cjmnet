"""
Implementation of Where2comm fusion.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
from opencood.utils.weather_reliability import (
    build_weather_reliability,
    save_weather_debug_npz,
    summarize_tensor
)


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def _build_mask(self, communication_maps, H, W, K=None):
        L = communication_maps.shape[0]
        if self.training:
            # Official training proxy objective. Reuse K when comparing original
            # and weather-adjusted rates in the same forward pass.
            if K is None:
                K = int(H * W * random.uniform(0, 1))
            flat_maps = communication_maps.reshape(L, H * W)
            _, indices = torch.topk(flat_maps, k=K, sorted=False)
            communication_mask = torch.zeros_like(flat_maps).to(flat_maps.device)
            ones_fill = torch.ones(L, K, dtype=flat_maps.dtype, device=flat_maps.device)
            communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
        elif self.threshold:
            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
            communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
        else:
            communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)
        return communication_mask, K

    def forward(self, batch_confidence_maps, B, batch_reliability_maps=None):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
            batch_reliability_maps: optional [(L1, 1, H, W), ...]
        """

        _, _, H, W = batch_confidence_maps[0].shape

        communication_masks = []
        communication_rates = []
        communication_rates_before_weather = []
        debug_confidence_maps = []
        debug_effective_maps = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            L = ori_communication_maps.shape[0]

            effective_communication_maps = ori_communication_maps
            if batch_reliability_maps is not None:
                reliability_map = batch_reliability_maps[b].to(
                    device=ori_communication_maps.device,
                    dtype=ori_communication_maps.dtype)
                if reliability_map.shape[-2:] != ori_communication_maps.shape[-2:]:
                    reliability_map = F.interpolate(
                        reliability_map,
                        size=ori_communication_maps.shape[-2:],
                        mode='bilinear',
                        align_corners=False)
                effective_communication_maps = ori_communication_maps * reliability_map

            original_maps_for_mask = self.gaussian_filter(ori_communication_maps) \
                if self.smooth else ori_communication_maps
            communication_maps = self.gaussian_filter(effective_communication_maps) \
                if self.smooth else effective_communication_maps

            original_mask, K = self._build_mask(original_maps_for_mask, H, W)
            communication_mask, _ = self._build_mask(communication_maps, H, W, K=K)

            communication_rate = communication_mask.sum() / (L * H * W)
            communication_rate_before_weather = original_mask.sum() / (L * H * W)
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
            communication_rates_before_weather.append(communication_rate_before_weather)
            debug_confidence_maps.append(ori_communication_maps)
            debug_effective_maps.append(effective_communication_maps)
        communication_rates = sum(communication_rates) / B
        communication_rates_before_weather = sum(communication_rates_before_weather) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        debug_details = {
            'confidence_map': torch.cat(debug_confidence_maps, dim=0),
            'effective_confidence': torch.cat(debug_effective_maps, dim=0),
            'communication_mask': communication_masks,
            'comm_rate_before_weather': communication_rates_before_weather,
            'comm_rate_after_weather': communication_rates
        }
        return communication_masks, communication_rates, debug_details


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

        self.fully = args['fully']
        if self.fully:
            print('constructing a fully connected communication graph')
        else:
            print('constructing a partially connected communication graph')

        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttentionFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttentionFusion(args['in_channels'])

        self.naive_communication = Communication(args['communication'])
        self.weather_reliability = args.get('weather_reliability', {})
        self.weather_enable = bool(self.weather_reliability.get('enable', False))
        self.weather_apply_to = self.weather_reliability.get('apply_to', 'comm')
        self.weather_comm_enable = self.weather_enable and \
            self.weather_apply_to in ['comm', 'both']
        self.weather_feature_enable = self.weather_enable and \
            self.weather_apply_to in ['feature', 'both']
        self.weather_loss_enable = self.weather_enable and \
            self.weather_apply_to in ['loss']
        self.lidar_range = args.get('lidar_range', None)
        self.voxel_size = args.get('voxel_size', [self.discrete_ratio, self.discrete_ratio, 1])
        self.grid_size = args.get('grid_size', None)
        self.debug_counter = 0

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def _build_weather_maps(self, psm_single, record_len, weather_inputs):
        if not self.weather_enable or \
                (not self.weather_comm_enable and
                 not self.weather_feature_enable and
                 not self.weather_loss_enable):
            return None, None
        if weather_inputs is None:
            warnings.warn('weather_reliability is enabled, but no weather_inputs were provided.')
            return None, None

        voxel_coords = weather_inputs.get('voxel_coords', None)
        voxel_num_points = weather_inputs.get('voxel_num_points', None)
        if voxel_coords is None or voxel_num_points is None:
            warnings.warn('weather_reliability is enabled, but voxel statistics are missing.')
            return None, None

        try:
            weather_maps = build_weather_reliability(
                voxel_coords=voxel_coords,
                voxel_num_points=voxel_num_points,
                record_len=record_len,
                confidence_map=psm_single,
                config=self.weather_reliability,
                lidar_range=self.lidar_range,
                voxel_size=self.voxel_size,
                grid_size=self.grid_size)
            reliability_floor = float(self.weather_reliability.get(
                'reliability_floor', 0.0))
            reliability_floor = min(max(reliability_floor, 0.0), 1.0)
            if reliability_floor > 0.0:
                # Soft gate: keep a configurable confidence floor so empty or
                # sparse BEV cells are down-weighted instead of hard-suppressed.
                weather_maps['R_final_resized'] = reliability_floor + \
                    (1.0 - reliability_floor) * weather_maps['R_final_resized']
            batch_reliability_maps = self.regroup(
                weather_maps['R_final_resized'], record_len)
            return weather_maps, batch_reliability_maps
        except Exception as exc:
            warnings.warn('Failed to build weather reliability map: %s' % exc)
            return None, None

    def _apply_feature_reliability(self, x, record_len, batch_reliability_maps):
        if not self.weather_feature_enable or batch_reliability_maps is None:
            return x

        weighted_features = []
        batch_node_features = self.regroup(x, record_len)
        for feature, reliability in zip(batch_node_features, batch_reliability_maps):
            reliability = reliability.to(device=feature.device,
                                         dtype=feature.dtype)
            if reliability.shape[-2:] != feature.shape[-2:]:
                reliability = F.interpolate(
                    reliability,
                    size=feature.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            # Keep ego unchanged; only down-weight cooperative agents.
            if reliability.shape[0] > 0:
                reliability = reliability.clone()
                reliability[0] = 1.0
            weighted_features.append(feature * reliability)
        return torch.cat(weighted_features, dim=0)

    def _apply_spatial_agent_weights(self, x, spatial_agent_weights):
        if spatial_agent_weights is None:
            return x
        weights = spatial_agent_weights.to(device=x.device, dtype=x.dtype)
        if weights.dim() == 3:
            weights = weights.unsqueeze(1)
        if weights.shape[0] != x.shape[0] or weights.shape[1] != 1:
            raise ValueError(
                'spatial_agent_weights must have shape [N,1,H,W]')
        if weights.shape[-2:] != x.shape[-2:]:
            weights = F.interpolate(
                weights, size=x.shape[-2:], mode='bilinear',
                align_corners=False)
        return x * weights.clamp(0.0, 1.0)

    def _finalize_weather_debug(self, weather_maps, comm_details):
        if not self.weather_enable or weather_maps is None or comm_details is None:
            return None

        stats = {
            'R': summarize_tensor(weather_maps['R_final_resized']),
            'R_density': summarize_tensor(weather_maps['R_density']),
            'R_isolation': summarize_tensor(weather_maps['R_isolation']),
            'R_isolation_low_frac': (
                weather_maps['R_isolation'] < 0.999
            ).to(weather_maps['R_isolation'].dtype).mean(),
            'R_map': weather_maps['R_final_resized'],
            'confidence_map': summarize_tensor(comm_details['confidence_map']),
            'effective_confidence': summarize_tensor(comm_details['effective_confidence']),
            'comm_rate_before_weather': comm_details['comm_rate_before_weather'],
            'comm_rate_after_weather': comm_details['comm_rate_after_weather']
        }

        if bool(self.weather_reliability.get('debug_print', False)):
            print('Weather reliability: R min %.4f max %.4f mean %.4f | '
                  'conf min %.4f max %.4f mean %.4f | '
                  'eff min %.4f max %.4f mean %.4f | '
                  'comm before %.4f after %.4f' %
                  (stats['R']['min'].detach().cpu().item(),
                   stats['R']['max'].detach().cpu().item(),
                   stats['R']['mean'].detach().cpu().item(),
                   stats['confidence_map']['min'].detach().cpu().item(),
                   stats['confidence_map']['max'].detach().cpu().item(),
                   stats['confidence_map']['mean'].detach().cpu().item(),
                   stats['effective_confidence']['min'].detach().cpu().item(),
                   stats['effective_confidence']['max'].detach().cpu().item(),
                   stats['effective_confidence']['mean'].detach().cpu().item(),
                   stats['comm_rate_before_weather'].detach().cpu().item(),
                   stats['comm_rate_after_weather'].detach().cpu().item()))

        if bool(self.weather_reliability.get('debug_save', False)):
            interval = int(self.weather_reliability.get('debug_interval', 1))
            if interval <= 0:
                interval = 1
            if self.debug_counter % interval == 0:
                debug_dir = self.weather_reliability.get(
                    'debug_dir', 'weather_reliability_debug')
                save_weather_debug_npz(debug_dir, self.debug_counter,
                                       weather_maps, comm_details)
        self.debug_counter += 1
        return stats

    def _regroup_agent_mask(self, agent_mask, record_len, device):
        """Split a flat counterfactual mask into per-sample agent masks."""
        if agent_mask is None:
            return None
        if not torch.is_tensor(agent_mask):
            agent_mask = torch.as_tensor(agent_mask, device=device)
        agent_mask = agent_mask.to(device=device, dtype=torch.bool).reshape(-1)
        expected = int(record_len.sum().item())
        if agent_mask.numel() != expected:
            raise ValueError(
                'agent_mask has %d entries, expected %d from record_len' %
                (agent_mask.numel(), expected))

        batch_masks = self.regroup(agent_mask, record_len)
        checked_masks = []
        for mask in batch_masks:
            mask = mask.clone()
            if mask.numel() == 0:
                raise ValueError('counterfactual mask cannot describe an empty sample')
            if not bool(mask[0].item()):
                raise ValueError('counterfactual mask must always retain the ego agent')
            if not bool(mask.any().item()):
                raise ValueError('counterfactual mask must retain at least one agent')
            checked_masks.append(mask)
        return checked_masks

    def forward(self, x, psm_single, record_len, pairwise_t_matrix, backbone=None,
                weather_inputs=None, agent_mask=None,
                spatial_agent_weights=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """

        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0]
        batch_agent_masks = self._regroup_agent_mask(
            agent_mask, record_len, x.device)
        weather_maps, batch_reliability_maps = self._build_weather_maps(
            psm_single, record_len, weather_inputs)
        weather_info = None

        if self.multi_scale:
            ups = []

            for i in range(self.num_levels):
                x = backbone.blocks[i](x)

                # 1. Communication (mask the features)
                if i == 0:
                    if self.fully:
                        communication_rates = torch.tensor(1).to(x.device)
                    else:
                        # Prune
                        batch_confidence_maps = self.regroup(psm_single, record_len)
                        communication_masks, communication_rates, comm_details = \
                            self.naive_communication(
                                batch_confidence_maps, B,
                                batch_reliability_maps if self.weather_comm_enable else None)
                        weather_info = self._finalize_weather_debug(
                            weather_maps, comm_details)
                        if x.shape[-1] != communication_masks.shape[-1]:
                            communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                                mode='bilinear', align_corners=False)
                        x = x * communication_masks
                    x = self._apply_feature_reliability(
                        x, record_len, batch_reliability_maps)
                    x = self._apply_spatial_agent_weights(
                        x, spatial_agent_weights)

                # 2. Split the features
                # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
                # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
                batch_node_features = self.regroup(x, record_len)

                # 3. Fusion
                x_fuse = []
                for b in range(B):
                    neighbor_feature = batch_node_features[b]
                    if batch_agent_masks is not None:
                        neighbor_feature = neighbor_feature[batch_agent_masks[b]]
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)

                # 4. Deconv
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            # 1. Communication (mask the features)
            if self.fully:
                communication_rates = torch.tensor(1).to(x.device)
            else:
                # Prune
                batch_confidence_maps = self.regroup(psm_single, record_len)
                communication_masks, communication_rates, comm_details = \
                    self.naive_communication(
                        batch_confidence_maps, B,
                        batch_reliability_maps if self.weather_comm_enable else None)
                weather_info = self._finalize_weather_debug(
                    weather_maps, comm_details)
                x = x * communication_masks
            x = self._apply_feature_reliability(
                x, record_len, batch_reliability_maps)
            x = self._apply_spatial_agent_weights(
                x, spatial_agent_weights)

            # 2. Split the features
            # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
            # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
            batch_node_features = self.regroup(x, record_len)

            # 3. Fusion
            x_fuse = []
            for b in range(B):
                neighbor_feature = batch_node_features[b]
                if batch_agent_masks is not None:
                    neighbor_feature = neighbor_feature[batch_agent_masks[b]]
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
        return x_fuse, communication_rates, weather_info
