"""Reliability-aware PointPillar feature encoder."""

import torch

from opencood.models.sub_modules.pillar_vfe import PillarVFE


class ReliabilityPillarVFE(PillarVFE):
    """Apply soft point weights after feature decoration, before PFN pooling."""

    def forward(self, batch_dict):
        voxel_features = batch_dict["voxel_features"]
        voxel_num_points = batch_dict["voxel_num_points"]
        coords = batch_dict["voxel_coords"]
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x +
            self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y +
            self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z +
            self.z_offset)
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance:
            features.append(torch.norm(
                voxel_features[:, :, :3], 2, 2, keepdim=True))
        features = torch.cat(features, dim=-1)
        valid = self.get_paddings_indicator(
            voxel_num_points, features.shape[1], axis=0)
        mask = valid.unsqueeze(-1).type_as(voxel_features)
        reliability = batch_dict["point_reliability"].unsqueeze(-1)
        if reliability.shape[:2] != features.shape[:2]:
            raise ValueError("point reliability is not aligned with voxel slots")
        features = features * mask * reliability
        for pfn in self.pfn_layers:
            features = pfn(features)
        batch_dict["pillar_features"] = features.squeeze(1)
        return batch_dict
