"""AttFuse with a pre-PillarVFE GSPR reliability front end."""

import torch.nn as nn

from attfuse_code.att_bev_backbone import AttBEVBackbone
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

from .pillar_vfe import ReliabilityPillarVFE
from .reliability import GeometryFirstPillarReliability


class PointPillarGsprAttfuse(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gspr = GeometryFirstPillarReliability(
            args["gspr"], args["voxel_size"], args["lidar_range"])
        self.pillar_vfe = ReliabilityPillarVFE(
            args["pillar_vfe"], num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"])
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = AttBEVBackbone(args["base_bev_backbone"], 64)
        self.cls_head = nn.Conv2d(128 * 3, args["anchor_number"], 1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args["anchor_num"], 1)

    def forward(self, data_dict):
        processed = data_dict["processed_lidar"]
        batch_dict = {
            "voxel_features": processed["voxel_features"],
            "voxel_coords": processed["voxel_coords"],
            "voxel_num_points": processed["voxel_num_points"],
            "record_len": data_dict["record_len"],
        }
        reliability = self.gspr(
            batch_dict["voxel_features"], batch_dict["voxel_num_points"],
            batch_dict["voxel_coords"])
        batch_dict.update(reliability)
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        output = {
            "psm": self.cls_head(batch_dict["spatial_features_2d"]),
            "rm": self.reg_head(batch_dict["spatial_features_2d"]),
        }
        # Stable interface for later communication/quality-aware fusion work.
        output.update(reliability)
        if "point_reliability_target" in data_dict:
            output["point_reliability_target"] = \
                data_dict["point_reliability_target"]
        return output
