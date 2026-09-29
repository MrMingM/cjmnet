from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.loss.un_map_loss import HybridUncertaintyLoss
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.utils.pyramid_fusion_modulev1 import MSUGPrecisionFusionV1
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class UncertaintyHead(nn.Module):
    """Predict a BEV uncertainty map from LiDAR BEV features."""

    def __init__(self, input_channels: int, output_channels: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BEVtextv1(nn.Module):
    """
    LiDAR-only UECP model used by the public release.

    The original research workspace also contained camera and exploratory fusion branches.
    This release keeps the ECCV UAPF runnable path: PointPillars encoder, uncertainty-map
    prediction, uncertainty-aware pyramid fusion, and anchor-based detection heads.
    """

    def __init__(self, args: Dict):
        super().__init__()
        self.args = args
        self.modality = args.get("modality", ["lidar"])
        if self.modality != ["lidar"]:
            raise ValueError("The UECP release model supports only modality: ['lidar'].")

        self.voxel_size = args["voxel_size"]
        self.cav_range = args["lidar_range"]
        self.feature_embedding = args["feature_embedding"]

        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.lidar_backbone = BaseBEVBackbone(args["backbone_args"], 64)

        self.shrink_flag = "shrink_header" in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args["shrink_header"])

        self.compress = "compressor" in args
        if self.compress:
            self.compressor = NaiveCompressor(
                args["compressor"]["input_dim"],
                args["compressor"]["compress_ratio"],
            )
            self.model_train_init()

        fusion_method = args.get("fusion_method", "PyramidFuserv1")
        if fusion_method != "PyramidFuserv1":
            raise ValueError(f"Unsupported UECP fusion method: {fusion_method}")
        self.fusion_net = MSUGPrecisionFusionV1(
            feature_dim=self.feature_embedding,
            scales=args.get("pyramid_scales", (4, 2, 1)),
            fusion_operation=args.get("fusion_operation", "sum"),
        )

        self.uncertainty_head = UncertaintyHead(input_channels=self.feature_embedding)
        self.uncertainty_loss_func = HybridUncertaintyLoss(alpha=0.5, gamma=2.0)
        self.use_gt_unmap = args.get("use_gt_unmap", False)

        self.supervise_single = bool(args.get("supervise_single", False))
        if self.supervise_single:
            self.cls_head_single = nn.Conv2d(
                args["in_head_single"], args["anchor_number"], kernel_size=1
            )
            self.reg_head_single = nn.Conv2d(
                args["in_head_single"], args["anchor_number"] * 7, kernel_size=1
            )
            self.dir_head_single = nn.Conv2d(
                args["in_head_single"],
                args["anchor_number"] * args["dir_args"]["num_bins"],
                kernel_size=1,
            )

        self.cls_head = nn.Conv2d(args["in_head"], args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(args["in_head"], 7 * args["anchor_number"], kernel_size=1)
        self.dir_head = nn.Conv2d(
            args["in_head"],
            args["dir_args"]["num_bins"] * args["anchor_number"],
            kernel_size=1,
        )

        self.H = self.cav_range[4] - self.cav_range[1]
        self.W = self.cav_range[3] - self.cav_range[0]
        self.fake_voxel_size = 1

    def model_train_init(self) -> None:
        if not self.compress:
            return
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.compressor.train()
        for parameter in self.compressor.parameters():
            parameter.requires_grad_(True)

    def extract_pts_feat(self, data_dict: Dict) -> torch.Tensor:
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": data_dict["record_len"],
            "pairwise_t_matrix": data_dict["pairwise_t_matrix"],
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.lidar_backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compress:
            spatial_features_2d = self.compressor(spatial_features_2d)
        return spatial_features_2d

    @staticmethod
    def _format_uncertainty_map(un_maps: torch.Tensor) -> torch.Tensor:
        if un_maps.dim() == 3:
            return un_maps.unsqueeze(1)
        if un_maps.dim() == 4 and un_maps.shape[1] == 1:
            return un_maps
        raise ValueError(f"Unexpected uncertainty map shape: {tuple(un_maps.shape)}")

    def forward(self, data_dict: Dict) -> Dict[str, torch.Tensor]:
        output_dict: Dict[str, torch.Tensor] = {}
        record_len = data_dict["record_len"]

        feature = self.extract_pts_feat(data_dict)
        pred_un_maps = self.uncertainty_head(feature)

        gt_u_low = None
        if "un_maps" in data_dict:
            gt_un_maps = self._format_uncertainty_map(data_dict["un_maps"])
            gt_u_low = F.interpolate(
                gt_un_maps,
                size=pred_un_maps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            output_dict["aux_loss"] = 5 * self.uncertainty_loss_func(pred_un_maps, gt_u_low)

        if self.supervise_single:
            output_dict.update(
                {
                    "cls_preds_single": self.cls_head_single(feature),
                    "reg_preds_single": self.reg_head_single(feature),
                    "dir_preds_single": self.dir_head_single(feature),
                }
            )

        affine_matrix = normalize_pairwise_tfm(
            data_dict["pairwise_t_matrix"],
            self.H,
            self.W,
            self.fake_voxel_size,
        )
        fusion_u_map = gt_u_low if self.use_gt_unmap and gt_u_low is not None else pred_un_maps
        fused_feature = self.fusion_net(
            feature,
            u_map=fusion_u_map,
            record_len=record_len,
            affine_matrix=affine_matrix,
        )

        output_dict.update(
            {
                "cls_preds": self.cls_head(fused_feature),
                "reg_preds": self.reg_head(fused_feature),
                "dir_preds": self.dir_head(fused_feature),
            }
        )
        return output_dict
