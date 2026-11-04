"""Capture the original PFN activation and the actual voxel projection origin."""
import numpy as np
import torch

from attfuse_gspr.point_pillar_gspr_attfuse import PointPillarGsprAttfuse
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from aggregation import A1


class Dataset(IntermediateFusionDataset):
    def __getitem__(self, index):
        self.origins = []
        sample = super().__getitem__(index)
        sample["ego"]["a1_origins"] = np.asarray(self.origins, dtype=np.float32)
        return sample

    def get_item_single_car(self, selected_cav_base, *args, **kwargs):
        result = super().get_item_single_car(selected_cav_base, *args, **kwargs)
        matrix = np.asarray(selected_cav_base["params"]["transformation_matrix"])
        self.origins.append(matrix[:3, 3].copy() if self.proj_first else np.zeros(3))
        return result

    def collate_batch_train(self, batch):
        result = super().collate_batch_train(batch)
        if result is not None:
            result["ego"]["a1_origins"] = torch.from_numpy(
                np.concatenate([item["ego"]["a1_origins"] for item in batch]))
        return result


class Model(PointPillarGsprAttfuse):
    def __init__(self, args):
        super().__init__(args)
        layers = self.pillar_vfe.pfn_layers
        if len(layers) != 1 or not layers[0].use_norm or layers[0].linear.out_features != 64:
            raise ValueError("Expected one normalized 64-channel OPV2V PFN")
        if list(args["voxel_size"]) != [.4, .4, 4]:
            raise ValueError("Unexpected voxel geometry")
        rng = torch.get_rng_state()
        self.a1 = A1(dict(neighbors=4, relation_dim=8, pillar_chunk=256,
                          relation_logit_bound=.5, residual_bound=.5))
        torch.set_rng_state(rng)
        self.enabled = True
        self.origins = self.point_features = None
        layers[0].norm.register_forward_hook(self.capture)
        self.pillar_vfe.register_forward_hook(self.aggregate)

    def capture(self, module, inputs, output):
        if self.enabled:
            self.point_features = output.transpose(1, 2).relu()

    def aggregate(self, module, inputs, output):
        if not self.enabled:
            return output
        xyz = output["voxel_features"][..., :3]
        agents = output["voxel_coords"][:, 0].long()
        valid = output["point_valid_mask"]
        counts = output["voxel_num_points"]
        expected = torch.arange(xyz.shape[1], device=xyz.device)[None] < counts[:, None]
        r, u = output["point_reliability"], output["point_uncertainty"]
        if not torch.equal(valid, expected) or r.shape != valid.shape or u.shape != valid.shape:
            raise ValueError("GSPR point slots do not match OPV2V voxel slots")
        if self.point_features.shape[:2] != valid.shape:
            raise ValueError("PFN point slots do not match")
        ranges = ((xyz - self.origins[agents, None]).square().sum(-1) + 1e-6).sqrt()
        features = self.a1(self.point_features, output["pillar_features"], xyz, valid, r, u, ranges)
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite A1 output")
        if getattr(self, "check_zero_residual", False):
            torch.testing.assert_close(features, output["pillar_features"], rtol=0, atol=0)
        output["pillar_features"] = features
        return output

    def forward(self, data):
        if self.enabled:
            self.origins = data["a1_origins"]
            if self.origins.shape != (int(data["record_len"].sum()), 3):
                raise ValueError("Sensor origins must follow collated CAV order")
        try:
            return super().forward(data)
        finally:
            self.origins = self.point_features = None
