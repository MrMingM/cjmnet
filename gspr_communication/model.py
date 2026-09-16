"""Frozen GSPR/AttFuse composition; shared source files are never patched."""
import torch
from torch import nn
import torch.nn.functional as F

from attfuse_gspr.point_pillar_gspr_attfuse import PointPillarGsprAttfuse
from . import codec
from .heads import RequestHead, ResponseHead
from .observation_stats import block_statistics
from .selection import foreground, hard_topk, soft_topk
from .masked_attfuse import fuse
from .residual import PriorityCorrection

VARIANTS = ("a0b0", "a1b0", "a0b1", "a1b1", "residual", "quality", "random", "none", "full")


class CommunicationModel(nn.Module):
    def __init__(self, model_args, config):
        super().__init__()
        self.base = PointPillarGsprAttfuse(model_args)
        self.base.requires_grad_(False)
        self.base.eval()
        self.request = RequestHead(hidden=int(config.get("hidden", 32)))
        self.response = ResponseHead(hidden=int(config.get("hidden", 32)))
        self.config = dict(config)
        self.variant = config["variant"]
        # Construct only for the new variant: old variants retain RNG/init behavior.
        self.correction = PriorityCorrection(int(config.get('correction_hidden', 16)),
                                            float(config.get('max_adjustment', .1))) if self.variant == 'residual' else None
        self.budget = int(config["budget_bytes"])
        self.temperature = float(config.get("temperature", .15))
        self.value_bytes = int(config.get("value_bytes", 4))
        if self.variant not in VARIANTS or self.budget < 0 or self.temperature <= 0:
            raise ValueError("Invalid communication configuration")
        arch = model_args["base_bev_backbone"]
        if (arch["layer_strides"] != [2, 2, 2] or arch["num_filters"] != [64, 128, 256]
                or arch["upsample_strides"] != [1, 2, 4] or arch.get("compression", 0)):
            raise ValueError("v1 requires the frozen three-scale uncompressed AttFuse layout")
        nx, ny, nz = model_args["point_pillar_scatter"]["grid_size"]
        if nx % 8 or ny % 8 or nz != 1:
            raise ValueError("v1 requires an XY grid divisible by 8 and nz=1")
        self.grid = (int(ny)//8, int(nx)//8)
        self.spatial_shape = (int(ny), int(nx))
        self.cost = codec.block_bytes([64, 128, 256], [4, 2, 1], self.value_bytes)
        self.request.requires_grad_(self.variant in ("a1b0", "a1b1"))
        self.response.requires_grad_(self.variant in ("a0b1", "a1b1"))
        self.last_diagnostics = []

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # prevents BN buffers from drifting
        return self

    def load_frontend(self, state):
        # Exact joint GSPR checkpoint required, never silently partially load.
        self.base.load_state_dict(state, strict=True)

    @torch.no_grad()
    def encode(self, data):
        processed = data["processed_lidar"]
        n = int(data["record_len"].sum().item())
        rel = self.base.gspr(processed["voxel_features"], processed["voxel_num_points"], processed["voxel_coords"])
        batch = dict(processed)
        batch.update(rel)
        batch = self.base.pillar_vfe(batch)
        # Explicit scatter cardinality also supports a trailing empty CAV.
        h, w = self.spatial_shape
        coords = processed["voxel_coords"].long()
        canvas = batch["pillar_features"].new_zeros((n, 64, h, w))
        if coords.numel():
            canvas[coords[:, 0], :, coords[:, 2], coords[:, 3]] = batch["pillar_features"]
        levels = []
        x = canvas
        for block in self.base.backbone.blocks:
            x = block(x)
            levels.append(x)
        local = torch.cat([d(x) for d, x in zip(self.base.backbone.deblocks, levels)], 1)
        confidence = F.max_pool2d(foreground(self.base.cls_head(local)), 4)
        semantics = F.avg_pool2d(levels[0], 4)
        stats = block_statistics(rel, processed, n, self.grid)
        return levels, semantics, stats, confidence

    def _request(self, semantics, stats, confidence):
        if self.variant in ("a1b0", "a1b1"):
            request = self.request(semantics, stats, confidence)
        else:
            request = 1 - confidence
        quantized = (request * 255).round().clamp(0, 255) / 255
        # Hard uint8 values in both train/eval; STE only in training.
        decoded = quantized.detach() + (request - request.detach()) if self.training else quantized
        return decoded

    def _score(self, semantics, stats, confidence, decoded):
        if self.variant == 'residual':
            return self.correction(stats, confidence, decoded)
        if self.variant in ("a0b1", "a1b1"):
            return self.response(semantics, stats, confidence, decoded)
        if self.variant == "random":
            return torch.rand_like(confidence)
        score = decoded * confidence
        if self.variant == "quality":
            score = score * stats[:, :1] * stats[:, 5:6]
        return score

    def forward(self, data):
        levels, semantics, stats, confidence = self.encode(data)
        counts = data["record_len"].detach().cpu().tolist()
        outputs, diagnostics = [], []
        start = 0
        for count in counts:
            count = int(count)
            if count < 1:
                raise ValueError("Each sample must contain ego")
            sl = slice(start, start + count)
            local_levels = [x[sl] for x in levels]
            sem, obs, conf = semantics[sl], stats[sl], confidence[sl]
            active, quotas = codec.allocate(self.budget, count-1, *self.grid, self.cost)
            if self.variant == "none":
                active, quotas = False, [0]*(count-1)
            if self.variant == "full":
                active, quotas = count > 1, [self.grid[0]*self.grid[1]]*(count-1)
            request = self._request(sem[:1], obs[:1], conf[:1]) if active else torch.zeros_like(conf[:1])
            hard = [torch.ones_like(request)]
            soft = [torch.ones_like(request)]
            received = [[x[:1]] for x in local_levels]
            request_cost = response_cost = wire_actual = 0
            replaced_blocks, adjustment_sum, adjustment_count = 0, 0., 0
            if active:
                request_cost = (count-1)*codec.request_bytes(*self.grid)
                if not self.training:
                    packet = codec.pack_request(request[0, 0].detach().cpu().numpy())
                    request = torch.as_tensor(codec.unpack_request(packet), device=sem.device, dtype=sem.dtype)[None, None]
                    wire_actual += (count-1)*len(packet)
            for peer, k in enumerate(quotas, 1):
                # No request was delivered when inactive: B is not invoked.
                score = self._score(sem[peer:peer+1], obs[peer:peer+1], conf[peer:peer+1], request) if active else torch.zeros_like(request)
                mask = hard_topk(score, k)
                if self.variant == 'residual' and active:
                    reference_score = request * conf[peer:peer+1]
                    reference_mask = hard_topk(reference_score, k)
                    replaced_blocks += int(((mask > 0) & (reference_mask == 0)).sum().item())
                    adjustment_sum += float((score-reference_score).detach().abs().sum())
                    adjustment_count += score.numel()
                hard.append(mask)
                soft.append(soft_topk(score, mask, self.temperature) if active else mask)
                if active:
                    response_cost += codec.HEADER.size + k*self.cost
                if self.training:
                    for target, x in zip(received, local_levels):
                        values = x[peer:peer+1]
                        if self.value_bytes == 2:
                            values = values.half().float()
                        target.append(values)
                elif active:
                    ids = mask.flatten().nonzero().flatten().cpu().numpy()
                    packet = codec.pack_response([x[peer].cpu().numpy() for x in local_levels], ids, self.grid, self.value_bytes)
                    decoded, _ = codec.unpack_response(packet, [tuple(x.shape[1:]) for x in local_levels], self.grid)
                    wire_actual += len(packet)
                    for target, array in zip(received, decoded):
                        target.append(torch.as_tensor(array, device=sem.device, dtype=sem.dtype)[None])
                else:
                    for target, x in zip(received, local_levels):
                        target.append(torch.zeros_like(x[peer:peer+1]))
            total = request_cost + response_cost
            if self.variant != "full" and total > self.budget:
                raise AssertionError("Communication exceeded per-scene budget")
            if not self.training and wire_actual != total:
                raise AssertionError("Serialized and predicted byte counts disagree")
            hard = torch.cat(hard)
            soft = torch.cat(soft) if self.training else None
            fused = [fuse(torch.cat(x), hard, soft) for x in received]
            joined = torch.cat([d(x) for d, x in zip(self.base.backbone.deblocks, fused)], 1)
            outputs.append(joined)
            diagnostics.append({
                "agents": count, "request_bytes": request_cost, "response_bytes": response_cost,
                "total_bytes": total, "selected_blocks": sum(quotas),
                "available_blocks": (count-1)*self.grid[0]*self.grid[1],
                "request_mean": float(request.detach().mean()),
                "selected_quality": self._selected_mean(obs[:, :1], hard),
                "selected_log_support": self._selected_mean(obs[:, 2:3], hard),
            })
            if self.variant == 'residual':
                diagnostics[-1].update({'replaced_blocks_vs_a0b0': replaced_blocks,
                    'replacement_fraction': replaced_blocks / max(sum(quotas), 1),
                    'mean_abs_score_adjustment': adjustment_sum / max(adjustment_count, 1)})
            start += count
        features = torch.cat(outputs)
        self.last_diagnostics = diagnostics
        return {"psm": self.base.cls_head(features), "rm": self.base.reg_head(features)}

    @staticmethod
    def _selected_mean(values, mask):
        return float(((values[1:]*mask[1:]).sum() / mask[1:].sum().clamp_min(1)).detach())
