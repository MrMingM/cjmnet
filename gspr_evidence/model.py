"""Frozen frontend, independent agents, real request/feature packet transport."""
import hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from gspr_communication.masked_attfuse import fuse
from gspr_communication.selection import foreground
from . import protocol
from .geometry import describe
from .heads import RiskHead, GainHead


def top_indices(scores, count):
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite selection scores')
    return torch.argsort(scores.flatten(), descending=True, stable=True)[:int(count)]


def flat(x):
    return x.flatten(1).T


def pose_features(own_pose, request_pose):
    relative = torch.linalg.inv(request_pose) @ own_pose
    yaw = torch.atan2(relative[1, 0], relative[0, 0])
    return torch.cat([relative[:3, 3]/100, yaw.sin()[None], yaw.cos()[None]])[None]


class EvidenceModel(nn.Module):
    def __init__(self, model_args, options):
        super().__init__()
        from gspr_communication.model import CommunicationModel
        self.engine = CommunicationModel(model_args, dict(options['communication'], variant='a0b0'))
        self.engine.requires_grad_(False)
        self.options = options
        self.lidar_range = model_args['lidar_range']
        self.grid = self.engine.grid
        self.heads = nn.ModuleDict({'risk': RiskHead(), 'gain': GainHead(matching=options.get('matching', True))})
        self.disable_u = False

    def train(self, mode=True):
        super().train(mode)
        self.engine.eval()
        return self

    @torch.no_grad()
    def encode(self, data):
        if len(data['record_len']) != 1:
            raise ValueError('Use one scene per batch for exact byte accounting')
        base = self.engine.base
        processed = data['processed_lidar']
        n = int(data['record_len'].sum())
        rel = base.gspr(processed['voxel_features'], processed['voxel_num_points'], processed['voxel_coords'])
        batch = base.pillar_vfe(dict(processed, **rel))
        h, w = self.engine.spatial_shape
        canvas = batch['pillar_features'].new_zeros((n, 64, h, w))
        coords = processed['voxel_coords'].long()
        if coords.numel():
            canvas[coords[:, 0], :, coords[:, 2], coords[:, 3]] = batch['pillar_features']
        levels = []
        for block in base.backbone.blocks:
            canvas = block(canvas)
            levels.append(canvas)
        joined = torch.cat([d(x) for d, x in zip(base.backbone.deblocks, levels)], 1)
        conf = F.max_pool2d(foreground(base.cls_head(joined)), 4)
        obs = describe(rel, processed, conf, data['transforms'], data['clouds'], self.grid, self.lidar_range)
        return dict(levels=levels, semantics=F.avg_pool2d(levels[0], 4), obs=obs, poses=data['transforms'])

    def profile(self, encoded, no_u=False):
        obs = encoded['obs'][:1].clone()
        if no_u:
            obs[:, 1] = 0
        risk = self.heads['risk'](encoded['semantics'][:1], obs)
        return torch.cat([obs, risk], 1)[0]

    def request(self, encoded, mode='learned'):
        profile = self.profile(encoded, no_u=mode == 'no_u' or self.disable_u)
        if mode == 'no_u' or self.disable_u:
            profile[1] = 0
        # Keep both high-risk and blind low-confidence queries; selection is ego-only.
        priority = (profile[13:].amax(0) + .25*(1-profile[12])
                    + .1*(1-profile[4:8].amax(0)))
        count = min(int(self.options['queries']), priority.numel())
        ids = top_indices(priority, count)
        packet = protocol.pack_request(profile.detach().cpu().numpy(), ids.cpu().numpy(), encoded['poses'][0].cpu().numpy())
        values, indices, pose, grid = protocol.unpack_request(packet)
        if grid != self.grid:
            raise AssertionError('Request grid mismatch')
        device = profile.device
        return packet, torch.as_tensor(values, device=device), torch.as_tensor(indices, device=device), torch.as_tensor(pose, device=device)

    def scores(self, encoded, peer, request, ids, request_pose, mode):
        own = flat(encoded['obs'][peer]).index_select(0, ids).clone()
        if mode == 'no_u' or self.disable_u:
            own[:, 1] = 0
        if mode == 'protocol':
            return (1-request[:, 12])*own[:, 12]
        semantics = flat(encoded['semantics'][peer]).index_select(0, ids)
        return self.heads['gain'](request, own, semantics, pose_features(encoded['poses'][peer], request_pose))

    @torch.no_grad()
    def detect(self, encoded, masks, serialize=True):
        levels = encoded['levels']
        received = [[x[:1]] for x in levels]
        packets = []
        for peer in range(1, len(masks)):
            ids = masks[peer].flatten().nonzero().flatten().cpu().numpy()
            if serialize:
                packet = protocol.pack_response([x[peer].cpu().numpy() for x in levels], ids, self.grid, self.engine.value_bytes)
                arrays, decoded_ids = protocol.unpack_response(packet, [tuple(x.shape[1:]) for x in levels], self.grid)
                if not np.array_equal(ids, decoded_ids):
                    raise AssertionError('Decoded response indices changed')
                packets.append(packet)
                for dst, array in zip(received, arrays):
                    dst.append(torch.as_tensor(array, device=levels[0].device)[None])
            else:
                for dst, x in zip(received, levels):
                    value = x[peer:peer+1]
                    if self.engine.value_bytes == 2:
                        value = value.half().float()
                    dst.append(value)
        fused = [fuse(torch.cat(x), masks) for x in received]
        base = self.engine.base
        joined = torch.cat([d(x) for d, x in zip(base.backbone.deblocks, fused)], 1)
        return dict(psm=base.cls_head(joined), rm=base.reg_head(joined)), packets

    def empty_masks(self, encoded):
        masks = encoded['obs'].new_zeros((len(encoded['obs']), 1, *self.grid))
        masks[0] = 1
        return masks

    @torch.no_grad()
    def run(self, encoded, mode='learned', exchange=None):
        if mode not in ('learned', 'protocol', 'no_u', 'none', 'full', 'a0b0'):
            raise ValueError(mode)
        n = len(encoded['obs'])
        if mode in ('a0b0', 'full'):
            from gspr_communication import codec
            q = self.grid[0]*self.grid[1]
            active, quotas = codec.allocate(self.engine.budget, n-1, *self.grid, self.engine.cost)
            if mode == 'full':
                active, quotas = n > 1, [q]*(n-1)
            # Original full-grid 1-confidence request and original byte cost.
            packet = codec.pack_request((1-encoded['obs'][0, 12]).cpu().numpy())
            req = torch.as_tensor(codec.unpack_request(packet), device=encoded['obs'].device, dtype=encoded['obs'].dtype).flatten()
            ids = torch.arange(q, device=req.device)
        else:
            packet, req, ids, pose = exchange or self.request(encoded, mode)
            active, quotas = protocol.allocate(self.engine.budget, n-1, len(ids), self.engine.cost)
        if mode == 'none':
            active, quotas = False, [0]*(n-1)
        masks = self.empty_masks(encoded)
        if active:
            for peer, count in enumerate(quotas, 1):
                score = (req*encoded['obs'][peer, 12].flatten() if mode in ('a0b0', 'full')
                         else self.scores(encoded, peer, req, ids, pose, mode))
                selected = ids[top_indices(score, count)]
                masks[peer].view(-1)[selected] = 1
        output, responses = self.detect(encoded, masks, serialize=active)
        request_bytes = len(packet)*(n-1) if active else 0
        total = request_bytes + sum(map(len, responses))
        if mode != 'full' and total > self.engine.budget:
            raise AssertionError('Total request+indices+three-scale payload exceeds budget')
        diag = dict(total_bytes=total, request_bytes=request_bytes, feature_bytes=total-request_bytes,
                    selected_blocks=int(masks[1:].sum()), query_blocks=len(ids) if active else 0,
                    message_sha256=hashlib.sha256((packet if active else b'')+b''.join(responses)).hexdigest(),
                    selected_ids=[x.flatten().nonzero().flatten().cpu().tolist() for x in masks[1:]])
        return output, diag
