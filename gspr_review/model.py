"""Cooperative evidence feedback BEFORE ego pillar encoding, with real packets."""
import hashlib
import torch
import torch.nn.functional as F

from gspr_communication.model import CommunicationModel
from gspr_communication import codec as bev_codec
from gspr_communication.selection import foreground, hard_topk
from gspr_communication.observation_stats import block_statistics
from gspr_communication.masked_attfuse import fuse
from . import codec
from .evidence import PointReviewer, summarize, combine

MODES = ('review', 'protocol', 'shuffled', 'a0b0', 'none', 'full')


class ReviewModel(CommunicationModel):
    def __init__(self, model_args, config, review):
        super().__init__(model_args, dict(config, variant='a0b0'))
        self.review_config = dict(review)
        self.mode = 'review'
        self.z_range = (float(model_args['lidar_range'][2]), float(model_args['lidar_range'][5]))
        self.z_bins = int(review['z_bins'])
        if self.z_bins < 1 or self.z_range[1] <= self.z_range[0]:
            raise ValueError('Invalid evidence height bins')
        architecture = review.get('architecture', 'legacy')
        if architecture == 'legacy':
            self.reviewer = PointReviewer(int(review['hidden']), float(review['max_adjustment']), self.base.gspr.reliability_floor)
        elif architecture in ('contrast', 'contrast_gain', 'bev_contrast'):
            from .counterfactual import CounterfactualReviewer
            self.reviewer = CounterfactualReviewer(int(review['hidden']), float(review['max_adjustment']),
                                                  self.base.gspr.reliability_floor, architecture == 'contrast_gain')
        else:
            raise ValueError('Unknown review architecture: ' + str(architecture))
        self.reviewer.intervention = 'normal'
        self.bev_location = architecture == 'bev_contrast'
        self.correction_penalty = None

    def _levels(self, processed, reliability, agents):
        batch = dict(processed)
        batch['point_reliability'] = reliability
        pillars = self.base.pillar_vfe(batch)['pillar_features']
        h, w = self.spatial_shape
        coords = processed['voxel_coords'].long()
        x = pillars.new_zeros((agents, 64, h, w))
        if len(coords):
            x[coords[:, 0], :, coords[:, 2], coords[:, 3]] = pillars
        result = []
        for block in self.base.backbone.blocks:
            x = block(x)
            result.append(x)
        return result

    @staticmethod
    def _agent(processed, rel, agent):
        keep = processed['voxel_coords'][:, 0].long() == agent
        local = {key: processed[key][keep] for key in ('voxel_features', 'voxel_num_points', 'voxel_coords')}
        local['voxel_coords'] = local['voxel_coords'].clone()
        local['voxel_coords'][:, 0] = 0
        quality = {key: value[keep] for key, value in rel.items() if torch.is_tensor(value) and value.ndim and value.shape[0] == len(keep)}
        return local, quality

    def forward(self, data):
        exchange = getattr(self, 'packet_exchange', lambda kind, packet: packet)
        if self.mode not in MODES:
            raise ValueError('Unknown review mode')
        if self.mode in ('a0b0', 'none', 'full'):
            self.variant = self.mode
            self.correction_penalty = None
            return super().forward(data)
        self.variant = 'a0b0'
        processed = data['processed_lidar']
        counts = [int(n) for n in data['record_len'].tolist()]
        if not counts or min(counts) < 1:
            raise ValueError('Every scene requires ego')
        with torch.no_grad():
            rel = self.base.gspr(processed['voxel_features'], processed['voxel_num_points'], processed['voxel_coords'])
            levels = self._levels(processed, rel['point_reliability'], sum(counts))
            local = torch.cat([d(x) for d, x in zip(self.base.backbone.deblocks, levels)], 1)
            confidence = F.max_pool2d(foreground(self.base.cls_head(local)), 4)
            stats = block_statistics(rel, processed, sum(counts), self.grid)
        output, diagnostics, penalties = [], [], []
        start = 0
        for count in counts:
            peers = count-1
            ego, ego_rel = self._agent(processed, rel, start)
            device = processed['voxel_features'].device
            # Fixed query mechanism: existing points with uncertain reliability.
            priority = stats[start, 1]*(1-stats[start, 0])*stats[start, 5]
            eligible_blocks = int((priority > 0).sum())
            n = codec.query_count(self.budget, peers, float(self.review_config['budget_fraction']),
                                  int(self.review_config['max_query_blocks']), eligible_blocks, self.z_bins)
            ids = torch.argsort(priority.flatten(), descending=True, stable=True)[:n]
            ids_np = ids.cpu().numpy()
            evidence = processed['voxel_features'].new_zeros((n, 8, 8, self.z_bins, 4))
            review_bytes = 0
            message_digest = hashlib.sha256()
            # Sender API receives only its local data and decoded query indices.
            if n:
                query = exchange('review_query', codec.pack_query(ids_np, self.grid))
                # Canonical query defines the same physical regions in every control.
                ids_np = codec.unpack_query(query, self.grid)
                if len(ids_np) != n:
                    raise ValueError('Replayed review query changes the allocated count')
                ids = torch.as_tensor(ids_np, device=device)
                for peer in range(1, count):
                    decoded_ids = codec.unpack_query(query, self.grid)
                    sender, sender_rel = self._agent(processed, rel, start+peer)
                    sender_ids = torch.as_tensor(decoded_ids, device=device)
                    values = summarize(sender, sender_rel, sender_ids, self.grid, self.z_range, self.z_bins)
                    packet = exchange('review_evidence', codec.pack_evidence(decoded_ids, values.cpu().numpy(), self.grid, self.z_bins))
                    received = codec.unpack_evidence(packet, self.grid, self.z_bins, ids_np)
                    evidence = combine(evidence, torch.as_tensor(received, device=device))
                    review_bytes += len(query)+len(packet)
                    message_digest.update(query)
                    message_digest.update(packet)
            if self.mode == 'shuffled' and n:
                # Same received byte count, deterministic spatially wrong evidence.
                evidence = evidence.roll(shifts=(3, 3), dims=(1, 2))
            corrected, supported = self.reviewer(ego, ego_rel, ids, evidence, self.grid, self.z_range, self.z_bins)
            if self.mode == 'protocol':
                corrected = ego_rel['point_reliability']
            delta = corrected-ego_rel['point_reliability']
            if supported.any():
                penalties.append(delta[supported].square().mean())
            # Recompute only the receiver. Frozen weights still pass gradients to point weights.
            bev_maps = []
            if self.bev_location:
                ego_levels = [x[start:start+1] for x in levels]
                if self.mode != 'protocol':
                    from .bev_location import correct_bev
                    fixed = getattr(self, 'fixed_bev_adjustment', None)
                    if fixed is not None and self.training:
                        raise ValueError('Fixed BEV control is evaluation-only')
                    ego_levels, bev_maps = correct_bev(ego_levels, ego, delta, supported, fixed)
                # Delta is a proposed scalar used at BEV, never a point-weight update.
                delta = torch.zeros_like(delta)
            else:
                ego_levels = self._levels(ego, corrected, 1) if self.mode != 'protocol' else [x[start:start+1] for x in levels]
            active, quotas = bev_codec.allocate(self.budget-review_bytes, peers, *self.grid, self.cost)
            conf = confidence[start:start+count]
            request = (1-conf[:1]).detach()
            feature_bytes = 0
            hard = [torch.ones_like(request)]
            received_levels = [[x] for x in ego_levels]
            if active:
                query = exchange('bev_query', bev_codec.pack_request(request[0, 0].cpu().numpy()))
                request = torch.as_tensor(bev_codec.unpack_request(query), device=device, dtype=conf.dtype)[None, None]
                feature_bytes += peers*len(query)
                message_digest.update(query)
            for peer, quota in enumerate(quotas, 1):
                mask = hard_topk(request*conf[peer:peer+1], quota)
                hard.append(mask)
                if active:
                    selected = mask.flatten().nonzero().flatten().cpu().numpy()
                    packet = exchange('bev_features', bev_codec.pack_response([x[start+peer].cpu().numpy() for x in levels], selected, self.grid, self.value_bytes))
                    arrays, actual_ids = bev_codec.unpack_response(packet, [tuple(x.shape[1:]) for x in levels], self.grid)
                    if len(actual_ids) != quota:
                        raise ValueError('Replayed feature packet changes the allocated quota')
                    # The receiver mask must describe received blocks, not fresh scores.
                    mask = torch.zeros_like(mask)
                    mask.flatten()[torch.as_tensor(actual_ids.astype('int64'), device=device)] = 1
                    hard[-1] = mask
                    feature_bytes += len(packet)
                    message_digest.update(packet)
                    for target, array in zip(received_levels, arrays):
                        target.append(torch.as_tensor(array, device=device, dtype=conf.dtype)[None])
                else:
                    for target, x in zip(received_levels, levels):
                        target.append(torch.zeros_like(x[start+peer:start+peer+1]))
            if review_bytes+feature_bytes > self.budget:
                raise AssertionError('Review + BEV messages exceeded total budget')
            masks = torch.cat(hard)
            fused = [fuse(torch.cat(x), masks) for x in received_levels]  # exact attention, NO soft selection proxy
            joined = torch.cat([d(x) for d, x in zip(self.base.backbone.deblocks, fused)], 1)
            output.append(joined)
            valid_points = int(ego_rel['point_valid_mask'].sum())
            diagnostics.append({'agents': count, 'total_bytes': review_bytes+feature_bytes,
                'message_sha256': message_digest.hexdigest(),
                'review_bytes': review_bytes, 'feature_bytes': feature_bytes, 'query_blocks': n,
                'selected_blocks': sum(quotas), 'eligible_points': int(supported.sum()),
                'valid_ego_points': valid_points, 'evidence_coverage': int(supported.sum())/max(valid_points, 1),
                'mean_abs_weight_change': float(delta[supported].detach().abs().mean()) if supported.any() else 0.,
                'changed_points': int((delta.detach().abs() > 1e-6).sum()),
                'raised_points': int((delta.detach() > 1e-6).sum()),
                'lowered_points': int((delta.detach() < -1e-6).sum())})
            if self.bev_location:
                diagnostics[-1].update(
                    bev_changed_cells=sum(int((m.detach().abs() > 1e-6).sum()) for m in bev_maps),
                    bev_abs_adjustment_sum=sum(float(m.detach().abs().sum()) for m in bev_maps))
            start += count
        self.correction_penalty = torch.stack(penalties).mean() if penalties else next(self.reviewer.parameters()).sum()*0
        self.last_diagnostics = diagnostics
        features = torch.cat(output)
        return {'psm': self.base.cls_head(features), 'rm': self.base.reg_head(features)}
