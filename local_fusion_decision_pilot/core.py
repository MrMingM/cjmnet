"""Candidate construction and ranking; neither consumes GT at inference."""
import bisect

import torch
from torch import nn
import torch.nn.functional as F

from local_fusion_utility_v2.fusion import (
    DESCRIPTOR_NAMES, expand_action_map, predict_from_levels, region_pool,
)
from local_fusion_utility_v2.supervision import sample_action_pairs


FEATURE_NAMES = (
    *('mean_'+name for name in DESCRIPTOR_NAMES),
    *('max_'+name for name in DESCRIPTOR_NAMES),
    'activity', 'gate_s0', 'gate_s1', 'feature_change_s0',
    'feature_change_s1', 'base_cls_max', 'candidate_cls_max',
    'cls_abs_change', 'reg_abs_change',
)


def choose_frames(scene_ends, seed, scene_limit, frames_per_scene):
    """Sample whole scene IDs, then frame IDs, without consulting GT."""
    ends = [int(v) for v in scene_ends]
    if not ends or any(b <= a for a, b in zip([0, *ends[:-1]], ends)):
        raise ValueError('Invalid cumulative scene lengths')
    generator = torch.Generator().manual_seed(int(seed))
    scenes = torch.randperm(len(ends), generator=generator)[:scene_limit].sort().values.tolist()
    chosen = []
    for scene in scenes:
        start = 0 if scene == 0 else ends[scene-1]
        width = ends[scene]-start
        offsets = torch.randperm(width, generator=generator)[:frames_per_scene].tolist()
        chosen.extend(start+offset for offset in offsets)
    chosen.sort()
    if not chosen:
        raise ValueError('Selected frame set is empty')
    if any(bisect.bisect_right(ends, index) not in scenes for index in chosen):
        raise AssertionError('Scene selection drifted')
    return scenes, chosen


def residual_with_gate_maps(module, levels):
    """Capture the actual v3 gate activations without modifying its checkpoint."""
    if len(levels[0]) == 1:
        fused, _ = module(levels)
        return fused, {}
    raw, handles = {}, []
    for scale in module.scales:
        key = str(scale)
        handles.append(module.gates[key].register_forward_hook(
            lambda _layer, _input, output, key=key: raw.__setitem__(key, output)))
    try:
        fused, _ = module(levels)
    finally:
        for handle in handles:
            handle.remove()
    if set(raw) != {str(s) for s in module.scales}:
        raise RuntimeError('Could not capture every v3 gate')
    return fused, {int(key): module.max_gate*value.sigmoid() for key, value in raw.items()}


def candidate_features(context, fused, gates, base):
    """Tile descriptors available before GT labels and before action choice."""
    desc = context['descriptors']
    gh, gw = context['grid']
    if desc.shape[0] == 0:
        return desc.new_empty((0, len(FEATURE_NAMES), gh, gw)), None
    reference = context['reference_hw']
    tile_size = context['tile_size']

    def pool(value, maximum=False):
        return region_pool(value, reference, tile_size, maximum)

    proposed = predict_from_levels(base, fused)
    base_pred = context['baseline_prediction']
    base_cls = torch.sigmoid(base_pred['psm']).amax(1, keepdim=True)
    proposed_cls = torch.sigmoid(proposed['psm']).amax(1, keepdim=True)
    maps = [desc.mean(0, keepdim=True), desc.amax(0, keepdim=True),
            context['activity'][None, None]]
    for scale in (0, 1):
        gate = gates.get(scale)
        maps.append(pool(gate) if gate is not None else desc.new_zeros((1, 1, gh, gw)))
    for scale in (0, 1):
        delta = (fused[scale]-context['baseline_levels'][scale]).abs().mean(1, keepdim=True)
        maps.append(pool(delta))
    maps += [pool(base_cls, True), pool(proposed_cls, True),
             pool((proposed_cls-base_cls).abs()),
             pool((proposed['rm']-base_pred['rm']).abs().mean(1, keepdim=True))]
    result = torch.cat(maps, dim=1)
    if result.shape[1] != len(FEATURE_NAMES) or not torch.isfinite(result).all():
        raise RuntimeError('Candidate descriptor shape or finiteness failure')
    return result, proposed


def sampled_tiles(activity, settings, seed):
    pairs = sample_action_pairs(activity, 1, settings['tiles_per_frame'], seed,
                                settings['low_score_floor'], settings['high_score_floor'])
    return pairs[:, 1].long()


def patch_prediction(base, context, fused, tile_ids, scales):
    """Only chosen tiles use v3; every other cell is exactly original AttFuse."""
    grid = context['grid']
    action = torch.zeros(grid[0]*grid[1], dtype=torch.bool, device=fused[0].device)
    if len(tile_ids):
        action[torch.as_tensor(tile_ids, device=action.device).long()] = True
    action = action.reshape(grid)
    mixed = []
    for scale, (old, new) in enumerate(zip(context['baseline_levels'], fused)):
        if scale not in scales or not action.any():
            mixed.append(old)
            continue
        mask = expand_action_map(action, old.shape[-2:], context['reference_hw'], context['tile_size'])
        mixed.append(torch.where(mask[None, None], new, old))
    return predict_from_levels(base, mixed)


def scores_for_rows(features, gate, confidence, predictor, mean, std):
    x = ((features.float()-mean)/std).clamp(-20, 20)
    with torch.no_grad():
        probabilities = predictor(x).softmax(-1)
    return {'learned': (probabilities[:, 2]-probabilities[:, 0]).cpu(),
            'confidence': confidence.cpu(), 'v3_gate': gate.cpu()}


def selected_indices(scores, budget, keep_positive=False):
    values = torch.as_tensor(scores).flatten()
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    if keep_positive:
        order = [i for i in order if float(values[i]) > 0]
    return order[:min(int(budget), len(order))]


class DecisionSelector(nn.Module):
    """Three outcomes: harmful, unchanged, beneficial single-tile action."""

    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden),
                                    nn.SiLU(), nn.Linear(hidden, 3))

    def forward(self, features):
        return self.layers(features)


def outcome_class(target):
    utility = target[:, 0]-target[:, 1]-target[:, 2]
    return torch.where(utility > 0, 2, torch.where(utility < 0, 0, 1)).long()


def weighted_cross_entropy(logits, classes):
    counts = torch.bincount(classes, minlength=3).float()
    weights = (counts.sum()/counts.clamp_min(1)).sqrt()
    weights = (weights/weights.min()).clamp(max=10.)
    return F.cross_entropy(logits, classes, weight=weights), counts.long()
