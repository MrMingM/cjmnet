"""Frozen-feature counterfactual fusion and deployable descriptor extraction."""
import math
import torch
import torch.nn.functional as F


DESCRIPTOR_NAMES = (
    's0_peer_mean', 's0_peer_rms', 's0_peer_base_abs_mean',
    's0_peer_base_rms', 's0_peer_ego_cosine', 's0_original_peer_weight',
    's1_peer_mean', 's1_peer_rms', 's1_peer_base_abs_mean',
    's1_peer_base_rms', 's1_peer_ego_cosine', 's1_original_peer_weight',
    'peer_cls_mean', 'peer_cls_max', 'full_cls_mean', 'full_cls_max',
    'peer_full_cls_abs_mean', 'peer_full_cls_abs_max',
    'peer_full_reg_abs_mean', 'peer_full_reg_abs_max',
    'peer_reg_abs_mean', 'full_reg_abs_mean',
)


def attention_fusion(features, query_index=0):
    """Original AttFuse rule, with a selectable source as the query."""
    if features.ndim != 4 or not 0 <= query_index < len(features):
        raise ValueError('Expected [sources,C,H,W] and a valid query index')
    logits = (features[query_index:query_index+1]*features).sum(1, keepdim=True)
    logits = logits/math.sqrt(features.shape[1])
    weights = torch.softmax(logits, dim=0)
    return (features*weights).sum(0, keepdim=True), weights


def predict_from_levels(base, fused_levels):
    if len(fused_levels) != len(base.backbone.deblocks):
        raise ValueError('Fused level count differs from detector backbone')
    joined = torch.cat([deblock(value) for deblock, value in zip(base.backbone.deblocks, fused_levels)], 1)
    return {'psm': base.cls_head(joined), 'rm': base.reg_head(joined)}


def predict_each_source(base, levels):
    """Run the frozen heads for all already-received source features in one call."""
    joined = torch.cat([deblock(value) for deblock, value in zip(base.backbone.deblocks, levels)], 1)
    return {'psm': base.cls_head(joined), 'rm': base.reg_head(joined)}


def region_pool(value, reference_hw, tile_size, maximum=False):
    """Pool exactly the cells selected by expand_action_map, including edge tiles.

    The frozen layout has integral scale ratios. Padding is excluded from means
    and maxima; adaptive pooling would shift the final partial tile boundaries.
    """
    h, w = value.shape[-2:]
    rh, rw = reference_hw
    if rh % h or rw % w:
        raise ValueError('Only integral spatial scale ratios are supported')
    sy, sx = rh // h, rw // w
    if tile_size % sy or tile_size % sx:
        raise ValueError('Decision tiles must align with all feature scales')
    th, tw = tile_size // sy, tile_size // sx
    gh, gw = math.ceil(h / th), math.ceil(w / tw)
    padding = (0, gw * tw - w, 0, gh * th - h)
    padded = F.pad(value, padding, value=-float('inf') if maximum else 0.)
    blocks = padded.reshape(*value.shape[:2], gh, th, gw, tw)
    if maximum:
        return blocks.amax(dim=(3, 5))
    valid = F.pad(value.new_ones((1, 1, h, w)), padding)
    counts = valid.reshape(1, 1, gh, th, gw, tw).sum(dim=(3, 5))
    return blocks.sum(dim=(3, 5)) / counts


def _feature_descriptors(level, baseline, weights, pool):
    peers = level[1:]
    ego = level[:1].expand_as(peers)
    base = baseline.expand_as(peers)
    delta = peers-base
    cosine = (peers*ego).sum(1, keepdim=True) / (
        peers.square().sum(1, keepdim=True).sqrt()*ego.square().sum(1, keepdim=True).sqrt()
    ).clamp_min(1e-6)
    maps = (
        peers.mean(1, keepdim=True),
        peers.square().mean(1, keepdim=True).sqrt(),
        delta.abs().mean(1, keepdim=True),
        delta.square().mean(1, keepdim=True).sqrt(),
        cosine,
        weights[1:],
    )
    return [pool(value) for value in maps]


def _prediction_descriptors(local, full, pool):
    peer_cls = torch.sigmoid(local['psm'][1:]).amax(1, keepdim=True)
    full_cls = torch.sigmoid(full['psm']).amax(1, keepdim=True).expand_as(peer_cls)
    cls_delta = (peer_cls-full_cls).abs()
    peer_reg = local['rm'][1:]
    full_reg = full['rm'].expand_as(peer_reg)
    reg_delta = (peer_reg-full_reg).abs()
    return [
        pool(peer_cls), pool(peer_cls, True),
        pool(full_cls), pool(full_cls, True),
        pool(cls_delta), pool(cls_delta, True),
        pool(reg_delta.mean(1, keepdim=True)),
        pool(reg_delta.amax(1, keepdim=True), True),
        pool(peer_reg.abs().mean(1, keepdim=True)),
        pool(full_reg.abs().mean(1, keepdim=True)),
    ]


def build_context(base, levels, tile_size=8):
    """Build all selector inputs before applying any action.

    No GT, decoded box, weather label, or hindsight action is used here.
    """
    if len(levels) != 3 or any(value.ndim != 4 for value in levels):
        raise ValueError('Expected the frozen three-scale source features')
    if any(len(value) != len(levels[0]) for value in levels):
        raise ValueError('Source count differs between scales')
    if tile_size < 1:
        raise ValueError('tile_size must be positive')
    h0, w0 = levels[0].shape[-2:]
    grid = (math.ceil(h0/tile_size), math.ceil(w0/tile_size))
    def pool(value, maximum=False):
        return region_pool(value, (h0, w0), tile_size, maximum)
    baseline, original_weights = [], []
    for value in levels:
        fused, weights = attention_fusion(value, 0)
        baseline.append(fused)
        original_weights.append(weights)
    full = predict_from_levels(base, baseline)
    local = predict_each_source(base, levels)
    peer_count = len(levels[0])-1
    if peer_count:
        descriptors = []
        for scale in (0, 1):
            descriptors.extend(_feature_descriptors(
                levels[scale], baseline[scale], original_weights[scale], pool))
        descriptors.extend(_prediction_descriptors(local, full, pool))
        descriptors = torch.cat(descriptors, 1)
        activity = torch.maximum(
            descriptors[:, DESCRIPTOR_NAMES.index('peer_cls_max')],
            descriptors[:, DESCRIPTOR_NAMES.index('full_cls_max')],
        ).amax(0)
    else:
        descriptors = levels[0].new_empty((0, len(DESCRIPTOR_NAMES), *grid))
        activity = levels[0].new_zeros(grid)
    if descriptors.shape[1] != len(DESCRIPTOR_NAMES):
        raise AssertionError('Descriptor specification drifted')
    return {
        'levels': levels,
        'reference_hw': (h0, w0),
        'grid': grid,
        'tile_size': int(tile_size),
        'baseline_levels': baseline,
        'original_weights': original_weights,
        'baseline_prediction': full,
        'local_prediction': local,
        'descriptors': descriptors,
        'activity': activity,
    }


def expand_action_map(action_map, target_hw, reference_hw, tile_size):
    """Map fixed scale-0 tiles to another scale by physical-grid position."""
    if action_map.ndim != 2:
        raise ValueError('action_map must be [grid_h,grid_w]')
    target_h, target_w = map(int, target_hw)
    ref_h, ref_w = map(int, reference_hw)
    y0 = torch.floor((torch.arange(target_h, device=action_map.device)+.5)*ref_h/target_h).long()
    x0 = torch.floor((torch.arange(target_w, device=action_map.device)+.5)*ref_w/target_w).long()
    gy = (y0//int(tile_size)).clamp_max(action_map.shape[0]-1)
    gx = (x0//int(tile_size)).clamp_max(action_map.shape[1]-1)
    return action_map.index_select(0, gy).index_select(1, gx)


def fused_levels_for_actions(context, action_map, changed_scales=(0, 1)):
    """Apply disjoint tile actions; 0 means preserve the original full fusion."""
    peer_count = len(context['levels'][0])-1
    if action_map.shape != context['grid']:
        raise ValueError('Action map shape differs from decision grid')
    if action_map.numel() and (int(action_map.min()) < 0 or int(action_map.max()) > peer_count):
        raise ValueError('Action map contains an unavailable source')
    changed_scales = tuple(int(value) for value in changed_scales)
    output = []
    for scale, (level, original) in enumerate(zip(context['levels'], context['baseline_levels'])):
        if scale not in changed_scales or peer_count == 0 or not torch.any(action_map):
            output.append(original)
            continue
        actions = expand_action_map(action_map, level.shape[-2:], context['reference_hw'], context['tile_size'])
        fused = original.clone()
        for peer in range(1, peer_count+1):
            if not torch.any(actions == peer):
                continue
            alternative, _ = attention_fusion(level, peer)
            mask = (actions == peer)[None, None]
            fused = torch.where(mask, alternative, fused)
        output.append(fused)
    return output


def prediction_for_actions(base, context, action_map, changed_scales=(0, 1)):
    return predict_from_levels(base, fused_levels_for_actions(context, action_map, changed_scales))


def single_action_prediction(base, context, peer, tile_flat, changed_scales=(0, 1)):
    action = torch.zeros(context['grid'], dtype=torch.long, device=context['levels'][0].device)
    action.view(-1)[int(tile_flat)] = int(peer)
    return prediction_for_actions(base, context, action, changed_scales)


def select_actions(scores, threshold=0.0, allow_keep=True):
    """Select the best peer per tile; peer IDs start at one."""
    if scores.ndim != 3:
        raise ValueError('scores must be [peers,H,W]')
    if not scores.shape[0]:
        return torch.zeros(scores.shape[1:], dtype=torch.long, device=scores.device)
    best, index = scores.max(0)
    actions = index.long()+1
    if allow_keep:
        actions = torch.where(best > float(threshold), actions, torch.zeros_like(actions))
    return actions


def confidence_scores(descriptors):
    """Simple no-learning baseline: peer object confidence minus full confidence."""
    peer = descriptors[:, DESCRIPTOR_NAMES.index('peer_cls_max')]
    full = descriptors[:, DESCRIPTOR_NAMES.index('full_cls_max')]
    return peer-full
