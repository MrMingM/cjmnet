"""Prediction-driven tiles and two genuinely different fusion actions.

No function in this file reads GT, weather labels, or post-NMS outcomes.
"""
import torch

from local_fusion_utility_v2.fusion import (
    DESCRIPTOR_NAMES, attention_fusion, expand_action_map, predict_from_levels,
    region_pool,
)
from local_fusion_decision_pilot.core import FEATURE_NAMES


ACTION_FEATURE_NAMES = (
    *FEATURE_NAMES,
    *('source_'+name for name in DESCRIPTOR_NAMES),
    'is_residual', 'is_peer_query', 'action_cls_gain',
    'action_cls_abs_change', 'action_reg_abs_change',
)


def _normal(value):
    value = value.float().clamp_min(0)
    return value/value.amax().clamp_min(1e-6)


def select_tiles(context, residual_features, settings):
    """Choose six tiles by prediction activity and prospective response change.

    Active, low-score and background regions compete separately. Selection is
    deterministic and does not use GT or the later counterfactual outcome.
    """
    if not len(context['descriptors']):
        return []
    activity = context['activity'].float()
    names = {name:i for i,name in enumerate(FEATURE_NAMES)}
    impact = (_normal(residual_features[0,names['cls_abs_change']]) +
              _normal(context['descriptors'][:,DESCRIPTOR_NAMES.index('peer_full_cls_abs_max')].amax(0)) +
              .25*_normal(residual_features[0,names['reg_abs_change']]) +
              .25*_normal(context['descriptors'][:,DESCRIPTOR_NAMES.index('peer_full_reg_abs_max')].amax(0)))
    if impact.shape != activity.shape or not torch.isfinite(impact).all():
        raise RuntimeError('Invalid prediction-driven tile priority')
    score = impact.detach().cpu().flatten().tolist()
    values = activity.detach().cpu().flatten().tolist()
    low, high = settings['low_score_floor'], settings['high_score_floor']
    strata = [lambda v:v >= high,
              lambda v:low <= v < high,
              lambda v:v < low]
    quotas = [2, 2, 1]
    chosen = []
    for allowed, quota in zip(strata, quotas):
        ranked = sorted((i for i,v in enumerate(values) if allowed(v)),
                        key=lambda i:(-score[i], i))
        chosen.extend(ranked[:quota])
    remaining = sorted((i for i in range(len(values)) if i not in chosen),
                       key=lambda i:(-score[i], i))
    chosen.extend(remaining[:max(0, settings['tiles_per_frame']-len(chosen))])
    return chosen[:settings['tiles_per_frame']]


def peer_choices(context, tile, limit):
    """One peer with strongest score disagreement; one with geometry disagreement."""
    desc = context['descriptors']
    if not len(desc):
        return []
    row, col = divmod(int(tile), context['grid'][1])
    cls = desc[:,DESCRIPTOR_NAMES.index('peer_full_cls_abs_max'),row,col]
    reg = desc[:,DESCRIPTOR_NAMES.index('peer_full_reg_abs_max'),row,col]
    order = []
    for values in (cls,reg):
        ranked = sorted(range(len(values)), key=lambda i:(-float(values[i]),i))
        for peer_index in ranked:
            peer = peer_index+1
            if peer not in order:
                order.append(peer)
                break
    return order[:limit]


def actions_for_tiles(context, tiles, settings):
    """code 1 = v3 residual; code peer+1 = peer-query AttFuse."""
    rows = []
    for tile in tiles:
        rows.append((int(tile), 1))
        rows.extend((int(tile), peer+1) for peer in peer_choices(
            context, tile, settings['max_peer_actions_per_tile']))
    return rows


def levels_for_choices(context, residual_levels, actions, changed_scales=(0,1)):
    """One action per tile; zero means exact original full fusion."""
    grid = context['grid']
    choice = torch.zeros(grid[0]*grid[1], dtype=torch.long,
                         device=context['levels'][0].device)
    for tile, code in actions:
        tile, code = int(tile), int(code)
        if not 0 <= tile < choice.numel() or not 1 <= code <= len(context['levels'][0]):
            raise ValueError('Invalid tile/source action')
        if choice[tile]:
            raise ValueError('More than one action was selected for a tile')
        choice[tile] = code
    choice = choice.reshape(grid)
    output = []
    for scale, (level, original) in enumerate(zip(context['levels'], context['baseline_levels'])):
        if scale not in changed_scales or not actions:
            output.append(original)
            continue
        expanded = expand_action_map(choice, original.shape[-2:],
                                     context['reference_hw'], context['tile_size'])
        fused = torch.where((expanded == 1)[None,None], residual_levels[scale], original)
        for peer in range(1,len(level)):
            code = peer+1
            if not torch.any(expanded == code):
                continue
            alternative, _ = attention_fusion(level, peer)
            fused = torch.where((expanded == code)[None,None], alternative, fused)
        output.append(fused)
    return output


def prediction_for_choices(base, context, residual_levels, actions, changed_scales=(0,1)):
    return predict_from_levels(base, levels_for_choices(
        context, residual_levels, actions, changed_scales))


def action_features(context, residual_features, action, prediction):
    """Describe the exact action's pre-NMS response at its chosen tile."""
    tile, code = action
    row, col = divmod(int(tile),context['grid'][1])
    base = residual_features[0,:,row,col]
    desc = context['descriptors']
    source = (desc.mean(0) if code == 1 else desc[code-2])[:,row,col]
    old = context['baseline_prediction']
    old_cls = old['psm'].sigmoid().amax(1,keepdim=True)
    new_cls = prediction['psm'].sigmoid().amax(1,keepdim=True)
    difference = new_cls-old_cls
    def pool(value, maximum=False):
        return region_pool(value,context['reference_hw'],context['tile_size'],maximum)[0,0,row,col]
    extra = torch.stack((base.new_tensor(float(code == 1)),
                         base.new_tensor(float(code != 1)),
                         pool(difference),pool(difference.abs(),True),
                         pool((prediction['rm']-old['rm']).abs().mean(1,keepdim=True))))
    features = torch.cat((base,source,extra))
    if features.numel() != len(ACTION_FEATURE_NAMES) or not torch.isfinite(features).all():
        raise RuntimeError('Invalid source/action descriptor')
    return features


def simple_scores(context, action, prediction):
    tile, code = action
    row, col = divmod(int(tile),context['grid'][1])
    desc = context['descriptors']
    full = desc[:,DESCRIPTOR_NAMES.index('full_cls_max'),row,col]
    peer = desc[:,DESCRIPTOR_NAMES.index('peer_cls_max'),row,col]
    confidence = (peer-full).amax() if code == 1 else peer[code-2]-full[code-2]
    old = context['baseline_prediction']['psm'].sigmoid().amax(1,keepdim=True)
    new = prediction['psm'].sigmoid().amax(1,keepdim=True)
    response = region_pool((new-old).abs(),context['reference_hw'],
                           context['tile_size'],True)[0,0,row,col]
    return float(confidence),float(response)


def choose_distinct_rows(scores, actions, budget, require_positive=False):
    """Equal action budget; never request two conflicting actions in one tile."""
    order = sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))
    selected, used = [], set()
    for index in order:
        if require_positive and float(scores[index]) <= 0:
            continue
        tile = int(actions[index][0])
        if tile not in used:
            selected.append(index)
            used.add(tile)
        if len(selected) >= budget:
            break
    return selected
