"""Non-GT action sampling and GT-only training target construction."""
import torch

from .fusion import single_action_prediction
from .outcomes import detection_outcome, count_new_false_positives


def sample_action_pairs(activity, peer_count, count, seed, low_floor=.02, low_ceiling=.2):
    """Select count distinct regions; label EVERY available peer in each region.

    count denotes regions, not actions. Default three regions cost 3*peers
    counterfactual detector passes per branch. GT never selects the regions.
    """
    if activity.ndim != 2 or peer_count < 0 or count < 0:
        raise ValueError('Invalid activity map or sampling request')
    total_tiles = activity.numel()
    count = min(int(count), total_tiles)
    if not count or not peer_count:
        return torch.empty((0, 2), dtype=torch.long)
    values = activity.detach().cpu().flatten()
    generator = torch.Generator().manual_seed(int(seed))
    active = (values >= float(low_ceiling)).nonzero().flatten()
    if len(active):
        active = active[torch.randperm(len(active), generator=generator)].tolist()
    low = ((values >= float(low_floor)) & (values < float(low_ceiling))).nonzero().flatten()
    if len(low):
        low = low[torch.randperm(len(low), generator=generator)].tolist()
    background = (values < float(low_floor)).nonzero().flatten()
    if len(background):
        background = background[torch.randperm(len(background), generator=generator)].tolist()
    pools = [list(active), list(low), list(background)]
    chosen = []
    # Round-robin strata: default one active, one low-score, one background.
    while len(chosen) < count and any(pools):
        for pool in pools:
            if pool and len(chosen) < count:
                chosen.append(int(pool.pop()))
    return torch.tensor([(peer, tile) for tile in chosen
                         for peer in range(1, peer_count+1)], dtype=torch.long)


@torch.no_grad()
def action_targets(base, context, batch, dataset, label_dict, criterion, settings, seed):
    """Evaluate sampled single-tile actions after the original post-processing/NMS."""
    pairs = sample_action_pairs(
        context['activity'], context['descriptors'].shape[0], settings['regions_per_view'], seed,
        settings['sampling_low_floor'], settings['sampling_low_ceiling'])
    baseline_output = context['baseline_prediction']
    base_boxes, base_scores, gt = dataset.post_process(batch, {'ego': baseline_output})
    base_match, base_fp_boxes = detection_outcome(base_boxes, base_scores, gt, settings['target_iou'])
    baseline_loss = float(criterion(baseline_output, label_dict).detach())
    rows = []
    gh, gw = context['grid']
    for peer, tile in pairs.tolist():
        output = single_action_prediction(base, context, peer, tile, settings['changed_scales'])
        boxes, scores, action_gt = dataset.post_process(batch, {'ego': output})
        if tuple(gt.shape) != tuple(action_gt.shape) or not torch.allclose(gt, action_gt):
            raise RuntimeError('GT changed during a paired counterfactual')
        matched, fp_boxes = detection_outcome(boxes, scores, gt, settings['target_iou'])
        action_loss = float(criterion(output, label_dict).detach())
        row, col = divmod(tile, gw)
        rows.append({
            'peer': peer,
            'tile': tile,
            'row': row,
            'col': col,
            'descriptor': context['descriptors'][peer-1, :, row, col].detach().cpu().float(),
            'target': torch.tensor([
                len(matched-base_match),
                len(base_match-matched),
                count_new_false_positives(fp_boxes, base_fp_boxes, settings['fp_identity_iou']),
            ], dtype=torch.float16),
            'loss_gain': baseline_loss-action_loss,
        })
    return {
        'rows': rows,
        'baseline_loss': baseline_loss,
        'baseline_tp': len(base_match),
        'baseline_fp': len(base_fp_boxes),
        'grid': (gh, gw),
        'peers': context['descriptors'].shape[0],
    }
