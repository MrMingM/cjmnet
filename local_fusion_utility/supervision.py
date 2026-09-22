"""Non-GT action sampling and GT-only training target construction."""
import torch

from .fusion import single_action_prediction
from .outcomes import detection_outcome, count_new_false_positives


def sample_action_pairs(activity, peer_count, count, seed, low_floor=.02, low_ceiling=.2):
    """Sample top, low-score, and background tiles without looking at GT."""
    if activity.ndim != 2 or peer_count < 0 or count < 0:
        raise ValueError('Invalid activity map or sampling request')
    total_tiles = activity.numel()
    maximum = total_tiles*peer_count
    count = min(int(count), maximum)
    if not count:
        return torch.empty((0, 2), dtype=torch.long)
    values = activity.detach().cpu().flatten()
    generator = torch.Generator().manual_seed(int(seed))
    descending = torch.argsort(values, descending=True, stable=True).tolist()
    low = ((values >= float(low_floor)) & (values < float(low_ceiling))).nonzero().flatten()
    if len(low):
        low = low[torch.randperm(len(low), generator=generator)].tolist()
    background = (values < float(low_floor)).nonzero().flatten()
    if len(background):
        background = background[torch.randperm(len(background), generator=generator)].tolist()
    pools = (descending, low, background)
    chosen, seen = [], set()
    quota = max(1, (count+2)//3)
    peer_offset = int(seed) % max(peer_count, 1)

    def add(tile):
        # Cycle sources so source identity is not inferred from sampling category.
        for shift in range(peer_count):
            peer = 1+(peer_offset+len(chosen)+shift) % peer_count
            key = (peer, int(tile))
            if key not in seen:
                seen.add(key)
                chosen.append(key)
                return

    for pool in pools:
        before = len(chosen)
        for tile in pool:
            add(tile)
            if len(chosen)-before >= quota or len(chosen) >= count:
                break
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        all_pairs = [(peer, tile) for peer in range(1, peer_count+1) for tile in range(total_tiles)]
        order = torch.randperm(len(all_pairs), generator=generator).tolist()
        for index in order:
            key = all_pairs[index]
            if key not in seen:
                seen.add(key)
                chosen.append(key)
            if len(chosen) >= count:
                break
    return torch.tensor(chosen, dtype=torch.long)


@torch.no_grad()
def action_targets(base, context, batch, dataset, label_dict, criterion, settings, seed):
    """Evaluate sampled single-tile actions after the original post-processing/NMS."""
    pairs = sample_action_pairs(
        context['activity'], context['descriptors'].shape[0], settings['actions_per_view'], seed,
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
            'descriptor': context['descriptors'][peer-1, :, row, col].detach().cpu().half(),
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
