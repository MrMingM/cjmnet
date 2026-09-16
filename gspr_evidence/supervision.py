"""GT is confined to these training/diagnostic targets, not network inference."""
import torch
import torch.nn.functional as F
from .model import top_indices


def dense_targets(output, labels, grid):
    pos = labels['pos_equal_one'].permute(0, 3, 1, 2).float()
    negative = labels.get('neg_equal_one')
    neg = (1-pos) if negative is None else negative.permute(0, 3, 1, 2).float()
    prob = output['psm'].sigmoid()
    missing = ((1-prob)*pos).amax(1, keepdim=True)
    false_positive = (prob*neg).amax(1, keepdim=True)
    b, a, h, w = pos.shape
    predicted = output['rm'].permute(0, 2, 3, 1).reshape(b, h, w, a, 7)
    target = labels['targets'].reshape_as(predicted)
    delta = predicted-target
    delta = torch.cat([delta[..., :6], torch.sin(delta[..., 6:7])], -1)
    error = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction='none', beta=1/9).mean(-1)
    localization = (error.permute(0, 3, 1, 2)*pos).amax(1, keepdim=True).clamp(0, 1)
    risk = F.adaptive_max_pool2d(torch.cat([missing, false_positive, localization], 1), grid)
    fg = F.adaptive_max_pool2d(pos.amax(1, keepdim=True), grid)
    return risk.detach(), fg.detach()


@torch.no_grad()
def candidate_targets(model, encoded, labels, criterion, seed):
    """Same context for all candidates within a peer group; every candidate costs one block.

    Context is A0B0 with one slot removed from that peer. Labels measure actual
    serialized AttFuse task-loss improvement. They are NOT clean-restoration labels.
    """
    from gspr_communication import codec
    n = len(encoded['obs'])
    query_count = min(model.options['queries'], model.grid[0]*model.grid[1])
    from .protocol import allocate
    active, quotas = allocate(model.engine.budget, n-1, query_count, model.engine.cost)
    if not active:
        return []
    device = encoded['obs'].device
    request = torch.as_tensor(codec.unpack_request(codec.pack_request((1-encoded['obs'][0, 12]).cpu().numpy())), device=device, dtype=encoded['obs'].dtype).flatten()
    masks = model.empty_masks(encoded)
    ranking = []
    for peer, quota in enumerate(quotas, 1):
        order = top_indices(request*encoded['obs'][peer, 12].flatten(), request.numel())
        ranking.append(order)
        masks[peer].view(-1)[order[:quota]] = 1
    generator = torch.Generator(device=device).manual_seed(seed)
    groups = []
    for peer, quota in enumerate(quotas, 1):
        context = masks.clone()
        removed = ranking[peer-1][quota-1]
        context[peer].view(-1)[removed] = 0
        allowed = context[peer].flatten() == 0
        limit = min(int(model.options['teacher_candidates']), int(allowed.sum()))
        # Include original slot, promising unused slots and uniform exploration, no GT sampling.
        promising = ranking[peer-1][quota:quota+max(0, limit//2-1)]
        pool = torch.arange(len(allowed), device=device)[allowed]
        pool = pool[torch.randperm(len(pool), generator=generator, device=device)]
        candidates = []
        for value in torch.cat([removed[None], promising, pool]).cpu().tolist():
            if value not in candidates:
                candidates.append(value)
            if len(candidates) == limit:
                break
        original, _ = model.detect(encoded, context, serialize=True)
        context_loss = float(criterion(original, labels))
        gains, wires = [], []
        for block in candidates:
            test = context.clone()
            test[peer].view(-1)[block] = 1
            prediction, packets = model.detect(encoded, test, serialize=True)
            gains.append(context_loss-float(criterion(prediction, labels)))
            wires.append(sum(map(len, packets)))
        if len(set(wires)) != 1:
            raise AssertionError('Candidate alternatives must use equal feature bytes')
        groups.append(dict(peer=peer, ids=torch.tensor(candidates), gains=torch.tensor(gains),
                           context_loss=context_loss, feature_bytes=wires[0],
                           context_ids=[m.flatten().nonzero().flatten().cpu().tolist() for m in context[1:]]))
    return groups
