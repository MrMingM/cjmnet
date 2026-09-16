"""Cached, serialized A0B0 messages and receiver-side absence interventions."""
from dataclasses import dataclass
import torch
from gspr_communication import codec
from gspr_communication.selection import hard_topk
from gspr_communication.masked_attfuse import fuse


@dataclass
class Messages:
    levels: list
    mask: torch.Tensor
    request: bytes
    responses: dict
    total_bytes: int


@torch.no_grad()
def capture(model, encoded):
    if model.training or model.variant != 'a0b0':
        raise ValueError('Capture requires evaluation-mode A0B0')
    levels, sem, obs, conf = encoded
    n = len(conf)
    active, quotas = codec.allocate(model.budget, n-1, *model.grid, model.cost)
    request = model._request(sem[:1], obs[:1], conf[:1])
    packet = codec.pack_request(request[0, 0].cpu().numpy()) if active else b''
    request = torch.as_tensor(codec.unpack_request(packet), device=conf.device,
                              dtype=conf.dtype)[None, None] if active else torch.zeros_like(request)
    masks = [torch.ones_like(request)]
    received = [[x[:1]] for x in levels]
    responses = {}
    for peer, k in enumerate(quotas, 1):
        score = model._score(sem[peer:peer+1], obs[peer:peer+1], conf[peer:peer+1], request)
        mask = hard_topk(score, k)
        masks.append(mask)
        if active:
            ids = mask.flatten().nonzero().flatten().cpu().numpy()
            responses[peer] = codec.pack_response([x[peer].cpu().numpy() for x in levels],
                                                  ids, model.grid, model.value_bytes)
            arrays, decoded_ids = codec.unpack_response(responses[peer],
                [tuple(x.shape[1:]) for x in levels], model.grid)
            assert list(decoded_ids) == list(ids)
            for target, array in zip(received, arrays):
                target.append(torch.as_tensor(array, device=conf.device, dtype=conf.dtype)[None])
        else:
            for target, x in zip(received, levels):
                target.append(torch.zeros_like(x[peer:peer+1]))
    total = (n-1)*len(packet) + sum(map(len, responses.values()))
    assert total <= model.budget
    return Messages([torch.cat(x) for x in received], torch.cat(masks), packet, responses, total)


def drop_mask(original, pairs):
    mask = original.clone()
    for peer, block in pairs:
        if peer <= 0 or peer >= len(mask) or block < 0 or block >= mask[peer].numel():
            raise ValueError('Invalid sender/block; ego cannot be removed')
        if not original[peer].flatten()[block]:
            raise ValueError('Cannot delete an untransmitted block')
        mask[peer].view(-1)[block] = 0
    return mask


@torch.no_grad()
def predict(model, messages, mask=None):
    mask = messages.mask if mask is None else mask
    if mask.shape != messages.mask.shape or not torch.all(mask[0] == 1):
        raise ValueError('Mask must preserve ego and original shape')
    if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask > messages.mask):
        raise ValueError('Only deletion of received blocks is supported')
    fused = [fuse(x, mask) for x in messages.levels]
    joined = torch.cat([d(x) for d, x in zip(model.base.backbone.deblocks, fused)], 1)
    return {'psm': model.base.cls_head(joined), 'rm': model.base.reg_head(joined)}


def groups(mask, side=1):
    """Group actual selected native blocks in fixed ego-grid squares, per sender."""
    if side < 1:
        raise ValueError('region-blocks must be positive')
    w = mask.shape[-1]
    result = {}
    for peer in range(1, len(mask)):
        for block in mask[peer].flatten().nonzero().flatten().tolist():
            key = (peer, (block//w)//side, (block % w)//side)
            result.setdefault(key, []).append((peer, block))
    return list(result.values())


def random_deletions(mask, harmful_pairs, seed):
    """Match removed block counts/bytes per sender, without using region losses."""
    generator = torch.Generator().manual_seed(seed)
    pairs = []
    for peer in range(1, len(mask)):
        count = sum(p == peer for p, _ in harmful_pairs)
        ids = mask[peer].flatten().nonzero().flatten().cpu()
        pairs.extend((peer, int(ids[i])) for i in torch.randperm(len(ids), generator=generator)[:count])
    return pairs


def loss_values(criterion, prediction, labels):
    if any(not torch.isfinite(x).all() for x in prediction.values()):
        raise RuntimeError('Nonfinite detector output')
    criterion(prediction, labels)
    values = {k: float(v) for k, v in criterion.loss_dict.items()}
    import math
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError('Nonfinite detection loss')
    return values


def classify(delta, epsilon):
    return 'harmful' if delta > epsilon else 'beneficial' if delta < -epsilon else 'neutral'
