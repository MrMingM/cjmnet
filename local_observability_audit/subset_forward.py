"""Frozen true subset forwards; cached reroute is a diagnostic, never the outcome source."""
import itertools
import numpy as np

def plans(n):
    return [(0,)] + [(0,i) for i in range(1,n)] + [(0,i,j) for i,j in itertools.combinations(range(1,n),2)]

def subset_input(processed, selected):
    import torch
    if not selected or selected[0] != 0 or len(set(selected)) != len(selected):
        raise ValueError('Subset must preserve ego first and unique sources')
    coords = processed['voxel_coords']
    mask = torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
    for i in selected:
        mask |= coords[:,0] == i
    out = {k: v[mask].clone() for k,v in processed.items()}
    for new, old in enumerate(selected):
        out['voxel_coords'][coords[mask,0] == old,0] = new
    return {'processed_lidar': out,
            'record_len': torch.tensor([len(selected)], device=coords.device, dtype=torch.long)}

def capture_full(model, inputs):
    captured, levels, handles = {}, {}, []
    def capture_gspr(_m, _i, output):
        if captured:
            raise ValueError('Unexpected repeated GSPR call')
        captured.update({k: v.detach().clone() for k,v in output.items()})
    handles.append(model.gspr.register_forward_hook(capture_gspr))
    for i, module in enumerate(model.backbone.fuse_modules):
        def capture_level(_m, args, index=i):
            levels[index] = args[0].detach().clone()
        handles.append(module.register_forward_pre_hook(capture_level))
    try:
        output = model(inputs)
    finally:
        for h in handles:
            h.remove()
    return output, captured, [levels[i] for i in range(len(levels))]

def reroute(model, levels, selected):
    import torch
    b = model.backbone
    count = torch.tensor([len(selected)], device=levels[0].device)
    ups = []
    for i, x in enumerate(levels):
        f = b.fuse_modules[i](x[list(selected)].contiguous(), count)
        ups.append(b.deblocks[i](f) if len(b.deblocks) else f)
    x = torch.cat(ups, 1) if len(ups)>1 else ups[0]
    if len(b.deblocks)>len(b.blocks):
        x = b.deblocks[-1](x)
    return {'psm': model.cls_head(x), 'rm': model.reg_head(x)}

def compare(a, b, cfg):
    import torch
    result = {}
    for k in ('psm','rm'):
        if a[k].shape != b[k].shape or not torch.isfinite(a[k]).all() or not torch.isfinite(b[k]).all():
            raise ValueError('Nonfinite or wrong-shaped detector output')
        result[k] = {'max_abs': float((a[k]-b[k]).abs().max().item()),
                     'allclose': bool(torch.allclose(a[k],b[k],atol=cfg['subset_atol'],rtol=cfg['subset_rtol']))}
    result['accepted'] = all(result[k]['allclose'] for k in ('psm','rm'))
    return result

def all_predictions(model, batch, postprocessor, cfg, verify=False):
    import torch
    processed = batch['ego']['processed_lidar']
    n = int(batch['ego']['record_len'].item())
    observed = sorted(torch.unique(processed['voxel_coords'][:,0]).cpu().tolist())
    if observed != list(range(n)):
        raise ValueError('EMPTY_CAV: frozen scatter cannot preserve all source slots; no silent source removal')
    full_input = subset_input(processed, tuple(range(n)))
    full, reliability, levels = capture_full(model, full_input)
    diagnostics = {'outcome_backend': 'true_subset_forward', 'checks': {}, 'cached_reroute_used': False}
    results = {}
    # The decoder gets only anchors + identity transform, no labels or GT tensors.
    decode_input = {'ego': {k: batch['ego'][k] for k in ('anchor_box','transformation_matrix')}}
    for selected in plans(n):
        key = ','.join(map(str,selected))
        actual = model(subset_input(processed, selected))
        if any(not torch.isfinite(actual[k]).all() for k in ('psm','rm')):
            raise ValueError('Nonfinite true subset output: '+key)
        if verify:
            cached = reroute(model, levels, selected)
            check = compare(cached, actual, cfg)
            diagnostics['checks'][key] = {'cached_vs_true': check}
            repeated = model(subset_input(processed, selected))
            repeat = compare(actual, repeated, cfg)
            diagnostics['checks'][key]['true_repeat'] = repeat
            if not repeat['accepted']:
                raise ValueError('True subset forward not reproducible: '+key)
        corners, scores = postprocessor.post_process(decode_input, {'ego': actual})
        results[key] = {'selected_indices': selected,
            'corners': np.empty((0,8,3)) if corners is None else corners.detach().cpu().numpy(),
            'scores': np.empty(0) if scores is None else scores.detach().cpu().numpy()}
    diagnostics['cache_equivalence'] = (all(x['cached_vs_true']['accepted'] for x in diagnostics['checks'].values())
                                         if verify else None)
    # Full forward reliability is the explicitly fixed audit reference, not claimed source-independent.
    return results, {k:v.cpu().numpy() for k,v in reliability.items()}, diagnostics
