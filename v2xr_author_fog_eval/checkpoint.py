"""Safe external-checkpoint parsing and strict tensor-by-tensor loading."""
from collections.abc import Mapping


def unpack(raw, is_tensor):
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError('Checkpoint must be a nonempty parameter dictionary or standard state_dict wrapper')
    if all(isinstance(k, str) and is_tensor(v) for k, v in raw.items()):
        state, wrapper, ignored = dict(raw), None, []
    else:
        candidates = [k for k in ('state_dict', 'model_state_dict') if k in raw
                      and isinstance(raw[k], Mapping) and raw[k]
                      and all(isinstance(n, str) and is_tensor(t) for n, t in raw[k].items())]
        if len(candidates) != 1:
            raise ValueError('Unknown/ambiguous checkpoint format; no architecture or key guessing allowed')
        wrapper = candidates[0]
        state, ignored = dict(raw[wrapper]), sorted(map(str, set(raw) - {wrapper}))
    flags = [k.startswith('module.') for k in state]
    if any(flags) and not all(flags):
        raise ValueError('Mixed module. prefixes refused; naming-only equivalence is unproven')
    mapping = {k: k[7:] if all(flags) else k for k in state}
    if len(set(mapping.values())) != len(mapping) or any(not k for k in mapping.values()):
        raise ValueError('Checkpoint key conversion is not bijective')
    return {mapping[k]: v for k, v in state.items()}, dict(
        format='plain_state_dict' if wrapper is None else 'standard_wrapper', wrapper=wrapper,
        unconsumed_wrapper_fields=ignored, module_prefix_removed=all(flags),
        conversion='identity' if not all(flags) else 'uniform module. prefix only; bijective',
        key_mapping=mapping)


def schema(state, expected):
    common = sorted(state.keys() & expected.keys())
    return dict(missing=sorted(expected.keys() - state.keys()),
                unexpected=sorted(state.keys() - expected.keys()),
                tensors=[dict(key=k, checkpoint_shape=list(state[k].shape),
                              model_shape=list(expected[k].shape), checkpoint_dtype=str(state[k].dtype),
                              model_dtype=str(expected[k].dtype),
                              compatible=state[k].shape == expected[k].shape and state[k].dtype == expected[k].dtype)
                         for k in common])


def strict_load(model, state, torch, audit):
    result = schema(state, model.state_dict())
    audit['parameter_comparison'] = result
    if result['missing'] or result['unexpected'] or any(not r['compatible'] for r in result['tensors']):
        raise ValueError('Checkpoint incompatible with official pure LiDAR AttFuse; see checkpoint_audit.json')
    bad = [k for k, t in state.items() if (t.is_floating_point() or t.is_complex())
           and not bool(torch.isfinite(t).all())]
    if bad:
        audit['nonfinite_tensors'] = bad
        raise ValueError('Nonfinite checkpoint tensors refused')
    with torch.no_grad():
        model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    for row in result['tensors']:
        row['loaded_exactly'] = bool(torch.equal(loaded[row['key']].detach().cpu(), state[row['key']]))
    if not all(r['loaded_exactly'] for r in result['tensors']):
        raise ValueError('A loaded tensor differs from the author checkpoint')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    audit['strict'] = True
    audit['all_tensors_loaded_exactly'] = True
    audit['tensor_count'] = len(state)
    return model


def safe_read(torch, path):
    # Never retry with weights_only=False or add arbitrary pickle globals.
    raw = torch.load(path, map_location='cpu', weights_only=True)
    return unpack(raw, torch.is_tensor)
