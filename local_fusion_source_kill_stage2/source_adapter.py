"""Adapters execute the original Oracle code object without global mutation."""
from types import FunctionType
import hashlib
import json
import math
from .common import KEEP, safe_path
from local_fusion_source_kill_stage1.oracle_adapter import KeyPolicy, partition


def make_runtime(protocol):
    from local_fusion_source_kill_stage1.k1_0_oracle_decomposition import make_runtime as original
    # Historical inputs remain preserved in the snapshot; execution uses the
    # verified canonical paths so changing an input symlink cannot redirect IO.
    runtime = original({**protocol['stage1_protocol'], 'inputs':protocol['inputs']})
    for key in ('root_dir','validate_dir'):
        if str(safe_path(runtime[3][key], dataset=True)) != protocol['dataset_roots'][key]:
            raise ValueError('Runtime dataset root differs or resolves to test: '+key)
    return runtime


def inference_input_hash(ego):
    """Fingerprint model inputs only, excluding all GT/evaluation fields."""
    result=hashlib.sha256()
    def visit(value):
        if isinstance(value,dict):
            for key in sorted(value):
                result.update(str(key).encode())
                visit(value[key])
        elif isinstance(value,(list,tuple)):
            for row in value:
                visit(row)
        elif hasattr(value,'detach'):
            array=value.detach().cpu().contiguous().numpy()
            result.update(str((array.dtype,array.shape)).encode())
            result.update(array.tobytes())
        elif hasattr(value,'dtype') and hasattr(value,'tobytes'):
            result.update(str((value.dtype,value.shape)).encode())
            result.update(value.tobytes())
        else:
            result.update(json.dumps(value,sort_keys=True).encode())
    for name in ('processed_lidar','processed_lidar_weather','record_len','communication_transforms',
                 'evidence_clouds','evidence_weather_clouds'):
        if name in ego:
            result.update(name.encode())
            visit(ego[name])
    return result.hexdigest()


def run_with_hooks(b, variant, args, on_target=None):
    policy = KeyPolicy(b._action_key, variant)
    holder = {}
    def state(post, target_index, threshold):
        holder['target_index'] = target_index
        return b._state(post, target_index, threshold)
    def proxy(dataset, batch, prediction):
        holder['current'] = prediction
        return b._proxy_cache(dataset, batch, prediction)
    def shortlist(*values, **kwargs):
        rows, info = b._shortlist_actions(*values, **kwargs)
        if on_target is not None:
            on_target(holder['target_index'], holder['current'], rows, info)
        return rows, info
    original = b.run_oracle
    scope = dict(original.__globals__, _action_key=policy, _state=state,
                 _proxy_cache=proxy, _shortlist_actions=shortlist)
    adapted = FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
    prediction, post, records = adapted(*args)
    if len(policy.targets) != len(records):
        raise ValueError('Oracle callback alignment lost')
    for record, audit in zip(records, policy.targets):
        audit['target_index'] = record['target_index']
        audit['shortlisted'] = dict(audit['shortlisted'])
    return prediction, post, records, policy.targets


def original_args(b, spec, dataset, batch, model, levels, shared, pool, proxy, lidar_range, mode):
    return (dataset, batch, model.engine.base, levels, shared, pool, proxy, lidar_range,
            spec['roi_expansions'], mode, spec['target_iou'], spec['shortlist_k'])


def frame_stats(b, post):
    stats = b.empty_stats()
    for threshold in stats:
        b.eval_utils.caluclate_tp_fp(*post, stats, threshold)
    return {str(t): row for t, row in stats.items()}


def total_ap(b, frames):
    stats = b.empty_stats()
    for frame in frames:
        for t in stats:
            row = frame['stats'][str(t)]
            stats[t]['gt'] += row['gt']
            for key in ('tp', 'fp', 'score'):
                stats[t][key].extend(row[key])
    return b.ap_values(stats, b.eval_utils)


def action_counts(records, audits):
    from local_fusion_source_kill_stage1.oracle_adapter import action_statistics
    return action_statistics(records, audits)


def assert_shortlist_contract(records, audits, pool_size, expansions, mode, k):
    expected = len(expansions) * (pool_size-1 if mode == 'same' else pool_size**2-1)
    for record, audit in zip(records, audits):
        info = record['shortlist']
        if info['all_proxy_actions'] != expected or info['shortlisted_actions'] != len(audit['actions']):
            raise ValueError('Proxy action count / complete shortlist fairness mismatch')
        if info['shortlisted_actions'] > (k if mode == 'same' else 3*k):
            raise ValueError('Original shortlist capacity differs')
        if any(row['cls_source'] == row['reg_source'] == KEEP for row in audit['actions']):
            raise ValueError('KEEP cannot be counted as a non-KEEP shortlisted candidate')


def masks_at_multiplier(anchors, corners, levels, lidar_range, expansion):
    """Allow the preregistered 0.97 ROI without modifying the old >=1 guard.

    Scaling the footprint by e, followed by original expansion=1, selects the
    same physical region as expansion=e. GT passed to evaluation stays fixed.
    """
    import numpy as np
    from qa_local_intervention.operators import masks_for_frame
    if not math.isfinite(expansion) or expansion<=0:
        raise ValueError('ROI multiplier must be finite and positive')
    if expansion>=1:
        return masks_for_frame(anchors,corners,levels,lidar_range,expansion)
    scaled=np.asarray(corners,dtype=np.float64).copy()
    center=scaled[:4,:2].mean(0)
    scaled[:,:2]=center+(scaled[:,:2]-center)*expansion
    head,context,stats=masks_for_frame(anchors,scaled,levels,lidar_range,1.)
    stats=dict(stats,effective_roi_multiplier=float(expansion),adapter='scaled footprint; original masks expansion=1')
    return head,context,stats


def evaluate_target(b, dataset, batch, current, pool, proxy_pool, gt_corners,
                    mask_rows, expansions, mode, target_index, threshold=.7, k=6):
    """Real post-NMS outcomes in ONE immutable baseline current state."""
    post = dataset.post_process(batch, {'ego': current})
    before = b._state(post, target_index, threshold)
    keep_key = b._action_key(before, before, KEEP, KEEP)
    shortlist, info = b._shortlist_actions(b._proxy_cache(dataset, batch, current), proxy_pool,
        gt_corners, mask_rows, tuple(sorted(pool)), expansions, mode, threshold, k)
    actions = [dict(cls_source=KEEP, reg_source=KEEP, expansion=None,
                    key=list(keep_key), group='KEEP')]
    for row in shortlist:
        prediction = b.compose_prediction(current, pool, row['mask'], row['cls_source'], row['reg_source'])
        after = b._state(dataset.post_process(batch, {'ego': prediction}), target_index, threshold)
        key = b._action_key(before, after, row['cls_source'], row['reg_source'])
        actions.append(dict(cls_source=row['cls_source'], reg_source=row['reg_source'],
            expansion=row['expansion'], key=list(key), group=partition(key, keep_key)))
    return dict(actions=actions, shortlist=info, keep_key=list(keep_key))


def reproduction(b, actual, expected, label):
    from local_fusion_source_kill_stage1.k1_0_oracle_decomposition import reproduction_gate
    try:
        return reproduction_gate(actual, expected, label)
    except RuntimeError as exc:
        raise RuntimeError('REPRODUCTION_FAIL: ' + str(exc)) from exc
