"""Generate LR0 predictions and hand them to the frozen O/O*/F replay."""
from contextlib import contextmanager
from collections import Counter
import math
import time

from common import atomic_json, identity, read_json, sha256
from evaluate import replay, statistics
from predict import live_gt_observer
import evaluate as frozen
import support
from runtime import build, loader
from empty_input import empty_slots


@contextmanager
def observe_ground_truth(native, holder):
    processor = native.post_processor
    original = processor.generate_gt_bbx

    def observed(batch):
        gt, rows = live_gt_observer(processor, batch, original)
        holder['rows'] = rows
        return gt

    with support.patched(processor, 'generate_gt_bbx', observed):
        yield


@contextmanager
def valid_nms_candidates(counters):
    """Reject degenerate decoded rectangles before calling the official NMS."""
    import numpy as np
    import torch
    from opencood.utils import box_utils, common_utils

    original = box_utils.nms_rotated

    def nms(boxes, scores, threshold):
        finite = (torch.isfinite(boxes).all(dim=(1, 2)) & torch.isfinite(scores)).cpu().numpy()
        indices = np.flatnonzero(finite)
        corners = boxes[indices].detach().cpu().numpy()
        polygons = common_utils.convert_format(corners)
        valid = np.asarray([p.is_valid and np.isfinite(p.area) and p.area > 0 for p in polygons], dtype=bool)
        indices = indices[valid]
        rejected = len(boxes) - len(indices)
        counters['invalid_nms_candidates'] += rejected
        if not rejected:
            return original(boxes, scores, threshold)
        counters['frames_with_invalid_nms_candidates'] += 1
        return indices[original(boxes[indices], scores[indices], threshold)]

    with support.patched(box_utils, 'nms_rotated', nms):
        yield


def evaluate(model, criterion, config, out, conditions, limit, contract_id, weight_hash, loss=False):
    import torch
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils_opv2v

    destination = out / 'summary.json'
    run_id = identity([contract_id, weight_hash, list(conditions), limit, loss])
    if destination.is_file():
        result = read_json(destination)
        if result['identity'] != run_id or any(sha256(out / p) != h for p, h in result['artifacts'].items()):
            raise ValueError(f'Completed evaluation changed: {out}')
        cache = read_json(out / 'prediction_manifest.json')
        if any(sha256(row['path']) != row['sha256'] for row in cache['files']):
            raise ValueError(f'Completed prediction changed: {out}')
        return result

    model.eval()
    evidence, manifests, performance, artifacts = {'frames': [], 'd1b_artifact_sha256': {}}, {}, {}, []
    frame_keys, val_losses = None, []
    for condition in conditions:
        dataset, _ = build(config, condition=condition, visualize=True)
        keys = dataset.keys[:limit]
        if frame_keys is None:
            frame_keys = keys
            evidence['frames'] = [{'conditions': {}} for _ in keys]
        if keys != frame_keys or len(keys) != limit:
            raise ValueError('Weather evaluation frame order differs')
        contract = dict(identity=run_id, frame_keys=keys)
        raw = statistics()
        location = out / 'official_runs' / condition
        location.mkdir(parents=True, exist_ok=True)
        prediction_root = out / 'predictions' / condition
        original = location / 'official_result_stat.json'
        marker = location / 'complete.json'
        condition_artifacts = [original, location / 'eval_epochNone.yaml']
        if marker.is_file():
            done = read_json(marker)
            if (done['identity'] != run_id or len(done['manifest']['entries']) != limit
                    or any(sha256(out / name) != digest for name, digest in done['artifacts'].items())):
                raise ValueError(f'Completed weather evaluation changed: {condition}')
            manifests[condition] = done['manifest']
            performance[condition] = done['performance']
            val_losses.extend(done['losses'])
            for index, key in enumerate(keys):
                path = prediction_root / f'{index:06d}.json'
                record = support.load_prediction(path, contract, condition, index, key)
                evidence['frames'][index]['conditions'][condition] = dict(final_gt=record['official_gt'])
            artifacts.extend(condition_artifacts + [marker])
            continue
        entries, holder = {}, {}
        empty = Counter()
        geometry = Counter()
        condition_losses = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started, infer_seconds = time.monotonic(), 0.
        with torch.no_grad(), observe_ground_truth(dataset.native, holder), valid_nms_candidates(geometry):
            for index, batch in enumerate(loader(dataset, evaluation=True, limit=limit)):
                empty.update(empty_slots(batch['ego']))
                holder.clear()
                batch = train_utils.to_device(batch, torch.device('cuda:0'))
                torch.cuda.synchronize()
                tick = time.monotonic()
                outputs = {}

                def capture(_, __, output):
                    outputs.update(output)

                hook = model.register_forward_hook(capture) if loss else None
                try:
                    boxes, scores, gt = inference_utils.inference_intermediate_fusion(
                        batch, model, dataset.native)
                finally:
                    if hook is not None:
                        hook.remove()
                torch.cuda.synchronize()
                infer_seconds += time.monotonic() - tick
                if loss:
                    value = float(criterion(outputs, batch['ego']['label_dict']))
                    if not math.isfinite(value):
                        raise ValueError('Nonfinite validation loss')
                    condition_losses.append(value)
                if any(t is not None and not torch.isfinite(t).all() for t in (boxes, scores, gt)):
                    raise ValueError('Nonfinite official prediction or ground truth')
                if boxes is not None:
                    for threshold in support.THRESHOLDS:
                        eval_utils_opv2v.caluclate_tp_fp(boxes, scores, gt, raw, threshold)
                rows = holder['rows']
                evidence['frames'][index]['conditions'][condition] = dict(final_gt=rows)
                key = keys[index]
                record = dict(contract_identity=run_id, condition=condition, index=index, frame_key=key,
                    prediction_is_none=boxes is None, prediction_empty=boxes is not None and not len(boxes),
                    prediction_count=0 if boxes is None else len(boxes),
                    prediction_state='none' if boxes is None else 'empty' if not len(boxes) else 'boxes',
                    prediction_boxes=None if boxes is None else boxes.cpu().tolist(),
                    scores=None if scores is None else scores.cpu().tolist(),
                    box_dtype=None if boxes is None else str(boxes.cpu().numpy().dtype),
                    score_dtype=None if scores is None else str(scores.cpu().numpy().dtype),
                    official_gt=rows)
                record['record_identity'] = identity(record)
                path = prediction_root / f'{index:06d}.json'
                atomic_json(path, record)
                entries[str(index)] = dict(sha256=sha256(path), path=str(path))
                if (index + 1) % 50 == 0 or index + 1 == limit:
                    print(f'EVAL {condition} {index + 1}/{limit}', flush=True)
        if len(entries) != limit:
            raise ValueError('Evaluation did not consume every requested frame')
        performance[condition] = dict(frames=limit, inference_seconds=infer_seconds,
            empty_slots=empty,
            rejected_geometry=geometry,
            inference_fps=limit / infer_seconds, end_to_end_seconds=time.monotonic() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(original, raw)
        ap = eval_utils_opv2v.eval_final_results(raw, str(location))
        manifests[condition] = dict(entries=entries, official_exact_AP=dict(
            zip(('ap30', 'ap50', 'ap70'), map(float, ap))))
        atomic_json(marker, dict(identity=run_id, manifest=manifests[condition],
            performance=performance[condition], losses=condition_losses,
            artifacts={str(p.relative_to(out)): sha256(p) for p in condition_artifacts}))
        val_losses.extend(condition_losses)
        artifacts.extend(condition_artifacts + [marker])

    definitions = dict(support.DEFINITIONS)
    definitions['F'] = dict(definitions['F'],
        gt='LR0 live official dual-modal Clean ground truth, fixed across weather')
    with support.patched(frozen, 'CONDITIONS', tuple(conditions)), support.patched(frozen, 'DEFINITIONS', definitions):
        results, _, _ = replay(out, evidence, contract, manifests)
    for condition, row in results.items():
        row['matches'] = {}
        for protocol in definitions:
            suffix = 'Ostar' if protocol == 'O*' else protocol
            path = out / f'stat_{condition}_{suffix}.json'
            stat = read_json(path)
            row['matches'][protocol] = {
                key: dict(tp=sum(value['tp']), fp=sum(value['fp']), fn=value['gt'] - sum(value['tp']),
                          gt=value['gt']) for key, value in stat.items()}
            artifacts.append(path)
    artifacts += [out / name for name in ('gt_manifest.json', 'prediction_manifest.json',
                                           'per_frame_metrics.jsonl', 'removed_gt_metrics.jsonl')]
    result = dict(status='complete', identity=run_id, frames=limit, conditions=list(conditions),
        checkpoint_sha256=weight_hash, results=results, performance=performance,
        validation_loss=sum(val_losses) / len(val_losses) if val_losses else None,
        artifacts={str(p.relative_to(out)): sha256(p) for p in artifacts})
    atomic_json(destination, result)
    return result


def compare_lidar_gt(out):
    """Compare official object IDs on identical weather/frame keys."""
    from inputs import DATA

    reference = DATA / 'logs/v2xr-dminus1c-protocol-v1/gt_manifest.json'
    mono = {(r['condition'], r['index']): r for r in read_json(reference)['frames']}
    diff = {}
    rows = []
    for r in read_json(out / 'gt_manifest.json')['frames']:
        key = r['condition'], r['index']
        old = mono[key]
        if r['frame_key'] != old['frame_key']:
            raise ValueError(f'LiDAR-only comparison frame key differs: {key}')
        stats = diff.setdefault(key[0], dict(lidar_gt=0, lr0_gt=0, lidar_only_ids=0,
                                            lr0_only_ids=0, frames_with_difference=0))
        a, b = set(old['official_ids']), set(r['official_ids'])
        stats['lidar_gt'] += len(a)
        stats['lr0_gt'] += len(b)
        stats['lidar_only_ids'] += len(a - b)
        stats['lr0_only_ids'] += len(b - a)
        stats['frames_with_difference'] += a != b
        rows.append(dict(condition=key[0], frame_key=r['frame_key'],
                         lidar_only=sorted(a - b), lr0_only=sorted(b - a)))
    atomic_json(out / 'gt_comparison.json', dict(reference_sha256=sha256(reference), totals=diff, frames=rows))
    return diff

