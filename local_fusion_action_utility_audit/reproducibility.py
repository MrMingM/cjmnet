"""Exact per-frame inference snapshots and numeric baseline reproduction checks."""
import hashlib
from pathlib import Path
import shutil
import numpy as np
from .common import (KEEP, WEATHERS, Runtime, atomic_json, atomic_torch,
                     load_cache, parser, read_json)

NUMERIC_ATOL = 2e-5
NUMERIC_RTOL = 2e-5
AP_ATOL = 1e-6
POLICY = {'schema': 2, 'method': 'exact_inference_snapshot',
          'score_atol': NUMERIC_ATOL, 'score_rtol': NUMERIC_RTOL, 'ap_atol': AP_ATOL,
          'evidence': 'same-input spectral reduction drift; unchanged TP/FP; existing prepare_context tolerance',
          'selection': 'fixed diagnostic engineering tolerance; no validation optimization'}


def numeric_difference(left, right):
    """Measure repeated inference noise; artifact integrity still uses SHA256."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return {'same_shape': False, 'left_shape': list(left.shape), 'right_shape': list(right.shape)}
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return {'same_shape': True, 'finite': False, 'exact': False, 'allclose_2e5': False}
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {'same_shape': True, 'finite': True, 'exact': bool(np.array_equal(left, right)),
            'max_absolute_difference': float(difference.max()) if difference.size else 0.,
            'mean_absolute_difference': float(difference.mean()) if difference.size else 0.,
            'different_elements': int(np.count_nonzero(difference)),
            # Same tolerance already used by prepare_context's implementation check.
            'allclose_2e5': bool(np.allclose(left, right, atol=NUMERIC_ATOL, rtol=NUMERIC_RTOL))}


def tensor_tree_hash(value):
    """Include field names, shape, dtype and bytes in stable recursive order."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                result.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                result.update(str(index).encode())
                visit(child)
        elif hasattr(item, 'detach'):
            array = item.detach().cpu().contiguous().numpy()
            result.update(str((array.dtype, array.shape)).encode())
            result.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            result.update(str((item.dtype, item.shape)).encode())
            result.update(item.tobytes())
        else:
            result.update(repr(item).encode())

    visit(value)
    return result.hexdigest()


def stats_difference(expected, actual):
    result = {}
    for threshold in (.3, .5, .7):
        before, after = expected[threshold], actual[threshold]
        result[str(threshold)] = {
            'same_gt_count': before['gt'] == after['gt'],
            'same_tp_sequence': before['tp'] == after['tp'],
            'same_fp_sequence': before['fp'] == after['fp'],
            'scores': numeric_difference(before['score'], after['score']),
        }
    return result


def assert_frame_reproduction(expected, actual):
    result = stats_difference(expected, actual)
    for threshold, value in result.items():
        scores = value['scores']
        if (not all(value[key] for key in ('same_gt_count', 'same_tp_sequence', 'same_fp_sequence'))
                or not scores.get('allclose_2e5', False)):
            raise RuntimeError(f'Shared detection consequences changed at IoU {threshold}: {value}')
    return result


def assert_ap_reproduction(expected, actual):
    differences = {}
    for metric in ('ap30', 'ap50', 'ap70'):
        difference = abs(actual[metric] - expected[metric])
        if not np.isfinite(difference) or difference > AP_ATOL:
            raise RuntimeError(f'Shared AP reproduction failed {metric}: {difference}')
        differences[metric] = difference
    return differences


def inference_input_hash(batch, weather):
    branch = 'processed_lidar' if weather == 'clean' else 'processed_lidar_weather'
    return tensor_tree_hash({key: batch['ego'][key] for key in
                             (branch, 'record_len', 'communication_transforms',
                              'anchor_box', 'transformation_matrix')})


def reference_path(run, weather, frame):
    return Path(run) / 'cache' / 'inference_reference' / weather / f'{int(frame):08d}.pt'


def load_reference(runtime, batch, weather, frame):
    from opencood.tools.train_utils import to_device
    if not runtime.manifest.complete('reference-frame', 'validation', weather, frame):
        raise RuntimeError('Missing exact inference reference for ' + str(frame))
    reference = load_cache(reference_path(runtime.run, weather, frame))
    if reference['input_hash'] != inference_input_hash(batch, weather):
        raise RuntimeError('Replay input changed; check loader/environment before using cached inference')
    if reference['weather'] != weather or reference['frame'] != int(frame):
        raise RuntimeError('Inference reference frame/weather mismatch')
    return to_device(reference['inference_pool'], runtime.target), reference


def build_references(run):
    """Cheap validation pass before labels: save exact FP32 outputs, verify AP."""
    import torch
    from ceif_audit.scoring import ap_values, empty_stats, merge
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from .s0_counterfactual import frame_ap_stats, prediction_hash
    runtime = Runtime(run)
    if not runtime.manifest.complete('baseline'):
        raise RuntimeError('Original complete B0 baseline reproduction is required')
    results = {}
    disk_checked = False
    with torch.no_grad():
        for weather in WEATHERS:
            folder = reference_path(run, weather, 0).parent
            folder.mkdir(parents=True, exist_ok=True)
            summary_path = folder / 'summary.json'
            if runtime.manifest.complete('reference-weather', 'validation', weather):
                results[weather] = read_json(summary_path)
                continue
            dataset, loader = runtime.loader('validation', weather)
            stats, seen = empty_stats(), []
            with runtime.manifest.work('reference-weather', 'validation', weather):
                for ordinal, batch in enumerate(loader):
                    index = int(batch['ego']['communication_sample_index'][0])
                    seen.append(index)
                    if index != runtime.protocol['validation_indices'][ordinal]:
                        raise RuntimeError('Reference loader order differs from B0')
                    path = reference_path(run, weather, index)
                    if runtime.manifest.complete('reference-frame', 'validation', weather, index):
                        reference = load_cache(path)
                    else:
                        with runtime.manifest.work('reference-frame', 'validation', weather, index):
                            batch = to_device(batch, runtime.target)
                            before = inference_input_hash(batch, weather)
                            context, shared, pool = runtime.predict(batch, weather, verify=ordinal == 0)
                            post = dataset.post_process(batch, {'ego': shared})
                            current_stats = frame_ap_stats(post)
                            if not runtime.manifest.complete('baseline-frame', 'validation', weather, index):
                                raise RuntimeError('Missing original baseline frame')
                            baseline = load_cache(runtime.run / 'cache' / 'baseline' / weather / f'{index:08d}.pt')
                            check = assert_frame_reproduction(baseline['stats'], current_stats)
                            cpu_pool = {name: {key: tensor.detach().cpu().clone() for key, tensor in prediction.items()}
                                        for name, prediction in pool.items()}
                            if before != inference_input_hash(batch, weather):
                                raise RuntimeError('Inference mutated frame inputs')
                            if not disk_checked:
                                bytes_one = sum(t.numel() * t.element_size() for t in cpu_pool[KEEP].values())
                                maximum_sources = max(len(cpu_pool), 1 + 2 * int(
                                    runtime.hypes.get('train_params', {}).get('max_cav', 7)))
                                remaining = sum(not runtime.manifest.complete('reference-frame', 'validation', w, i)
                                                for w in WEATHERS for i in runtime.protocol['validation_indices'])
                                estimated = bytes_one * maximum_sources * remaining
                                free = shutil.disk_usage(runtime.run).free
                                print(f'Exact inference snapshot storage estimate <= {estimated / 2**30:.2f} GiB; free={free / 2**30:.2f} GiB', flush=True)
                                if free < estimated + 512 * 2**20:
                                    raise RuntimeError('Insufficient disk space for exact FP32 inference snapshots')
                                disk_checked = True
                            reference = {'schema': 2, 'frame': index, 'weather': weather,
                                         'input_hash': before, 'inference_pool': cpu_pool,
                                         'prediction_hash': prediction_hash(shared),
                                         'evaluation_only': {'stats': current_stats, 'reproduction': check,
                                             'original_raw_hash_equal': prediction_hash(shared) == baseline['prediction_hash']}}
                            atomic_torch(path, reference)
                            del context, shared, pool, cpu_pool, post
                        runtime.manifest.mark('reference-frame', 'complete', 'validation', weather, index, [path])
                    merge(stats, reference['evaluation_only']['stats'])
                    if ordinal == 0 or (ordinal + 1) % 10 == 0:
                        print(f'validation-reference {weather} {ordinal + 1}/{len(loader)}', flush=True)
                if seen != runtime.protocol['validation_indices']:
                    raise RuntimeError('Incomplete validation reference pass')
                ap = ap_values(stats, eval_utils)
                expected = runtime.protocol['b0_saved_results']['conditions'][weather]['results']['Shared']
                differences = assert_ap_reproduction(expected, ap)
                results[weather] = {'Shared': ap, 'absolute_difference_vs_B0': differences, 'frames': len(seen)}
                atomic_json(summary_path, results[weather])
            runtime.manifest.mark('reference-weather', 'complete', 'validation', weather, artifacts=[summary_path])
            del dataset, loader
    atomic_json(runtime.run / 'validation_reference_results.json', {'policy': POLICY, 'conditions': results})


def main():
    args = parser(__doc__).parse_args()
    build_references(args.run)


if __name__ == '__main__':
    main()
