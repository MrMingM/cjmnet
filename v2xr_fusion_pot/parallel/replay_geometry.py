"""CPU-only server replay of cached formal validation predictions."""
import json
import os
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run

from common import atomic_json, identity, read_json, sha256
from evaluate import statistics
from geometry_warnings import checked_intersections
from support import THRESHOLDS, corner_array, load_prediction, prediction_tensors


def verify_failure_path():
    import shapely
    from shapely.geometry import box

    native = shapely.intersection

    def invalid(*args, **kwargs):
        warnings.warn_explicit('invalid value encountered in intersection', RuntimeWarning,
            filename='/test/shapely/set_operations.py', lineno=133, module='shapely.set_operations')
        return None

    try:
        shapely.intersection = invalid
        with checked_intersections():
            try:
                box(0, 0, 1, 1).intersection(box(0, 0, 1, 1))
            except FloatingPointError:
                return
        raise AssertionError('Invalid warned result was accepted')
    finally:
        shapely.intersection = native


def replay(group, epoch):
    import torch
    from opencood.utils import eval_utils_opv2v

    path = run.OUT / group / 'epochs' / f'{epoch:02d}'
    marker = read_json(path / 'complete.json')
    evaluation = path / 'validation'
    summary = read_json(evaluation / 'summary.json')
    for file in (evaluation / 'summary.json',):
        if sha256(file) != marker['artifacts'][str(file)]:
            raise ValueError('Completed epoch validation changed')
    manifest = read_json(evaluation / 'prediction_manifest.json')['files']
    if len(manifest) != 829:
        raise ValueError('Expected a complete 829-frame validation set')
    stats = statistics()
    with checked_intersections() as counts:
        for entry in manifest:
            prediction = Path(entry['path'])
            if sha256(prediction) != entry['sha256']:
                raise ValueError(f'Cached prediction changed: {prediction}')
            key = read_json(prediction)
            record = load_prediction(prediction, dict(identity=summary['identity']),
                                     'clean', key['index'], key['frame_key'])
            boxes, scores = prediction_tensors(record)
            gt = torch.from_numpy(corner_array(record['official_gt']))
            if boxes is not None:
                for threshold in THRESHOLDS:
                    eval_utils_opv2v.caluclate_tp_fp(boxes, scores, gt, stats, threshold)
    expected = evaluation / 'official_runs/clean/official_result_stat.json'
    if sha256(expected) != summary['artifacts'][str(expected.relative_to(evaluation))]:
        raise ValueError('Official stored statistics changed')
    if identity(stats) != identity(read_json(expected)):
        raise AssertionError(f'{group} replay differs from the original official AP inputs')
    result = dict(frames=len(manifest), epoch=epoch, warning_counts=dict(counts),
                  official_statistics_sha256=sha256(expected),
                  original_O_AP=summary['results']['clean']['AP']['O'], exact_official_statistics=True)
    print(group, json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    if os.environ.get('ROCR_VISIBLE_DEVICES') != '' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Disable visible GPUs for this cached CPU replay')
    verify_failure_path()
    reports = {group: replay(group, epoch) for group, epoch in (('P1', 6), ('P2', 1), ('P3', 1))}
    if not sum(row['warning_counts'].get('validated', 0) for row in reports.values()):
        raise AssertionError('Real validation did not reproduce the warning')
    atomic_json(run.OUT / 'geometry-warning-check' / 'REPLAY_RESULTS.json',
                dict(status='complete', groups=reports, invalid_geometry_rejected=True))
    print('GEOMETRY REPLAY PASSED', flush=True)
