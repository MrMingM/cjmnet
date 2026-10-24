"""Server-only replay of an existing reference-weather smoke, with geometry checks.

Lives outside reference_weather so its frozen generation identity stays unchanged.
Reuses generated PCDs and the recorded configuration/checkpoint. Writes a separate
audit directory; the original metrics and point clouds are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import random
import traceback
import warnings


class GeometryAudit:
    def __init__(self, common_utils, np):
        self.common, self.np = common_utils, np
        self.original_convert = common_utils.convert_format
        self.original_iou = common_utils.compute_iou
        self.polygons = 0
        self.iou_pairs = 0

    def convert(self, boxes):
        if not self.np.isfinite(boxes).all():
            raise ValueError('Nonfinite decoded box coordinates before polygon conversion')
        polygons = self.original_convert(boxes)
        for i, polygon in enumerate(polygons):
            if (not polygon.is_valid or polygon.is_empty
                    or not self.np.isfinite(polygon.area) or polygon.area <= 0):
                from shapely.validation import explain_validity
                raise ValueError(f'Invalid polygon at index {i}: {explain_validity(polygon)}; '
                                 f'area={polygon.area}; WKT={polygon.wkt}')
        self.polygons += len(polygons)
        return polygons

    def iou(self, box, boxes):
        values = self.original_iou(box, boxes)
        if not self.np.isfinite(values).all():
            raise ValueError(f'Nonfinite IoU: {values!r}; selected polygon={box.wkt}')
        if ((values < -1e-6) | (values > 1 + 1e-6)).any():
            raise ValueError(f'IoU outside [0,1]: {values!r}')
        self.iou_pairs += len(values)
        return values

    def __enter__(self):
        self.common.convert_format, self.common.compute_iou = self.convert, self.iou
        return self

    def __exit__(self, *_):
        self.common.convert_format, self.common.compute_iou = self.original_convert, self.original_iou


def point_summary(root, manifest, read_json, np):
    result = {}
    for mode, aggregate in manifest['splits']['validate']['conditions'].items():
        rows = [read_json(p) for p in sorted((root / 'records' / mode / 'validate').rglob('*.json'))]
        fractions = [r['output']['count'] / r['input']['count'] for r in rows if r['input']['count']]
        input_count = aggregate['input_points']
        result[mode] = dict(aggregate=aggregate,
            retained_fraction=aggregate['output_points'] / input_count if input_count else None,
            per_cloud_retained_fraction_min_median_max=(
                [float(np.min(fractions)), float(np.median(fractions)), float(np.max(fractions))]
                if fractions else []),
            per_cloud_mean_output_intensity_min_median_max=(
                [float(np.min([r['output']['mean_intensity'] for r in rows])),
                 float(np.median([r['output']['mean_intensity'] for r in rows])),
                 float(np.max([r['output']['mean_intensity'] for r in rows]))] if rows else []),
            note='Retained count includes weather replacement returns; this is not a target-survival fraction.')
    return result


def run(root):
    import numpy as np
    import torch
    import shapely
    from torch.utils.data import DataLoader
    from attfuse_code.point_pillar_intermediate import PointPillarIntermediate
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import common_utils, eval_utils
    from reference_weather.common import (atomic_json, digest, object_hash, project_hashes,
        read_json, validate_paths, verify_sources)
    from reference_weather.evaluate import SeededFrames
    from reference_weather.generate import verify_dataset
    from reference_weather.run import exclusive_run

    with exclusive_run(root):
        stored = read_json(root / 'manifest.json')
        config = copy.deepcopy(stored['protocol']['config'])
        if Path(config['output_root']).resolve() != root:
            raise ValueError('Dataset was relocated; source identity/path verification is required')
        validate_paths(config)
        limit = stored['protocol']['limit_frames']
        manifest = verify_dataset(config, limit)
        expected = manifest['splits']['validate']['selected_frames']
        if expected > 90:
            raise ValueError('This diagnostic is limited to a smoke of at most 90 frames')
        saved_protocol = read_json(root / 'evaluation' / 'protocol.json')
        baseline_protocol = saved_protocol['protocol']
        if (digest(config['checkpoint']) != baseline_protocol['checkpoint_sha256']
                or digest(config['model_config']) != baseline_protocol['model_config_sha256']):
            raise ValueError('Recorded model configuration/checkpoint changed')
        if not torch.cuda.is_available():
            raise RuntimeError('No visible HIP GPU; set only ROCR_VISIBLE_DEVICES')
        output = root / ('geometry_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        output.mkdir()
        state = dict(status='in_progress', dataset_identity=manifest['identity'],
            audit_source_sha256=digest(__file__), package_sources=verify_sources(),
            launcher_sha256=digest(Path(__file__).with_name('run_reference_weather_audit.sh')),
            project_sources=project_hashes(), checkpoint_sha256=digest(config['checkpoint']),
            model_config_sha256=digest(config['model_config']),
            versions={n: version(n) for n in ('numpy', 'scipy', 'shapely', 'torch')},
            geos_version=getattr(shapely, 'geos_version_string', 'unavailable'),
            gpu_name=torch.cuda.get_device_name(0), expected_frames=expected,
            point_statistics=point_summary(root, manifest, read_json, np), results={},
            warning_counts={}, warning_examples=[], current_frame=None,
            warning_policy='Per-frame default filter; repeated warnings at the same location may be coalesced.')
        destination = output / 'audit.json'
        atomic_json(destination, state)
        print(f'Audit output: {destination}', flush=True)
        print(f"Point statistics: {state['point_statistics']}", flush=True)
        counter = Counter()
        try:
            hypes = load_yaml(config['model_config'])
            if hypes['model']['core_method'] != 'point_pillar_intermediate':
                raise ValueError('Expected the pure AttFuse configuration')
            for key in ('weather_augmentation', 'prediction_distillation', 'DG_params'):
                hypes.pop(key, None)
            hypes['data_augment'] = []
            if 'wild_setting' in hypes:
                hypes['wild_setting']['async'] = False
                hypes['wild_setting']['loc_err'] = False
            model = PointPillarIntermediate(hypes['model']['args'])
            model.load_state_dict(torch.load(config['checkpoint'], map_location='cpu', weights_only=False), strict=True)
            model = model.to('cuda:0').eval()
            def check_heads(module, inputs, result):
                for name in ('psm', 'rm'):
                    if not torch.isfinite(result[name]).all().item():
                        raise ValueError(f'Nonfinite raw detector output: {name}')
            model.register_forward_hook(check_heads)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            with GeometryAudit(common_utils, np) as audit:
                for mode in ['clean', *config['conditions']]:
                    params = copy.deepcopy(hypes)
                    params['root_dir'] = str(Path(config['clean_root']) / 'train')
                    params['validate_dir'] = str(root / mode / 'validate')
                    dataset = build_dataset(params, visualize=False, train=False)
                    actual_keys = [[Path(v['yaml']).parent.parent.name, stamp]
                        for scene in dataset.scenario_database.values()
                        for stamp, v in next(iter(scene.values())).items() if stamp != 'ego']
                    if len(dataset) != expected or actual_keys != manifest['splits']['validate']['frame_keys']:
                        raise ValueError('Audit dataset frame identities/order differ from the original smoke')
                    seed = int(config['seed'])
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    loader = DataLoader(SeededFrames(dataset, seed), batch_size=1,
                        num_workers=int(config['loader_workers']), collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=True, drop_last=False)
                    statistics = {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)}
                    frame_count = 0
                    with torch.inference_mode():
                        for frame_count, batch in enumerate(loader, 1):
                            state['current_frame'] = [mode, frame_count, *actual_keys[frame_count - 1]]
                            with warnings.catch_warnings(record=True) as captured:
                                warnings.simplefilter('default', RuntimeWarning)
                                try:
                                    batch = train_utils.to_device(batch, torch.device('cuda:0'))
                                    boxes, scores, gt = inference_utils.inference_intermediate_fusion(batch, model, dataset)
                                    for name, tensor in (('boxes', boxes), ('scores', scores), ('ground_truth', gt)):
                                        if tensor is not None and not torch.isfinite(tensor).all().item():
                                            raise ValueError(f'Nonfinite final {name}')
                                    for threshold in statistics:
                                        eval_utils.caluclate_tp_fp(boxes, scores, gt, statistics, threshold)
                                finally:
                                    for warning in captured:
                                        key = f'{mode}: {warning.category.__name__}: {warning.message}'
                                        counter[key] += 1
                                        if len(state['warning_examples']) < 20:
                                            state['warning_examples'].append(dict(frame=state['current_frame'],
                                                category=warning.category.__name__, message=str(warning.message),
                                                file=warning.filename, line=warning.lineno))
                            print(f'Audit {mode}: {frame_count}/{expected}; '
                                  f'checked IoU pairs={audit.iou_pairs}; warnings={sum(counter.values())}', flush=True)
                    if frame_count != expected:
                        raise ValueError('Incomplete audit frame count')
                    ap = {f'ap{round(t * 100)}': float(eval_utils.calculate_ap(copy.deepcopy(statistics), t, False)[0])
                          for t in statistics}
                    original = read_json(root / 'evaluation' / mode / 'result.json')
                    if original['identity'] != saved_protocol['identity'] or original['frames'] != expected:
                        raise ValueError('Original result identity/frame count mismatch')
                    deltas = {key: 100 * (value - original['primary_ap'][key]) for key, value in ap.items()}
                    state['results'][mode] = dict(primary_ap=ap, original_primary_ap=original['primary_ap'],
                        replay_minus_original_pp=deltas, frame_order_sha256=object_hash(actual_keys))
                    if any(not np.isfinite(v) for v in ap.values()):
                        raise ValueError('Nonfinite audit AP')
                    state['warning_counts'] = dict(counter)
                    atomic_json(destination, state)
                    print(f'Audit {mode}: AP={ap}; replay-original/pp={deltas}', flush=True)
                state['checked_polygons'] = audit.polygons
                state['checked_iou_pairs'] = audit.iou_pairs
            state['status'] = 'complete'
            state['geometry_checks_passed'] = True
            state['max_replay_difference_pp'] = max(abs(v) for row in state['results'].values()
                for v in row['replay_minus_original_pp'].values())
            state['replay_matches_within_0_01pp'] = state['max_replay_difference_pp'] <= .01
            state['interpretation'] = ('Checks apply to this smoke only. Warnings with finite valid geometry '
                'do not certify the simulator physics or the full dataset.')
        except Exception as error:
            state['status'] = 'failed'
            state['geometry_checks_passed'] = False
            state['error'] = repr(error)
            state['traceback'] = traceback.format_exc()
            raise
        finally:
            state['warning_counts'] = dict(counter)
            atomic_json(destination, state)
            print(f'Audit {state["status"]}: {destination}', flush=True)
        print(f"Geometry valid; replay matches within 0.01pp={state['replay_matches_within_0_01pp']}", flush=True)


if __name__ == '__main__':
    from reference_weather.common import server_only
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/data/cjm/datasets/opv2v-reference-weather-smoke-v1')
    args = parser.parse_args()
    run(Path(args.dataset_root).expanduser().resolve(strict=True))
