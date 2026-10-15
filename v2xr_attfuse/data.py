"""Verify extracted payload, labels and paired views without changing source data."""
from __future__ import annotations

import math
from pathlib import Path
import re

from reference_weather.common import atomic_json, digest, object_hash, read_json
from v2xr_subset.prepare import pcd_header, verify_sources as subset_sources


def vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f'{name} must have {count} elements')
    if not all(math.isfinite(float(x)) for x in value):
        raise ValueError(f'{name} contains nonfinite values')


def validate_annotation(value):
    if not isinstance(value, dict):
        raise ValueError('Annotation must be a mapping')
    vector(value.get('lidar_pose'), 6, 'lidar_pose')
    if 'ego_speed' not in value or not math.isfinite(float(value['ego_speed'])):
        raise ValueError('Missing/invalid ego_speed required by the existing dataset')
    vehicles = value.get('vehicles')
    if not isinstance(vehicles, dict):
        raise ValueError('Expected vehicles mapping in CARLA world coordinates')
    for object_id, obj in vehicles.items():
        if not isinstance(obj, dict):
            raise ValueError(f'vehicles[{object_id}] must be a mapping')
        for key in ('location', 'angle', 'center', 'extent'):
            vector(obj.get(key), 3, f'vehicles[{object_id}].{key}')
        if any(float(x) <= 0 for x in obj['extent']):
            raise ValueError(f'Nonpositive bounding-box half extents: {object_id}')
    return len(vehicles)


def frame_indices(keys, per_scene):
    from v2xr_subset.prepare import select_stamps
    scenes = {}
    for index, (scene, stamp) in enumerate(keys):
        scenes.setdefault(scene, []).append((stamp, index))
    selected = []
    for items in scenes.values():
        wanted = set(select_stamps([t for t, _ in items], per_scene))
        selected.extend(index for stamp, index in items if stamp in wanted)
    return selected


def verify_subset(root, destination):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    root = Path(root).resolve(strict=True)
    plan, manifest = read_json(root / 'plan.json'), read_json(root / 'manifest.json')
    p = plan['protocol']
    if (plan['identity'] != object_hash(p) or manifest['identity'] != plan['identity']
            or manifest['status'] != 'complete' or p['split'] != 'validate'
            or p['output_root'] != str(root) or p['sources'] != subset_sources()):
        raise ValueError('Subset source/plan/complete status mismatch')
    if len(p['frame_keys']) != p['selected_cooperative_frames']:
        raise ValueError('Prepared frame count mismatch')
    all_keys = [tuple(k) for k in p['frame_keys']]
    if all_keys != sorted(set(all_keys)):
        raise ValueError('Prepared frame keys are duplicated or unordered')
    report = dict(status='in_progress', subset_identity=plan['identity'],
        cooperative_frames=len(all_keys), variants=p['selected_variants'],
        verified_payload_files=0, point_counts=dict(clean=0, fog=0, snow=0),
        empty_clouds=dict(clean=0, fog=0, snow=0), annotations=0, scene_agents={},
        label_schema='lidar_pose: x,y,z,roll,yaw,pitch; vehicles: world location, angle, center, half extent',
        source_subset_package=p['sources'])
    sample_headers, annotation_examples, seen = {}, [], set()
    try:
        for index, row in enumerate(p['archives'], 1):
            scene, cav = row['scene'], row['cav']
            if (not re.fullmatch(r'[A-Za-z0-9_-]+', scene) or not re.fullmatch(r'-?\d+', cav)
                    or (scene, cav) in seen):
                raise ValueError('Invalid or duplicated scene/agent entry')
            seen.add((scene, cav))
            report['scene_agents'].setdefault(scene, []).append(cav)
            payload = root / 'payload' / 'validate' / scene / cav
            record = read_json(root / 'records' / 'validate' / scene / (cav + '.json'))
            if record['identity'] != plan['identity'] or set(record['files']) != {m['name'] for m in row['members']}:
                raise ValueError('Extraction record identity/member selection mismatch')
            actual_stamps = set()
            for member in row['members']:
                name, stamp, mode = member['name'], member['timestamp'], member['condition']
                if (Path(name).name != name or not stamp.isdigit()
                        or mode not in ('yaml', 'clean', 'fog', 'snow')):
                    raise ValueError('Invalid selected filename/timestamp/condition')
                source = payload / name
                if source.is_symlink() or source.stat().st_size != member['size'] or digest(source) != record['files'][name]:
                    raise ValueError(f'Extracted file hash/size mismatch: {source}')
                report['verified_payload_files'] += 1
                modes = ('clean', 'fog', 'snow') if mode == 'yaml' else (mode,)
                for condition in modes:
                    suffix = '.yaml' if mode == 'yaml' else '.pcd'
                    view = root / condition / 'validate' / scene / cav / (stamp + suffix)
                    if not view.is_symlink() or view.resolve(strict=True) != source.resolve(strict=True):
                        raise ValueError(f'Weather view link mismatch: {view}')
                if mode == 'yaml':
                    # Same loader as the actual dataset, including any NumPy/tuple
                    # serialization used by the author's trusted annotations.
                    annotation = load_yaml(str(source))
                    count = validate_annotation(annotation)
                    report['annotations'] += 1
                    actual_stamps.add(stamp)
                    if len(annotation_examples) < 3:
                        annotation_examples.append(dict(path=str(source), top_keys=list(annotation),
                            lidar_pose=annotation['lidar_pose'], vehicles=count))
                else:
                    header = pcd_header(source)
                    fields = {line.split()[0]: line.split()[1:] for line in header if line and not line.startswith('#')}
                    if fields.get('FIELDS') != ['x', 'y', 'z', 'rgb']:
                        raise ValueError(f'Unverified intensity encoding in {source}')
                    n = int(fields['POINTS'][0])
                    if n < 0:
                        raise ValueError('Negative PCD point count')
                    report['point_counts'][mode] += n
                    report['empty_clouds'][mode] += n == 0
                    sample_headers.setdefault(mode, dict(path=str(source), header=header))
            wanted = {t for s, t in all_keys if s == scene}
            if actual_stamps != wanted:
                raise ValueError(f'Agent timestamps differ from planned frames: {scene}/{cav}')
            atomic_json(destination, report)
            print(f'Verify V2X-R: {index}/{len(p["archives"])} agents; '
                  f'{report["verified_payload_files"]} files', flush=True)
        if any(len(v) > 5 for v in report['scene_agents'].values()):
            raise ValueError('Prepared scene has more than the checkpoint maximum of five agents')
        report.update(status='complete', sample_headers=sample_headers,
            annotation_examples=annotation_examples,
            point_count_ratio_to_clean={k: v / report['point_counts']['clean']
                for k, v in report['point_counts'].items()} if report['point_counts']['clean'] else {})
        atomic_json(destination, report)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        atomic_json(destination, report)
        raise
    return plan, report


class SelectedFrames:
    def __init__(self, dataset, indices, keys, seed):
        self.dataset, self.indices, self.keys, self.seed = dataset, indices, keys, seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        import numpy as np
        import random
        from reference_weather.common import stable_seed
        original = self.indices[index]
        seed = stable_seed(self.seed, 'v2xr-eval-frame', original)
        np.random.seed(seed)
        random.seed(seed)
        try:
            return self.dataset[original]
        except Exception as error:
            raise RuntimeError(f'V2X-R reader failed at global index {original}, frame={self.keys[original]}: {error}') from error
