"""Observe real official reads. Sampling, augmentation and GT functions are untouched."""
import hashlib
import importlib
from pathlib import Path
import random

from auditlib import identity, no_test, patched, sha256


def pair_key(path):
    path = Path(path)
    name = path.name
    if name.endswith('_fog_0.060.pcd'):
        stamp, condition = name[:-len('_fog_0.060.pcd')], 'fog'
    elif name.endswith('.pcd') and '_' not in name[:-4]:
        stamp, condition = name[:-4], 'clean'
    else:
        raise ValueError('Unexpected training LiDAR filename: ' + str(path))
    return (path.parent.parent.name, path.parent.name, stamp), condition


def paired_inventory(manifest, train_root):
    root = Path(train_root) / 'train'
    rows = {r['relative']: r for r in manifest['files']}
    pairs = []
    keys = []
    for scene in manifest['scenes']:
        for stamp in scene['timestamps']:
            keys.append([scene['scene'], stamp])
            for cav in scene['agents']:
                prefix = scene['scene'] + '/' + cav + '/' + stamp
                clean, fog = prefix + '.pcd', prefix + '_fog_0.060.pcd'
                if clean not in rows or fog not in rows:
                    raise ValueError('Missing mandatory Clean/Fog pair: ' + prefix)
                if pair_key(clean)[0] != pair_key(fog)[0]:
                    raise ValueError('Mismatched scene/CAV/timestamp')
                for rel in (clean, fog):
                    no_test(root / rel)
                pairs.append(dict(key=[scene['scene'], cav, stamp], clean=clean, fog=fog,
                    clean_sha256=rows[clean]['sha256'], fog_sha256=rows[fog]['sha256']))
    return dict(frames=keys, pairs=pairs, identity=identity(pairs))


def tensor_digest(value):
    """Hash actual post-augmentation voxel/label tensors without consuming RNG."""
    import numpy as np
    import torch
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, dict):
            for key in sorted(x):
                h.update(str(key).encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
        elif isinstance(x, (np.ndarray, torch.Tensor)):
            a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            h.update(str((a.shape, a.dtype)).encode())
            h.update(a.tobytes())
        elif x is not None:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


class AuditedDataset:
    def __init__(self, dataset, manifest, mode='mixed'):
        self.dataset, self.mode = dataset, mode
        root = Path(dataset.params['root_dir'])
        self.expected_lidar = {str(no_test(root / row['relative'])): row['sha256'] for row in manifest['files']
            if row['relative'].endswith('.pcd') and not row['relative'].endswith(('_radar.pcd', '_snow_2.5_2.0.pcd'))}
        self.allowed = set()
        self.keys = []
        for i, scene in enumerate(manifest['scenes']):
            content = dataset.scenario_database[i]
            if list(content) != scene['agents'][:7] or not content[scene['ego']]['ego']:
                raise ValueError('Official scene/CAV order differs from accepted extraction')
            self.keys += [[scene['scene'], t] for t in scene['timestamps']]
            for cav, records in content.items():
                if [t for t in records if t != 'ego'] != scene['timestamps']:
                    raise ValueError('Official timestamp alignment differs')
                for stamp in scene['timestamps']:
                    self.allowed.add((scene['scene'], cav, stamp))
        if len(dataset) != len(self.keys) or dataset.max_cav != 7 or not dataset.proj_first:
            raise ValueError('Official frame count/max_cav/proj_first differs')
        if mode == 'mixed' and (not dataset.train or not dataset.train_sim or dataset.sim_weather != '_fog_0.060'):
            raise ValueError('Mixed dataset must execute official train_sim branch')
        if mode == 'fog' and (dataset.train or not dataset.eval_sim or dataset.data_augmentor.train):
            raise ValueError('Fog calibration requires inference dataset with augmentation disabled')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import numpy as np
        from opencood.utils import pcd_utils
        reads, gt_calls, native_filter, augmentation_calls = [], [], [], []
        original = pcd_utils.pcd_to_np
        retrieve = self.dataset.retrieve_base_data
        get_single = self.dataset.get_item_single_car
        native_module = importlib.import_module(type(self.dataset).__module__)
        native_points = native_module.points_in_boxes_cpu
        augment_forward = self.dataset.data_augmentor.forward
        before = identity(dict(python=random.getstate(), numpy=np.random.get_state()))
        def read(path, *a, **kw):
            original_path = path  # Open3D's native IO receives the original str, never our audit Path.
            if kw.get('get_weather', False):
                path = no_test(path)
                key, condition = pair_key(path)
                if key not in self.allowed or (self.mode == 'fog' and condition != 'fog'):
                    raise ValueError('Actual read outside accepted training pair: ' + str(path))
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError('Missing/empty point cloud; fallback forbidden: ' + str(path))
                digest = sha256(path)
                if self.expected_lidar.get(str(path)) != digest:
                    raise ValueError('Actually read training point cloud bytes changed: ' + str(path))
                reads.append(dict(key=key, condition=condition, path=str(path), sha256=digest))
            return original(original_path, *a, **kw)
        def retrieve_observer(idx, *a, **kw):
            gt_calls.append(dict(requested_index=int(idx)))
            return retrieve(idx, *a, **kw)
        def single(base, *a, **kw):
            result = get_single(base, *a, **kw)
            gt_calls.append(dict(cav=str(base['id']), candidate_ids_before_native_filter=list(map(str, result['object_ids']))))
            return result
        def points_filter(*a, **kw):
            result = native_points(*a, **kw)
            counts = result.sum(axis=1)
            native_filter.append(dict(points_per_box=counts.tolist(), zero_point_boxes=int((counts == 0).sum())))
            return result
        def augment(*a, **kw):
            augmentation_calls.append(True)
            return augment_forward(*a, **kw)
        with patched(pcd_utils, 'pcd_to_np', read), patched(self.dataset, 'retrieve_base_data', retrieve_observer), patched(self.dataset, 'get_item_single_car', single), patched(native_module, 'points_in_boxes_cpu', points_filter), patched(self.dataset.data_augmentor, 'forward', augment):
            item = self.dataset[index]
        if not reads:
            raise ValueError('Official loader did not read any LiDAR')
        ego = item['ego']
        item['_audit'] = dict(index=int(index), frame_key=self.keys[index], reads=reads,
            gt_calls=gt_calls, final_gt_ids=list(map(str, ego['object_ids'])),
            native_point_filter=native_filter,
            configured_world_augmentation_calls=len(augmentation_calls),
            final_gt_count=int(ego['object_bbx_mask'].sum()), rng_before=before,
            rng_after=identity(dict(python=random.getstate(), numpy=np.random.get_state())),
            input_sha256=tensor_digest(dict(voxels=ego['processed_lidar'], labels=ego['label_dict'])))
        return item

    def collate_batch_train(self, items):
        audits = [x.pop('_audit') for x in items]
        result = self.dataset.collate_batch_train(items)
        if result is None:
            raise ValueError('Official collate returned None')
        result['_audit'] = audits
        return result


def build_train(hypes, manifest, fog=False):
    import copy
    from opencood.data_utils.datasets import build_dataset
    config = copy.deepcopy(hypes)
    if fog:
        config.update(validate_dir=config['root_dir'], train_sim=False, eval_sim=True)
    dataset = build_dataset(config, visualize=False, train=not fog)
    return AuditedDataset(dataset, manifest, mode='fog' if fog else 'mixed')


class PairedTrainingDataset:
    """Clean reference from TRAIN only, isolated RNG and explicit resampling mask."""
    def __init__(self, weather, clean):
        self.weather, self.clean = weather, clean
        self.dataset = weather.dataset

    def __len__(self):
        return len(self.weather)

    def __getitem__(self, index):
        import numpy as np
        item = self.weather[index]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        calls = []
        retrieve = self.clean.retrieve_base_data
        def observed(idx, *args, **kwargs):
            calls.append(int(idx))
            return retrieve(idx, *args, **kwargs)
        try:
            with patched(self.clean, 'retrieve_base_data', observed):
                reference = self.clean[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        weather_calls = [c['requested_index'] for c in item['_audit']['gt_calls'] if 'requested_index' in c]
        item['_clean_reference'] = reference
        item['_reference_aligned'] = calls == [index] and weather_calls == [index]
        return item

    def collate_batch_train(self, items):
        references = [x.pop('_clean_reference') for x in items]
        aligned = all(x.pop('_reference_aligned') for x in items)
        clean = self.clean.collate_batch_train(references)
        batch = self.weather.collate_batch_train(items)
        if clean is not None and aligned:
            import torch
            if torch.equal(clean['ego']['record_len'], batch['ego']['record_len']):
                batch['ego']['repe_clean_lidar'] = clean['ego']['processed_lidar']
        return batch


_unpaired_build_train = build_train

def build_train(hypes, manifest, fog=False):
    if fog:
        raise ValueError('Fog calibration is outside REPE')
    import copy
    from opencood.data_utils.datasets import build_dataset
    weather = _unpaired_build_train(hypes, manifest)
    cfg = copy.deepcopy(hypes)
    cfg.update(train_sim=False, eval_sim=False)
    clean = build_dataset(cfg, visualize=False, train=True)
    return PairedTrainingDataset(weather, clean)
