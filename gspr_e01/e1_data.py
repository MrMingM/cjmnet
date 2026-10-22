"""Per-instance sampling and additional observations around the original reader."""
from pathlib import Path
from adapters import install_sampler, pair_key
from e01_common import SETTINGS, identity, safe, sha


def inventory(manifest, train_root):
    rows = {r['relative']: r for r in manifest['files']}
    pairs = []
    for scene in manifest['scenes']:
        for stamp in scene['timestamps']:
            for cav in scene['agents']:
                prefix = scene['scene']+'/'+cav+'/'+stamp
                paths = {w: prefix+s+'.pcd' for w, s in
                    (('clean', ''), ('fog', '_fog_0.060'), ('snow', '_snow_2.5_2.0'))}
                if any(p not in rows for p in paths.values()):
                    raise ValueError('Missing accepted three-weather training files: '+prefix)
                for p in paths.values():
                    safe(Path(train_root) / 'train' / p)
                pairs.append(dict(key=[scene['scene'], cav, stamp], paths=paths,
                    sha256={w: rows[p]['sha256'] for w, p in paths.items()}))
    if len(pairs) != 24088 or sum(len(s['timestamps']) for s in manifest['scenes']) != 8083:
        raise ValueError('E1 requires the accepted 8083 cooperative / 24088 CAV frames')
    return dict(identity=identity(pairs), pairs=pairs, cooperative_frames=8083)


def build_adapter(old, hypes, manifest, recipe, source):
    """Each DataLoader worker observes its own original dataset; no global source edits."""
    import copy
    from opencood.data_utils.datasets import build_dataset
    import torch
    class Audited(old.AuditedDataset):
        def __init__(self, dataset):
            super().__init__(dataset, manifest)
            root = Path(dataset.params['root_dir'])
            self.expected_lidar = {str(safe(root / r['relative'])): r['sha256']
                for r in manifest['files'] if r['relative'].endswith('.pcd')
                and not r['relative'].endswith('_radar.pcd')}

        def __getitem__(self, index):
            worker = torch.utils.data.get_worker_info()
            before = sha_tensor(torch.get_rng_state())
            item = super().__getitem__(index)
            audit = item['_audit']
            audit['worker'] = dict(id=worker.id if worker else None,
                seed=worker.seed if worker else torch.initial_seed(),
                torch_rng_before=before, torch_rng_after=sha_tensor(torch.get_rng_state()))
            audit['recipe'] = recipe
            audit['requested_frames_including_empty_GT_retries'] = [r['requested_index']
                for r in audit['gt_calls'] if 'requested_index' in r]
            audit['empty_GT_frames_resampled'] = len(audit['requested_frames_including_empty_GT_retries'])-1
            audit['retained_CAV_count'] = int(item['ego']['cav_num'])
            audit['retained_CAV_order'] = [r['cav'] for r in audit['gt_calls'] if 'cav' in r][-audit['retained_CAV_count']:]
            audit['zero_point_GT_boxes'] = sum(r['zero_point_boxes'] for r in audit['native_point_filter'])
            audit['read_CAV_count_including_retries'] = len(audit['reads'])
            if set(r['condition'] for r in audit['reads']) - set(SETTINGS['weather_order']):
                raise ValueError('Unregistered weather')
            return item
    dataset = build_dataset(copy.deepcopy(hypes), visualize=False, train=True)
    install_sampler(dataset, recipe, source)
    # The legacy observer resolves pair_key at call time; only its private process is adapted.
    old.pair_key = pair_key
    return Audited(dataset)


def sha_tensor(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def compare_epoch(left, right, expected_frames):
    from e01_common import jsonl
    count = 0
    fields = ('index', 'frame_key', 'reads', 'gt_calls', 'final_gt_ids', 'native_point_filter',
        'configured_world_augmentation_calls', 'final_gt_count', 'rng_before', 'rng_after',
        'input_sha256', 'worker', 'recipe', 'requested_frames_including_empty_GT_retries',
        'zero_point_GT_boxes', 'read_CAV_count_including_retries', 'empty_GT_frames_resampled',
        'retained_CAV_count', 'retained_CAV_order')
    import itertools
    for batch, (a, b) in enumerate(itertools.zip_longest(jsonl(left), jsonl(right))):
        if a is None or b is None or len(a) != len(b):
            raise ValueError('C/D trace batch count differs')
        for i, (x, y) in enumerate(zip(a, b)):
            if any(k not in x or k not in y or x[k] != y[k] for k in fields):
                differences = [k for k in fields if x.get(k) != y.get(k)]
                raise ValueError(f'C/D data pairing differs batch={batch} slot={i} fields={differences}')
            count += 1
    if count != expected_frames:
        raise ValueError('C/D trace consumed frame count differs: '+str(count))
    return dict(status='complete', exact=True, consumed_frames=count,
        fields=list(fields), C_trace_sha256=sha(left), D_trace_sha256=sha(right))
