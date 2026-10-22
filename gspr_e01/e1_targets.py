"""Passive real-input geometry and target metrics, reusing the official matcher."""
from contextlib import ExitStack
from pathlib import Path
import sys
from e01_common import PROJECT, WEATHERS, atomic, identity, jsonl, read, sha


def native_dataset(dataset):
    """The frozen generator wraps its 12-frame smoke in a prefix Subset."""
    while not hasattr(dataset, 'get_item_single_car'):
        if not hasattr(dataset, 'dataset') or not hasattr(dataset, 'indices'):
            raise ValueError('Unrecognized official inference dataset wrapper')
        if list(dataset.indices) != list(range(len(dataset.indices))):
            raise ValueError('Frozen inference must use the original frame-order prefix')
        dataset = dataset.dataset
    return dataset


def observed_evaluation(cli, args, evaluation):
    destination = Path(cli.destination)
    if cli.clean_only:
        return evaluation.evaluate(args, Path(cli.checkpoint), destination, clean_only=True, smoke=cli.smoke)
    import numpy as np
    import torch
    import predict
    from support import patched
    sys.path.insert(0, str(PROJECT / 'gspr_e0_diagnosis'))
    from e0_runtime import points_in_targets, tensor_numpy
    from e0_numeric import array_hash
    original_generate = predict.generate
    native_loader = torch.utils.data.DataLoader
    context = {}
    class Dataset:
        def __init__(self, dataset):
            self.dataset = dataset
            self.native = native_dataset(dataset)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            arrays, fallback = [], []
            original = self.native.get_item_single_car
            def single(*a, **kw):
                calls = []
                preprocess = self.native.pre_processor.preprocess
                def observe(points):
                    calls.append(np.asarray(points).copy())
                    return preprocess(points)
                with patched(self.native.pre_processor, 'preprocess', observe):
                    result = original(*a, **kw)
                if len(calls) < 2:
                    raise ValueError('Expected native Radar then LiDAR preprocessing')
                # Pinned source: the second call is the real post-FOV LiDAR, before fallback.
                arrays.append(calls[1][:, :3].astype(np.float32))
                fallback.append(not np.array_equal(calls[1], result['projected_lidar']))
                return result
            with patched(self.native, 'get_item_single_car', single):
                item = self.dataset[index]
            targets = context['evidence']['frames'][index]['conditions']['clean']['final_gt']
            points = np.concatenate(arrays) if arrays else np.zeros((0, 3), dtype=np.float32)
            counts = points_in_targets(points, targets).sum(axis=1).tolist()
            item['_e01_geometry'] = dict(index=index, targets=[dict(object_id=t['object_id'],
                box=t['box'], observed_points=int(n)) for t, n in zip(targets, counts)],
                real_post_FOV_points=len(points), fallback_CAVs=sum(fallback),
                point_scope='real post-FOV/pre-voxel LiDAR, summed CAV observations; fallback excluded')
            return item
    class Loader:
        def __init__(self, dataset, *a, **kw):
            collate = kw['collate_fn']
            def collect(items):
                metadata = [x.pop('_e01_geometry') for x in items]
                batch = collate(items)
                batch['_e01_geometry'] = metadata
                return batch
            kw['collate_fn'] = collect
            self.loader = native_loader(Dataset(dataset), *a, **kw)

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            for i, batch in enumerate(self.loader):
                geometry = batch.pop('_e01_geometry')
                if len(geometry) != 1 or geometry[0]['index'] != i:
                    raise ValueError('Inference geometry / loader order differs')
                row = dict(**geometry[0], weather=context['weather'],
                    frame_key=context['evidence']['b']['selected_frame_keys'][i],
                    input_sha256=identity({k: array_hash(tensor_numpy(v))
                        for k, v in batch['ego']['processed_lidar'].items()}))
                atomic(destination / 'input_geometry' / context['weather'] / f'{i:06d}.json', row)
                yield batch
    def generate(weather, output, run_args, evidence, contract):
        context.update(weather=weather, evidence=evidence)
        return original_generate(weather, output, run_args, evidence, contract)
    with ExitStack() as stack:
        stack.enter_context(patched(predict, 'generate', generate))
        stack.enter_context(patched(torch.utils.data, 'DataLoader', Loader))
        result = evaluation.evaluate(args, Path(cli.checkpoint), destination, smoke=cli.smoke)
    count = 12 if cli.smoke else 829
    files = {}
    for w in WEATHERS:
        for i in range(count):
            path = destination / 'input_geometry' / w / f'{i:06d}.json'
            prediction = destination / 'predictions' / w / f'{i:06d}.json'
            geometry = read(path)
            if geometry['frame_key'] != read(prediction)['frame_key'] or geometry['index'] != i:
                raise ValueError('Passive geometry / prediction identity differs')
            files[str(path.relative_to(destination))] = sha(path)
    atomic(destination / 'input_geometry_manifest.json', dict(status='complete',
        checkpoint_sha256=result['checkpoint_sha256'], evaluation_contract=result['contract_identity'],
        prediction_cache_identity=result['prediction_cache_identity'], files=files))
    # Additions are separately committed; the frozen evaluator's registered outputs remain checked.
    return result


def target_analysis(directory, count):
    directory = Path(directory)
    frames = list(jsonl(directory / 'per_frame_metrics.jsonl'))
    if len(frames) != 3*count or len({(r['condition'], r['index']) for r in frames}) != 3*count:
        raise ValueError('Incomplete target analysis')
    manifest = read(directory / 'input_geometry_manifest.json')
    for name, digest in manifest['files'].items():
        if sha(directory / name) != digest:
            raise ValueError('Input geometry changed')
    groups, totals = {}, {}
    for row in frames:
        w, i = row['condition'], row['index']
        geometry = read(directory / 'input_geometry' / w / f'{i:06d}.json')
        for p, values in row['protocols'].items():
            for t, stat in values['thresholds'].items():
                total = totals.setdefault(w, {}).setdefault(p, {}).setdefault(t, dict(tp=0, fp=0, fn=0))
                for k in total:
                    total[k] += int(stat[k])
        matches = row['protocols']['F']['matched_pairs']
        for target in geometry['targets']:
            x, y, _, h, width, length, _ = target['box']
            radius = (x*x+y*y)**.5
            distance = '<20m' if radius < 20 else '20-40m' if radius < 40 else '40-70m' if radius < 70 else '>=70m'
            volume = h*width*length
            size = '<10m3' if volume < 10 else '10-30m3' if volume < 30 else '>=30m3'
            points = target['observed_points']
            density = '0' if not points else '1-5' if points <= 5 else '6-20' if points <= 20 else '>20'
            for kind, label in (('distance', distance), ('size_volume', size), ('point_count', density)):
                value = groups.setdefault(w, {}).setdefault(kind, {}).setdefault(label,
                    dict(targets=0, tp={'0.3': 0, '0.5': 0, '0.7': 0}))
                value['targets'] += 1
                for t in value['tp']:
                    value['tp'][t] += any(str(m['object_id']) == str(target['object_id']) for m in matches[t])
    for weather in groups.values():
        for kind in weather.values():
            for value in kind.values():
                value['recall'] = {t: tp/value['targets'] for t, tp in value['tp'].items()}
    return dict(status='complete', frames_per_weather=count, counts=totals, fixed_GT_groups=groups,
        matching='frozen original per_frame_metrics F matched_pairs; BEV IoU',
        density='real post-FOV/pre-voxel point count across CAVs; no fallback points',
        limitation='Stratified target recall only; FP has no unique GT stratum; no rewritten group AP',
        geometry_manifest_sha256=sha(directory / 'input_geometry_manifest.json'))
