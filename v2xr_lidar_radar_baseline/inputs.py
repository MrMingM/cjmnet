"""Bind both official sensors to the accepted scene/vehicle/timestamp inventory."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import atomic_json, identity, no_test, read_json, sha256

DATA = Path('/data/cjm/datasets')
TRAIN_MANIFEST = DATA / 'v2xr-official-train-v2/manifest.json'
VAL_MANIFEST = DATA / 'logs/v2xr-official-gt-audit-v1/input_manifest.json'
TRAIN = TRAIN_MANIFEST.parent / 'train'
VALIDATE = VAL_MANIFEST.parent / 'data_view/validate'
SUFFIX = {'clean': '.pcd', 'fog': '_fog_0.060.pcd', 'snow': '_snow_2.5_2.0.pcd'}
RADAR = '_radar.pcd'
WEATHER = {'clean': '', 'fog': '_fog_0.060', 'snow': '_snow_2.5_2.0'}


def header(path):
    fields = {}
    with path.open('rb') as stream:
        for _ in range(25):
            line = stream.readline(1024).decode('ascii').strip().split()
            if not line or line[0].startswith('#'):
                continue
            fields[line[0]] = line[1:]
            if line[0] == 'DATA':
                return ' '.join(fields['FIELDS']), int(fields['POINTS'][0])
    raise ValueError(f'Invalid PCD header: {path}')


def inventory(output):
    summary = {}
    for split, root, source, frames, cars in (
            ('train', TRAIN, TRAIN_MANIFEST, 8083, 24088),
            ('validate', VALIDATE, VAL_MANIFEST, 829, 3041)):
        manifest = read_json(source)
        catalog = {r['relative']: r for r in manifest['files']}
        suffixes = ['.yaml', '.pcd', SUFFIX['fog'], RADAR, '_camera0.png']
        if split == 'validate':
            suffixes.append(SUFFIX['snow'])
        pairs = [(s['scene'], cav, ts) for s in manifest['scenes']
                 for ts in s['timestamps'] for cav in s['agents']]
        keys = {(scene, ts) for scene, _, ts in pairs}
        if len(keys) != frames or len(pairs) != cars or len(set(pairs)) != cars:
            raise ValueError(f'{split}: accepted frame inventory differs')
        paths = [('/'.join(pair) + suffix, suffix) for pair in pairs for suffix in suffixes]
        missing = [rel for rel, _ in paths if rel not in catalog or not (root / rel).is_file()]
        if missing:
            raise ValueError(f'{split}: missing {len(missing)} required files: {missing[:10]}')

        def check(entry):
            rel, suffix = entry
            path = no_test(root / rel)
            expected = catalog[rel]
            if path.stat().st_size != expected['size'] or sha256(path) != expected['sha256']:
                raise ValueError(f'{split}: accepted payload changed: {path}')
            if suffix.endswith('.pcd'):
                fields, count = header(path)
                if not {'x', 'y', 'z', 'rgb'} <= set(fields.split()):
                    raise ValueError(f'Missing official point fields: {path}')
                return suffix, fields, count == 0
            return suffix, None, False

        observed, empty = Counter(), Counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, (suffix, fields, is_empty) in enumerate(pool.map(check, paths), 1):
                if fields:
                    observed[f'{suffix}: {fields}'] += 1
                if is_empty:
                    empty[suffix] += 1
                if index % 5000 == 0:
                    print(f'INPUT {split} {index}/{len(paths)}', flush=True)
        summary[split] = dict(frames=frames, single_vehicle_frames=cars, files=len(paths),
            input_manifest_sha256=sha256(source), paired_keys_sha256=identity(pairs),
            header_fields=observed, raw_empty_files=empty)
    train_keys = {(s['scene'], ts) for s in read_json(TRAIN_MANIFEST)['scenes'] for ts in s['timestamps']}
    if train_keys & {tuple(k) for k in read_json(VAL_MANIFEST)['frame_keys']}:
        raise ValueError('Train/validation split overlaps')
    atomic_json(output, dict(status='complete', splits=summary))
    return read_json(output)


def check_dataset(dataset, split, condition):
    source = TRAIN_MANIFEST if split == 'train' else VAL_MANIFEST
    scenes = read_json(source)['scenes']
    root = TRAIN if split == 'train' else VALIDATE
    if (not dataset.use_Lidar or not dataset.use_Radar or dataset.use_DeLidar
            or not dataset.fov or not dataset.proj_first or dataset.async_flag
            or dataset.max_cav != 7 or len(dataset.scenario_database) != len(scenes)):
        raise ValueError('Official paired dataset flags differ')
    keys = []
    for i, scene in enumerate(scenes):
        content = dataset.scenario_database[i]
        if list(content) != scene['agents'] or not content[scene['ego']]['ego']:
            raise ValueError(f'Vehicle order differs: {scene["scene"]}')
        for car, records in content.items():
            if [key for key in records if key != 'ego'] != scene['timestamps']:
                raise ValueError(f'Timestamp order differs: {scene["scene"]}/{car}')
            for stamp in scene['timestamps']:
                prefix = root / scene['scene'] / car / stamp
                row = records[stamp]
                if (Path(row['lidar']) != Path(str(prefix) + SUFFIX[condition])
                        or Path(row['radar']) != Path(str(prefix) + RADAR)
                        or Path(row['yaml']) != prefix.with_suffix('.yaml')):
                    raise ValueError(f'Modality/time pairing differs: {prefix}')
        keys.extend([scene['scene'], stamp] for stamp in scene['timestamps'])
    if len(dataset) != len(keys):
        raise ValueError('Official dataset length differs')
    return keys

