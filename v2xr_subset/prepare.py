"""Server-only selective extraction. Original archives and test stay untouched."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n',
                   encoding='utf-8', newline='\n')
    os.replace(tmp, path)


def sources():
    return {p.relative_to(PACKAGE).as_posix(): digest(p) for p in sorted(PACKAGE.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.sh', '.md', '.json')
            and p.name != 'source_snapshot.json' and '__pycache__' not in p.parts}


def verify_sources():
    current = sources()
    if current != read_json(PACKAGE / 'source_snapshot.json')['sources']:
        raise ValueError('Sync the entire v2xr_subset directory including its source snapshot')
    return current


def safe_member(name):
    name = name.replace('\\', '/')
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or ':' in name or '\n' in name
            or '\r' in name or '\0' in name
            or any(p.lower() in ('test', 'testing') for p in path.parts)):
        raise ValueError(f'Unsafe archive member: {name!r}')
    return path


def classify(name):
    name = safe_member(name).name
    match = re.fullmatch(r'(\d+)\.yaml', name)
    if match:
        return match[1], 'yaml'
    match = re.fullmatch(r'(\d+)(?:_([A-Za-z][A-Za-z0-9_.-]*))?\.pcd', name)
    if not match:
        return None
    stamp, suffix = match.groups()
    if suffix is None:
        return stamp, 'clean'
    if suffix in ('fog', 'snow') or suffix.startswith(('fog_', 'snow_')):
        return stamp, suffix
    return None


def parse_listing(text):
    """7z l -slt: ignore archive header, parse only member records after separator."""
    if '\n----------\n' not in text.replace('\r\n', '\n'):
        raise ValueError('Unrecognized 7z technical listing; preserve the listing log')
    body = text.replace('\r\n', '\n').split('\n----------\n', 1)[1]
    result, entry = [], {}
    for line in body.splitlines() + ['']:
        if not line.strip():
            if 'Path' in entry:
                safe_member(entry['Path'])
                directory = entry.get('Folder') == '+' or entry.get('Attributes', '').startswith('D')
                if not directory:
                    if (any(k in entry for k in ('Symbolic Link', 'Hard Link'))
                            or any(token.startswith('l') for token in entry.get('Attributes', '').split())):
                        raise ValueError('Archive links are unsupported')
                    size = int(entry['Size'])
                    if size < 0:
                        raise ValueError('Negative member size')
                    result.append(dict(member=entry['Path'], size=size, crc=entry.get('CRC')))
            entry = {}
        elif ' = ' in line:
            key, value = line.split(' = ', 1)
            if key in entry:
                raise ValueError(f'Duplicate listing field: {key}')
            entry[key] = value
    if not result:
        raise ValueError('No regular files in archive listing')
    return result


def choose_variant(family, available, requested):
    candidates = sorted(x for x in available if x == family or x.startswith(family + '_'))
    if requested:
        if requested not in candidates:
            raise ValueError(f'{family} variant {requested!r} is not paired in every archive: {candidates}')
        return requested
    if len(candidates) != 1:
        raise ValueError(f'Choose --{family}-variant explicitly; common variants={candidates}. '
                         'See inventory.json; no severity is selected automatically when ambiguous.')
    return candidates[0]


def select_stamps(stamps, cap):
    stamps = sorted(stamps)
    if not cap or cap >= len(stamps):
        return stamps
    if cap == 1:
        return [stamps[len(stamps) // 2]]
    return [stamps[round(i * (len(stamps) - 1) / (cap - 1))] for i in range(cap)]


def archive_inventory(seven, archive):
    print(f'List and hash: {archive}', flush=True)
    process = subprocess.run([seven, 'l', '-slt', '-sccUTF-8', str(archive)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding='utf-8', errors='strict', timeout=600)
    if process.returncode:
        raise RuntimeError(f'7z listing failed: {archive}\n{process.stdout}\n{process.stderr}')
    entries = parse_listing(process.stdout)
    frames = {}
    for item in entries:
        kind = classify(item['member'])
        if kind is None:
            continue
        stamp, variant = kind
        if variant in frames.setdefault(stamp, {}):
            raise ValueError(f'Duplicate timestamp/variant in {archive}: {stamp}/{variant}')
        frames[stamp][variant] = item
    if not frames:
        raise ValueError(f'No recognized LiDAR/YAML frames in {archive}')
    if any('clean' not in value or 'yaml' not in value for value in frames.values()):
        raise ValueError(f'Incomplete Clean/YAML pairing in {archive}')
    return dict(path=str(archive), sha256=digest(archive), compressed_bytes=archive.stat().st_size,
                seven_zip_banner=process.stdout.splitlines()[:3],
                archive_uncompressed_bytes=sum(x['size'] for x in entries), frames=frames,
                ignored_examples=[x['member'] for x in entries if classify(x['member']) is None][:8])


def make_plan(args, seven, source, output):
    archives, protocols = [], []
    split_root = source / args.split
    scenes = sorted(p for p in split_root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    for scene in scenes:
        if scene.is_symlink():
            raise ValueError(f'Scene symlinks are unsupported: {scene}')
        files = sorted(scene.glob('*.7z'))
        if not files:
            raise ValueError(f'No per-agent .7z archives in {scene}')
        for file in files:
            if not re.fullmatch(r'-?\d+', file.stem) or file.is_symlink():
                raise ValueError(f'Expected a regular numeric agent archive: {file}')
            row = archive_inventory(seven, file)
            row.update(scene=scene.name, cav=file.stem)
            archives.append(row)
        protocol = scene / 'data_protocol.yaml'
        if protocol.is_file():
            protocols.append(dict(source=str(protocol), scene=scene.name, sha256=digest(protocol)))
    if not archives:
        raise ValueError(f'No archives found in {split_root}')
    inventory = dict(split=args.split, archives=archives, source_root=str(source))
    atomic_json(output / 'inventory.json', inventory)
    common_variants = None
    for row in archives:
        for frame in row['frames'].values():
            common_variants = set(frame) if common_variants is None else common_variants & set(frame)
    fog = choose_variant('fog', common_variants, args.fog_variant)
    snow = choose_variant('snow', common_variants, args.snow_variant)
    scene_frames, total_full_frames = {}, 0
    for scene in scenes:
        rows = [r for r in archives if r['scene'] == scene.name]
        stamps = sorted(rows[0]['frames'])
        if any(sorted(r['frames']) != stamps for r in rows):
            raise ValueError(f'Agent timestamp mismatch in {scene}; no silent intersection/subsampling')
        total_full_frames += len(stamps)
        scene_frames[scene.name] = select_stamps(stamps, args.frames_per_scene)
    items = []
    totals = Counter()
    for row in archives:
        chosen, basenames = [], set()
        for stamp in scene_frames[row['scene']]:
            for variant in ('clean', fog, snow, 'yaml'):
                entry = row['frames'][stamp][variant]
                basename = safe_member(entry['member']).name
                if basename in basenames:
                    raise ValueError('Flattening would cause a filename collision')
                basenames.add(basename)
                family = variant.split('_')[0]
                chosen.append(dict(**entry, name=basename, timestamp=stamp, condition=family))
                totals[family + '_files'] += 1
                totals[family + '_bytes'] += entry['size']
        items.append({k: row[k] for k in ('path', 'sha256', 'compressed_bytes', 'scene', 'cav', 'seven_zip_banner')}
                     | {'members': chosen})
    selected_bytes = sum(x['size'] for row in items for x in row['members'])
    selected_bytes += sum(Path(x['source']).stat().st_size for x in protocols)
    payload = dict(schema=1, source_root=str(source), output_root=str(output), split=args.split,
        sources=verify_sources(), selected_variants=dict(fog=fog, snow=snow),
        frames_per_scene=args.frames_per_scene, frame_keys=[[s, t] for s, ts in scene_frames.items() for t in ts],
        source_cooperative_frames=total_full_frames, selected_cooperative_frames=sum(map(len, scene_frames.values())),
        archives=items, protocols=protocols, selected_bytes=selected_bytes, totals=dict(totals),
        archive_uncompressed_bytes=sum(r['archive_uncompressed_bytes'] for r in archives),
        interpretation='Paired V2X-R LiDAR development subset; not an OPV2V-W reproduction; no Rain.')
    identity = object_hash(payload)
    plan = dict(identity=identity, protocol=payload)
    path = output / 'plan.json'
    if path.exists() and read_json(path)['identity'] != identity:
        raise ValueError('Selection/source changed; use a new output root')
    atomic_json(path, plan)
    gib = 1024 ** 3
    print(json.dumps(dict(plan=str(path), variants=payload['selected_variants'],
        cooperative_frames=payload['selected_cooperative_frames'], agents_archives=len(items),
        selected_GiB=round(selected_bytes / gib, 3),
        all_archive_contents_GiB=round(payload['archive_uncompressed_bytes'] / gib, 3),
        free_GiB=round(shutil.disk_usage(output).free / gib, 3), file_counts=dict(totals)), indent=2), flush=True)
    return plan


def pcd_header(path):
    lines = []
    with Path(path).open('rb') as f:
        for _ in range(100):
            line = f.readline(4096)
            if not line:
                raise ValueError(f'Missing DATA field in {path}')
            text = line.decode('ascii').strip()
            lines.append(text)
            if text.startswith('DATA '):
                return lines
    raise ValueError(f'PCD header too long: {path}')


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve(strict=True) == source.resolve(strict=True):
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(source.resolve(strict=True))


def unpack_selected(seven, archive, members, stage):
    stage.mkdir(parents=True, exist_ok=True)
    include = stage / 'selected.txt'
    include.write_text(''.join(x['member'] + '\n' for x in members), encoding='utf-8', newline='\n')
    with (stage / '7z.log').open('w', encoding='utf-8') as log:
        run = subprocess.run([seven, 'e', str(archive), '@' + str(include), '-o' + str(stage),
            '-y', '-aoa', '-spd', '-scsUTF-8', '-sccUTF-8', '-mmt=2'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError(f'7z extraction failed; see {stage / "7z.log"}')


def extract(args, seven, source, output):
    plan = read_json(output / 'plan.json')
    p = plan['protocol']
    if (plan['identity'] != object_hash(p) or p['sources'] != verify_sources()
            or p['source_root'] != str(source) or p['output_root'] != str(output) or p['split'] != args.split):
        raise ValueError('Plan identity/source mismatch')
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old['identity'] != plan['identity'] or not args.resume:
            raise ValueError('Existing extraction requires matching --resume')
    manifest = dict(identity=plan['identity'], status='in_progress', completed_archives=0)
    atomic_json(manifest_path, manifest)
    needed = sum(item['size'] for row in p['archives'] for item in row['members']
                 if not (output / 'payload' / p['split'] / row['scene'] / row['cav'] / item['name']).exists())
    # One selected archive is staged at a time. Camera/radar files are never written.
    needed += max(sum(item['size'] for item in r['members']) for r in p['archives'])
    needed += 512 * 1024 ** 2
    if shutil.disk_usage(output).free < needed:
        raise OSError(f'Insufficient output disk space: require conservative {needed / 1024**3:.2f} GiB free')
    samples, completed_files = {}, {}
    for index, row in enumerate(p['archives'], 1):
        archive = Path(row['path'])
        if archive.is_symlink() or digest(archive) != row['sha256']:
            raise ValueError(f'Archive changed: {archive}')
        target = output / 'payload' / p['split'] / row['scene'] / row['cav']
        target.mkdir(parents=True, exist_ok=True)
        record_path = output / 'records' / p['split'] / row['scene'] / (row['cav'] + '.json')
        if record_path.exists():
            record = read_json(record_path)
            if record['identity'] != plan['identity']:
                raise ValueError('Archive record identity mismatch')
            expected_names = {item['name'] for item in row['members']}
            if set(record['files']) != expected_names:
                raise ValueError('Archive record selection mismatch')
            for name, sha in record['files'].items():
                if (target / name).is_symlink() or digest(target / name) != sha:
                    raise ValueError(f'Extracted file changed: {target / name}')
            file_hashes = record['files']
        else:
            stage = output / '.partial' / p['split'] / row['scene'] / row['cav']
            print(f'Extract {index}/{len(p["archives"])}: {archive}', flush=True)
            unpack_selected(seven, archive, row['members'], stage)
            file_hashes = {}
            for item in row['members']:
                temporary, final = stage / item['name'], target / item['name']
                if temporary.is_symlink() or temporary.stat().st_size != item['size']:
                    raise ValueError(f'Selected member missing/wrong size: {temporary}')
                sha = digest(temporary)
                if final.exists() or final.is_symlink():
                    if final.is_symlink() or digest(final) != sha:
                        raise FileExistsError(f'Existing extraction differs: {final}')
                    temporary.unlink()
                else:
                    os.replace(temporary, final)
                file_hashes[item['name']] = sha
            atomic_json(record_path, dict(identity=plan['identity'], files=file_hashes))
        for item in row['members']:
            src = target / item['name']
            relative = Path(p['split']) / row['scene'] / row['cav'] / (item['timestamp'] + '.pcd')
            if item['condition'] == 'yaml':
                for mode in ('clean', 'fog', 'snow'):
                    link(src, (output / mode / relative).with_suffix('.yaml'))
                if 'yaml_preview' not in samples:
                    samples['yaml_preview'] = src.read_text(encoding='utf-8').splitlines()[:120]
                    samples['yaml_path'] = str(src)
            else:
                link(src, output / item['condition'] / relative)
                if item['condition'] not in samples:
                    samples[item['condition']] = dict(path=str(src), header=pcd_header(src))
        completed_files[f'{row["scene"]}/{row["cav"]}'] = len(file_hashes)
        manifest['completed_archives'] = index
        atomic_json(output / 'samples.json', samples)
        atomic_json(manifest_path, manifest)
        print(f'Verified {index}/{len(p["archives"])} archives', flush=True)
    for protocol in p['protocols']:
        src = Path(protocol['source'])
        if digest(src) != protocol['sha256']:
            raise ValueError('Data protocol changed')
        dst = output / 'protocols' / p['split'] / (protocol['scene'] + '.yaml')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and digest(dst) != protocol['sha256']:
            raise FileExistsError(dst)
        dst.write_bytes(src.read_bytes())
    manifest.update(status='complete', selected_cooperative_frames=p['selected_cooperative_frames'],
                    variants=p['selected_variants'], file_counts=completed_files)
    atomic_json(manifest_path, manifest)
    print(f'Complete: {manifest_path}; inspect headers/schema in {output / "samples.json"}', flush=True)


@contextmanager
def lock(output):
    import fcntl
    path = output.with_name(output.name + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Data preparation runs only on the Linux research server')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'extract', 'check'), default='plan')
    parser.add_argument('--source', default='/data/cjm/datasets/V2X-R')
    parser.add_argument('--output', default='/data/cjm/datasets/v2xr-lidar-weather-validate-v1')
    parser.add_argument('--split', choices=('train', 'validate'), default='validate')
    parser.add_argument('--frames-per-scene', type=int, default=0)
    parser.add_argument('--fog-variant')
    parser.add_argument('--snow-variant')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.frames_per_scene < 0:
        parser.error('Frame cap must be nonnegative')
    verify_sources()
    if args.stage == 'check':
        subprocess.run([sys.executable, '-m', 'unittest', 'v2xr_subset.test_contract'], check=True)
        return
    source = Path(args.source).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    split_root = (source / args.split).resolve(strict=True)
    if split_root.parent != source:
        raise ValueError('Split symlinks outside the dataset root are unsupported')
    if (source == output or source in output.parents or output in source.parents
            or output == REPO or REPO in output.parents
            or any(x.lower() in ('test', 'testing') for x in (*output.parts, *source.parts))):
        raise ValueError('Use a separate data-disk output root; test paths are refused')
    seven = shutil.which('7zz') or shutil.which('7z') or shutil.which('7za')
    if not seven:
        raise RuntimeError('Missing 7z command; ask the server administrator to provide 7zz/7z/7za')
    with lock(output):
        if output.exists() and not (output / 'inventory.json').exists() and any(output.iterdir()):
            raise FileExistsError('Refusing an unrelated nonempty output directory')
        output.mkdir(parents=True, exist_ok=True)
        if args.stage == 'plan':
            subprocess.run([sys.executable, '-m', 'unittest', 'v2xr_subset.test_contract'], check=True)
            make_plan(args, seven, source, output)
        else:
            extract(args, seven, source, output)


if __name__ == '__main__':
    main()
