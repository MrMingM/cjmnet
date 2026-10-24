"""Read-only source conversion: preserve all non-image bytes and exact PNG sizes.

Default materialization uses hard links on Linux. Original data is never deleted,
renamed, chmod'ed or rewritten. Treat both trees as immutable after conversion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import zlib

DATASETS = ('v2xr-official-train-v2', 'v2xr-lidar-weather-validate-v1',
            'v2xr-official-validate-v1')
PROOF = 'POINTCLOUD_ONLY.json'


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe(path):
    path = path.expanduser().resolve(strict=True)
    if any(part.lower() == 'test' for part in path.parts):
        raise ValueError('Formal test is outside this operation: ' + str(path))
    return path


def png_size(path):
    # PNG stores the exact width and height in IHDR; no pixel decoding required.
    with path.open('rb') as stream:
        head = stream.read(33)
    if (len(head) != 33 or head[:8] != b'\x89PNG\r\n\x1a\n'
            or head[8:16] != b'\x00\x00\x00\rIHDR'
            or zlib.crc32(head[12:29]) & 0xffffffff != struct.unpack('>I', head[29:33])[0]):
        raise ValueError('Invalid PNG header: ' + str(path))
    width, height = struct.unpack('>II', head[16:24])
    if not width or not height:
        raise ValueError('Zero image dimensions: ' + str(path))
    return [width, height]


def walk(folder, ancestors=frozenset()):
    real = safe(folder)
    if real in ancestors:
        raise ValueError('Directory symlink cycle: ' + str(folder))
    with os.scandir(folder) as iterator:
        entries = sorted(iterator, key=lambda e: e.name)
    for entry in entries:
        path = Path(entry.path)
        if entry.is_dir(follow_symlinks=True):
            yield from walk(path, ancestors | {real})
        elif entry.is_file(follow_symlinks=True):
            safe(path)
            yield path
        else:
            raise ValueError('Broken link or unsupported entry: ' + str(path))


def stable_hash(path, cache):
    before = path.stat()
    key = tuple(getattr(before, k) for k in
                ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
    if key not in cache:
        cache[key] = digest(path)
    after = path.stat()
    if key != tuple(getattr(after, k) for k in
                    ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
        raise ValueError('Source changed during hashing: ' + str(path))
    return before, cache[key]


def prepare(base, output, copy=False):
    base, output = safe(base), output.expanduser().resolve()
    for name in DATASETS:
        safe(base / name)
    if output.exists():
        raise FileExistsError('Choose a new empty output path: ' + str(output))
    for name in DATASETS:
        source = safe(base / name)
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Output overlaps an input dataset')
    output.mkdir(parents=True)
    files, images, cache, unique = {}, {}, {}, {}
    count = 0
    for name in DATASETS:
        for source in walk(base / name):
            relative = source.relative_to(base).as_posix()
            stat, sha = stable_hash(source, cache)
            row = dict(size=stat.st_size, sha256=sha)
            suffix = source.suffix.lower()
            if suffix in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
                if suffix != '.png':
                    raise ValueError('Only the original PNG format is supported: ' + str(source))
                images[relative] = dict(row, dimensions=png_size(source))
            else:
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if copy:
                    shutil.copyfile(safe(source), target)
                else:
                    try:
                        os.link(safe(source), target)
                    except OSError as error:
                        raise RuntimeError('Hard link failed; retain this incomplete output for inspection. '
                                           'Use a NEW output and --copy only if enough space is available: '
                                           + str(error)) from error
                # Creating a hard link changes inode ctime. Refresh the hash cache
                # for later references to the same source inode without rereading it.
                current = source.stat()
                cache[tuple(getattr(current, k) for k in
                    ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))] = sha
                if target.stat().st_size != row['size']:
                    raise ValueError('Materialized file size differs: ' + relative)
                files[relative] = row
                unique[(stat.st_dev, stat.st_ino)] = stat.st_size
            count += 1
            if count % 1000 == 0:
                print('PROGRESS inspected=%d kept=%d image_metadata=%d' %
                      (count, len(files), len(images)), flush=True)
    if not images:
        raise ValueError('No original images found; refuse an unverified conversion')
    proof = dict(schema=1, status='complete', source_base=str(base), datasets=list(DATASETS),
                 mode='copy' if copy else 'hardlink', files=files, images=images,
                 image_pixels_included=False, fov_rule_changed=False,
                 unique_source_nonimage_bytes=sum(unique.values()),
                 logical_nonimage_bytes=sum(r['size'] for r in files.values()),
                 logical_original_image_bytes=sum(r['size'] for r in images.values()))
    proof['identity'] = identity(proof)
    temp = output / (PROOF + '.tmp')
    temp.write_text(json.dumps(proof, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, output / PROOF)
    print(json.dumps(dict(status='SLIM_PREPARED', root=str(output), kept_files=len(files),
        image_metadata_records=len(images), original_data_deleted=False,
        unique_nonimage_GiB=proof['unique_source_nonimage_bytes']/1024**3,
        logical_nonimage_GiB=proof['logical_nonimage_bytes']/1024**3,
        logical_original_image_GiB=proof['logical_original_image_bytes']/1024**3), indent=2), flush=True)


def load(root):
    root = safe(root)
    proof = json.loads((root / PROOF).read_text(encoding='utf-8'))
    if (proof.get('schema') != 1 or proof.get('status') != 'complete'
            or proof.get('datasets') != list(DATASETS)
            or identity({k: v for k, v in proof.items() if k != 'identity'}) != proof.get('identity')):
        raise ValueError('Slim metadata identity/status differs')
    for group in ('files', 'images'):
        for name in proof[group]:
            rel = PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in name or rel.parts[0] not in DATASETS:
                raise ValueError('Unsafe metadata path: ' + name)
    return proof


def verify(root):
    root = safe(root)
    proof, cache = load(root), {}
    actual = {p.relative_to(root).as_posix() for name in DATASETS for p in walk(root / name)}
    if actual != set(proof['files']):
        raise ValueError('Missing/extra slim files: missing=%d extra=%d' %
                         (len(set(proof['files']) - actual), len(actual - set(proof['files']))))
    for i, (name, expected) in enumerate(proof['files'].items(), 1):
        stat, sha = stable_hash(root / name, cache)
        if dict(size=stat.st_size, sha256=sha) != expected:
            raise ValueError('Slim content changed: ' + name)
        if i % 2000 == 0:
            print('VERIFIED', i, '/', len(actual), flush=True)
    print(json.dumps(dict(status='SLIM_CONTENT_PASSED', files=len(actual),
                          metadata_identity=proof['identity'], training_executed=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('prepare')
    create.add_argument('--base', type=Path, default=Path('/data/cjm/datasets'))
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--copy', action='store_true')
    check = sub.add_parser('verify')
    check.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.base, args.output, args.copy)
    else:
        verify(args.root)


if __name__ == '__main__':
    main()
