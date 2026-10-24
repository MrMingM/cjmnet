"""Independent, read-only SHA256 inventory of the three migrated datasets.

Run make separately on each server, copy only the completed manifest, then compare.
Uses the Python standard library; does not import models or touch dataset files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

DATASETS = ('v2xr-official-train-v2', 'v2xr-lidar-weather-validate-v1',
            'v2xr-official-validate-v1')


def emit(stream, value):
    stream.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + '\n')


def walk(folder, ancestors=frozenset()):
    real = folder.resolve(strict=True)
    if real in ancestors:
        raise ValueError('Directory link cycle: ' + str(folder))
    with os.scandir(folder) as entries:
        children = sorted(entries, key=lambda p: p.name)
    for entry in children:
        path = Path(entry.path)
        if entry.is_dir(follow_symlinks=True):
            yield from walk(path, ancestors | {real})
        elif entry.is_file(follow_symlinks=True):
            yield path
        else:
            raise ValueError('Broken link or unsupported file: ' + str(path))


def make(base, output):
    base = base.resolve(strict=True)
    output = output.expanduser().resolve()
    roots = [base / name for name in DATASETS]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(str(root))
        if output == root.resolve() or root.resolve() in output.parents:
            raise ValueError('Write manifest outside the dataset directories')
    if output.exists():
        raise FileExistsError('Completed manifest exists; choose a new output: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.partial')
    count, total = 0, 0
    # Exclusive creation prevents two workers from writing the same manifest.
    with partial.open('x', encoding='utf-8', newline='\n') as stream:
        emit(stream, dict(type='header', schema=1, datasets=list(DATASETS)))
        for root in roots:
            local = 0
            for path in walk(root):
                before = path.stat()
                sha = hashlib.sha256()
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        sha.update(block)
                after = path.stat()
                attributes = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
                if any(getattr(before, a) != getattr(after, a) for a in attributes):
                    raise ValueError('File changed during hashing: ' + str(path))
                emit(stream, dict(type='file', path=path.relative_to(base).as_posix(),
                                  size=after.st_size, sha256=sha.hexdigest()))
                count += 1
                local += 1
                total += after.st_size
                if count % 1000 == 0:
                    stream.flush()
                    print('HASHED files=%d bytes=%d' % (count, total), flush=True)
            if not local:
                raise ValueError('Empty dataset: ' + str(root))
        emit(stream, dict(type='complete', files=count, bytes=total))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, output)
    print(json.dumps(dict(status='MANIFEST_COMPLETE', path=str(output), files=count,
                          bytes=total, training_executed=False)), flush=True)


def load(path):
    rows, footer = {}, None
    with path.open(encoding='utf-8') as stream:
        header = json.loads(next(stream))
        if header != dict(type='header', schema=1, datasets=list(DATASETS)):
            raise ValueError('Unexpected manifest header: ' + str(path))
        for line in stream:
            row = json.loads(line)
            if footer is not None:
                raise ValueError('Unexpected content after completion marker')
            if row['type'] == 'complete':
                footer = row
            elif row['type'] == 'file' and row['path'] not in rows:
                rows[row['path']] = (row['size'], row['sha256'])
            else:
                raise ValueError('Invalid or duplicate manifest entry')
    if footer != dict(type='complete', files=len(rows), bytes=sum(r[0] for r in rows.values())):
        raise ValueError('Incomplete manifest: ' + str(path))
    return rows


def compare(source, target):
    left, right = load(source), load(target)
    missing = sorted(left.keys() - right.keys())
    extra = sorted(right.keys() - left.keys())
    changed = sorted(k for k in left.keys() & right.keys() if left[k] != right[k])
    passed = not (missing or extra or changed)
    print(json.dumps(dict(status='CONTENT_CHECK_PASSED' if passed else 'DIFFERENCES_FOUND',
        source_files=len(left), target_files=len(right),
        missing_count=len(missing), extra_count=len(extra), changed_count=len(changed),
        missing_examples=missing[:30], extra_examples=extra[:30], changed_examples=changed[:30],
        scope=list(DATASETS), official_loader_verified=False, training_executed=False), indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('make')
    create.add_argument('--base', type=Path, default=Path('/data/cjm/datasets'))
    create.add_argument('--output', type=Path, required=True)
    diff = commands.add_parser('compare')
    diff.add_argument('--source', type=Path, required=True)
    diff.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'make':
        make(args.base, args.output)
        return 0
    return compare(args.source, args.target)


if __name__ == '__main__':
    raise SystemExit(main())
