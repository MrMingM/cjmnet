"""Create scene-level development/locked validation roots using symlinks.

No point-cloud files are copied or modified. Each scenario directory is placed
entirely in one split so consecutive frames cannot leak across splits.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare deterministic scene-level synthetic validation.')
    parser.add_argument('--clean_validate_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dev_ratio', default=0.5, type=float)
    parser.add_argument('--seed', default=20, type=int)
    return parser.parse_args()


def ensure_empty_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(
            'Refusing to modify non-empty directory: %s' % path)


def link_scenarios(names, source_root, split_root):
    for name in names:
        source = (source_root / name).resolve()
        destination = split_root / name
        os.symlink(str(source), str(destination), target_is_directory=True)


def main():
    opt = parse_args()
    if not 0.0 < opt.dev_ratio < 1.0:
        raise ValueError('--dev_ratio must be between 0 and 1')

    source_root = Path(opt.clean_validate_dir).resolve()
    output_root = Path(opt.output_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError('output_dir must not be inside clean_validate_dir')

    scenario_names = sorted(
        entry.name for entry in source_root.iterdir() if entry.is_dir())
    if len(scenario_names) < 2:
        raise ValueError('at least two scenario directories are required')

    rng = np.random.default_rng(opt.seed)
    shuffled = np.asarray(scenario_names, dtype=object)
    rng.shuffle(shuffled)
    dev_count = int(round(len(shuffled) * opt.dev_ratio))
    dev_count = min(max(dev_count, 1), len(shuffled) - 1)
    dev_names = sorted(str(name) for name in shuffled[:dev_count])
    locked_names = sorted(str(name) for name in shuffled[dev_count:])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / 'scene_split_manifest.json'
    if manifest_path.exists():
        raise FileExistsError(
            'Refusing to overwrite existing manifest: %s' % manifest_path)
    dev_root = output_root / 'development'
    locked_root = output_root / 'locked'
    ensure_empty_directory(dev_root)
    ensure_empty_directory(locked_root)

    link_scenarios(dev_names, source_root, dev_root)
    link_scenarios(locked_names, source_root, locked_root)
    manifest = {
        'source_root': str(source_root),
        'seed': int(opt.seed),
        'dev_ratio': float(opt.dev_ratio),
        'development': dev_names,
        'locked': locked_names
    }
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write('\n')

    print('development scenarios: %d -> %s' %
          (len(dev_names), dev_root))
    print('locked scenarios: %d -> %s' %
          (len(locked_names), locked_root))
    print('manifest: %s' % manifest_path)


if __name__ == '__main__':
    main()
