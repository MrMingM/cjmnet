"""Read-only V2X-R input inventory. No Torch, model, GPU or Git required."""
import argparse
import json
from pathlib import Path


def inspect(root, split, weather):
    root = root.expanduser().resolve()
    if (root / split).is_dir():
        root = root / split
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    missing, count, scenes = [], 0, []
    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        local = 0
        for cav in sorted(p for p in scene.iterdir() if p.is_dir()):
            for yaml in sorted(cav.glob('*.yaml')):
                if 'additional' in yaml.name or yaml.name == 'data_protocol.yaml':
                    continue
                stamp = yaml.stem
                names = [stamp + '.pcd', stamp + '_radar.pcd']
                names += [stamp + '_camera' + str(i) + '.png' for i in range(4)]
                if weather:
                    names += [stamp + '_fog_0.060.pcd', stamp + '_snow_2.5_2.0.pcd']
                for name in names:
                    file = cav / name
                    if not file.is_file() or file.stat().st_size == 0:
                        missing.append(str(file))
                count += 1
                local += 1
        scenes.append(dict(scene=scene.name, single_cav_frames=local))
    if not count:
        raise ValueError('No per-frame YAML files found under scene/CAV directories: ' + str(root))
    return dict(root=str(root), single_cav_frames=count, scenes=scenes,
                missing_count=len(missing), missing_examples=missing[:30])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--validate', required=True, type=Path)
    args = parser.parse_args()
    results = {key: inspect(path, key, True) for key, path in
               [('train', args.train), ('validate', args.validate)]}
    passed = not any(row['missing_count'] for row in results.values())
    print(json.dumps(dict(status='file_inventory_passed' if passed else 'missing_files',
        results=results, content_hashes_verified=False, official_loader_verified=False,
        training_executed=False), ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
