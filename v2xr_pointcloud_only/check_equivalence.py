"""SERVER ONLY: compare sampled official CPU batches and RNG, without a model.

Requires original images and the server's existing official native runtime.
No training, no optimizer updates and no GPU computation are performed here.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from slim_data import DATASETS, PROOF, digest, load, safe


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Run real data checks only on the remote Linux server')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slim-root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path,
                        default=Path('/data/cjm/datasets/v2xr-official-runtime-v1'))
    parser.add_argument('--frames', type=int, default=3)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError('frames must be positive')
    slim = safe(args.slim_root)
    proof = load(slim)
    source = safe(Path(proof['source_base']))
    runtime = safe(args.runtime)
    project = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project / 'gspr_repe'))
    sys.path.insert(0, str(runtime))
    import numpy as np
    import torch
    import opencood
    if runtime not in Path(opencood.__file__).resolve().parents:
        raise ValueError('Wrong opencood import: ' + opencood.__file__)
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from camera_metadata import install
    def fingerprint(value):
        h = hashlib.sha256()
        def visit(x):
            if isinstance(x, torch.Tensor):
                visit(x.detach().cpu().numpy())
            elif isinstance(x, np.ndarray):
                h.update(str((x.shape, x.dtype)).encode())
                h.update(x.tobytes())
            elif isinstance(x, dict):
                for k in sorted(x):
                    h.update(str(k).encode()); visit(x[k])
            elif isinstance(x, (tuple, list)):
                h.update(str(type(x)).encode())
                for part in x:
                    visit(part)
            else:
                h.update(repr(x).encode())
        visit(value)
        return h.hexdigest()
    def sample(dataset, index):
        random.seed(20261017 + index)
        np.random.seed(20261017 + index)
        torch.manual_seed(20261017 + index)
        item = dataset[index]
        batch = dataset.collate_batch_train([item]) if item is not None else None
        return dict(batch=fingerprint(batch), rng=fingerprint(
            [random.getstate(), np.random.get_state(), torch.get_rng_state()]))
    hypes = yaml_utils.load_yaml(str(runtime / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    os.environ['POINTCLOUD_ONLY_ROOT'] = str(slim)
    observed = []
    for mode in ('mixed_train', 'clean_validate'):
        datasets = []
        for base in (source, slim):
            cfg = copy.deepcopy(hypes)
            cfg.update(root_dir=str(base / DATASETS[0] / 'train'),
                       validate_dir=str(base / DATASETS[2] / 'validate'),
                       train_sim=mode == 'mixed_train', eval_sim=False, sim_weather='_fog_0.060')
            datasets.append(build_dataset(cfg, visualize=False, train=mode == 'mixed_train'))
        if len(datasets[0]) != len(datasets[1]):
            raise ValueError('Frame count differs')
        count = min(args.frames, len(datasets[0]))
        indices = sorted({int(i * (len(datasets[0])-1) / max(count-1, 1)) for i in range(count)})
        for index in indices:
            original = sample(datasets[0], index)
            from argparse import Namespace
            config = Namespace(train_root=slim / DATASETS[0], paired_root=slim / DATASETS[1],
                               view_root=slim / DATASETS[2])
            _, undo = install(config, require_equivalence=False)
            try:
                optimized = sample(datasets[1], index)
            finally:
                undo()
            if original != optimized:
                raise ValueError('Batch tensors/labels or RNG differ at %s/%d' % (mode, index))
            observed.append(dict(mode=mode, index=index, **original))
            print('EQUAL', mode, index, flush=True)
    report = dict(status='SAMPLED_INPUTS_EQUAL', metadata_identity=proof['identity'],
        camera_adapter_sha256=digest(project / 'gspr_repe/camera_metadata.py'), samples=observed,
        scope='sampled official collated tensors/labels and post-sample RNG; not full training equivalence',
        runtime=str(runtime), image_pixels_included=False, training_executed=False)
    path = slim / 'EQUIVALENCE.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != report:
            raise ValueError('Existing equivalence report differs; preserve it and inspect before replacing')
    else:
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
