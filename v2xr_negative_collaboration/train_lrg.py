"""Run the existing LR-G recipe on GPU 3, without starting LR1 training."""
import os
from pathlib import Path
import sys

from audit import devices
from v2xr_lr1 import run as native

sys.path.insert(0, str(native.HERE))


if __name__ == '__main__':
    devices.idle(['E1'])
    device = devices.device('E1')
    print('E1 training device', device, flush=True)
    target = Path(os.environ['LRG_RUN_DIR'])
    if target.name != 'LRG':
        raise ValueError('Expected training directory named LRG')
    native.ROOT = target.parent
    native.ROOT.mkdir(parents=True, exist_ok=True)
    with native.lock(native.ROOT / '.lrg-training.lock'):
        sources = native.source_contract()
        data_file = native.ROOT / 'DATA.json'
        data = native.read_json(data_file) if data_file.is_file() else native.inventory(data_file)
        for split, manifest, frames in (('train', native.TRAIN_MANIFEST, 8083),
                                        ('validate', native.VAL_MANIFEST, 829)):
            row = data['splits'][split]
            if (data['status'] != 'complete' or row['frames'] != frames
                    or row['input_manifest_sha256'] != native.sha256(manifest)):
                raise ValueError(f'Invalid {split} inventory')
        frozen = native.read_json(native.LR0 / 'CONTRACT.json')
        if native.sha256(native.LR0 / 'DATA.json') != frozen['data_sha256']:
            raise ValueError('Frozen LR0 inventory changed')
        preflight = native.ROOT / 'PREFLIGHT.json'
        if not preflight.is_file():
            native.preflight(native.LR0 / 'config.yaml', preflight)
        check = native.read_json(preflight)
        if check['status'] != 'complete' or not check['gspr_initialization_identical']:
            raise ValueError('Invalid LR-G initialization preflight')
        native.run_variant('LRG', sources, dict(device, python=sys.version,
                                               hash_seed=os.environ['PYTHONHASHSEED']))
