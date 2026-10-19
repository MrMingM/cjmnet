"""Reuse the frozen D-1D training loop/commit/resume machinery with explicit GSPR metadata."""
import copy
import json
from pathlib import Path
import re

from auditlib import SETTINGS, load_frozen

inherited = load_frozen('training', 'training.py')


def train_config(args, output):
    import yaml
    from opencood.hypes_yaml import yaml_utils
    source = Path(args.runtime_root) / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'
    text = source.read_text(encoding='utf-8')
    expected = copy.deepcopy(yaml.safe_load(text))
    for key, value in dict(root_dir=str(Path(args.train_root) / 'train'),
                           validate_dir=str(Path(args.view_root) / 'validate')).items():
        text, count = re.subn('^' + key + r':[^\n]*$', lambda m: key + ': ' + json.dumps(value), text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError('Expected one official path field')
        expected[key] = value
    # Explicit extra metadata; official config's model/loss/data/optimizer/scheduler values stay intact.
    expected['gspr_experiment'] = SETTINGS
    text += '\ngspr_experiment: ' + json.dumps(SETTINGS, ensure_ascii=False) + '\n'
    if yaml.safe_load(text) != expected:
        raise ValueError('Unexpected training configuration mutation')
    destination = output / 'config.yaml'
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        raise ValueError('GSPR training configuration changed')
    destination.write_text(text, encoding='utf-8', newline='\n')
    hypes = yaml_utils.load_yaml(str(destination))
    if (hypes['train_params'] != dict(batch_size=8, epoches=20, eval_freq=1, save_freq=1)
            or hypes['fusion']['core_method'] != 'IntermediateFusionDatasetV2XR'
            or not hypes['fusion']['args']['proj_first'] or not hypes['fov']
            or hypes['train_sim'] or hypes['eval_sim'] or 'max_cav' in hypes['train_params']):
        raise ValueError('Official Clean training flags changed')
    return destination, hypes


def run(args, own):
    inherited.train_config = train_config
    print('V2XR-GSPR: random-init Clean training; official O Clean AP selects epoch.', flush=True)
    return inherited.run(args, own)
