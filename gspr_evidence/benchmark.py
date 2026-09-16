"""Historical OPV2V/OPV2V-W test protocol; never synthesizes weather."""
import copy
from pathlib import Path

ROOTS = {
    'clean': '/data/scd/datasets/opv2v_official_data_dumping/test',
    'fog': '/data/cjm/datasets/opv2v-w/fog/test',
    'rain': '/data/cjm/datasets/opv2v-w/rain/test',
    'snow': '/data/cjm/datasets/opv2v-w/snow/test',
}


def test_hypes(hypes, weather):
    result = copy.deepcopy(hypes)
    result['validate_dir'] = ROOTS[weather]
    result.pop('weather_augmentation', None)
    result['data_augment'] = []
    return result


def load_config(config, frontend, weather):
    import yaml
    from opencood.hypes_yaml.yaml_utils import load_yaml
    # Keep training options byte-for-byte semantically intact for checkpoint validation.
    # In particular, do not resolve/require training-time fog lookup tables here.
    options = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    hypes = load_yaml(str(frontend))
    if hypes['fusion']['core_method'] != 'IntermediateFusionDataset':
        raise ValueError('Expected IntermediateFusionDataset')
    hypes['fusion']['args'] = {'proj_first': True}
    hypes = test_hypes(hypes, weather)
    if not Path(hypes['validate_dir']).is_dir():
        raise FileNotFoundError('Historical test dataset missing: '+hypes['validate_dir'])
    return options, hypes
