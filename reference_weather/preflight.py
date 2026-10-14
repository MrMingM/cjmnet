"""Fail early on the server before large downloads or expensive simulation."""
from __future__ import annotations

from pathlib import Path


def preflight(config, require_gpu=False):
    import numpy as np
    import scipy.integrate
    if not hasattr(scipy.integrate, 'trapz'):
        scipy.integrate.trapz = np.trapz
    try:
        import PyMieScatt
        import open3d
        import sklearn
        import torch
        from reference_weather.vendor.snow import simulation
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.data_utils.datasets import build_dataset
        from opencood.tools import inference_utils
        from attfuse_code.point_pillar_intermediate import PointPillarIntermediate
    except ImportError as error:
        raise RuntimeError('Missing server dependency. In the opencood environment run: '
                           'python -m pip install -r reference_weather/requirements-server.txt') from error
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError('Selected HIP GPU is unavailable; check GPU/ROCR_VISIBLE_DEVICES.')
    config_path = Path(config['model_config']).resolve(strict=True)
    checkpoint = Path(config['checkpoint']).resolve(strict=True)
    hypes = load_yaml(str(config_path))
    if hypes['model']['core_method'] != 'point_pillar_intermediate':
        raise ValueError('A pure AttFuse configuration is required')
    model = PointPillarIntermediate(hypes['model']['args'])
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    del model, state
    print(f'Preflight OK: strict pure AttFuse checkpoint={checkpoint}', flush=True)
