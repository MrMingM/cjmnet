"""Only the audited V2X-R runtime may own the top-level opencood namespace."""
from pathlib import Path

from v2xr_raygen_e0e3.runtime import arguments, legacy_paths, input_stamps, audit_imports
from v2xr_raygen_e0e3.runtime import bootstrap as native_bootstrap, hypes as native_hypes

from .common import read

__all__ = ['arguments', 'legacy_paths', 'input_stamps', 'audit_imports', 'bootstrap', 'host_config', 'loader_config']


def bootstrap(output, gpu=True):
    return native_bootstrap(output, gpu)


def host_config(output):
    from .upstream import SOURCE
    import yaml
    c = read(Path(output) / 'config.json')
    h = yaml.safe_load((SOURCE / 'hypes_yaml/v2v4real/LiDAR_only/UAPF.yaml').read_text())
    a = h['model']['args']
    a.update(lidar_range=c['range'], voxel_size=c['voxel'], use_gt_unmap=False)
    a['point_pillar_scatter']['grid_size'] = [704, 200, 1]
    a['pillar_vfe']['use_absolute_xyz'] = True
    h['optimizer'] = dict(core_method='AdamW', **c['optimizer'])
    h['lr_scheduler'] = c['scheduler']
    return h


def loader_config(output):
    h = native_hypes(output)
    # Native postprocessor already implements UECP's direction correction when dm is present.
    h['postprocess']['dir_args'] = host_config(output)['model']['args']['dir_args']
    return h
