"""Explicit no-pixel camera input adapter; frozen official source stays unchanged."""
import os
from pathlib import Path
from functools import lru_cache


class CameraSize:
    """The official LiDAR dataset only accesses camera0.size, never its pixels."""
    def __init__(self, dimensions):
        self.size = tuple(dimensions)


def tools():
    from auditlib import ROOT, load_frozen
    return load_frozen('pointcloud_only_tools', 'slim_data.py', ROOT / 'v2xr_pointcloud_only')


def root():
    value = os.environ.get('POINTCLOUD_ONLY_ROOT')
    return Path(value).expanduser().resolve(strict=True) if value else None


def proof():
    slim = root()
    if slim is None:
        return None
    stat = (slim / tools().PROOF).stat()
    return _cached_proof(str(slim), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2)
def _cached_proof(path, mtime_ns, size):
    return tools().load(Path(path))


def remap_original(path):
    path = Path(path)
    data = proof()
    if data is None:
        return path
    try:
        relative = path.relative_to(data['source_base'])
    except ValueError:
        raise ValueError('Historical view path outside recorded source base: ' + str(path))
    if relative.parts[0] not in tools().DATASETS:
        raise ValueError('Historical view path outside selected datasets: ' + str(path))
    return root() / relative


def removed_image(path, expected, data=None):
    if root() is None or Path(path).suffix.lower() != '.png':
        return False
    data = data if data is not None else proof()
    name = Path(path).relative_to(root()).as_posix()
    row = data['images'].get(name)
    if (row is None or row['size'] != expected['size'] or row['sha256'] != expected['sha256']
            or Path(path).exists()):
        raise ValueError('Removed image is not bound to original extraction: ' + str(path))
    return True


def install(args, require_equivalence=True):
    """Return evidence and an undo function; installing does not mutate source files."""
    slim = root()
    if slim is None:
        return None, lambda: None
    data = proof()
    if require_equivalence:
        import json
        report = json.loads((slim / 'EQUIVALENCE.json').read_text(encoding='utf-8'))
        if (report.get('status') != 'SAMPLED_INPUTS_EQUAL'
                or report.get('metadata_identity') != data['identity']
                or report.get('camera_adapter_sha256') != tools().digest(Path(__file__))):
            raise ValueError('Run check_equivalence.py on the original server with this adapter first')
    for attribute, name in (('train_root', 'v2xr-official-train-v2'),
                            ('view_root', 'v2xr-official-validate-v1'),
                            ('paired_root', 'v2xr-lidar-weather-validate-v1')):
        if Path(getattr(args, attribute)).resolve() != slim / name:
            raise ValueError('Slim mode requires matching ' + attribute)
    from opencood.data_utils.datasets.opv2v import basedataset
    previous = basedataset.load_camera_data
    if getattr(previous, '_pointcloud_only', False):
        raise ValueError('Camera metadata adapter already installed in this process')
    overlay = Path(os.environ.get('CUDA_INPUT_ROOT', '/nonexistent-overlay')) / 'validate'
    view = slim / 'v2xr-official-validate-v1/validate'
    def sizes(camera_files, preload=True):
        result = []
        for filename in camera_files:
            path = Path(filename)
            if path.is_relative_to(overlay):
                path = view / path.relative_to(overlay)
            try:
                key = path.relative_to(slim).as_posix()
            except ValueError:
                raise ValueError('Camera path outside slim inventory: ' + str(path))
            row = data['images'].get(key)
            if row is None:
                raise ValueError('Missing exact camera dimensions: ' + key)
            result.append(CameraSize(row['dimensions']))
        return result
    sizes._pointcloud_only = True
    basedataset.load_camera_data = sizes
    evidence = dict(mode='exact_png_dimensions_no_pixels', metadata_identity=data['identity'],
                    metadata_sha256=tools().digest(slim / tools().PROOF),
                    fov_enabled_unchanged=True, radar_unchanged=True)
    def undo():
        basedataset.load_camera_data = previous
    return evidence, undo
