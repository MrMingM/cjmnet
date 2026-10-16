"""Execute upstream setup, selecting native extensions for the AttFuse build.

Original extension objects, sources, flags and BuildExtension are retained.
Only the ext_modules selection passed into setuptools.setup is narrowed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

from audit_native_dependencies import audit
from common import PACKAGE, atomic_json, read_json, server_only, sha256

SELECTED = (
    'opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_cuda',
)
SKIPPED = (
    'opencood.pcdet_utils.iou3d_nms.iou3d_nms_cuda',
    'opencood.pcdet_utils.pointnet2.pointnet2_stack.pointnet2_stack_cuda',
    'opencood.pcdet_utils.pointnet2.pointnet2_batch.pointnet2_batch_cuda',
)


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    root = Path(args.runtime).resolve(strict=True)
    setup_path = root / 'opencood/pcdet_utils/setup.py'
    expected = read_json(PACKAGE / 'UPSTREAM.json')['files']
    if sha256(setup_path) != expected['opencood/pcdet_utils/setup.py']:
        raise ValueError('Upstream setup.py changed')
    dependency_audit = audit(root)
    import setuptools
    original_setup = setuptools.setup

    def selected_setup(*positional, **keywords):
        extensions = keywords['ext_modules']
        names = [extension.name for extension in extensions]
        if set(names) != set(SELECTED + SKIPPED) or len(names) != 4:
            raise ValueError(f'Unexpected official extension definitions: {names}')
        chosen = [extension for extension in extensions if extension.name in SELECTED]
        # Retain the official CUDAExtension objects, including automatic HIP translation.
        record = dict(mode='official_setup_with_AttFuse_extension_selection',
            upstream_setup_sha256=sha256(setup_path), selected=list(SELECTED), skipped=list(SKIPPED),
            reason='The official pure LiDAR AttFuse import closure requires roiaware_pool3d and '
                   'the separately built box_overlaps; PointNet2 and pcdet iou3d_nms are not reached.',
            dependency_audit=dependency_audit,
            compiled_sources={extension.name: [dict(path=str(Path(p).resolve()),
                sha256=sha256(p)) for p in extension.sources] for extension in chosen},
            algorithm_source_changes=[], detection_or_metric_changes=[])
        atomic_json(root / 'native_build_selection.json', record)
        print('Official native extensions selected: ' + ', '.join(SELECTED), flush=True)
        print('PointNet2/pcdet iou3d_nms excluded from this AttFuse-only build.', flush=True)
        keywords['ext_modules'] = chosen
        return original_setup(*positional, **keywords)

    setuptools.setup = selected_setup
    old_argv = sys.argv
    try:
        sys.argv = [str(setup_path), 'build_ext', '--inplace']
        runpy.run_path(str(setup_path), run_name='__main__')
    finally:
        setuptools.setup = original_setup
        sys.argv = old_argv


if __name__ == '__main__':
    main()
