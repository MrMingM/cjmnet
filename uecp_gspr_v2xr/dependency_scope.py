"""Direct GSPR/UECP runtime files; historical native provenance remains fully pinned."""
import ast
from pathlib import Path


DIRECT_PREFIXES = ('attfuse_gspr/', 'related_code/UECP-main/')


def direct_files(project):
    root = Path(project)
    folder = root / 'related_code/UECP-main/opencood'
    required = {'attfuse_gspr/reliability.py',
        'related_code/UECP-main/opencood/hypes_yaml/v2v4real/LiDAR_only/UAPF.yaml'}
    pending = ['models.bev_text_v1', 'loss.point_pillar_depth_loss']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        path = folder.joinpath(*name.split('.')).with_suffix('.py')
        required.add(path.relative_to(root).as_posix())
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('opencood.'):
                pending.append(node.module[len('opencood.'):])
            elif isinstance(node, ast.Import):
                pending.extend(a.name[len('opencood.'):] for a in node.names if a.name.startswith('opencood.'))
    return required


def validate_scope(project, pins):
    actual = {name for name in pins if name.startswith(DIRECT_PREFIXES)}
    expected = direct_files(project)
    if actual != expected:
        raise ValueError('Direct runtime dependency scope differs; missing=' + repr(sorted(expected-actual))
                         + '; unrelated=' + repr(sorted(actual-expected)))
