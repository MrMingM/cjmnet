"""Static import closure only; safe locally, never import upstream modules."""
from __future__ import annotations

import ast
from pathlib import Path

ROOTS = ('opencood.tools.inference', 'opencood.tools.inference_utils',
         'opencood.tools.train_utils', 'opencood.data_utils.datasets',
         'opencood.models.point_pillar_intermediate', 'opencood.utils.eval_utils_opv2v')


def audit(root):
    root = Path(root)
    pending, seen, native, dynamic = list(ROOTS), {}, set(), []

    def source(module):
        path = root.joinpath(*module.split('.'))
        if path.with_suffix('.py').is_file():
            return path.with_suffix('.py')
        if (path / '__init__.py').is_file():
            return path / '__init__.py'
        return None

    def add(module):
        if module != 'opencood' and not module.startswith('opencood.'):
            return
        parts = module.split('.')
        # Importing a leaf executes its parent package initializers too.
        for size in range(1, len(parts) + 1):
            name = '.'.join(parts[:size])
            if name not in seen and source(name):
                pending.append(name)
        if module.endswith('_cuda') or module == 'opencood.utils.box_overlaps':
            native.add(module)

    while pending:
        module = pending.pop()
        if module in seen:
            continue
        path = source(module)
        if path is None:
            continue
        seen[module] = path.relative_to(root).as_posix()
        add(module)
        package = module if path.name == '__init__.py' else module.rsplit('.', 1)[0]
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    parent = package.split('.')
                    base = '.'.join(parent[:len(parent) - node.level + 1] + ([base] if base else []))
                add(base)
                for alias in node.names:
                    if alias.name != '*':
                        add(base + '.' + alias.name)
            elif isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in ('importlib.import_module', '__import__'):
                    dynamic.append(dict(module=module, line=node.lineno, expression=ast.unparse(node)))
    pointnet = sorted(n for n in set(seen) | native if '.pointnet2' in n)
    if pointnet:
        raise ValueError(f'Official AttFuse static import closure reaches PointNet2: {pointnet}')
    return dict(roots=list(ROOTS), python_modules=dict(sorted(seen.items())),
        native_imports=sorted(native), pointnet2_reached=False, dynamic_import_sites=dynamic,
        boundary='Conservative AST closure includes function/conditional imports. '
                 'Dynamic model dispatch is explicitly seeded with official point_pillar_intermediate; '
                 'actual imported modules must also be checked on the server.')


if __name__ == '__main__':
    import json
    print(json.dumps(audit(Path(__file__).resolve().parent / 'upstream'), indent=2))
