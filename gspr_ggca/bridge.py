"""Load byte-pinned engines without editing their source or generic module names."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def load(name, relative):
    # The historical engine uses the same short names in several directories.
    key = '_ggca_frozen_' + name + '_' + hashlib.sha256(str(relative).encode()).hexdigest()[:12]
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def adapt_training(module):
    """Three reviewed insertion points; keep the optimizer/data/commit loop."""
    tree = ast.parse((ROOT / 'gspr_repe/training.py').read_text(encoding='utf-8'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    count = 0
    for index, node in enumerate(run.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Name) and t.value.id == 'contract'
                and isinstance(t.slice, ast.Constant) and t.slice.value == 'identity' for t in node.targets):
            run.body.insert(index, ast.parse("contract['ggca_initialization'] = initialization_reference(args)").body[0])
            count += 1
            break
    for index, node in enumerate(run.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model' for t in node.targets):
            node.value = ast.Call(func=ast.Name(id='initialize_model', ctx=ast.Load()),
                                  args=[node.value, ast.Name(id='args', ctx=ast.Load())], keywords=[])
            count += 1
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'writer' for t in node.targets):
            run.body.insert(index, ast.parse('restore_initial_rng(args)').body[0])
            count += 1
            break
    if count != 3:
        raise ValueError('Frozen training insertion points changed')
    code = ast.fix_missing_locations(ast.Module(body=[run], type_ignores=[]))
    exec(compile(code, str(ROOT / 'gspr_repe/training.py') + ':GGCA', 'exec'), module.__dict__)

