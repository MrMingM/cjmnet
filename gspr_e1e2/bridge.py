"""Reuse immutable training/evaluation engines via reviewed insertion points."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

def load(name, relative):
    key = '_e12_' + name + '_' + hashlib.sha256(str(relative).encode()).hexdigest()[:12]
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]

def adapt_training(module):
    tree = ast.parse((ROOT/'gspr_repe/training.py').read_text(encoding='utf-8'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    inserted = 0
    for index, node in enumerate(run.body):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "contract['identity']":
            run.body.insert(index, ast.parse("contract['e0_initialization'] = initialization_reference(args)").body[0])
            inserted += 1
            break
    for index, node in enumerate(run.body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'model':
                node.value = ast.Call(func=ast.Name(id='initialize_model', ctx=ast.Load()),
                    args=[node.value, ast.Name(id='args', ctx=ast.Load())], keywords=[])
                inserted += 1
            if node.targets[0].id == 'writer':
                run.body.insert(index, ast.parse('restore_initial_rng(args)').body[0])
                inserted += 1
                break
    if inserted != 3:
        raise ValueError('Unexpected frozen training AST: expected exactly three insertions')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[run], type_ignores=[])),
                 str(ROOT/'gspr_repe/training.py')+':E12', 'exec'), module.__dict__)
