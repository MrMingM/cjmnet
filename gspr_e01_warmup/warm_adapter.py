"""Two commit-boundary return guards around the unchanged original formal training body."""
import ast
import copy
from pathlib import Path


def pause_tree(tree):
    tree = copy.deepcopy(tree)
    method = tree.body[0]
    assignments = [node for node in method.body if isinstance(node, ast.Assign)
                   and ast.unparse(node) == 'completed = len(rows)']
    if len(assignments) != 1:
        raise ValueError('Expected one original committed-epoch count')
    guard = ast.parse('if _formal_warmup_limit is not None and completed >= _formal_warmup_limit:\n    return').body[0]
    method.body.insert(method.body.index(assignments[0]) + 1, guard)
    loops = [node for node in ast.walk(method) if isinstance(node, ast.For)
             and ast.unparse(node.target) == 'epoch' and ast.unparse(node.iter).startswith('range(completed,')]
    if len(loops) != 1 or 'EPOCH COMMITTED' not in ast.unparse(loops[0].body[-1]):
        raise ValueError('Original final epoch commit/progress boundary differs')
    loops[0].body.extend(ast.parse(
        'if _formal_warmup_limit is not None and epoch + 1 >= _formal_warmup_limit:\n    return').body)
    return ast.fix_missing_locations(tree)


def adapted_run(module, limit):
    from adapters import training_tree
    from e01_common import identity
    original, edits = training_tree(Path(module.__file__).read_text(encoding='utf-8'))
    tree = pause_tree(original)
    namespace = dict(module.__dict__, _formal_warmup_limit=limit)
    exec(compile(tree, module.__file__ + ' [formal warmup commit boundary]', 'exec'), namespace)
    # Metadata is identical in the 2-epoch and continuation processes, as is the 20-epoch contract.
    return namespace['run'], dict(frozen_E01_edits=edits,
        adapted_ast_identity=identity(ast.dump(tree, include_attributes=False)),
        formal_startup='Optional return after full epoch commit/progress; total formal epochs remain 20',
        extra_return_guards=2)
