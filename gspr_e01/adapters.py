"""Reviewed, asserted AST edits of two frozen functions; never edit their files."""
import ast
from pathlib import Path
import types
from e01_common import SETTINGS, identity

SUFFIX = dict(clean='', fog='_fog_0.060', snow='_snow_2.5_2.0')


def pair_key(path):
    path = Path(path)
    for weather in ('fog', 'snow', 'clean'):
        ending = SUFFIX[weather]+'.pcd'
        if path.name.endswith(ending):
            stamp = path.name[:-len(ending)]
            if not stamp or '_' in stamp:
                break
            return (path.parent.parent.name, path.parent.name, stamp), weather
    raise ValueError('Unexpected training LiDAR path: '+str(path))


def select_path(path, draw, recipe):
    if recipe not in SETTINGS['draw_order'] or not 0 <= draw < 7:
        raise ValueError('Invalid registered per-CAV weather draw')
    if pair_key(path)[1] != 'clean':
        raise ValueError('Native retrieve must supply the clean source filename')
    return str(path)[:-4]+SUFFIX[SETTINGS['draw_order'][recipe][draw]]+'.pcd'


def retrieve_tree(source):
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseDataset']
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == 'retrieve_base_data'] if len(classes) == 1 else []
    if len(methods) != 1:
        raise ValueError('Frozen retrieve_base_data signature differs')
    method = methods[0]
    hits = []
    for node in ast.walk(method):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'self.train_sim and self.train':
            hits.append(node)
    if len(hits) != 1 or hits[0].orelse:
        raise ValueError('Expected one original per-CAV train_sim branch')
    expected = ast.parse("aug = random.randint(0, 6)\nif aug <= 2:\n    lidar_file = lidar_file.replace('.pcd', self.sim_weather + '.pcd')").body
    if ast.dump(ast.Module(body=hits[0].body, type_ignores=[]), include_attributes=False) != ast.dump(ast.Module(body=expected, type_ignores=[]), include_attributes=False):
        raise ValueError('Original sampling branch differs; inspect before adapting')
    hits[0].body = ast.parse("aug = random.randint(0, 6)\nlidar_file = _e01_select_path(lidar_file, aug, self._e01_recipe)").body
    result = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    return result


def install_sampler(dataset, recipe, source):
    tree = retrieve_tree(Path(source).read_text(encoding='utf-8'))
    native = dataset.retrieve_base_data.__func__
    namespace = dict(native.__globals__, _e01_select_path=select_path)
    exec(compile(tree, str(source)+' [E01 per-instance sampler]', 'exec'), namespace)
    dataset._e01_recipe = recipe
    dataset.retrieve_base_data = types.MethodType(namespace['retrieve_base_data'], dataset)
    return dict(method='BaseDataset.retrieve_base_data', scope='this training dataset instance only',
        transformation='one randint(0,6) per CAV; only selected LiDAR suffix changes',
        adapted_ast_identity=identity(ast.dump(tree, include_attributes=False)))


def training_tree(source):
    """Keep the frozen training loop, optimizer, loss, scheduler and commit order."""
    tree = ast.parse(source)
    methods = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run']
    if len(methods) != 1:
        raise ValueError('Frozen training entry changed')
    counts = dict(epochs=0, subset=0, weather=0)
    class Adapt(ast.NodeTransformer):
        def visit_IfExp(self, node):
            if ast.unparse(node) == '1 if args.smoke else epochs':
                counts['epochs'] += 1
                node.body = ast.Constant(SETTINGS['smoke_epochs'])
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node) == 'range(16)':
                counts['subset'] += 1
                node.args = [ast.Constant(SETTINGS['smoke_train_frames'])]
            if ast.unparse(node) == 'dict(clean=0, fog=0)':
                counts['weather'] += 1
                node.keywords.append(ast.keyword(arg='snow', value=ast.Constant(0)))
            return self.generic_visit(node)
    result = ast.fix_missing_locations(Adapt().visit(ast.Module(body=methods, type_ignores=[])))
    if counts != dict(epochs=3, subset=2, weather=1):
        raise ValueError('Unexpected frozen training AST edit counts: '+repr(counts))
    return result, counts


def adapted_run(module):
    tree, counts = training_tree(Path(module.__file__).read_text(encoding='utf-8'))
    namespace = dict(module.__dict__)
    exec(compile(tree, module.__file__+' [E01 weather/smoke adapter]', 'exec'), namespace)
    return namespace['run'], dict(edits=counts, adapted_ast_identity=identity(ast.dump(tree, include_attributes=False)))
