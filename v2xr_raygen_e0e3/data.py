"""Frame-level Mix3, stateless RNG, legal background frames, frozen host target selection."""
import ast
import contextlib
import copy
import inspect
from pathlib import Path
import random
import textwrap
import types

import numpy as np

from .common import identity, read, safe, stream_seed
from .geometry import labels, windows


@contextlib.contextmanager
def rng_scope(seed):
    py, n = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(n)


def adapt_native(dataset):
    """Two narrow AST replacements, checked structurally; no mutation of runtime source/modules."""
    cls = type(dataset)
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__getitem__)))
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and 'self.training and len(object_stack) == 0' == ast.unparse(node.test):
            node.test = ast.parse('len(object_stack) == 0', mode='eval').body
            node.body = ast.parse('object_stack = [np.zeros((0, 7))]').body
            node.orelse = []
            changes += 1
    if changes != 1:
        raise ValueError('Zero-GT resampling source no longer matches the reviewed native loader')
    namespace = dict(cls.__getitem__.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<raygen-background-adapter>', 'exec'), namespace)
    item = namespace['__getitem__']
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.get_item_single_car)))
    class RemoveFill(ast.NodeTransformer):
        count = 0
        def visit_While(self, node):
            if 'voxel_features' not in ast.unparse(node.test):
                raise ValueError('Unexpected native while loop')
            self.count += 1
            return None
    visitor = RemoveFill()
    tree = visitor.visit(tree)
    if visitor.count != 3:
        raise ValueError('Native empty-voxel filler changed')
    namespace2 = dict(cls.get_item_single_car.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<raygen-empty-adapter>', 'exec'), namespace2)
    dataset.get_item_single_car = types.MethodType(namespace2['get_item_single_car'], dataset)
    preprocess = dataset.pre_processor.preprocess
    def process(points):
        out = preprocess(points) if len(points) else {'voxel_features': np.empty((0, 1, 4))}
        if not len(out['voxel_features']):
            slots = dataset.params['preprocess']['args']['max_points_per_voxel']
            # Sparse batch IDs require one coordinate per empty CAV. Zero intensity/data padding,
            # never returned as projected points, so it cannot create a GT hit or a ray endpoint.
            out = dict(voxel_features=np.zeros((1, slots, 4), dtype=np.float32),
                       voxel_coords=np.zeros((1, 3), dtype=np.int32),
                       voxel_num_points=np.ones(1, dtype=np.int32))
        return out
    dataset.pre_processor.preprocess = process
    original_label = dataset.post_processor.generate_label
    def generate_label(**kw):
        if np.asarray(kw['mask']).sum() == 0:
            a = kw['anchors']
            shape = a.shape[:3]
            return dict(pos_equal_one=np.zeros(shape), neg_equal_one=np.ones(shape),
                        targets=np.zeros((*shape[:2], shape[2] * 7)))
        return original_label(**kw)
    dataset.post_processor.generate_label = generate_label
    def no_world_augment(*a, **kw):
        raise ValueError('Unexpected world augmentation; cached rays would require a new geometry contract')
    dataset.data_augmentor.forward = no_world_augment
    return types.MethodType(item, dataset)


class Dataset:
    def __init__(self, output, epoch=0, auxiliary=False):
        from .runtime import arguments, hypes
        from opencood.data_utils.datasets import build_dataset
        self.root, self.epoch, self.auxiliary = Path(output), epoch, auxiliary
        self.config = read(self.root / 'config.json')
        self.manifest = read(self.root / 'preflight/train_manifest.json')
        self.native = build_dataset(hypes(output), visualize=False, train=True)
        self.keys, self.records = [], []
        for i, s in enumerate(self.manifest['scenes']):
            db = self.native.scenario_database[i]
            if list(db) != s['agents'][:7] or not db[s['ego']]['ego']:
                raise ValueError('Native CAV/ego registration differs')
            for t in s['timestamps']:
                self.keys.append([s['scene'], t, s['ego']])
                self.records.append((i, t))
            for cav, rows in db.items():
                if [t for t in rows if t != 'ego'] != s['timestamps']:
                    raise ValueError('Native CAV timestamps differ')
        if len(self.native) != len(self.keys) or self.native.max_cav != 7:
            raise ValueError('Native frame count or max_cav differs')
        self.native_item = adapt_native(self.native)
        self.last_base = None
        retrieve = self.native.retrieve_base_data
        def observe(idx, **kw):
            result = retrieve(idx, **kw)
            self.last_base = copy.deepcopy(result[0])
            return result
        self.native.retrieve_base_data = observe

    def __len__(self):
        return len(self.keys)

    def schedule(self, index, force=None):
        uid = self.keys[index]
        r = np.random.RandomState(stream_seed(self.config['seed'], self.epoch, uid, 'weather'))
        return force if force is not None else list(self.config['weather'])[int(r.randint(3))]

    def get(self, index, force=None):
        self.current_index = index
        condition = self.schedule(index, force)
        scenario, t = self.records[index]
        for cav, rows in self.native.scenario_database[scenario].items():
            path = Path(rows[t]['yaml']).with_name(t + self.config['weather'][condition] + '.pcd')
            rows[t]['lidar'] = str(safe(path))
        with rng_scope(stream_seed(self.config['seed'], self.epoch, self.keys[index], 'base')):
            item = self.native_item(index)
        item['_audit'] = dict(uid=self.keys[index], index=index, weather=condition,
                              cavs=[str(k) for k, v in self.last_base.items()
                                    if np.linalg.norm(np.asarray(v['params']['lidar_pose'])[:2]
                                                      - np.asarray(next(x for x in self.last_base.values() if x['ego'])['params']['lidar_pose'])[:2]) <= 70],
                              geometry='identity', seed=stream_seed(self.config['seed'], self.epoch, self.keys[index], 'base'))
        if self.auxiliary:
            from .cache import load_windows
            item['_aux'] = load_windows(self.root, index, self.epoch)
        return item

    def __getitem__(self, index):
        try:
            return self.get(index)
        except Exception as e:
            raise RuntimeError('Sample UID=' + repr(self.keys[index])) from e

    def collate(self, items):
        audits = [x.pop('_audit') for x in items]
        aux = [x.pop('_aux', []) for x in items]
        out = self.native.collate_batch_train(items)
        if out is None:
            raise ValueError('Background samples must not be dropped')
        # Only tensors consumed by the original detector cross into its forward path.
        inputs = {k: out['ego'][k] for k in ('processed_lidar', 'record_len')}
        return dict(inputs=inputs, labels=out['ego']['label_dict'], audits=audits, auxiliary=aux)

    def order(self):
        r = np.random.RandomState(stream_seed(self.config['seed'], self.epoch, 'all', 'order'))
        return r.permutation(len(self)).tolist()
