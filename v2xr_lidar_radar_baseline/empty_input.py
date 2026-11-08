"""Optional empty-observation representation. Enable only after protocol approval."""
import ast
import inspect
import textwrap
from types import MethodType


def allow_empty_observations(dataset):
    """Remove only the two active synthetic-point retry loops from the native method."""
    method = type(dataset).get_item_single_car
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    removed = []
    for statement in tree.body[0].body:
        if not isinstance(statement, ast.While):
            continue
        node_names = {node.id for node in ast.walk(statement.test) if isinstance(node, ast.Name)}
        if node_names not in ({'processed_lidar'}, {'processed_radar'}):
            continue
        draws = [node for node in ast.walk(statement)
                 if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                 and node.value.id == 'random' and node.attr == 'choices']
        if len(draws) != 6:
            raise RuntimeError('Official empty-voxel fallback changed')
        statement.body = [ast.copy_location(ast.Break(), statement.body[0])]
        removed.extend(node_names)
    if sorted(removed) != ['processed_lidar', 'processed_radar']:
        raise RuntimeError('Expected exactly the two official modality fallback loops')
    namespace = {}
    code = compile(ast.fix_missing_locations(tree), inspect.getfile(method), 'exec')
    exec(code, method.__globals__, namespace)
    dataset.get_item_single_car = MethodType(namespace[method.__name__], dataset)


def preserve_empty_vehicle_slots(model):
    """Keep native encoding/scatter for real pillars and zero-fill unobserved grid slots."""
    import torch
    import torch.nn.functional as F

    for encoder in (model.lidar_pillar_vfe, model.radar_pillar_vfe):
        native = encoder.forward

        def encode(batch, forward=native):
            voxels = batch['voxel_features']
            if not len(voxels):
                batch['pillar_features'] = voxels.new_empty((0, 64))
                return batch
            result = forward(batch)
            result['pillar_features'] = result['pillar_features'].reshape(-1, 64)
            return result

        encoder.forward = encode

    scatter = model.scatter
    native_scatter = scatter.forward

    def scatter_with_slots(batch):
        coords = batch['voxel_coords']
        count = int(batch['record_len'].sum())
        if not len(coords):
            features = batch['pillar_features'].new_zeros((count, 64, scatter.ny, scatter.nx))
        else:
            result = native_scatter(batch)
            features = result['spatial_features']
            if features.shape[0] > count:
                raise ValueError('Voxel vehicle index exceeds actual vehicle count')
            features = F.pad(features, (0, 0, 0, 0, 0, 0, 0, count - features.shape[0]))
        batch['spatial_features'] = features
        batch['spatial_features_3d'] = features.unsqueeze(2)
        return batch

    scatter.forward = scatter_with_slots


def empty_slots(ego):
    """Count true empty vehicle/modalities before any feature processing."""
    count = int(ego['record_len'].sum())
    return {sensor: count - len(ego[f'processed_{sensor}']['voxel_coords'][:, 0].unique())
            for sensor in ('lidar', 'radar')}

