"""Additive hooks on original GSPR and PFN; no frozen source/weight mutation."""
import numpy as np
from e0_common import read, sha
from e0_numeric import (array_hash, author_mask_proxy, distribution, nearest_proxy, range_group,
                        replace_weights)
from e0_runtime import npz_write, points_in_targets, state_hash, tensor_numpy


class Observer:
    def __init__(self, model, arm, context, branch_support=True):
        self.model, self.arm, self.context = model, arm, context
        self.handles = []
        self.original_support = model.gspr.return_branch_support if hasattr(model, 'gspr') else None
        self.before_state = None
        self.last = None
        self.raw = None
        if hasattr(model, 'gspr'):
            model.gspr.return_branch_support = branch_support
            self.handles.append(model.gspr.register_forward_hook(self.gspr))
            pfn = model.pillar_vfe.pfn_layers[0]
            self.handles.append(pfn.register_forward_pre_hook(self.before_pfn, prepend=True))
            self.handles.append(pfn.register_forward_pre_hook(self.after_weighting))
        self.handles.append(model.register_forward_hook(self.after_model))

    def close(self):
        for handle in self.handles:
            handle.remove()
        if self.original_support is not None:
            self.model.gspr.return_branch_support = self.original_support

    def before_pfn(self, module, inputs):
        self.raw = inputs[0].detach().clone()

    def gspr(self, module, inputs, output):
        import torch
        features, counts, coords = inputs
        r, valid = output['point_reliability'], output['point_valid_mask']
        xyz, num, co = (tensor_numpy(x) for x in (features, counts, coords))
        original = {k: tensor_numpy(v).copy() for k, v in output.items()}
        if (xyz.ndim != 3 or xyz.shape[-1] != 4 or num.shape != (len(xyz),)
                or num.dtype.kind not in 'iu' or co.shape != (len(xyz), 4) or co.dtype.kind not in 'iu'
                or not np.isfinite(xyz).all() or (num <= 0).any() or (num > xyz.shape[1]).any()):
            raise ValueError('Invalid official voxel inputs')
        shape = xyz.shape[:2]
        for name, expected in [('point_reliability', shape), ('point_uncertainty', shape),
                ('point_evidence', (*shape, 2)), ('pillar_reliability', (len(xyz),)), ('pillar_uncertainty', (len(xyz),))]:
            if original[name].shape != expected or original[name].dtype != original['point_reliability'].dtype:
                raise ValueError('Invalid diagnostic output shape/dtype: '+name)
        if original['point_valid_mask'].dtype != np.bool_ or original['point_valid_mask'].shape != shape:
            raise ValueError('Invalid point validity mask')
        if ((original['point_uncertainty'] < 0).any() or (original['point_uncertainty'] > 1).any()
                or (original['point_evidence'] < 0).any() or (original['point_evidence'] > module.evidence_cap).any()):
            raise ValueError('Out-of-range diagnostic uncertainty/evidence')
        expected_mask = np.arange(xyz.shape[1])[None, :] < num[:, None]
        if not np.array_equal(expected_mask, original['point_valid_mask']):
            raise ValueError('GSPR validity does not match official voxel counts')
        for name in ('point_uncertainty', 'point_evidence', 'pillar_reliability', 'pillar_uncertainty'):
            if not np.isfinite(original[name]).all():
                raise ValueError('Nonfinite diagnostic output: ' + name)
        for name in ('geometry_support', 'radiometric_support', 'physics_support'):
            if name in original and (original[name].shape != (*shape, 2)
                    or original[name].dtype != original['point_reliability'].dtype or not np.isfinite(original[name]).all()):
                raise ValueError('Invalid auxiliary branch output: '+name)
        ctx = self.context
        agents = ctx['cavs']
        if np.any(co[:, 0] < 0) or np.any(co[:, 0] >= len(agents)):
            raise ValueError('Voxel CAV index out of range')
        key = [ctx['weather'], *ctx['frame_key']]
        applied, audit = replace_weights(original['point_reliability'], original['point_valid_mask'],
            co[:, 0], self.arm, key, agent_ids=agents)
        for item in audit:
            item['cav'] = str(agents[item['agent_slot']])
        if self.arm != 'R0':
            output = dict(output)
            output['point_reliability'] = torch.from_numpy(applied).to(device=r.device, dtype=r.dtype)
        if not torch.equal(output['point_valid_mask'], valid) or output['point_reliability'].shape != r.shape:
            raise ValueError('Intervention changed slot/mask structure')
        self.last = dict(features=xyz.copy(), counts=num.copy(), coords=co.copy(),
            original=original, applied=applied, intervention=audit)
        return output

    def after_weighting(self, module, inputs):
        import torch
        if self.last is None or self.raw is None:
            raise ValueError('PFN executed without observed GSPR output')
        actual = inputs[0]
        applied = torch.from_numpy(self.last['applied']).to(device=actual.device, dtype=actual.dtype)
        expected = self.raw * applied.unsqueeze(-1)
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or not torch.isfinite(actual).all() or not torch.equal(actual, expected)):
            raise ValueError('Actual PFN input does not equal decorated features times applied reliability')
        self.last['pfn_weight_verified'] = True
        self.last['pfn_input_sha256'] = array_hash(tensor_numpy(actual))
        self.raw = None

    def after_model(self, module, inputs, output):
        if self.before_state is None:
            # First forward is in eval/no_grad and the strict official loader already checked every tensor.
            self.before_state = self.context.get('state_before_forward')
            if self.before_state is None:
                raise ValueError('Missing loaded-state immutability proof')
        if self.original_support is not None and not self.last.get('pfn_weight_verified'):
            raise ValueError('GSPR never reached PFN')
        self.context['heads'] = {k: tensor_numpy(output[k]).copy() for k in ('psm', 'rm')}

    def verify_unchanged(self):
        actual = state_hash(self.model)
        if self.before_state is not None and self.before_state != actual:
            raise ValueError('Frozen parameters or buffers changed during inference')
        return actual


def summarize_points(observer, batch, pair_folder, evidence, index, save_folder):
    from opencood.utils.transformation_utils import x1_to_x2
    state = observer.last
    ctx = observer.context
    original = state['original']
    valid = original['point_valid_mask']
    xyz = state['features'][..., :3][valid]
    intensity = state['features'][..., 3][valid]
    agent = np.broadcast_to(state['coords'][:, 0, None], valid.shape)[valid]
    pillar_density = np.broadcast_to(state['counts'][:, None]/valid.shape[1], valid.shape)[valid]
    ego_range = np.linalg.norm(xyz, axis=1)
    poses = tensor_numpy(batch['ego']['lidar_pose']).reshape(-1, 6)
    if len(poses) != len(ctx['cavs']):
        raise ValueError('Sensor pose/CAV mapping differs')
    sensor_range = np.zeros(len(xyz), np.float64)
    proxy = np.zeros(len(xyz), np.bool_)
    analysis = read(pair_folder / 'frames' / f'{index:06d}.json')
    blob = pair_folder / analysis['clean_cloud']
    if sha(blob) != analysis['clean_cloud_sha256']:
        raise ValueError('Spatial proxy Clean cloud changed')
    with np.load(blob, allow_pickle=False) as stored:
        clean_xyz, clean_agent = stored['xyz'], stored['agents']
    author = analysis['author_mask_clouds'][ctx['weather']]
    author_path = pair_folder / author['path']
    if sha(author_path) != author['sha256']:
        raise ValueError('Author color-mask reference changed')
    with np.load(author_path, allow_pickle=False) as stored:
        author_status = author_mask_proxy(xyz, agent, stored['xyz'], stored['agents'], stored['nonzero'])
    real = np.ones(len(xyz), np.bool_)
    for a in range(len(poses)):
        selected = agent == a
        local = np.c_[xyz[selected], np.ones(selected.sum())] @ x1_to_x2(poses[0], poses[a]).T
        sensor_range[selected] = np.linalg.norm(local[:, :3], axis=1)
        _, nearest = nearest_proxy(clean_xyz[clean_agent == a], xyz[selected], (.1,))
        proxy[selected] = nearest > .1
        info = analysis['conditions'][ctx['weather']]['agents'][a]
        if str(info['cav']) != str(ctx['cavs'][a]):
            raise ValueError('Proxy cloud CAV identity mismatch')
        if info['synthetic_lidar_fallback']:
            real[selected] = False
    fixed = evidence['frames'][index]['conditions']['clean']['final_gt']
    inside = points_in_targets(xyz, fixed)
    in_box = inside.any(0) if len(fixed) else np.zeros(len(xyz), np.bool_)
    variables = dict(r=original['point_reliability'][valid], u=original['point_uncertainty'][valid],
        density=pillar_density, sensor_range=sensor_range, intensity=intensity,
        ego_range=ego_range, suspicious_proxy=proxy.astype(np.float64))
    coords = state['coords']
    occupied = {tuple(c[[0, 2, 3]]) for c in coords}
    neighbor_density = np.asarray([sum((int(c[0]), int(c[2])+dy, int(c[3])+dx) in occupied
        for dy in (-1, 0, 1) for dx in (-1, 0, 1))/9. for c in coords])
    variables['local_pillar_density'] = np.broadcast_to(neighbor_density[:, None], valid.shape)[valid]
    evidence_values = original['point_evidence'][valid]
    variables.update(evidence_reliable=evidence_values[:, 0], evidence_noise=evidence_values[:, 1])
    for name in ('geometry_support', 'radiometric_support', 'physics_support'):
        if name in original:
            support = original[name][valid]
            variables[name+'_difference'] = support[:, 0]-support[:, 1]
    names = list(variables)
    matrix = np.column_stack([variables[n][real] for n in names]).astype(np.float64)
    ranges = range_group(sensor_range)
    ego_ranges = range_group(ego_range)
    groups = {'all_real': real, 'inside_fixed_gt': real & in_box, 'outside_fixed_gt': real & ~in_box,
              'near_clean_proxy': real & ~proxy, 'unmatched_clean_proxy': real & proxy}
    for value, label in [(-1, 'unmapped'), (0, 'zero'), (1, 'nonzero'), (2, 'ambiguous')]:
        groups['author_color/'+label] = real & (author_status == value)
    for group in range(4):
        groups['sensor_range/'+str(group)] = real & (ranges == group)
        groups['ego_range/'+str(group)] = real & (ego_ranges == group)
    count_points = (pillar_density*valid.shape[1]).round().astype(int)
    for label, mask in [('1_3', count_points <= 3), ('4_15', (count_points >= 4) & (count_points < 16)), ('16_32', count_points >= 16)]:
        groups['pillar_count/'+label] = real & mask
    for group in range(3):
        local_group = np.minimum((variables['local_pillar_density']*3).astype(int), 2)
        groups['local_pillar_density/'+str(group)] = real & (local_group == group)
    for a, cav in enumerate(ctx['cavs']):
        groups['cav/'+str(cav)] = real & (agent == a)
    distributions = {group: {name: distribution(variables[name][mask]) for name in ('r', 'u')}
                     for group, mask in groups.items()}
    for name in ('evidence_reliable', 'evidence_noise'):
        distributions['all_real'][name] = distribution(variables[name][real], upper=32.)
    for name in ('pillar_reliability', 'pillar_uncertainty'):
        pillar_real = np.asarray([not analysis['conditions'][ctx['weather']]['agents'][a]['synthetic_lidar_fallback']
                                 for a in state['coords'][:, 0]])
        distributions['all_real'][name] = distribution(original[name][pillar_real])
    r = variables['r']
    floor = observer.model.gspr.reliability_floor
    cap = observer.model.gspr.evidence_cap
    lower, upper = floor+(1-floor)/(cap+2), floor+(1-floor)*(cap+1)/(cap+2)
    near_mean = 0
    for a in range(len(ctx['cavs'])):
        mask = real & (agent == a)
        if mask.any():
            near_mean += int((abs(r[mask]-r[mask].mean(dtype=np.float64)) <= .02).sum())
    target_rows = []
    for j, target in enumerate(fixed):
        mask = inside[j] & real
        target_rows.append(dict(object_id=target['object_id'], points=int(mask.sum()),
            r=distribution(r[mask]), u=distribution(variables['u'][mask])))
    record = dict(index=index, frame_key=ctx['frame_key'], weather=ctx['weather'],
        distributions=distributions, targets=target_rows,
        saturation=dict(real_points=int(real.sum()), excluded_synthetic_points=int((~real).sum()),
            attainable_lower=lower, attainable_upper=upper,
            near_lower=int(((r <= lower+.05) & real).sum()), near_upper=int(((r >= upper-.05) & real).sum()),
            within_vehicle_mean_002=near_mean),
        correlation=dict(names=names, n=len(matrix), sums=matrix.sum(0).tolist(), gram=(matrix.T @ matrix).tolist()))
    known = real & ((author_status == 0) | (author_status == 1))
    author_names = names+['author_color_nonzero']
    author_matrix = np.column_stack([*[variables[n][known] for n in names], author_status[known]]).astype(np.float64)
    record['author_color_correlation'] = dict(names=author_names, n=len(author_matrix),
        sums=author_matrix.sum(0).tolist(), gram=(author_matrix.T @ author_matrix).tolist())
    record['author_color_mapping'] = dict(tolerance_m=1e-4,
        counts={str(v): int((real & (author_status == v)).sum()) for v in (-1, 0, 1, 2)},
        interpretation='original green==0 de_lidar mask proxy; ambiguous/unmapped kept separate, physical origin semantics unverified')
    # Detailed spatial evidence only for preregistered two frames per scene.
    if index in ctx['sample_indices']:
        keep = []
        for a in range(len(ctx['cavs'])):
            selected = np.flatnonzero(real & (agent == a))
            keep.extend(selected[np.linspace(0, len(selected)-1, min(len(selected), 6000), dtype=int)].tolist() if len(selected) else [])
        keep = np.asarray(keep, dtype=int)
        path = save_folder / 'samples' / ctx['weather'] / f'{index:06d}.npz'
        npz_write(path, xyz=xyz[keep], r=r[keep], u=variables['u'][keep],
            agent=agent[keep], density=pillar_density[keep], suspicious_proxy=proxy[keep],
            fixed_corners=np.asarray([o['corners'] for o in fixed], np.float32).reshape(-1, 8, 3))
        record['sample'] = dict(path=str(path), sha256=sha(path), method='deterministic row-major evenly spaced up to 6000 real voxel slots per CAV')
    return record
