"""Train-only frozen forwards; no GT outcome, AP, optimizer or validation."""
from collections import defaultdict
import os
import uuid
import numpy as np
from .common import (WEATHERS, atomic_json, commit_unit, digest,
    now, object_hash, read_json, resource_check, resume, safe_path, settings)
from .nulls import (MAGNITUDES, alpha_distance, candidate_ids, magnitude_summary,
                    mix_predictions, pool_deltas, pool_magnitudes, rng_for)


def calibration_indices(indices, count=64):
    indices = sorted(indices)
    positions = np.linspace(0, len(indices)-1, min(count, len(indices)), dtype=int)
    return [indices[i] for i in positions]


def prepare(folder, protocol, runtime):
    final_path = folder / 'NULL_CONFIG.json'
    identity = protocol['identity']
    if final_path.exists():
        config = read_json(final_path)
        if config['root_identity'] != identity:
            raise ValueError('Null config root identity mismatch')
        for path, expected in config['artifacts'].items():
            if digest(folder / path) != expected:
                raise ValueError('Frozen train delta pool corrupted')
        if read_json(folder/'PROTOCOL.json')['null_config'] != config:
            raise ValueError('Immutable null protocol differs from frozen config')
        return config
    if (folder/'PROTOCOL.json').exists():
        # Recover only an interrupted final marker write, keeping the EXACT
        # previously frozen config/timestamp. Never reselect alpha in this RUN.
        config=read_json(folder/'PROTOCOL.json')['null_config']
        if config['root_identity']!=identity:
            raise ValueError('Frozen null protocol root differs')
        load_pools(folder,config)
        atomic_json(final_path,config)
        return config
    if (folder/'conditions').exists():
        raise ValueError('Scientific conditions exist without immutable null config; new RUN required')
    b, spec, options, hypes, target, model, arm = runtime
    from local_fusion_action_utility_audit.common import assert_loader_paths
    indices = calibration_indices(protocol['train_indices'], settings()['calibration']['train_frames_per_weather'])
    rows = {}
    for weather in WEATHERS:
        resource_check(protocol)
        unit = folder / 'calibration' / (weather + '.json')
        payload = resume(unit, identity)
        if payload is not None:
            rows[weather] = payload
            continue
        b.seed_all(settings()['calibration']['seed'])
        dataset, loader = b.selected_loader(hypes, options, 'train', weather, indices)
        assert_loader_paths(dataset)
        vectors = defaultdict(list)
        source_magnitude = {k: [] for k in MAGNITUDES}
        random_magnitude = {str(alpha): {k: [] for k in MAGNITUDES}
                            for alpha in settings()['calibration']['alpha_grid']}
        seen = []
        with b.torch.no_grad():
            for ordinal, batch in enumerate(loader):
                resource_check(protocol)
                index = int(batch['ego']['communication_sample_index'][0])
                seen.append(index)
                batch = b.to_device(batch, target)
                context = b.v3rt.context(model, batch['ego'], 'clean' if weather == 'clean' else 'weather', verify=ordinal == 0)
                shared, _ = arm.predict(model.engine.base, context['levels'])
                pool = b.build_candidate_pool(model.engine.base, context['levels'], shared, spec['candidate_families'])
                ids = candidate_ids(b, dataset, batch, shared, settings()['calibration']['anchors_per_frame'])
                for name, vector in pool_deltas(dataset, batch, shared, pool, ids).items():
                    if not np.isfinite(vector).all():
                        raise ValueError('Nonfinite train decoded output delta')
                    vectors[name.split(':')[0]].append(vector.astype(np.float32))
                values = pool_magnitudes(b, dataset, batch, shared, pool, ids)
                for key in MAGNITUDES:
                    source_magnitude[key].extend(values[key])
                for alpha in settings()['calibration']['alpha_grid']:
                    random = mix_predictions(b, model.engine.base, context['levels'], shared, alpha,
                        rng_for(settings()['calibration']['seed'], weather, index, 'N2-calibration'))
                    values = pool_magnitudes(b, dataset, batch, shared, random, ids)
                    for key in MAGNITUDES:
                        random_magnitude[str(alpha)][key].extend(values[key])
                print(f'train-only calibration {weather} {ordinal+1}/{len(indices)} frame={index}', flush=True)
        if sorted(seen) != indices:
            raise ValueError('Incomplete/duplicate train calibration frames')
        pool_path = unit.with_suffix('.npz')
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        temp = pool_path.with_name(pool_path.name + '.' + uuid.uuid4().hex + '.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **{k: np.concatenate(v) for k, v in vectors.items()})
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(pool_path)
        payload = dict(weather=weather, indices=indices, split='train', gt_outcome_used=False,
                       AP_used=False, validation_used=False, pool_file=pool_path.relative_to(folder).as_posix(),
                       source=magnitude_summary(source_magnitude),
                       random={k: magnitude_summary(v) for k, v in random_magnitude.items()})
        commit_unit(unit, identity, payload, (pool_path,))
        rows[weather] = payload
    scores = {str(alpha): float(np.mean([alpha_distance(row['source'], row['random'][str(alpha)],
        settings()['calibration']['epsilon']) for row in rows.values()]))
        for alpha in settings()['calibration']['alpha_grid']}
    alpha = min(settings()['calibration']['alpha_grid'], key=lambda value: scores[str(value)])
    config = dict(root_identity=identity, alpha=alpha, objective=scores, per_weather=rows,
        train_indices=indices, frozen_at=now(), validation_used=False, gt_outcome_used=False,
        artifacts={row['pool_file']: digest(folder / row['pool_file']) for row in rows.values()},
        rules=settings()['calibration'])
    config['config_hash'] = object_hash(config)
    atomic_json(folder / 'PROTOCOL.json', dict(root_identity=identity, null_config=config))
    atomic_json(final_path, config)  # immutable extension before any validation
    return config


def load_pools(folder, config):
    result = {}
    if config['config_hash'] != object_hash({k: v for k, v in config.items() if k != 'config_hash'}):
        raise ValueError('Frozen null config corrupted')
    for weather, row in config['per_weather'].items():
        path = safe_path(folder / row['pool_file'])
        if digest(path) != config['artifacts'][row['pool_file']]:
            raise ValueError('Train pool hash drift')
        with np.load(path, allow_pickle=False) as archive:
            for family in archive.files:
                result[f'{weather}/{family}'] = archive[family]
    return result
