"""One-frame/one-PCD streaming source-ID flow, original-position target association."""
from pathlib import Path
import numpy as np
from .core import read, write, identity, stage
from .data import catalogue, fixed_gt, cache_frame
from .voxel_trace import Tracer, trace, counts, REASONS


def membership(clean, pose, ego_pose, gt, tolerance):
    """Full 3D oriented parallelepiped, including tilt. Record overlaps explicitly."""
    from shared_weather_benchmark.geometry import inverse, pose_matrix
    m = np.asarray(inverse(pose_matrix(ego_pose))) @ np.asarray(pose_matrix(pose))
    xyz = clean[:, :3].astype(float) @ m[:3, :3].T + m[:3, 3]
    membership_count = np.zeros(len(clean), dtype=np.uint16); selected = []
    for box in gt:
        low, high = box.min(0)-tolerance, box.max(0)+tolerance
        candidates = np.flatnonzero(((xyz >= low) & (xyz <= high)).all(1))
        # corner convention: 0->1, 0->3, 0->4 span orthogonal box axes
        edges = np.stack([box[1]-box[0], box[3]-box[0], box[4]-box[0]], axis=1).astype(float)
        if abs(np.linalg.det(edges)) < 1e-10:
            raise ValueError('Degenerate fixed GT box')
        u = (xyz[candidates]-box[0]) @ np.linalg.inv(edges).T
        scaled_tol = tolerance / np.maximum(np.linalg.norm(edges, axis=0), 1e-10)
        keep = candidates[((u >= -scaled_tol) & (u <= 1+scaled_tol)).all(1)]
        selected.append(keep); membership_count[keep] += 1
    return selected, membership_count


def run(c, out, parent, smoke):
    from opencood.data_utils.pre_processor import build_preprocessor
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.utils import pcd_utils
    from scorw_dataset.io import read_points
    from scorw_r3.core import array_hash
    root, data, keys, rows = catalogue(c)
    h = load_yaml(read(root / 'DATASET_PROTOCOL.json')['config']['model']['config'])
    tracer = Tracer(build_preprocessor(h['preprocess'], False))
    layouts = {s['scene']: s for s in data['layout']}
    n = c['smoke_frames'] if smoke else len(keys)
    for i, (s, t) in enumerate(keys[:n]):
        folder = out / 'frames' / s / t
        def work():
            gt, ids, active, ego_pose = fixed_gt(c, layouts[s], t, rows)
            cache = cache_frame(root, s, t)
            if cache['cavs'] != active or cache['clean_GT_sha256'] != identity(gt.tolist()):
                raise ValueError('D6 active CAV/GT changed')
            clouds = []; objects = []; hashes = {'clean': [], 'fog': []}; examples = []
            for cav in layouts[s]['cavs']:
                r = rows[(s, t, cav)]; rec = root / 'records/validate' / s / 'clouds' / cav / t
                raw = read_points(root / 'clean' / r['relative']); fog = read_points(root / 'fog' / r['relative'])
                receipt = read(rec / 'RESULT.json')
                with np.load(rec / 'provenance.npz', allow_pickle=False) as z:
                    provenance = {k: z[k].copy() for k in ('original_index', 'emitted', 'origin', 'integral',
                                    'hard_signal', 'soft_signal', 'overlap_unknown')}
                idx = provenance['original_index']; emitted = provenance['emitted']; origin = provenance['origin']
                if not np.array_equal(idx, np.arange(len(raw))) or emitted.dtype != bool or any(
                        len(v) != len(raw) for v in provenance.values()) or len(fog) != int(emitted.sum()):
                    raise ValueError('Source row/order/length association failed')
                if not np.isin(origin, [0, 1, 2]).all() or any(not np.isfinite(provenance[k]).all()
                        for k in ('integral', 'hard_signal', 'soft_signal')):
                    raise ValueError('Invalid provenance values')
                expected_stats = dict(hard=int((emitted & (origin == 0)).sum()),
                    weather=int((emitted & (origin == 1)).sum()), unknown=int((emitted & (origin == 2)).sum()),
                    lost=int((~emitted).sum()), original=len(raw), output=len(fog))
                if any(receipt['stats'][k] != v for k, v in expected_stats.items()):
                    raise ValueError('Source category counts differ from frozen receipt')
                if not np.allclose(fog[origin[emitted] == 0, :3], raw[idx[emitted][origin[emitted] == 0], :3], atol=2e-5, rtol=0):
                    raise ValueError('Original hard return coordinates changed')
                a, clean_pc, caudit, cs4 = trace(raw, r['pose'], ego_pose, tracer, pcd_utils, cav in active)
                f, fog_pc, faudit, fs4 = trace(fog, r['pose'], ego_pose, tracer, pcd_utils, cav in active)
                b = np.full(len(raw), REASONS.index('no_return'), dtype=np.uint8); b[idx[emitted]] = f
                s4 = np.zeros(len(raw), dtype=bool); s4[idx[emitted]] = fs4
                if cav in active:
                    hashes['clean'].append(array_hash(clean_pc)); hashes['fog'].append(array_hash(fog_pc))
                distance = np.linalg.norm(raw[:, :3].astype(float), axis=1)
                near = np.zeros(len(raw), dtype=bool)
                near[idx[emitted]] = np.linalg.norm(fog[:, :3].astype(float), axis=1) < c['near_field_m']
                near_clean = distance < c['near_field_m']
                categories = {str(k): dict(total=int((origin == k).sum()),
                    clean=counts(a, origin == k), fog=counts(b, origin == k),
                    near_fog=counts(b, (origin == k) & near),
                    near_clean=counts(a, (origin == k) & near_clean),
                    near_fog_count=int(((origin == k) & near).sum()),
                    near_clean_count=int(((origin == k) & near_clean).sum())) for k in (0, 1, 2)}
                bins = []
                for lo, hi in zip(c['distance_edges_m'], c['distance_edges_m'][1:]):
                    mask = (distance >= lo) & (distance < hi)
                    bins.append(dict(low=lo, high=hi, original=int(mask.sum()), clean=counts(a, mask), fog=counts(b, mask)))
                associated, multiplicity = membership(raw, r['pose'], ego_pose, gt, c['point_box_tolerance_m'])
                for j, indices in enumerate(associated):
                    unique = indices[multiplicity[indices] == 1]; mask = np.zeros(len(raw), dtype=bool); mask[unique] = True
                    row = dict(scene=s, timestamp=t, cav=cav, object_id=ids[j], active_CAV=cav in active,
                        ego=cav == layouts[s]['cavs'][0], distance_ego_m=float(np.linalg.norm(gt[j].mean(0))),
                        raw_box_points=len(indices), unique_box_points=len(unique), ambiguous_points=int((multiplicity[indices]>1).sum()),
                        original_hard=int((mask & emitted & (origin == 0)).sum()),
                        weather_replaced=int((mask & emitted & (origin == 1)).sum()),
                        unknown=int((mask & emitted & (origin == 2)).sum()), lost=int((mask & ~emitted).sum()),
                        clean=counts(a, mask), fog=counts(b, mask),
                        hard_network=int((mask & (origin == 0) & (b == REASONS.index('PillarVFE_slot'))).sum()),
                        weather_flow=counts(b, mask & (origin == 1)),
                        fog_s4=int(s4[unique].sum()), clean_s4=int(cs4[unique].sum()),
                        tau_sum=float(provenance['integral'][unique].sum()),
                        detected_clean= j in cache['predictions']['clean']['matches']['0.7']['matched_gt'],
                        detected_fog=j in cache['predictions']['fog']['matches']['0.7']['matched_gt'])
                    if sum(row['clean'].values()) != len(unique) or sum(row['fog'].values()) != len(unique):
                        raise ValueError('Target flow count conservation failed')
                    objects.append(row)
                    # Predeclared lexicographic first six 40-60m active targets with unique points.
                    if cav in active and 40 <= row['distance_ego_m'] < 60 and len(unique) and len(examples) < c['example_count']:
                        sample_ids = unique[:12]
                        positions = np.full((len(raw), 4), np.nan, dtype=np.float32); positions[idx[emitted]] = fog
                        examples.append(dict(summary=row, point_cases=[dict(original_index=int(k), clean=raw[k].tolist(),
                            fog=positions[k].tolist() if emitted[k] else None, origin=int(origin[k]),
                            final_reason=REASONS[int(b[k])], clean_final_reason=REASONS[int(a[k])],
                            integral=float(provenance['integral'][k]), hard_signal=float(provenance['hard_signal'][k]),
                            soft_signal=float(provenance['soft_signal'][k])) for k in sample_ids]))
                clouds.append(dict(cav=cav, active=cav in active, original=len(raw), emitted=len(fog),
                    clean=counts(a), fog=counts(b), by_origin=categories, by_original_distance=bins,
                    clean_s4=int(cs4.sum()), fog_s4=int(s4.sum()),
                    overlap_GT_points=int((multiplicity > 1).sum()), no_GT_points=int((multiplicity == 0).sum()),
                    source_association_failed=0, clean_voxel_audit=caudit, fog_voxel_audit=faudit))
                if sum(counts(a).values()) != len(raw) or sum(counts(b).values()) != len(raw):
                    raise ValueError('Cloud flow count conservation failed')
            for mode in hashes:
                if identity(hashes[mode]) != cache['predictions'][mode]['input_hash']:
                    raise ValueError('D6 exact input cloud hash mismatch: '+mode)
            write(folder / 'RESULT.json', dict(scene=s, timestamp=t, clouds=clouds, objects=objects,
                  examples=examples, D6_input_hash_exact=True, actual_model_forward=False,
                  network_definition='valid four-feature slots in exact native-collated D6 PillarVFE input; not maxpool attribution'))
        stage(folder, dict(parent=parent, experiment='B', frame=[s, t]), work)
        if (i+1) % 10 == 0 or i+1 == n:
            print('B exact point/voxel trace %d/%d' % (i+1, n), flush=True)
