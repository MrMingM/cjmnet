"""Selected-weight inference only. GT is used after the model, never as input to GGCA."""
import copy
import json
from pathlib import Path
from auditlib import atomic_json, bound_json, complete, identity, read_json, sha256, sources


def inside(points, corners):
    """Post-hoc membership in the saved oriented cuboid, using its two BEV edges."""
    import numpy as np
    c = np.asarray(corners)
    a, b = c[1, :2]-c[0, :2], c[3, :2]-c[0, :2]
    local = points[:, :2] - c[0, :2]
    along_a, along_b = local @ a, local @ b
    return ((along_a >= 0) & (along_a <= a @ a) & (along_b >= 0) & (along_b <= b @ b)
            & (points[:, 2] >= c[:, 2].min()) & (points[:, 2] <= c[:, 2].max()))


def run(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from opencood.hypes_yaml import yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from models import create, OBSERVATIONS
    from checkpoint import seed
    from lineage import selected
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.experiment_root)
    reference = complete(run_root / 'baseline_reference.json')
    gt_rows = [json.loads(s) for s in (Path(args.gt_audit_root) / 'per_frame.jsonl').read_text(encoding='utf-8').splitlines()]
    weights = {'A0': reference}
    for variant in ('A1', 'A2', 'A3'):
        weights[variant] = selected(run_root, variant)
    contract = dict(sources=sources(), checkpoints={v: r['checkpoint_sha256'] for v, r in weights.items()},
                    indices=list(range(12)), scope='first 12 validation frames per condition; diagnostic only, no model selection')
    contract['identity'] = identity(contract)
    bound_json(root / 'contract.json', contract)
    manifest = {}
    for weather, suffix in (('clean', ''), ('fog', '_fog_0.060'), ('snow', '_snow_2.5_2.0')):
        h = yaml_utils.load_yaml(str(Path(args.runtime_root) / 'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
        h.update(validate_dir=args.prepared[1]['input_manifest']['overlay'], train_sim=False,
                 eval_sim=weather != 'clean', sim_weather=suffix)
        for variant in weights:
            seed(20261016)
            dataset = build_dataset(copy.deepcopy(h), visualize=False, train=False)
            if len(dataset) != 829:
                raise ValueError('Diagnostic validation partition differs')
            actual_keys = []
            for scene in dataset.scenario_database.values():
                ego = next(v for v in scene.values() if v['ego'])
                for stamp, value in ego.items():
                    if stamp != 'ego':
                        actual_keys.append([Path(value['yaml']).parent.parent.name, stamp])
            if actual_keys != args.prepared[1]['b']['selected_frame_keys']:
                raise ValueError('Diagnostic scene/frame order differs')
            loader = DataLoader(Subset(dataset, range(12)), batch_size=1, num_workers=0,
                                collate_fn=dataset.collate_batch_test, shuffle=False)
            model = create(h, variant).cuda().eval()
            model.load_state_dict(torch.load(weights[variant]['checkpoint'], map_location='cpu', weights_only=False), strict=True)
            observations = []
            for index, batch in enumerate(loader):
                path = root / variant / weather / (f'{index:06d}.json')
                if path.exists():
                    old = complete(path)
                    if old['contract'] != contract['identity']:
                        raise ValueError('Diagnostic record belongs to different sources/weights')
                    manifest[str(path)] = sha256(path)
                    continue
                data = train_utils.to_device(batch, torch.device('cuda:0'))['ego']
                from data import tensor_digest
                input_sha256 = tensor_digest(data['processed_lidar'])
                with torch.no_grad():
                    output = model(data)
                raw = data['processed_lidar']
                valid = output['point_valid_mask']
                xyz = raw['voxel_features'][valid][..., :3].detach().cpu().numpy()
                r = output['point_reliability'][valid].detach().cpu().numpy()
                u = output['point_uncertainty'][valid].detach().cpu().numpy()
                if variant != 'A0':
                    expanded = model.ggca.pillar_rows[:, None].expand(-1, valid.shape[1], -1)[valid].cpu().numpy()
                targets = []
                for gt in gt_rows[index]['conditions']['clean']['final_gt']:
                    mask = inside(xyz, gt['corners'])
                    count = int(mask.sum())
                    row = dict(object_id=gt['object_id'], point_count_sum_across_cavs=count,
                               r_mean=float(r[mask].mean()) if count else None,
                               u_mean=float(u[mask].mean()) if count else None)
                    if variant != 'A0':
                        row.update(mean_consensus=float(expanded[mask, 3].mean()) if count else None,
                            mean_strength=float(expanded[mask, 4].mean()) if count else None,
                            protected_points=int(expanded[mask, 6].sum()))
                    targets.append(row)
                record = dict(status='complete', contract=contract['identity'], model=variant, condition=weather,
                              index=index, frame_key=gt_rows[index]['frame_key'], targets=targets,
                              input_sha256=input_sha256, observation=OBSERVATIONS[-1], GT_used_only_after_forward=True)
                atomic_json(path, record); manifest[str(path)] = sha256(path)
                OBSERVATIONS.clear()
            del model
            torch.cuda.empty_cache()
    atomic_json(root / 'summary.json', dict(status='complete', contract=contract,
                record_sha256=manifest, new_optimization_steps=0, AP_recomputed=False))
