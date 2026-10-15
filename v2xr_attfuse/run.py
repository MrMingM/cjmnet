"""Server-only AttFuse transfer baseline: identical paired V2X-R validate frames."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
from importlib.metadata import version
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import warnings

from reference_weather.common import (atomic_json, digest, object_hash, project_hashes,
    read_json, server_only, verify_sources as reference_sources)
from reference_weather.run import exclusive_run
from v2xr_attfuse.data import SelectedFrames, frame_indices, verify_subset

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent


def jsonable(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def sources():
    current = {p.relative_to(PACKAGE).as_posix(): digest(p) for p in sorted(PACKAGE.rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.yaml', '.json', '.sh', '.md')
        and p.name != 'source_snapshot.json' and '__pycache__' not in p.parts}
    if current != read_json(PACKAGE / 'source_snapshot.json')['sources']:
        raise ValueError('Sync the entire v2xr_attfuse directory including its snapshot')
    return current


def hypes_and_model(config):
    import torch
    from attfuse_code.point_pillar_intermediate import PointPillarIntermediate
    from opencood.hypes_yaml.yaml_utils import load_yaml
    hypes = load_yaml(config['model_config'])
    if (hypes['model']['core_method'] != 'point_pillar_intermediate'
            or hypes['fusion']['core_method'] != 'IntermediateFusionDataset'):
        raise ValueError('Expected the original pure AttFuse / IntermediateFusionDataset configuration')
    for key in ('weather_augmentation', 'prediction_distillation', 'DG_params'):
        hypes.pop(key, None)
    hypes['data_augment'] = []
    if 'wild_setting' in hypes:
        hypes['wild_setting']['async'] = False
        hypes['wild_setting']['loc_err'] = False
    if int(hypes.get('train_params', {}).get('max_cav', 7)) != 5:
        raise ValueError('This baseline requires the recorded five-agent AttFuse configuration')
    model = PointPillarIntermediate(hypes['model']['args'])
    model.load_state_dict(torch.load(config['checkpoint'], map_location='cpu', weights_only=False), strict=True)
    return hypes, model


def evaluate(config, plan, verification, resume):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import opencood.data_utils.datasets as datasets
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import common_utils, eval_utils
    from v2xr_attfuse.geometry import CheckedGeometry

    if not torch.cuda.is_available():
        raise RuntimeError('No visible HIP GPU; use GPU=<physical card> in the launcher')
    hypes, model = hypes_and_model(config)
    root, output = Path(config['dataset_root']), Path(config['output_root'])
    hypes['validate_dir'] = str(root / 'clean' / 'validate')
    all_keys = plan['protocol']['frame_keys']
    indices = frame_indices(all_keys, int(config['frames_per_scene']))
    selected_keys = [all_keys[i] for i in indices]
    if not indices:
        raise ValueError('No selected frames')
    protocol = dict(schema=1, experiment='OPV2V-trained pure LiDAR AttFuse transferred to V2X-R',
        config=config, subset_identity=plan['identity'], evaluation_sources=sources(),
        reference_package_sources=reference_sources(), project_sources=project_hashes(),
        model_class='attfuse_code.point_pillar_intermediate.PointPillarIntermediate',
        checkpoint_sha256=digest(config['checkpoint']), model_config_sha256=digest(config['model_config']),
        evaluated_hypes=jsonable(hypes), selected_keys=selected_keys, selected_global_indices=indices,
        input_encoding='Original XYZI from Open3D RGB red channel, 0..1; no rewriting',
        geometry='Existing OpenCOOD world-object and CAV-to-ego projection',
        camera_fov_crop=False, point_dependent_gt_filter=False, radar_input=False,
        primary='BEV polygon IoU; global_sort=False', secondary='global_sort=True',
        gt_range=datasets.GT_RANGE, communication_range=datasets.COM_RANGE,
        variants=verification['variants'], versions={n: version(n) for n in ('torch', 'numpy', 'scipy', 'shapely', 'open3d')},
        hip_version=torch.version.hip, gpu_name=torch.cuda.get_device_name(0),
        official_v2xr_benchmark_reproduction=False, opv2vw_reproduction=False)
    identity = object_hash(protocol)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if read_json(protocol_path)['identity'] != identity or not resume:
            raise ValueError('Existing evaluation requires matching RESUME=1; changed settings need a new OUTPUT_ROOT')
    atomic_json(protocol_path, dict(identity=identity, protocol=protocol))
    model = model.to('cuda:0').eval()
    def finite_heads(module, inputs, result):
        for name in ('psm', 'rm'):
            if not torch.isfinite(result[name]).all().item():
                raise ValueError(f'Nonfinite detector output: {name}')
    model.register_forward_hook(finite_heads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows, baseline_gt = {}, None
    context = None
    try:
        for mode in ('clean', 'fog', 'snow'):
            destination = output / mode / 'result.json'
            if destination.exists():
                row = read_json(destination)
                if (not resume or row['identity'] != identity or row['frames'] != len(indices)
                        or not row['geometry_valid'] or len(row['gt_fingerprints']) != len(indices)):
                    raise ValueError('Cached result identity/count mismatch')
                if mode == 'clean':
                    baseline_gt = row['gt_fingerprints']
                elif row['gt_fingerprints'] != baseline_gt:
                    raise ValueError('Cached weather GT differs from Clean')
                rows[mode] = row
                print(f'Reuse checked AttFuse {mode}: {row["primary_ap"]}', flush=True)
                continue
            params = copy.deepcopy(hypes)
            params['validate_dir'] = str(root / mode / 'validate')
            dataset = datasets.build_dataset(params, visualize=False, train=False)
            actual_keys = [[Path(v['yaml']).parent.parent.name, stamp]
                for scene in dataset.scenario_database.values()
                for stamp, v in next(iter(scene.values())).items() if stamp != 'ego']
            if len(dataset) != len(all_keys) or actual_keys != all_keys:
                raise ValueError(f'{mode}: actual frame identities/order differ from the verified plan')
            seed = int(config['seed'])
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            loader = DataLoader(SelectedFrames(dataset, indices, all_keys, seed), batch_size=1,
                num_workers=int(config['loader_workers']), collate_fn=dataset.collate_batch_test,
                shuffle=False, pin_memory=True, drop_last=False)
            statistics = {t: dict(tp=[], fp=[], gt=0, score=[]) for t in (.3, .5, .7)}
            gt_fingerprints, frame_counts, examples = [], [], []
            counts = Counter()
            start = time.monotonic()
            with CheckedGeometry(common_utils, np) as geometry, torch.inference_mode():
                for index, batch in enumerate(loader):
                    context = dict(condition=mode, selected_index=index, global_index=indices[index], frame=selected_keys[index])
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('default', RuntimeWarning)
                        try:
                            if batch is None:
                                raise ValueError('Empty data batch; no synthetic/fallback point is inserted')
                            batch = train_utils.to_device(batch, torch.device('cuda:0'))
                            boxes, scores, gt = inference_utils.inference_intermediate_fusion(batch, model, dataset)
                            for name, tensor in (('predictions', boxes), ('scores', scores), ('GT', gt)):
                                if tensor is not None and not torch.isfinite(tensor).all().item():
                                    raise ValueError(f'Nonfinite {name}')
                            arr = gt.detach().cpu().numpy()
                            fingerprint = object_hash(dict(shape=list(arr.shape), dtype=str(arr.dtype),
                                bytes_sha256=hashlib.sha256(arr.tobytes()).hexdigest()))
                            gt_fingerprints.append(fingerprint)
                            if baseline_gt is not None and fingerprint != baseline_gt[index]:
                                raise ValueError('Weather GT geometry/order differs from Clean')
                            for threshold in statistics:
                                eval_utils.caluclate_tp_fp(boxes, scores, gt, statistics, threshold)
                            frame_counts.append(dict(gt=len(gt), predictions=len(boxes) if boxes is not None else 0))
                        finally:
                            for item in captured:
                                counts[f'{item.category.__name__}: {item.message}'] += 1
                                if len(examples) < 12:
                                    examples.append(dict(frame=context, message=str(item.message),
                                        category=item.category.__name__, file=item.filename, line=item.lineno))
                    if (index + 1) % 25 == 0 or len(indices) <= 20 or index + 1 == len(indices):
                        print(f'AttFuse V2X-R {mode}: {index + 1}/{len(indices)}; '
                              f'checked IoU pairs={geometry.pairs}; captured warnings={sum(counts.values())}', flush=True)
            if len(gt_fingerprints) != len(indices) or statistics[.7]['gt'] == 0:
                raise ValueError('Incomplete evaluation or no GT objects')
            ap = {f'ap{round(t * 100)}': float(eval_utils.calculate_ap(copy.deepcopy(statistics), t, False)[0]) for t in statistics}
            global_ap = {f'ap{round(t * 100)}': float(eval_utils.calculate_ap(copy.deepcopy(statistics), t, True)[0]) for t in statistics}
            row = dict(identity=identity, condition=mode, frames=len(indices), primary_ap=ap,
                secondary_global_sort_ap=global_ap, gt_fingerprints=gt_fingerprints, frame_counts=frame_counts,
                checked_polygons=geometry.polygons, checked_iou_pairs=geometry.pairs, geometry_valid=True,
                warning_counts=dict(counts), warning_examples=examples,
                warning_policy='Per-frame default filtering; repeated same-location warnings may be coalesced.',
                elapsed_seconds=time.monotonic() - start)
            atomic_json(destination, row)
            rows[mode] = row
            if mode == 'clean':
                baseline_gt = gt_fingerprints
            print(f'AttFuse V2X-R {mode}: {ap}', flush=True)
    except Exception as error:
        atomic_json(output / 'last_failure.json', dict(identity=identity, context=context,
            error=repr(error), traceback=traceback.format_exc()))
        raise
    gaps = {mode: 100 * (rows['clean']['primary_ap']['ap70'] - rows[mode]['primary_ap']['ap70']) for mode in ('fog', 'snow')}
    summary = dict(identity=identity, status='complete', frames=len(indices),
        full_prepared_validation=len(indices) == len(all_keys), ap70_gap_pp=gaps,
        gt_identical_across_conditions=True, geometry_valid=True, results=rows,
        interpretation='OPV2V checkpoint transfer baseline under existing OpenCOOD protocol; no Rain; not an official V2X-R score.')
    atomic_json(output / 'summary.json', summary)
    lines = ['# AttFuse 在 V2X-R 上的跨数据集天气评测', '',
        '使用 OPV2V 训练的纯 LiDAR epoch13；本轮仅评测，未进行 V2X-R 训练。',
        '沿用当前 OpenCOOD 检测范围、通信范围和处理顺序；不使用相机FOV裁剪或雷达。',
        '三条件真值框逐帧核对一致；所有参与 NMS/AP 的多边形与 IoU 均通过有效性检查。',
        '本结果属于跨数据集基线，不能与作者官方协议的分数直接等同。', '',
        '| 条件 | 帧数 | AP30 | AP50 | AP70 | Clean−天气 AP70 / pp |',
        '|---|---:|---:|---:|---:|---:|']
    for mode, row in rows.items():
        a = row['primary_ap']
        gap = '—' if mode == 'clean' else f'{gaps[mode]:.3f}'
        lines.append(f"| {mode} | {row['frames']} | {a['ap30']:.6f} | {a['ap50']:.6f} | {a['ap70']:.6f} | {gap} |")
    lines += ['', f"完整已准备 validate：{summary['full_prepared_validation']}。",
        '主AP：BEV IoU，global_sort=False；global_sort=True另存JSON。',
        '点数和天气文件后缀见 verification.json。本数据没有Rain，不能单独解决Rain开发门槛。',
        f"checkpoint SHA256：`{protocol['checkpoint_sha256']}`", '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    import yaml
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('evaluate', 'verify'), default='evaluate')
    parser.add_argument('--config', default=str(PACKAGE / 'config.yaml'))
    parser.add_argument('--output-root')
    parser.add_argument('--frames-per-scene', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    sources()
    reference_sources()
    with Path(args.config).open(encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.output_root is not None:
        config['output_root'] = args.output_root
    if args.frames_per_scene is not None:
        config['frames_per_scene'] = args.frames_per_scene
    if (config['schema'] != 1 or min(int(config[k]) for k in ('seed', 'frames_per_scene', 'loader_workers')) < 0):
        raise ValueError('Invalid evaluation configuration')
    root = Path(config['dataset_root']).expanduser().resolve(strict=True)
    output = Path(config['output_root']).expanduser().resolve()
    if (root == output or root in output.parents or output in root.parents
            or output == REPO or REPO in output.parents
            or any(x.lower() in ('test', 'testing') or 'opv2v-w' in x.lower() for x in (*root.parts, *output.parts))):
        raise ValueError('Use a separate development output directory on the data disk; test paths are refused')
    config['dataset_root'], config['output_root'] = str(root), str(output)
    for name in ('model_config', 'checkpoint'):
        config[name] = str(Path(config[name]).expanduser().resolve(strict=True))
    with exclusive_run(output):
        if output.exists() and any(output.iterdir()) and not (output / 'verification.json').exists():
            raise FileExistsError('Refusing an unrelated nonempty output directory')
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'unittest', 'v2xr_attfuse.test_contract'], check=True)
        plan, verification = verify_subset(root, output / 'verification.json')
        if args.stage == 'evaluate':
            evaluate(config, plan, verification, args.resume)


if __name__ == '__main__':
    main()
