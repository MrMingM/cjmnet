"""Evaluate the explicitly selected pure AttFuse checkpoint on development data."""
from __future__ import annotations

import copy
from importlib.metadata import version
from pathlib import Path
import random
import time

import numpy as np

from reference_weather.common import (atomic_json, digest, object_hash, project_hashes,
    read_json, stable_seed, verify_sources)
from reference_weather.generate import verify_dataset


class SeededFrames:
    """Make voxel point-order draws independent of DataLoader scheduling."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        seed = stable_seed(self.seed, 'eval-frame', index)
        np.random.seed(seed)
        random.seed(seed)
        return self.dataset[index]


def evaluate(config, limit_frames=0, resume=False):
    import torch
    from torch.utils.data import DataLoader
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils
    from attfuse_code.point_pillar_intermediate import PointPillarIntermediate

    if 'validate' not in config['splits']:
        raise ValueError('AttFuse development evaluation requires the validate split')
    if not torch.cuda.is_available():
        raise RuntimeError('No visible research GPU. Set only ROCR_VISIBLE_DEVICES via GPU in the launcher.')
    manifest = verify_dataset(config, limit_frames)
    checkpoint = Path(config['checkpoint']).resolve(strict=True)
    config_path = Path(config['model_config']).resolve(strict=True)
    hypes = load_yaml(str(config_path))
    if hypes['model']['core_method'] != 'point_pillar_intermediate':
        raise ValueError('Expected the pure AttFuse point_pillar_intermediate config, without GSPR')
    hypes.pop('weather_augmentation', None)
    hypes.pop('prediction_distillation', None)
    hypes.pop('DG_params', None)
    hypes['data_augment'] = []
    if 'wild_setting' in hypes:
        hypes['wild_setting']['async'] = False
        hypes['wild_setting']['loc_err'] = False
    config_sources = dict(checkpoint_sha256=digest(checkpoint),
                          model_config_sha256=digest(config_path))
    identity_data = dict(dataset_identity=manifest['identity'], **config_sources,
        checkpoint=str(checkpoint), model_config=str(config_path),
        model_class='attfuse_code.point_pillar_intermediate.PointPillarIntermediate',
        state_dict_loading='strict=True', primary_iou='BEV polygon IoU',
        primary_global_sort=False, secondary_global_sort=True, seed=config['seed'],
        package_sources=verify_sources(), project_sources=project_hashes(),
        torch_version=torch.__version__, numpy_version=np.__version__,
        dependency_versions={name: version(name) for name in
                             ('scipy', 'scikit-learn', 'PyMieScatt', 'open3d')},
        hip_version=torch.version.hip, gpu_name=torch.cuda.get_device_name(0),
        frame_keys=manifest['splits']['validate']['frame_keys'])
    identity = object_hash(identity_data)
    output = Path(config['output_root']) / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and read_json(protocol_path)['identity'] != identity:
        raise ValueError('Evaluation checkpoint/protocol changed. Use a new output root.')
    atomic_json(protocol_path, dict(identity=identity, protocol=identity_data))
    model = PointPillarIntermediate(hypes['model']['args'])
    # Explicit file, never latest checkpoint discovery or partial parameter loading.
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    del state
    model = model.to('cuda:0').eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    expected = manifest['splits']['validate']['selected_frames']
    results = {}
    for mode in ['clean', *config['conditions']]:
        destination = output / mode / 'result.json'
        if destination.exists():
            saved = read_json(destination)
            if not resume or saved['identity'] != identity:
                raise FileExistsError(f'Existing evaluation requires matching RESUME=1: {destination}')
            if saved['frames'] != expected:
                raise ValueError('Incomplete cached evaluation')
            results[mode] = saved
            continue
        params = copy.deepcopy(hypes)
        params['root_dir'] = str(Path(config['clean_root']) / 'train')
        params['validate_dir'] = str(Path(config['output_root']) / mode / 'validate')
        dataset = build_dataset(params, visualize=False, train=False)
        if len(dataset) != expected:
            raise ValueError(f'{mode}: {len(dataset)} frames, expected {expected}')
        actual_keys = []
        for scene in dataset.scenario_database.values():
            first = next(iter(scene.values()))
            for stamp, frame in first.items():
                if stamp != 'ego':
                    actual_keys.append([Path(frame['yaml']).parent.parent.name, stamp])
        if actual_keys != manifest['splits']['validate']['frame_keys']:
            raise ValueError(f'{mode}: loaded frame identities/order do not match the generation manifest')
        seed = int(config['seed'])
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        loader = DataLoader(SeededFrames(dataset, seed), batch_size=1,
            num_workers=int(config['loader_workers']), collate_fn=dataset.collate_batch_test,
            shuffle=False, pin_memory=True, drop_last=False)
        statistics = {threshold: dict(tp=[], fp=[], gt=0, score=[]) for threshold in (.3, .5, .7)}
        frame_count = 0
        start = time.monotonic()
        with torch.inference_mode():
            for frame_count, batch in enumerate(loader, 1):
                if batch is None:
                    raise ValueError(f'Empty data batch at {mode}/{frame_count - 1}')
                batch = train_utils.to_device(batch, torch.device('cuda:0'))
                boxes, scores, ground_truth = inference_utils.inference_intermediate_fusion(batch, model, dataset)
                for threshold in statistics:
                    eval_utils.caluclate_tp_fp(boxes, scores, ground_truth, statistics, threshold)
                if frame_count % 50 == 0 or frame_count == expected:
                    print(f'AttFuse {mode}: {frame_count}/{expected} frames', flush=True)
        if frame_count != expected:
            raise ValueError(f'Evaluation frame count mismatch: {mode}')
        ap, global_ap = {}, {}
        for threshold in statistics:
            if statistics[threshold]['gt'] == 0:
                raise ValueError('No validation GT boxes')
            label = f'ap{round(threshold * 100)}'
            ap[label] = float(eval_utils.calculate_ap(copy.deepcopy(statistics), threshold, False)[0])
            global_ap[label] = float(eval_utils.calculate_ap(copy.deepcopy(statistics), threshold, True)[0])
        row = dict(identity=identity, condition=mode, frames=frame_count,
                   frame_order_sha256=object_hash(actual_keys),
                   elapsed_seconds=time.monotonic() - start, primary_ap=ap,
                   secondary_global_sort_ap=global_ap, **config_sources)
        atomic_json(destination, row)
        results[mode] = row
        print(f'AttFuse {mode}: {ap}', flush=True)
    return report(config, manifest, results)


def report(config, manifest=None, results=None):
    root = Path(config['output_root'])
    manifest = manifest or read_json(root / 'manifest.json')
    modes = ['clean', *config['conditions']]
    results = results or {mode: read_json(root / 'evaluation' / mode / 'result.json') for mode in modes}
    if len({v['identity'] for v in results.values()}) != 1:
        raise ValueError('Cannot combine evaluations with different identities')
    saved_protocol = read_json(root / 'evaluation' / 'protocol.json')
    if (saved_protocol['protocol']['dataset_identity'] != manifest['identity']
            or any(r['identity'] != saved_protocol['identity'] for r in results.values())
            or any(r['frames'] != manifest['splits']['validate']['selected_frames'] for r in results.values())):
        raise ValueError('Result identity/frame selection does not match the dataset')
    clean = results['clean']['primary_ap']['ap70']
    gaps = {mode: 100 * (clean - results[mode]['primary_ap']['ap70']) for mode in modes[1:]}
    full = manifest['splits']['validate']['complete_split']
    gate = (gaps.get('fog', float('-inf')) >= 10 and gaps.get('rain', float('-inf')) >= 10)
    result = dict(dataset_identity=manifest['identity'], results=results, ap70_gap_pp=gaps,
                  full_validation=full, fog_rain_minimum_gap_gate=bool(gate and full),
                  interpretation='Minimum severity screen only; matching AP gaps does not establish distribution equivalence.',
                  official_opv2vw_recipe_reproduced=False)
    atomic_json(root / 'evaluation' / 'summary.json', result)
    lines = ['# AttFuse 原版天气模拟开发评测', '',
             '仅使用原始 train/validate；本报告不包含正式 test。', '',
             '主指标：BEV 多边形 IoU，按帧累计 AP（global_sort=False）。',
             '另存跨帧分数统一排序的 AP，两个协议单独列出。', '',
             '| 条件 | 帧数 | AP30 | AP50 | AP70 | Clean−天气 AP70 / pp |',
             '|---|---:|---:|---:|---:|---:|']
    for mode in modes:
        a = results[mode]['primary_ap']
        gap = '—' if mode == 'clean' else f'{gaps[mode]:.3f}'
        lines.append(f"| {mode} | {results[mode]['frames']} | {a['ap30']:.6f} | "
                     f"{a['ap50']:.6f} | {a['ap70']:.6f} | {gap} |")
    lines += ['', f'完整 validate：{full}。Fog/Rain 均 ≥10 pp 的最低筛选门槛：{bool(gate and full)}。',
              '小样本 smoke 结果不能用于通过开发集门槛。', '',
              '这些参数属于公开原版模拟器的开发配置，OPV2V-W 的完整生成配方仍未知。',
              'Snow 采用已授权的 OPV2V 适配：仰角重建通道、关闭 HDL 专属焦距/通道强度修正、',
              '根据 OPV2V 路面拟合噪声阈值；保留原版雪粒遮挡和回波波形计算。',
              '公开雪粒场半径为 80m；目标超过 80m 的数量记录在每个 PCD 的统计中。', '',
              '三种天气的点数、分距离点数、回波替换数和丢点数见 manifest.json 与 records/。',
              'PCD 强度采用 RGB 8bit 编码；没有补回 clean 点来掩盖空点云。', '',
              f"checkpoint SHA256：`{results['clean']['checkpoint_sha256']}`", '']
    (root / 'evaluation' / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return result
