"""Run byte-identical upstream inference.main with additive provenance observers.

No alternative detector, dataset, NMS, GT construction or AP implementation.
Only smoke may replace the DataLoader's sampler with a prefix Subset.
"""
from __future__ import annotations

import argparse
import copy
from importlib import metadata
import importlib.util
import os
from pathlib import Path
import random
import sys
import traceback

from common import (COMMIT, PACKAGE, REFERENCE, atomic_json, jsonable,
                    no_test, read_json, server_only, sha256)


def normalize_state(raw):
    # Exact key mapping from the reviewed official train_utils.load_model.
    if not isinstance(raw, dict):
        raise ValueError('Official checkpoint must be a plain state_dict')
    result = {}
    for key, tensor in raw.items():
        if key.startswith('module') and not key.startswith('module_list'):
            mapped = key[7:]
        elif key.startswith('cdd'):
            mapped = 'm' + key[1:]
        else:
            mapped = key
        if mapped in result or not hasattr(tensor, 'shape'):
            raise ValueError('Nested/non-tensor/colliding checkpoint keys are refused')
        result[mapped] = tensor
    return result


def bootstrap(runtime):
    runtime = Path(runtime).resolve(strict=True)
    # Never extend a cjmnet namespace package or use its extension modules.
    if any(n == 'opencood' or n.startswith('opencood.') for n in sys.modules):
        raise RuntimeError('OpenCOOD was imported before source isolation')
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() not in
                   (PACKAGE.parent, PACKAGE.parent.parent)]
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    os.environ['PYTHONPATH'] = str(runtime)
    return runtime


def module_inventory(runtime):
    result = {}
    expected = read_json(PACKAGE / 'UPSTREAM.json')['files']
    for name, module in list(sys.modules.items()):
        if name != 'opencood' and not name.startswith('opencood.'):
            continue
        if name.startswith('opencood.pcdet_utils.pointnet2'):
            raise RuntimeError('Unexpected PointNet2 dependency in the AttFuse-only runtime; audit before continuing')
        path = getattr(module, '__file__', None)
        if not path:
            continue
        path = Path(path).resolve(strict=True)
        if runtime not in path.parents:
            raise RuntimeError(f'Non-official OpenCOOD module was imported: {name}: {path}')
        relative = path.relative_to(runtime).as_posix()
        if path.suffix == '.py' and sha256(path) != expected.get(relative):
            raise RuntimeError(f'Imported source differs from official snapshot: {path}')
        result[name] = dict(path=str(path), sha256=sha256(path))
    return result


def source_differences(runtime):
    # Report pre-existing project/fork differences without importing project code.
    current = Path('/home/cjm/OpenCOOD-main/cjmnet/opencood')
    names = ['data_utils/datasets/__init__.py', 'data_utils/datasets/opv2v/basedataset.py',
        'data_utils/datasets/opv2v/intermediate_fusion_dataset.py',
        'models/point_pillar_intermediate.py', 'tools/inference.py',
        'tools/train_utils.py', 'utils/eval_utils.py', 'utils/eval_utils_opv2v.py',
        'data_utils/post_processor/base_postprocessor.py',
        'data_utils/post_processor/voxel_postprocessor.py']
    rows = []
    for name in names:
        official, local = runtime / 'opencood' / name, current / name
        rows.append(dict(path=name, official_sha256=sha256(official),
                         project_sha256=sha256(local) if local.is_file() else None,
                         same_bytes=local.is_file() and sha256(local) == sha256(official)))
    return rows


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runtime', 'model-dir', 'output-root', 'view-root'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--limit-frames', type=int, default=0)
    parser.add_argument('--loader-workers', type=int, default=16)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    runtime = bootstrap(args.runtime)
    output, model_dir = Path(args.output_root), Path(args.model_dir)
    try:
        import numpy as np
        import torch
        import opencood
        from opencood.data_utils import datasets
        from opencood.hypes_yaml import yaml_utils
        from opencood.tools import train_utils, inference_utils
        from opencood.utils import eval_utils_opv2v

        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError('Expected the opencood HIP environment and one visible physical GPU')
        random.seed(20261016)
        np.random.seed(20261016)
        torch.manual_seed(20261016)
        # Keep the upstream backend settings; do not force deterministic kernels.
        info = read_json(model_dir / 'provenance.json')
        hypes = yaml_utils.load_yaml(str(model_dir / 'config.yaml'))
        if (hypes['fusion']['core_method'] != 'IntermediateFusionDatasetV2XR'
                or hypes['fov'] is not True or hypes['fusion']['args']['proj_first'] is not True
                or hypes['eval_sim'] or hypes['train_sim']
                or hypes['model']['core_method'] != 'point_pillar_intermediate'
                or hypes['model']['args']['use_modality'] != 'processed_lidar'
                or hypes['postprocess']['target_args']['score_threshold'] != 0.20
                or hypes['postprocess']['nms_thresh'] != 0.15):
            raise ValueError('Actual parsed configuration does not match D-1A official Clean protocol')
        if sha256(model_dir / 'config.yaml') != info['runtime_yaml_sha256']:
            raise ValueError('Model config changed after preparation')
        for name in ('0.pth', 'net_epoch0.pth'):
            if sha256(no_test(model_dir / name)) != info['checkpoint_sha256']:
                raise ValueError('Checkpoint alias changed')
        raw = torch.load(model_dir / 'net_epoch0.pth', map_location='cpu', weights_only=False)
        checkpoint_state = normalize_state(raw)
        del raw
        original_create = train_utils.create_model
        original_load = train_utils.load_model
        actual = {}

        def checked_create(parameters):
            model = original_create(parameters)
            model_state = model.state_dict()
            missing = sorted(set(model_state) - set(checkpoint_state))
            extra = sorted(set(checkpoint_state) - set(model_state))
            shape = [k for k in set(model_state) & set(checkpoint_state)
                     if model_state[k].shape != checkpoint_state[k].shape
                     or model_state[k].dtype != checkpoint_state[k].dtype]
            if missing or extra or shape:
                atomic_json(output / 'checkpoint_mismatch.json',
                    dict(missing=missing, extra=extra, shape_or_dtype_mismatch=shape))
                raise ValueError('Official loader would skip/drop/fill checkpoint parameters; refusing reproduction')
            actual['model_class'] = type(model).__module__ + '.' + type(model).__name__
            actual['checkpoint_tensor_count'] = len(checkpoint_state)
            return model

        def checked_load(saved_path, model, epoch=None, start_from_best=True):
            if Path(saved_path).resolve() != model_dir.resolve() or epoch != 0 or start_from_best:
                raise ValueError('Unexpected official checkpoint selection')
            result = original_load(saved_path, model, epoch, start_from_best=start_from_best)
            for key, value in result[1].state_dict().items():
                if not torch.equal(value.detach().cpu(), checkpoint_state[key]):
                    raise ValueError(f'Official load did not reproduce checkpoint tensor: {key}')
            actual['checkpoint_all_tensors_loaded_exactly'] = True
            return result

        train_utils.create_model = checked_create
        train_utils.load_model = checked_load
        base_protocol = dict(stage='D-1A', condition='clean', upstream_commit=COMMIT,
            upstream_repository='https://github.com/ylwhxht/V2X-R',
            upstream_source_hashes=read_json(PACKAGE / 'UPSTREAM.json')['files'],
            upstream_manifest_sha256=sha256(PACKAGE / 'UPSTREAM.json'),
            native_build_selection=read_json(runtime / 'native_build_selection.json')
                if (runtime / 'native_build_selection.json').exists() else {'mode': 'legacy_full_extension_build'},
            model=info, evaluated_hypes=hypes,
            declared_GT_RANGE_OPV2V=datasets.GT_RANGE_OPV2V,
            effective_GT_range=hypes['postprocess']['anchor_args']['cav_lidar_range'],
            COM_RANGE=datasets.COM_RANGE,
            dataset_registry_name=hypes['fusion']['core_method'],
            actual_dataset_class=datasets.__all__['IntermediateFusionDatasetV2XR'].__module__ + '.' +
                                 datasets.__all__['IntermediateFusionDatasetV2XR'].__name__,
            eval_utils='opencood.utils.eval_utils_opv2v',
            AP='BEV polygon IoU, score-sorted greedy matching within frame; '
               'frame-order concatenation, global_sort_detections=False, VOC2010 integral',
            score_threshold=hypes['postprocess']['target_args']['score_threshold'],
            nms_threshold=hypes['postprocess']['nms_thresh'], fov=True, proj_first=True,
            point_dependent_gt_filter=True, fixed_gt=False,
            quirks_preserved=['inference.py always selects eval_utils_opv2v via truthy string condition',
                'inference.py skips AP accumulation when pred_box_tensor is None',
                'upstream empty-voxel random-point handling is unchanged'],
            operational_settings=dict(seed=20261016, PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'),
                loader_workers=args.loader_workers, upstream_loader_workers=16,
                limit_frames=args.limit_frames, device='cuda:0',
                ROCR_VISIBLE_DEVICES=os.environ.get('ROCR_VISIBLE_DEVICES')),
            source_differences_from_project=source_differences(runtime),
            versions={n: metadata.version(n) for n in ('torch', 'numpy', 'scipy', 'shapely', 'open3d')},
            hip_version=torch.version.hip, gpu_name=torch.cuda.get_device_name(0))
        if args.probe:
            model = checked_create(hypes).to('cuda:0')
            checked_load(str(model_dir), model, 0, start_from_best=False)
            atomic_json(output / 'preflight.json', dict(status='complete', **actual,
                protocol=base_protocol, imported_modules=module_inventory(runtime)))
            print('Official imports/checkpoint preflight passed; no dataset inference in this probe.', flush=True)
            return
        verification = read_json(Path(args.view_root) / 'manifest.json')
        if verification['status'] != 'complete' or verification['validation_frame_count'] != 829:
            raise ValueError('Official validation view is not verified complete')
        keys = verification['frame_keys']
        selected_keys = keys[:args.limit_frames] if args.limit_frames else keys
        inference_path = runtime / 'opencood/tools/inference.py'
        spec = importlib.util.spec_from_file_location('_official_d1a_inference', inference_path)
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)
        original_build = inference.build_dataset
        original_loader = inference.DataLoader
        original_infer = inference_utils.inference_intermediate_fusion
        original_final = eval_utils_opv2v.eval_final_results
        frames = []
        official_aps = None

        def checked_build(parameters, visualize=False, train=True):
            dataset = original_build(parameters, visualize=visualize, train=train)
            if len(dataset) != 829 or train or not visualize:
                raise ValueError('Official inference dataset flags/frame count changed')
            recovered = []
            scene_rows = []
            for content in dataset.scenario_database.values():
                agents = list(content)
                ego = agents[0]
                stamps = [t for t in content[ego] if t != 'ego']
                first = content[ego][stamps[0]]['yaml']
                scene = Path(first).parent.parent.name
                scene_rows.append(dict(scene=scene, agents=agents, ego=ego, timestamps=stamps))
                recovered.extend([scene, t] for t in stamps)
                for agent, records in content.items():
                    if [t for t in records if t != 'ego'] != stamps:
                        raise ValueError('Official loader changed CAV timestamp alignment')
                    for stamp in stamps:
                        expected = Path(args.view_root) / 'validate' / scene / agent / (stamp + '.pcd')
                        if Path(records[stamp]['lidar']).resolve() != expected.resolve():
                            raise ValueError('Official loader did not select the Clean point cloud')
            if recovered != keys or scene_rows != verification['scenes']:
                raise ValueError('Official dataset scene/CAV/ego/timestamp order differs from verified inventory')
            if (dataset.fov is not True or dataset.proj_first is not True or not dataset.use_Lidar
                    or dataset.use_Radar or dataset.use_DeLidar):
                raise ValueError('Unexpected actual dataset mode')
            actual['dataset_class'] = type(dataset).__module__ + '.' + type(dataset).__name__
            actual['postprocessor_class'] = type(dataset.post_processor).__module__ + '.' + type(dataset.post_processor).__name__
            actual['preprocessor_class'] = type(dataset.pre_processor).__module__ + '.' + type(dataset.pre_processor).__name__
            actual['max_cav'] = dataset.max_cav
            return dataset

        def observed_loader(dataset, *positional, **keywords):
            if (keywords.get('batch_size') != 1 or keywords.get('shuffle') is not False
                    or keywords.get('drop_last') is not False):
                raise ValueError('Official validation loader ordering changed')
            keywords['num_workers'] = args.loader_workers
            if args.limit_frames:
                dataset = torch.utils.data.Subset(dataset, range(len(selected_keys)))
            return original_loader(dataset, *positional, **keywords)

        def observed_infer(batch, model, dataset):
            result = original_infer(batch, model, dataset)
            predictions, scores, gt = result
            index = len(frames)
            row = dict(index=index, frame_key=selected_keys[index],
                gt_count=int(gt.shape[0]), prediction_count=int(predictions.shape[0]) if predictions is not None else 0,
                skipped_by_official_inference=predictions is None,
                participating_agents=int(batch['ego']['record_len'][0].item()))
            # Additive validity assertions; never drop, repair, or replace boxes.
            for label, tensor in (('predictions', predictions), ('scores', scores), ('GT', gt)):
                if tensor is not None and not torch.isfinite(tensor).all().item():
                    raise ValueError(f'Nonfinite official {label} at {row["frame_key"]}')
            frames.append(row)
            with (output / 'frames.jsonl').open('a', encoding='utf-8') as stream:
                import json
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            if len(frames) % 25 == 0 or len(selected_keys) <= 20 or len(frames) == len(selected_keys):
                print(f'D-1A official Clean: {len(frames)}/{len(selected_keys)}', flush=True)
            return result  # Return original objects unchanged.

        def observed_final(statistics, save_path, *positional, **keywords):
            nonlocal official_aps
            snapshot = copy.deepcopy(statistics)
            counted_gt = sum(r['gt_count'] for r in frames if not r['skipped_by_official_inference'])
            counted_predictions = sum(r['prediction_count'] for r in frames)
            for threshold in (.3, .5, .7):
                stat = snapshot[threshold]
                if (stat['gt'] != counted_gt or len(stat['tp']) != counted_predictions
                        or len(stat['fp']) != counted_predictions or len(stat['score']) != counted_predictions):
                    raise ValueError('Official AP counters differ from observed non-skipped frames')
            atomic_json(output / 'official_result_stat.json', snapshot)
            official_aps = original_final(statistics, save_path, *positional, **keywords)
            return official_aps

        inference.build_dataset = checked_build
        inference.DataLoader = observed_loader
        inference_utils.inference_intermediate_fusion = observed_infer
        eval_utils_opv2v.eval_final_results = observed_final
        (output / 'frames.jsonl').write_text('', encoding='utf-8')
        atomic_json(output / 'protocol.json', base_protocol)
        sys.argv = [str(inference_path), '--model_dir', str(model_dir),
                    '--fusion_method', 'intermediate', '--eval_epoch', '0']
        inference.main()  # The original official loop and metric dispatch execute here.
        if len(frames) != len(selected_keys) or official_aps is None:
            raise ValueError('Official inference did not complete the requested Clean frames')
        ap = dict(zip(('ap30', 'ap50', 'ap70'), map(float, official_aps)))
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in ap.values()):
            raise ValueError('Official AP is invalid')
        absolute = {f'{k.upper()}_diff_pp': abs(v - REFERENCE[k]) * 100 for k, v in ap.items()}
        full = len(frames) == 829
        alignment = ('OFFICIAL REPRODUCTION NOT YET ALIGNED' if full and absolute['AP70_diff_pp'] > 2
            else 'REFERENCE DISTANCE WITHIN 2pp; NOT A STATISTICAL PASS' if full
            else 'SMOKE ONLY; OFFICIAL ALIGNMENT NOT ASSESSED')
        base_protocol.update(actual=actual, imported_modules=module_inventory(runtime),
                             data_identity=verification['identity'], selected_frame_keys=selected_keys)
        atomic_json(output / 'protocol.json', base_protocol)
        summary = dict(status='complete', stage='D-1A', condition='clean', AP=ap,
            official_reference=REFERENCE, **absolute, alignment=alignment,
            validation_frame_count=len(frames), available_validation_frames=829,
            full_validation=full, GT_total=sum(r['gt_count'] for r in frames),
            prediction_total=sum(r['prediction_count'] for r in frames),
            official_metric_GT_total=sum(r['gt_count'] for r in frames if not r['skipped_by_official_inference']),
            official_skipped_frame_count=sum(r['skipped_by_official_inference'] for r in frames),
            checkpoint_sha256=info['checkpoint_sha256'], original_yaml_sha256=info['original_yaml_sha256'],
            runtime_yaml_sha256=info['runtime_yaml_sha256'], upstream_commit=COMMIT,
            actual=actual, eval_utils='opencood.utils.eval_utils_opv2v',
            later_stages_implemented=False, automatic_next_stage=False)
        # Official writes into model_dir; copy its exact metric artifact into this run.
        import shutil
        shutil.copyfile(model_dir / 'eval_epochNone.yaml', output / 'official_eval.yaml')
        lines = ['# D−1A：官方 AttFuse Clean validation', '', '**' + alignment + '**', '',
            '| 指标 | 本次 AP | 官方参考 | 绝对差值 / pp |', '|---|---:|---:|---:|']
        lines += [f'| {k.upper()} | {ap[k]:.8f} | {REFERENCE[k]:.4f} | {absolute[k.upper()+"_diff_pp"]:.4f} |'
                  for k in ap]
        lines += ['', f'validation帧数：{len(frames)} / 829；完整验证集：{full}。',
            f'GT总数：{summary["GT_total"]}；预测总数：{summary["prediction_total"]}。',
            f'官方AP实际累计GT：{summary["official_metric_GT_total"]}；'
            f'官方因预测None跳过帧数：{summary["official_skipped_frame_count"]}。',
            'GT_total包含所有已读取帧；保留官方跳帧行为，没有补计这些帧的漏检。',
            '实际dataset/model/eval、关键配置、完整源码与checkpoint哈希见protocol.json。',
            '仅执行Clean；本工具没有后续天气分析或训练阶段。',
            '2pp只用于提示复现偏差；接近参考值不等于统计意义PASS。',
            f'checkpoint SHA256：`{info["checkpoint_sha256"]}`', '']
        (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
        atomic_json(output / 'summary.json', summary)  # Completion marker is written last.
        print('\n'.join(lines), flush=True)
    except Exception as error:
        atomic_json(output / 'official_failure.json', dict(error=repr(error),
            traceback=traceback.format_exc(), runtime=str(runtime),
            advice='Use the official dependencies/compiled extensions in this isolated runtime; '
                   'do not fall back to cjmnet dataset/model/eval code.'))
        raise


if __name__ == '__main__':
    main()
