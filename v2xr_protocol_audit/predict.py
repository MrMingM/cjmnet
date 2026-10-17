"""Execute original inference.main once per condition, caching final predictions.

Resume replays dataset loading in original order, but never forwards cached frames.
All official inference/AP/model/postprocess functions remain the originals.
"""
import copy
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import random
import re
import sys

from support import (CONDITIONS, REPRO, atomic_json, compare_gt, corner_array, identity,
                     jsonable, load_prediction, patched, prediction_tensors, read_json,
                     sha256, tensor_array)


def condition_model(output, condition, args, evidence):
    """D-1A's original config text and checkpoint aliases; separate output model_dir."""
    from common import link
    from run import config_difference
    import yaml
    from opencood.hypes_yaml import yaml_utils
    path = output / 'official_runs' / condition
    path.mkdir(parents=True, exist_ok=True)
    text = (Path(args.model_dir) / 'config.yaml').read_text(encoding='utf-8')
    changes = dict(validate_dir=evidence['input_manifest']['overlay'])
    if condition != 'clean':
        changes.update(eval_sim=True, sim_weather={'fog': '_fog_0.060', 'snow': '_snow_2.5_2.0'}[condition])
    import json
    for key, value in changes.items():
        text, count = re.subn(r'^' + key + r':[^\n]*$', lambda m: key + ': ' + json.dumps(value), text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f'Expected one official config field: {key}')
    if any(k not in changes for k in config_difference(yaml.safe_load((Path(args.model_dir) / 'config.yaml').read_text()), yaml.safe_load(text))):
        raise ValueError('Unexpected behavioral configuration change')
    config_path = path / 'config.yaml'
    if config_path.exists() and config_path.read_text(encoding='utf-8') != text:
        raise ValueError('Existing condition config changed')
    if not config_path.exists():
        config_path.write_text(text, encoding='utf-8', newline='\n')
    if jsonable(yaml_utils.load_yaml(str(config_path))) != evidence['b']['evaluated_configs'][condition]:
        raise ValueError('Condition parsed config differs from successful D-1B')
    for name in ('0.pth', 'net_epoch0.pth'):
        link(Path(args.model_dir) / name, path / name)
    return path


def live_gt_observer(postprocessor, batch, original):
    import numpy as np
    from opencood.utils import box_utils
    native_ids = batch['ego']['object_ids']
    centers = tensor_array(batch['ego']['object_bbx_center'][batch['ego']['object_bbx_mask'] == 1])
    if len(centers) != len(native_ids) or len(set(map(str, native_ids))) != len(native_ids):
        raise ValueError('Official intermediate-fusion ID/center mapping changed')
    center_map = {str(k): v.tolist() for k, v in zip(native_ids, centers)}
    selected_ids = [str(x) for x in set(native_ids)]
    masks = []
    original_mask = box_utils.get_mask_for_boxes_within_range_torch

    def mask(boxes, *args, **kwargs):
        result = original_mask(boxes, *args, **kwargs)
        masks.append((tensor_array(boxes).copy(), tensor_array(result).copy()))
        return result

    with patched(box_utils, 'get_mask_for_boxes_within_range_torch', mask):
        gt = original(batch)
    if len(masks) != 1:
        raise ValueError('Unexpected official GT mask path')
    boxes, keep = masks[0]
    if len(boxes) != len(selected_ids) or not np.array_equal(tensor_array(gt), boxes[keep]):
        raise ValueError('GT observer differs from original official return')
    rows = [dict(object_id=k, box=center_map[k], corners=box.tolist())
            for k, box, valid in zip(selected_ids, boxes, keep) if valid]
    return gt, rows


def generate(condition, output, args, evidence, contract):
    import numpy as np
    import torch
    from observe import normalize_state, module_inventory
    from opencood.tools import train_utils, inference_utils
    from opencood.utils import eval_utils_opv2v

    selected = contract['frame_keys']
    cache_dir = output / 'predictions' / condition
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else dict(
        contract_identity=contract['identity'], condition=condition, entries={}, status='in_progress')
    if manifest['contract_identity'] != contract['identity'] or manifest['condition'] != condition:
        raise ValueError('Prediction cache manifest identity differs')
    if not set(manifest['entries']) <= {str(i) for i in range(len(selected))}:
        raise ValueError('Unexpected frame indices in cache manifest')
    known = {}
    for index, key in enumerate(selected):
        path = cache_dir / f'{index:06d}.json'
        if str(index) in manifest['entries'] and not path.exists():
            raise ValueError(f'Committed prediction missing: {path}')
        if path.exists():
            digest = sha256(path)
            if str(index) in manifest['entries'] and manifest['entries'][str(index)]['sha256'] != digest:
                raise ValueError(f'Committed prediction bytes changed: {path}')
            record = load_prediction(path, contract, condition, index, key)
            compare_gt(record['official_gt'], evidence['frames'][index]['conditions'][condition]['final_gt'], f'cache {condition}/{key}')
            known[index] = record
            # Adopt a valid atomically saved frame after interruption before journal commit.
            manifest['entries'][str(index)] = dict(path=str(path), sha256=digest)
    expected_paths = {f'{i:06d}.json' for i in range(len(selected))} | {'manifest.json', 'manifest.json.tmp'}
    if any(p.is_file() and p.name not in expected_paths and not re.fullmatch(r'\d{6}\.json\.tmp', p.name) for p in cache_dir.iterdir()):
        raise ValueError('Unexpected files in prediction cache directory')
    atomic_json(manifest_path, manifest)
    if manifest['status'] == 'complete':
        if len(known) != len(selected):
            raise ValueError('Completed prediction manifest has missing frames')
        for name, digest in manifest['official_output_sha256'].items():
            if sha256(output / name) != digest:
                raise ValueError(f'Official cache completion evidence changed: {name}')
        print(f'Reuse complete {condition} prediction cache: {len(known)} frames; no inference.', flush=True)
        return manifest
    if args.stage == 'evaluate':
        raise ValueError(f'{condition}: prediction stage incomplete; evaluate never creates predictions')
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('Prediction requires the successful D-1A HIP environment/GPU')
    hardware = dict(gpu_name=torch.cuda.get_device_name(0), physical_gpu=__import__('os').environ.get('ROCR_VISIBLE_DEVICES'))
    if 'generation_hardware' in manifest and manifest['generation_hardware'] != hardware:
        raise ValueError('Partial prediction cache must resume on the same recorded physical GPU')
    manifest['generation_hardware'] = hardware
    atomic_json(manifest_path, manifest)
    model_dir = condition_model(output, condition, args, evidence)
    # Reproduce D-1A seed and setup order, including 16 DataLoader workers.
    random.seed(20261016)
    np.random.seed(20261016)
    torch.manual_seed(20261016)
    state = normalize_state(torch.load(model_dir / 'net_epoch0.pth', map_location='cpu', weights_only=False))
    inference_path = Path(args.runtime_root) / 'opencood/tools/inference.py'
    spec = importlib.util.spec_from_file_location('_d1c_official_inference', inference_path)
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    original_build, original_loader = inference.build_dataset, inference.DataLoader
    original_infer = inference_utils.inference_intermediate_fusion
    original_create, original_load = train_utils.create_model, train_utils.load_model
    original_final = eval_utils_opv2v.eval_final_results
    seen, actual = [], {}
    gt_holder, result_holder = {}, {}

    with ExitStack() as stack:
        def build(parameters, visualize=False, train=True):
            dataset = original_build(parameters, visualize=visualize, train=train)
            if (len(dataset) != 829 or train or not visualize or dataset.max_cav != 7
                    or not dataset.fov or not dataset.proj_first or not dataset.use_Lidar
                    or dataset.use_Radar or dataset.use_DeLidar):
                raise ValueError('Official dataset flags changed')
            recovered = []
            for scene_index, scene in enumerate(evidence['input_manifest']['scenes']):
                content = dataset.scenario_database[scene_index]
                if list(content) != scene['agents'] or not content[scene['ego']]['ego']:
                    raise ValueError('Official ego/CAV selection changed')
                for cav, records in content.items():
                    if [t for t in records if t != 'ego'] != scene['timestamps']:
                        raise ValueError('Official CAV frame alignment changed')
                    suffix = '' if condition == 'clean' else {'fog': '_fog_0.060', 'snow': '_snow_2.5_2.0'}[condition]
                    for stamp in scene['timestamps']:
                        expected = Path(evidence['input_manifest']['overlay']) / scene['scene'] / cav / (stamp + suffix + '.pcd')
                        if Path(records[stamp]['lidar']) != expected:
                            raise ValueError('Actual official weather file selection changed')
                recovered.extend([scene['scene'], t] for t in scene['timestamps'])
            if recovered != evidence['b']['selected_frame_keys']:
                raise ValueError('Official frame order changed')
            original_gt = dataset.post_processor.generate_gt_bbx
            def gt(batch):
                result, rows = live_gt_observer(dataset.post_processor, batch, original_gt)
                gt_holder['rows'] = rows
                return result
            stack.enter_context(patched(dataset.post_processor, 'generate_gt_bbx', gt))
            actual['dataset_class'] = type(dataset).__module__ + '.' + type(dataset).__name__
            return dataset

        def loader(dataset, *args_, **kwargs):
            if kwargs != dict(batch_size=1, num_workers=16, collate_fn=dataset.collate_batch_test,
                              shuffle=False, pin_memory=False, drop_last=False):
                raise ValueError('Original inference DataLoader arguments changed')
            if len(selected) != 829:
                dataset = torch.utils.data.Subset(dataset, range(len(selected)))
            return original_loader(dataset, *args_, **kwargs)

        def create(parameters):
            model = original_create(parameters)
            own = model.state_dict()
            if own.keys() != state.keys() or any(own[k].shape != state[k].shape or own[k].dtype != state[k].dtype for k in own):
                raise ValueError('Checkpoint model keys/shape/dtype differ')
            actual['model_class'] = type(model).__module__ + '.' + type(model).__name__
            return model

        def load(path, model, epoch=None, start_from_best=True):
            if Path(path) != model_dir or epoch != 0 or start_from_best:
                raise ValueError('Unexpected official checkpoint selection')
            result = original_load(path, model, epoch=epoch, start_from_best=start_from_best)
            if any(not torch.equal(value.detach().cpu(), state[k]) for k, value in result[1].state_dict().items()):
                raise ValueError('Official checkpoint tensor loading differs')
            actual['checkpoint_all_tensors_loaded_exactly'] = True
            return result

        def infer(batch, model, dataset):
            index = len(seen)
            key = selected[index]
            expected = evidence['frames'][index]['conditions'][condition]
            if int(batch['ego']['record_len'][0]) != len(expected['context']['participating_cavs']):
                raise ValueError('Participating CAV count differs from D-1B')
            gt_holder.clear()
            if index in known:
                dataset.post_processor.generate_gt_bbx(batch)
                compare_gt(gt_holder['rows'], known[index]['official_gt'], f'resume live GT {condition}/{key}')
                boxes, scores = prediction_tensors(known[index])
                row = known[index]
                gt = torch.from_numpy(corner_array(row['official_gt']))
            else:
                boxes, scores, gt = original_infer(batch, model, dataset)
                compare_gt(gt_holder['rows'], expected['final_gt'], f'live GT vs D-1B {condition}/{key}')
                for name, tensor in (('boxes', boxes), ('scores', scores), ('GT', gt)):
                    if tensor is not None and not torch.isfinite(tensor).all().item():
                        raise ValueError(f'Nonfinite official {name}: {condition}/{key}')
                b = tensor_array(boxes) if boxes is not None else None
                s = tensor_array(scores) if scores is not None else None
                if boxes is not None and (scores is None or b.shape != (len(s), 8, 3)):
                    raise ValueError('Unexpected official prediction shape')
                row = dict(contract_identity=contract['identity'], condition=condition, index=index,
                    scene=key[0], timestamp=key[1], frame_key=key, ego=expected['context']['ego'],
                    participating_cavs=expected['context']['participating_cavs'],
                    prediction_is_none=boxes is None, prediction_empty=boxes is not None and len(boxes) == 0,
                    prediction_state='none' if boxes is None else 'empty' if len(boxes) == 0 else 'boxes',
                    prediction_count=0 if boxes is None else len(boxes),
                    prediction_boxes=None if b is None else b.tolist(), scores=None if s is None else s.tolist(),
                    box_dtype=None if b is None else str(b.dtype), score_dtype=None if s is None else str(s.dtype),
                    official_gt=gt_holder['rows'], official_gt_identity=identity(gt_holder['rows']),
                    d1b_gt_identity=identity(expected['final_gt']))
                row['record_identity'] = identity(row)
                path = cache_dir / f'{index:06d}.json'
                atomic_json(path, row)
                manifest['entries'][str(index)] = dict(path=str(path), sha256=sha256(path))
                atomic_json(manifest_path, manifest)
            seen.append(row)
            if index % 25 == 0 or len(selected) <= 20 or index+1 == len(selected):
                print(f'D-1C prediction {condition}: {index+1}/{len(selected)}, {row["prediction_state"]}', flush=True)
            return boxes, scores, gt

        def final(statistics, save_path, *args_, **kwargs):
            if Path(save_path) != model_dir or args_ or kwargs:
                raise ValueError('Official metric dispatch changed')
            raw = copy.deepcopy(statistics)
            expected_gt = sum(len(r['official_gt']) for r in seen if not r['prediction_is_none'])
            expected_predictions = sum(r['prediction_count'] for r in seen)
            if any(r['gt'] != expected_gt or len(r['tp']) != expected_predictions
                   or len(r['score']) != expected_predictions for r in raw.values()):
                raise ValueError('Original official AP counters differ from cached predictions/GT')
            result_holder['statistics'] = raw
            atomic_json(model_dir / 'official_result_stat.json', raw)
            result = original_final(statistics, save_path)
            result_holder['AP'] = dict(zip(('ap30', 'ap50', 'ap70'), map(float, result)))
            return result

        for obj, name, replacement in ((inference, 'build_dataset', build), (inference, 'DataLoader', loader),
            (train_utils, 'create_model', create), (train_utils, 'load_model', load),
            (inference_utils, 'inference_intermediate_fusion', infer), (eval_utils_opv2v, 'eval_final_results', final)):
            stack.enter_context(patched(obj, name, replacement))
        previous_argv = sys.argv
        try:
            sys.argv = [str(inference_path), '--model_dir', str(model_dir), '--fusion_method', 'intermediate', '--eval_epoch', '0']
            inference.main()
        finally:
            sys.argv = previous_argv
    if len(seen) != len(selected) or not result_holder or not actual.get('checkpoint_all_tensors_loaded_exactly'):
        raise ValueError('Original official inference did not complete all requested frames')
    manifest.update(status='complete', official_exact_AP=result_holder['AP'], actual=actual,
        imported_modules=module_inventory(Path(args.runtime_root)),
        gpu_name=torch.cuda.get_device_name(0), physical_gpu=__import__('os').environ.get('ROCR_VISIBLE_DEVICES'),
        official_output_sha256={str((model_dir / name).relative_to(output)): sha256(model_dir / name)
                               for name in ('config.yaml', 'official_result_stat.json', 'eval_epochNone.yaml')})
    atomic_json(manifest_path, manifest)
    return manifest
