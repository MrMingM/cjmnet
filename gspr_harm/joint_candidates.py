"""复用旧消息，检查单块改善候选一起删除后到底有没有 AP 收益。"""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import time

import torch
from gspr_communication import codec, runtime as rt
from .core import Messages, predict, drop_mask, random_deletions, loss_values
from .scan import fingerprint, identity
from .config import fusion_args_for_probe


def improving(row):
    # 和机会检查用同一个条件，不在这里偷偷调筛选标准。
    recovered, lost, fp = (row['recovered_after_drop_iou70'], row['lost_after_drop_iou70'],
                           row['fp_change_after_drop'])
    return lost == 0 and ((recovered > 0 and fp <= 0) or (recovered == 0 and fp < 0))


def candidates(rows, width, selected):
    selected = set(map(tuple, selected))
    covered, delta, improve = set(), set(), set()
    for row in rows:
        pairs = {(int(row['sender_index']), int(y)*width+int(x)) for y, x in row['blocks_yx']}
        if len(pairs) != row['removed_native_blocks'] or not pairs <= selected or covered & pairs:
            raise ValueError('旧区域记录与已发送块不一致或重复，停止比较')
        covered |= pairs
        if row['class'] == 'harmful':
            delta |= pairs
        if improving(row):
            improve |= pairs
    if covered != selected:
        raise ValueError('缺少逐块记录，不能把不完整扫描当成完整候选集')
    return sorted(delta), sorted(improve)


def replay(model, levels, directory, meta):
    n = len(levels[0])
    mask = levels[0].new_zeros((n, 1, *model.grid))
    mask[0] = 1
    received = [torch.zeros_like(x) for x in levels]
    for destination, source in zip(received, levels):
        destination[:1] = source[:1]
    request = (directory/'request.bin').read_bytes()
    if request and codec.unpack_request(request).shape != model.grid:
        raise ValueError('旧请求网格与当前模型不一致')
    responses, selected = {}, []
    # 邻车特征只取旧包里的值，不采用这次重新编码的邻车特征。
    for name, digest in meta['packet_sha256'].items():
        if Path(name).name != name or rt.sha256(directory/name) != digest:
            raise ValueError('原始通信包校验失败: '+name)
    if 'request.bin' not in meta['packet_sha256']:
        raise ValueError('缺少请求包哈希')
    for peer in range(1, n):
        name = f'sender_{peer}.bin'
        if name not in meta['packet_sha256']:
            if request:
                raise ValueError('有请求但缺少邻车响应包')
            continue
        packet = (directory/name).read_bytes()
        arrays, ids = codec.unpack_response(packet, [tuple(x.shape[1:]) for x in levels], model.grid)
        responses[peer] = packet
        mask[peer].view(-1)[torch.as_tensor(ids.copy(), dtype=torch.long, device=mask.device)] = 1
        selected.extend((peer, int(i)) for i in ids)
        for destination, array in zip(received, arrays):
            destination[peer] = torch.as_tensor(array, dtype=destination.dtype, device=destination.device)
    total = (n-1)*len(request)+sum(map(len, responses.values()))
    if total != meta['total_bytes'] or set(selected) != set(map(tuple, meta['selected_sender_blocks'])):
        raise ValueError('回放消息与原发送集合或字节数不一致')
    return Messages(received, mask, request, responses, total)


def packet_bytes(model, messages, mask):
    total = (len(mask)-1)*len(messages.request)
    for peer in messages.responses:
        ids = mask[peer].flatten().nonzero().flatten().cpu().numpy()
        total += len(codec.pack_response([x[peer].cpu().numpy() for x in messages.levels],
                                        ids, model.grid, model.value_bytes))
    removed = int((messages.mask-mask).sum())
    if total != messages.total_bytes-removed*model.cost:
        raise ValueError('实际打包字节数与删除数量不符')
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--frontend-config', required=True)
    parser.add_argument('--frontend-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--weather', nargs='+', choices=['fog', 'rain', 'snow'], default=['fog', 'rain', 'snow'])
    args = parser.parse_args()
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.loss.point_pillar_loss import PointPillarLoss
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from gspr_communication.dataset_adapter import CommunicationDataset
    from gspr_communication.evaluate import matched_objects

    source = args.source_run.resolve()
    audit = json.loads((source/'audit.json').read_text(encoding='utf-8'))
    old_summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if old_summary.get('status') != 'complete' or not old_summary.get('frozen_model_unchanged'):
        raise ValueError('需要完整完成且模型未变化的原实验')
    rt.verify_frozen()
    for filename, key in [(args.frontend_config, 'frontend_config_sha256'),
                          (args.frontend_checkpoint, 'checkpoint_sha256')]:
        if rt.sha256(filename) != audit[key]:
            raise ValueError('配置或权重与原实验不一致: '+key)
    for name in ('gspr_communication/model.py', 'gspr_communication/codec.py',
                 'gspr_communication/masked_attfuse.py', 'gspr_harm/core.py'):
        if rt.sha256(rt.ROOT/name) != audit['source_hashes'][name]:
            raise ValueError('影响前向的源码与原实验不一致: '+name)
    hypes = load_yaml(args.frontend_config)
    hypes['fusion']['args'] = fusion_args_for_probe(hypes['fusion'])
    hypes['data_augment'] = []
    hypes['validate_dir'] = audit['evaluation_root']
    options = {'communication': audit['communication'], 'lidar_key': 'processed_lidar_weather'}
    seed = int(audit['seed'])
    rt.seed_all(seed)
    target = rt.device()
    model, checkpoint_hash = rt.load_model(hypes, options, args.frontend_checkpoint, target)
    model.eval().requires_grad_(False)
    state_before = fingerprint(model.state_dict())
    criterion = PointPillarLoss(hypes['loss']['args']).to(target).eval()
    out = rt.new_output(args.output_dir)
    rt.write_json(out/'protocol.json', {'source_run': str(source), 'source_audit_sha256': rt.sha256(source/'audit.json'),
        'checkpoint_sha256': checkpoint_hash, 'seed': seed,
        'candidate_rule': 'lost=0 AND ((recovered>0 AND fp_change<=0) OR (recovered=0 AND fp_change<0)), IoU0.7',
        'purpose': 'GT-assisted joint diagnostic, not deployable, not an optimal oracle',
        'baseline_check': 'exact input/label and packet hashes; component losses atol=1e-5 rtol=1e-5',
        'ego': 'reencoded from identical hashed input; original ego features were not saved',
        'bytes': 'original requests and reply headers retained, no refill; random matched per sender to new candidates'})
    names = ['a0b0', 'delta_joint', 'improvement_joint', 'random_0', 'random_1', 'random_2']
    results = {}
    with torch.inference_mode():
        for weather in args.weather:
            protocol = json.loads((source/weather/'data_protocol.json').read_text(encoding='utf-8'))
            hypes['weather_augmentation'] = protocol['weather']
            if protocol.get('root') != hypes['validate_dir']:
                raise ValueError('数据目录与源实验不符')
            ds = CommunicationDataset(copy.deepcopy(hypes), train=False)
            indices = protocol['probe_indices']
            if len(indices) != old_summary['weather'][weather]['frames'] or len(set(indices)) != len(indices):
                raise ValueError('源实验帧数不一致')
            rows_by_frame = defaultdict(list)
            with (source/weather/'regions.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    rows_by_frame[row['sample_index']].append(row)
            with (source/weather/'frames.jsonl').open(encoding='utf-8') as stream:
                old_frames = {r['sample_index']: r for r in map(json.loads, stream)}
            if set(old_frames) != set(indices) or not set(rows_by_frame) <= set(indices):
                raise ValueError('旧结果帧索引缺失或多余')
            wd = out/weather
            wd.mkdir()
            metrics = {name: {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
                             for iou in (.3, .5, .7)} for name in names}
            totals = {name: {'bytes': 0, 'loss': 0., 'recovered_vs_a0b0': 0,
                            'lost_vs_a0b0': 0, 'fp_change_vs_a0b0': 0} for name in names}
            started = time.monotonic()
            max_baseline_drift = 0.
            with (wd/'frames.jsonl').open('w', encoding='utf-8') as stream:
                for position, index in enumerate(indices):
                    directory = source/weather/f'frame_{index}'
                    meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
                    rt.seed_all(meta['seed'])
                    sample = ds[index]
                    current = identity(ds, index, sample)
                    if any(current[k] != meta[k] for k in ('scene_path', 'timestamp', 'cav_ids_in_feature_order')):
                        raise ValueError('场景、帧或车辆顺序改变')
                    batch = to_device(ds.collate_batch_test([sample]), target)
                    inp, labels = rt.model_input(batch['ego'], options), batch['ego']['label_dict']
                    if fingerprint(inp) != meta['input_sha256'] or fingerprint(labels) != meta['labels_sha256']:
                        raise ValueError(f'{weather}/{index}: 输入或标签哈希不一致，停止；不要把它当成同一帧的公平比较')
                    encoded = model.encode(inp)
                    messages = replay(model, encoded[0], directory, meta)
                    delta, improve = candidates(rows_by_frame[index], model.grid[1], meta['selected_sender_blocks'])
                    old_delta = old_frames[index]['controls']['gt_single_deletion_joint']['removed_sender_blocks']
                    if delta != sorted(map(tuple, old_delta)):
                        raise ValueError('旧 ΔL 选择集合不一致')
                    choices = {'a0b0': [], 'delta_joint': delta, 'improvement_joint': improve}
                    for i in range(3):
                        choices[f'random_{i}'] = random_deletions(messages.mask, improve, seed+index+100003*(i+1))
                    frame = {'sample_index': index, 'scene': meta['scene_path'], 'controls': {}}
                    for name, pairs in choices.items():
                        mask = drop_mask(messages.mask, pairs)
                        output = predict(model, messages, mask)
                        loss = loss_values(criterion, output, labels)
                        if name in ('a0b0', 'delta_joint'):
                            old_name = 'a0b0' if name == 'a0b0' else 'gt_single_deletion_joint'
                            old_loss = old_frames[index]['controls'][old_name]['loss']
                            for k, value in loss.items():
                                max_baseline_drift = max(max_baseline_drift, abs(value-old_loss[k]))
                                if not math.isclose(value, old_loss[k], abs_tol=1e-5, rel_tol=1e-5):
                                    raise ValueError(f'{weather}/{index}/{name}: 旧分支损失无法复现，停止比较')
                        boxes, scores, gt = ds.post_process(batch, {'ego': output})
                        match, fp = matched_objects(boxes, scores, gt)
                        if name == 'a0b0':
                            base_match, base_fp, base_gt = match, fp, gt
                        else:
                            torch.testing.assert_close(gt, base_gt, atol=0, rtol=0)
                        actual = packet_bytes(model, messages, mask)
                        for iou in metrics[name]:
                            eval_utils.caluclate_tp_fp(boxes, scores, gt, metrics[name], iou)
                        totals[name]['bytes'] += actual
                        totals[name]['loss'] += loss['total_loss']
                        totals[name]['recovered_vs_a0b0'] += len(match-base_match)
                        totals[name]['lost_vs_a0b0'] += len(base_match-match)
                        totals[name]['fp_change_vs_a0b0'] += fp-base_fp
                        frame['controls'][name] = {'bytes': actual, 'loss': loss, 'removed_sender_blocks': pairs,
                            'recovered_gt_indices_iou70': sorted(match-base_match),
                            'lost_gt_indices_iou70': sorted(base_match-match), 'fp_change_vs_a0b0': fp-base_fp}
                    if any(frame['controls'][f'random_{i}']['bytes'] != frame['controls']['improvement_joint']['bytes'] for i in range(3)):
                        raise ValueError('随机对照与新候选的实际字节不一致')
                    stream.write(json.dumps(frame)+'\n')
                    stream.flush()
                    if position == 0 or (position+1) % 10 == 0:
                        elapsed = time.monotonic()-started
                        print(f'{weather}: {position+1}/{len(indices)} frames; elapsed {elapsed/60:.1f} min; '
                              f'rough remaining {elapsed/(position+1)*(len(indices)-position-1)/60:.1f} min', flush=True)
            result = {'frames': len(indices), 'controls': {}, 'max_replayed_control_loss_drift': max_baseline_drift,
                      'warning': 'GT-assisted diagnostic selected on this validation set; not independent test or deployable result'}
            for name in names:
                if metrics[name][.7]['gt'] == 0:
                    raise ValueError('没有 GT，无法计算 AP')
                folder = wd/name
                folder.mkdir()
                eval_utils.eval_final_results(metrics[name], str(folder), True)
                import yaml
                ap = yaml.safe_load((folder/'eval_global_sort.yaml').read_text())
                result['controls'][name] = dict(totals[name], mean_bytes=totals[name]['bytes']/len(indices),
                    mean_loss=totals[name]['loss']/len(indices), ap50=ap['ap_50'], ap70=ap['ap_70'])
            result['replayed_ap_difference_from_source'] = {name: {
                key: result['controls'][name][key]-old_summary['weather'][weather]['controls'][old][key]
                for key in ('ap50', 'ap70')} for name, old in [('a0b0', 'a0b0'), ('delta_joint', 'gt_single_deletion_joint')]}
            rt.write_json(wd/'summary.json', result)
            results[weather] = result
    if fingerprint(model.state_dict()) != state_before or rt.sha256(args.frontend_checkpoint) != checkpoint_hash:
        raise ValueError('冻结模型或 checkpoint 改变')
    rt.verify_frozen()
    rt.write_json(out/'summary.json', {'status': 'complete', 'source_run': str(source),
                                      'frozen_model_unchanged': True, 'weather': results})
    print('Send back: '+str(out/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
