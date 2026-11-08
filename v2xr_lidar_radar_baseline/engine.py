"""LR0 uses official OpenCOOD operations and the existing full-state checkpoint helper."""
from contextlib import ExitStack
from collections import Counter
from pathlib import Path
import time

from common import atomic_json, read_json, sha256
from checkpoint import restore, restore_rng, save, seed, state, rng
from runtime import EPOCHS, SEED, build, components, create_model, loader
from metrics import compare_lidar_gt, evaluate
from empty_input import empty_slots


def checked_forward(model, ego):
    import torch
    n = int(ego['record_len'].sum())
    ny, nx = model.scatter.ny, model.scatter.nx
    shapes, maps = {}, []

    def encoder(sensor):
        def check(_, args, out):
            x = args[0]['voxel_features']
            coords = out['voxel_coords']
            p = out['voxel_num_points']
            f = out['pillar_features']
            if (x.shape != (len(coords), 32, 4) or coords.shape != (len(p), 4)
                    or f.shape != (len(coords), 64) or not torch.isfinite(x).all()
                    or not torch.isfinite(f).all() or not ((p > 0) & (p <= 32)).all()
                    or not ((coords[:, 0] >= 0) & (coords[:, 0] < n)).all()):
                raise ValueError(f'{sensor} encoder shape or input differs')
            shapes[sensor] = dict(voxels=list(x.shape), pillars=list(f.shape))
        return check

    def scatter(_, __, out):
        spatial = out['spatial_features']
        if spatial.shape != (n, 64, ny, nx):
            raise ValueError('Vehicle count or modality feature map shape differs')
        maps.append(spatial)
        shapes['scatter'] = list(spatial.shape)

    def backbone(_, args):
        x = args[0]['spatial_features']
        if (len(maps) != 2 or x.shape != (n, 128, ny, nx)
                or not torch.equal(x[:, :64], maps[0]) or not torch.equal(x[:, 64:], maps[1])
                or not torch.equal(args[0]['record_len'], ego['record_len'])):
            raise ValueError('Official concatenation or attention vehicle grouping differs')
        shapes['attention_backbone_input'] = list(x.shape)

    with ExitStack() as stack:
        for layer, hook in ((model.lidar_pillar_vfe, encoder('lidar')),
                            (model.radar_pillar_vfe, encoder('radar')), (model.scatter, scatter)):
            stack.callback(layer.register_forward_hook(hook).remove)
        stack.callback(model.backbone.register_forward_pre_hook(backbone).remove)
        prediction = model(ego)
    for key, channels in (('psm', 2), ('rm', 14)):
        value = prediction[key]
        if value.shape != (len(ego['record_len']), channels, ny // 2, nx // 2) or not torch.isfinite(value).all():
            raise ValueError(f'Official detection head output differs: {key}')
        shapes[key] = list(value.shape)
    return prediction, shapes


def train_step(model, criterion, optimizer, batch, epoch, inspect=False):
    import torch
    from opencood.tools import train_utils

    model.train()
    optimizer.zero_grad()
    ego = train_utils.to_device(batch['ego'], torch.device('cuda:0'))
    ego['epoch'] = epoch
    prediction, shapes = checked_forward(model, ego) if inspect else (model(ego), None)
    loss = criterion(prediction, ego['label_dict'])
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    watched = (model.lidar_pillar_vfe.pfn_layers[0].linear.weight,
               model.radar_pillar_vfe.pfn_layers[0].linear.weight, model.cls_head.weight)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Missing/nonfinite training gradient')
    if inspect and any(not torch.count_nonzero(p.grad) for p in watched):
        raise ValueError('An encoder or detection head did not receive a gradient')
    before = [p.detach().clone() for p in watched] if inspect else None
    optimizer.step()
    if inspect and any(torch.equal(old, p) for old, p in zip(before, watched)):
        raise ValueError('Optimizer failed to update an encoder or detection head')
    return float(loss.detach()), shapes


def smoke(root, config, contract):
    from v2xr_weather_l012.data import tensor_digest

    report = root / 'SMOKE.json'
    if report.is_file():
        done = read_json(report)
        if (done['contract'] != contract or sha256(root / 'smoke/summary.json') != done['evaluation_sha256']
                or sha256(root / 'smoke_state.pth') != done['resume_sha256']):
            raise ValueError('Smoke evidence changed')
        return
    seed(SEED)
    dataset, cfg = build(config, split='train')
    batches = loader(dataset, training=True, limit=16)
    model, criterion, optimizer, scheduler = components(cfg, len(batches))
    losses, shapes, empty = [], None, Counter()
    for index, batch in enumerate(batches):
        empty.update(empty_slots(batch['ego']))
        loss, observed = train_step(model, criterion, optimizer, batch, 0, inspect=index == 0)
        shapes = observed or shapes
        losses.append(loss)
        print(f'SMOKE batch {index+1} loss={loss:.6f} shapes={observed}', flush=True)
    if len(losses) != 2:
        raise ValueError('Smoke must finish two real batch-8 updates')
    weights = root / 'smoke_weights.pth'
    save(weights, model.state_dict())
    evaluated = evaluate(model, criterion, config, root / 'smoke', ('clean', 'fog', 'snow'),
                         12, contract, sha256(weights), loss=True)
    scheduler.step(0)
    resume = root / 'smoke_state.pth'
    save(resume, state(model, optimizer, scheduler, criterion, 1, contract))
    expected = tensor_digest([model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), rng()])
    seed(SEED + 1)
    restore(resume, model, optimizer, scheduler, criterion, contract, 1)
    if expected != tensor_digest([model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), rng()]):
        raise ValueError('Full-state checkpoint round trip differs')
    if set(evaluated['results']) != {'clean', 'fog', 'snow'} or any(
            set(r['AP']) != {'O', 'O*', 'F'} for r in evaluated['results'].values()):
        raise ValueError('Frozen evaluator did not produce all requested protocols')
    atomic_json(report, dict(status='complete', contract=contract, batch_losses=losses,
        shapes=shapes, empty_slots=empty,
        evaluation_sha256=sha256(root / 'smoke/summary.json'), resume_sha256=sha256(resume),
        metrics=evaluated['results'], gpu=evaluated['performance']))


def committed(root, contract):
    rows = []
    for epoch in range(1, EPOCHS + 1):
        path = root / 'epochs' / f'{epoch:02d}' / 'complete.json'
        if not path.is_file():
            continue
        row = read_json(path)
        if (len(rows) + 1 != epoch or row['epoch'] != epoch or row['contract'] != contract
                or any(sha256(k) != v for k, v in row['artifacts'].items())):
            raise ValueError(f'Invalid committed epoch {epoch}')
        rows.append(row)
    return rows


def clean_validation(model, criterion, config, output, contract, weight_hash):
    training_rng = rng()
    try:
        seed(20261016)
        return evaluate(model, criterion, config, output, ('clean',), 829,
                        contract, weight_hash, loss=True)
    finally:
        restore_rng(training_rng)


def train(root, config, contract):
    import torch

    rows = committed(root, contract)
    if len(rows) == EPOCHS:
        return rows
    seed(SEED)
    dataset, cfg = build(config, split='train')
    batches = loader(dataset, training=True)
    if len(dataset) != 8083 or len(batches) != 1010:
        raise ValueError('Unexpected train frame/batch count')
    model, criterion, optimizer, scheduler = components(cfg, len(batches))
    if rows:
        restore(rows[-1]['resume'], model, optimizer, scheduler, criterion, contract, len(rows))
    for epoch in range(len(rows), EPOCHS):
        started = time.monotonic()
        destination = root / 'epochs' / f'{epoch+1:02d}'
        weights, resume = destination / 'weights.pth', destination / 'resume.pth'
        train_marker = destination / 'train_complete.json'
        pre_val = destination / 'before_validation.pth'
        if train_marker.is_file():
            step = read_json(train_marker)
            if (step['contract'] != contract or step['epoch'] != epoch+1
                    or sha256(weights) != step['checkpoint_sha256']
                    or sha256(pre_val) != step['pre_validation_sha256']):
                raise ValueError('Interrupted epoch training evidence changed')
            restore(pre_val, model, optimizer, scheduler, criterion, contract, epoch+1)
        else:
            losses, empty = [], Counter()
            torch.cuda.reset_peak_memory_stats()
            print(f'TRAIN START {epoch+1}/{EPOCHS}', flush=True)
            for index, batch in enumerate(batches):
                empty.update(empty_slots(batch['ego']))
                value, _ = train_step(model, criterion, optimizer, batch, epoch)
                losses.append(value)
                if (index + 1) % 25 == 0:
                    print(f'TRAIN {epoch+1} {index+1}/{len(batches)} loss={value:.5f}', flush=True)
            if len(losses) != 1010:
                raise ValueError('Incomplete epoch')
            save(weights, model.state_dict())
            save(pre_val, state(model, optimizer, scheduler, criterion, epoch+1, contract))
            step = dict(epoch=epoch+1, contract=contract, checkpoint_sha256=sha256(weights),
                pre_validation_sha256=sha256(pre_val), train_updates=len(losses), empty_slots=empty,
                train_loss=sum(losses)/len(losses), peak_train_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
            atomic_json(train_marker, step)
        weight_hash = step['checkpoint_sha256']
        torch.cuda.empty_cache()
        val = clean_validation(model, criterion, config, destination / 'validation', contract, weight_hash)
        scheduler.step(epoch)
        save(resume, state(model, optimizer, scheduler, criterion, epoch+1, contract))
        row = dict(step, status='complete', checkpoint=str(weights),
            resume=str(resume), validation_loss=val['validation_loss'],
            clean_AP=val['results']['clean']['AP'], next_lr=optimizer.param_groups[0]['lr'],
            seconds=time.monotonic() - started,
            artifacts={str(p): sha256(p) for p in (weights, resume, destination / 'validation/summary.json')})
        atomic_json(destination / 'complete.json', row)
        pre_val.unlink()
        rows.append(row)
        atomic_json(root / 'history.json', rows)
        print(f'EPOCH COMPLETE {epoch+1}/{EPOCHS} Clean O AP70={row["clean_AP"]["O"]["ap70"]}', flush=True)
    return rows


def finalize(root, config, contract, rows):
    import torch
    from opencood.hypes_yaml import yaml_utils
    if len(rows) != EPOCHS:
        raise ValueError('Selection requires 20 committed epochs')
    best = min(rows, key=lambda r: (-r['clean_AP']['O']['ap70'], r['epoch']))
    weight = Path(best['checkpoint'])
    digest = sha256(weight)
    if digest != best['checkpoint_sha256']:
        raise ValueError('Selected weight differs')
    selected = dict(epoch=best['epoch'], checkpoint=str(weight), sha256=digest,
        policy='maximum Clean O AP70 over 20 epochs; earliest exact tie',
        candidates=[dict(epoch=r['epoch'], ap70=r['clean_AP']['O']['ap70'],
                         sha256=r['checkpoint_sha256']) for r in rows])
    atomic_json(root / 'selected.json', selected)
    seed(20261016)
    model = create_model(yaml_utils.load_yaml(str(config)))
    model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=False), strict=True)
    out = root / 'final_evaluation'
    metrics = evaluate(model, None, config, out, ('clean', 'fog', 'snow'), 829, contract, digest)
    difference = compare_lidar_gt(out)
    parameters = sum(p.numel() for p in model.parameters())
    result = dict(status='complete', experiment='LR0', contract=contract, selected=selected,
        parameters=parameters, gt_comparison=difference, evaluation=metrics,
        input_data=read_json(root / 'DATA.json'))
    lines = ['# LR0 正式结果', '', f'选中第 {best["epoch"]} 轮，权重 SHA256 {digest}。',
             f'模型参数量 {parameters:,}。三种天气各 829 帧。', '',
             '| 天气 | 协议 | AP30 | AP50 | AP70 | 参与真值 | None帧 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for weather, r in metrics['results'].items():
        for protocol, ap in r['AP'].items():
            lines.append(f'| {weather} | {protocol} | {ap["ap30"]:.6f} | {ap["ap50"]:.6f} | '
                         f'{ap["ap70"]:.6f} | {r["counted_gt"][protocol]} | {r["counts"]["none_frames"]} |')
    lines += ['', '| 天气 | 协议 | 重叠阈值 | 匹配 | 误检 | 漏检 |',
              '|---|---|---:|---:|---:|---:|']
    for weather, r in metrics['results'].items():
        for protocol, matches in r['matches'].items():
            for threshold, m in matches.items():
                lines.append(f'| {weather} | {protocol} | {threshold} | {m["tp"]} | {m["fp"]} | {m["fn"]} |')
    lines += ['', '| 天气 | 推理帧/秒 | 峰值显存 GiB（已分配） |',
              '|---|---:|---:|']
    for weather, p in metrics['performance'].items():
        lines.append(f'| {weather} | {p["inference_fps"]:.2f} | {p["peak_gpu_allocated_bytes"]/2**30:.3f} |')
    lines += ['', '推理时间包括模型和官方后处理，按帧同步。显存为本进程峰值。', '',
              'O 跳过 None 帧，O* 保留天气真值，F 固定双模态 Clean 真值。', '',
              '| 天气 | 纯 LiDAR 真值 | 双模态真值 | 仅纯 LiDAR | 仅双模态 |',
              '|---|---:|---:|---:|---:|']
    for weather, r in difference.items():
        lines.append(f'| {weather} | {r["lidar_gt"]} | {r["lr0_gt"]} | '
                     f'{r["lidar_only_ids"]} | {r["lr0_only_ids"]} |')
    lines += ['', '双模态真值允许 LiDAR 或 Radar 支撑。分母变化需单独比较。']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    atomic_json(root / 'FINAL_RESULTS.json', result)

