"""One fail-closed sequential run using the LR0 epoch and evaluation engines."""
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import sys
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / p) for p in ('v2xr_lidar_radar_baseline', 'v2xr_official_repro',
                                        'v2xr_protocol_audit', 'v2xr_weather_l012')]
sys.path.append(str(REPO))
from common import atomic_json, identity, lock, read_json, sha256
from runtime import DATA, RUNTIME, build, loader
sys.path.insert(0, str(RUNTIME))
from v2xr_lidar_radar_baseline import engine
from v2xr_lidar_radar_baseline.run import source_contract
from checkpoint import rng, restore, save, seed, state
from v2xr_weather_l012.data import tensor_digest
from support import patched
from diagnostics import LR0, WEATHERS, aggregate, observe_eval, target_changes, verify_gt

OUT = DATA / 'logs/v2xr-radar-ra-rb-v1'
CONFIG = LR0 / 'config.yaml'


def bind(path, value):
    if path.is_file():
        if read_json(path) != value:
            raise ValueError(f'Frozen record differs: {path}')
    else:
        atomic_json(path, value)


def audit():
    import torch
    from inputs import TRAIN_MANIFEST, VAL_MANIFEST, inventory

    if socket.gethostname().split('.')[0] != 'admin1' or not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError('Require admin1 with exactly one visible ROCm card')
    if not Path(__import__('opencood').__file__).is_relative_to(RUNTIME):
        raise RuntimeError('Wrong OpenCOOD runtime')
    locked = read_json(LR0 / 'CONTRACT.json')
    lr0 = read_json(LR0 / 'FINAL_RESULTS.json')
    if (source_contract() != locked['sources'] or sha256(CONFIG) != locked['config_sha256']
            or lr0['status'] != 'complete' or sha256(LR0 / 'DATA.json') != locked['data_sha256']):
        raise ValueError('Frozen LR0 source, config or data identity differs')
    snapshot = read_json(HERE / 'source_snapshot.json')
    current = {p.name: sha256(p) for p in HERE.iterdir() if p.is_file() and p.name != 'source_snapshot.json'}
    if current != snapshot:
        raise ValueError('Transfer the complete source package')
    data = OUT / 'DATA.json'
    record = read_json(data) if data.is_file() else inventory(data)
    if record != read_json(LR0 / 'DATA.json') or any(
            record['splits'][s]['input_manifest_sha256'] != sha256(p)
            for s, p in (('train', TRAIN_MANIFEST), ('validate', VAL_MANIFEST))):
        raise ValueError('Audited training and validation inputs differ from LR0')
    deps = ['v2xr_lr1/lr1_model.py', 'gspr_ggca/modules.py', 'gspr_ggca/settings.json',
            'gspr_ggca/bridge.py', 'gspr_repe/modules.py', 'attfuse_gspr/reliability.py',
            'v2xr_gspr_clean/settings.json']
    contract = dict(source=snapshot, frozen={p: sha256(REPO / p) for p in deps},
                    lr0_contract=sha256(LR0 / 'CONTRACT.json'), lr0_results=sha256(LR0 / 'FINAL_RESULTS.json'),
                    fixed_gt=sha256(LR0 / 'final_evaluation/gt_manifest.json'), data=sha256(data),
                    physical_gpu=os.environ['ROCR_VISIBLE_DEVICES'], gpu=torch.cuda.get_device_name(0),
                    torch=str(torch.__version__), hip=torch.version.hip, python=sys.version,
                    hash_seed=os.environ['PYTHONHASHSEED'])
    bind(OUT / 'CONTRACT.json', contract)
    return identity(contract)


def make_model(cfg, settings):
    from model import Model
    return Model(cfg['model']['args'], settings).to('cuda:0')


def components(cfg, batches, settings):
    from model import Loss
    from opencood.tools import train_utils

    model = make_model(cfg, settings)
    criterion = Loss(train_utils.create_loss(cfg).to('cuda:0'), model, settings).to('cuda:0')
    optimizer = train_utils.setup_optimizer(cfg, model)
    scheduler = train_utils.setup_lr_schedular(cfg, optimizer, n_iter_per_epoch=batches)
    return model, criterion, optimizer, scheduler


def update(model, criterion, optimizer, batch, epoch, inspect=False):
    import torch
    from opencood.tools import train_utils

    model.train()
    optimizer.zero_grad(set_to_none=True)
    ego = train_utils.to_device(batch['ego'], torch.device('cuda:0'))
    ego['epoch'] = epoch
    pair = batch.get('paired_weather')
    criterion.pair = None if pair is None else tuple(a.to('cuda:0') for a in pair)
    try:
        loss = criterion(model(ego), ego['label_dict'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        watched = {}
        if inspect:
            watched = {'gspr': model.gspr, 'ggca': model.ggca, 'radar': model.radar_pillar_vfe}
            watched.update({k: getattr(model, k) for k in model.extra_used})
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise ValueError(f'Nonfinite gradient: {name}')
        changed = {}
        for name, layer in watched.items():
            usable = [p for p in layer.parameters() if p.grad is not None and torch.count_nonzero(p.grad)]
            if not usable:
                raise ValueError(f'{name} has no nonzero gradient')
            changed[name] = [(p, p.detach().clone()) for p in usable]
        optimizer.step()
        for name, before in changed.items():
            if not any(not torch.equal(p, old) for p, old in before):
                raise ValueError(f'{name} parameters did not update')
        return float(loss.detach()), None
    finally:
        criterion.pair = None


@contextmanager
def use_variant(settings, out):
    from pairing import PairedDataset

    def training_build(config, **kwargs):
        data, cfg = build(config, **kwargs)
        if settings['paired'] and kwargs.get('split') == 'train':
            grid = cfg['model']['args']['point_pillar_scatter']['grid_size']
            data = PairedDataset(data, int(grid[0]))
        return data, cfg

    counts, sums, seen = Counter(), Counter(), 0

    def training_update(model, criterion, optimizer, batch, epoch, **kwargs):
        nonlocal seen
        result = update(model, criterion, optimizer, batch, epoch, **kwargs)
        seen += 1
        if settings['paired']:
            counts.update('fog' if row[2] else 'clean' for row in batch['pair_audit'])
            counts['extra_reads'] += sum(row[3] > 0 for row in batch['pair_audit'])
            sums.update({k: criterion.last_terms.get(k, 0.) for k in ('weather', 'preserve', 'supervised', 'protected')})
        if seen == 1010:
            atomic_json(out / 'epochs' / f'{epoch+1:02d}' / 'auxiliary.json',
                        dict(paired_vehicle_reads=counts, loss_and_mask_sums=sums))
            counts.clear()
            sums.clear()
            seen = 0
        return result

    create = lambda cfg: make_model(cfg, settings)
    def final_eval(model, criterion, config, destination, *args, **kwargs):
        with observe_eval(model, destination):
            return engine_metrics(model, criterion, config, destination, *args, **kwargs)

    engine_metrics = engine.evaluate
    with patched(engine, 'create_model', create), patched(engine, 'build', training_build), \
         patched(engine, 'components', lambda cfg, batches: components(cfg, batches, settings)), \
         patched(engine, 'train_step', training_update):
        yield final_eval


def run_variant(name, settings, contract, final_reference):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / 'config.yaml'
    if not cfg.is_file():
        cfg.write_bytes(CONFIG.read_bytes())
    if sha256(cfg) != sha256(CONFIG):
        raise ValueError('Baseline training configuration changed')
    config = dict(experiment=contract, group=name, settings=settings, base_config=sha256(cfg),
                  initial=read_json(OUT / 'PREFLIGHT.json')['initial'][name],
                  historical_lr1_candidate=str(DATA / 'logs/v2xr-lr1-v1/LR1'))
    bind(out / 'CONTRACT.json', config)
    bind(out / 'DATA.json', read_json(OUT / 'DATA.json'))
    run_id = identity(config)
    done = out / 'GROUP_COMPLETE.json'
    if done.is_file():
        marker = read_json(done)
        if marker['contract'] != run_id or any(sha256(out / k) != v for k, v in marker['files'].items()):
            raise ValueError(f'Completed group changed: {name}')
        return read_json(out / 'FINAL_RESULTS.json')
    with use_variant(settings, out) as final_eval:
        print(f'START {name} 20-EPOCH TRAINING', flush=True)
        history = engine.train(out, cfg, run_id)
        with patched(engine, 'evaluate', final_eval):
            engine.finalize(out, cfg, run_id, history)
    verify_gt(out / 'final_evaluation')
    summary = read_json(out / 'FINAL_RESULTS.json')
    summary['experiment'] = name
    summary['diagnostics'] = aggregate(out / 'final_evaluation')
    summary['training_seconds'] = sum(r['seconds'] for r in history)
    summary['training_peak_gpu_allocated_bytes'] = max(r['peak_train_gpu_allocated_bytes'] for r in history)
    if name == 'F0':
        summary['historical_lr1_evidence'] = final_reference
    atomic_json(out / 'FINAL_RESULTS.json', summary)
    md = out / 'RESULTS.md'
    md.write_text(md.read_text(encoding='utf-8').replace('# LR0 正式结果', '# '+name+' 正式结果'), encoding='utf-8')
    bind(done, dict(contract=run_id, files={k:sha256(out/k) for k in
        ('FINAL_RESULTS.json', 'RESULTS.md', 'selected.json', 'history.json')}))
    return read_json(out / 'FINAL_RESULTS.json')


def interventions(settings, result, contract):
    import torch
    from opencood.hypes_yaml import yaml_utils

    seed(20261016)
    model = make_model(yaml_utils.load_yaml(str(CONFIG)), settings)
    checkpoint = result['selected']
    if sha256(checkpoint['checkpoint']) != checkpoint['sha256']:
        raise ValueError('F3 selected weights changed')
    model.load_state_dict(torch.load(checkpoint['checkpoint'], map_location='cpu', weights_only=False), strict=True)
    outputs = {}
    for name, mode, ra, rb in (
        ('radar_off', 'radar_off', True, True), ('radar_shuffle', 'radar_shuffle', True, True),
        ('ra_off', 'normal', False, True), ('rb_off', 'normal', True, False),
        ('both_off', 'normal', False, False)):
        seed(20261016)
        model.mode, model.ra_on, model.rb_on = mode, ra, rb
        out = OUT / 'mechanisms' / name
        with observe_eval(model, out):
            result_eval = engine.evaluate(model, None, CONFIG, out, WEATHERS, 829,
                                          identity([contract, name]), checkpoint['sha256'])
        verify_gt(out)
        outputs[name] = dict(evaluation=result_eval, diagnostics=aggregate(out))
    return outputs


def compare(results, mechanisms):
    contrasts = [('F1','F0'), ('F2','F1'), ('F3','F2'), ('F2','F2-ordinary')]
    all_results = dict(results, **mechanisms)
    contrasts.extend((k, 'F3') for k in mechanisms)
    rows = ['# R-A / R-B 消融结果', '',
            '| 对照 | 天气 | 协议 | 指标 | 当前 AP | 参照 AP | 差值（百分点） |',
            '|---|---|---|---|---:|---:|---:|']
    output = {}
    for current, previous in contrasts:
        newer, older = all_results[current], all_results[previous]
        contrast = output[f'{current}-{previous}'] = dict(ap=[], targets=target_changes(
            older['diagnostics'], newer['diagnostics']))
        for w in WEATHERS:
            x, y = (a['evaluation']['results'][w]['AP'] for a in (newer, older))
            for protocol, values in x.items():
                for k, v in values.items():
                    old = y[protocol][k]
                    delta = 100*(v-old)
                    contrast['ap'].append(dict(weather=w, protocol=protocol, metric=k,
                                               current=v, reference=old, delta_pp=delta))
                    rows.append(f'| {current}−{previous} | {w} | {protocol} | {k} | {v:.6f} | {old:.6f} | {delta:+.4f} |')
    rows.extend(['', '目标诊断按双模态固定 Clean 真值，重叠阈值 0.7。'])
    for name, record in output.items():
        rows.extend(['', f'## {name}', '', '| 天气 | 目标组 | 恢复 | 丢失 | 共同匹配 | 共同匹配平均 IoU 差 |',
                     '|---|---|---:|---:|---:|---:|'])
        for w, groups in record['targets'].items():
            for group, v in groups.items():
                mean = f'{v["iou_delta_sum"]/v["common"]:+.5f}' if v['common'] else '—'
                rows.append(f'| {w} | {group.replace("|", " / ")} | {v["recovered"]} | {v["lost"]} | {v["common"]} | {mean} |')
    (OUT / 'ABLATION_COMPARISON.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    atomic_json(OUT / 'FINAL_COMPARISON.json', dict(status='complete', contrasts=output,
        groups=results, mechanisms=mechanisms))
    atomic_json(OUT / 'FINAL_RESULTS.json', dict(status='complete', groups={n:sha256(OUT/n/'FINAL_RESULTS.json')
        for n in results}, comparison=sha256(OUT/'FINAL_COMPARISON.json')))
    (OUT / 'RESULTS.md').write_text('\n'.join(['# 双模态特征增强正式结果', '',
        *[f'- [{n}]({n}/RESULTS.md)' for n in results],
        '- [配对消融与目标诊断](ABLATION_COMPARISON.md)', '']), encoding='utf-8')


def main():
    import torch
    from v2xr_radar_ra_rb.preflight import preflight, smoke

    settings = read_json(HERE / 'settings.json')
    print('START ENVIRONMENT AND DATA', flush=True)
    contract = audit()
    if torch.cuda.mem_get_info()[0] < 50 * 2**30:
        raise RuntimeError('Require at least 50 GiB free for batch-8 acceptance')
    preflight(settings, contract)
    smoke(settings['F3'], contract)
    history = DATA / 'logs/v2xr-lr1-v1/LR1'
    if (history / 'FINAL_RESULTS.json').exists():
        raise ValueError('A historical LR1 run is now available. Audit its identity before deciding F0 reuse.')
    reference = dict(status='no_verified_historical_lr1', checked=str(history), action='train F0 from scratch')
    results = {name:run_variant(name, cfg, contract, reference) for name, cfg in settings.items()}
    compare(results, interventions(settings['F3'], results['F3'], contract))
    print('COMPLETE R-A R-B', flush=True)


if __name__ == '__main__':
    with lock(OUT / '.lock'):
        try:
            main()
        except Exception:
            atomic_json(OUT / 'FAILURE.json', dict(status='failed', traceback=traceback.format_exc()))
            raise

