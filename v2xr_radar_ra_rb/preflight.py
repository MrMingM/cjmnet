"""Fail before formal training unless synthetic and real-data checks pass."""
from pathlib import Path
import time

import numpy as np
import torch
from common import atomic_json, read_json, sha256
from checkpoint import rng, restore, save, seed, state
from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from v2xr_weather_l012.data import tensor_digest
from v2xr_lr1.lr1_model import LR1Model
from empty_input import preserve_empty_vehicle_slots
from enhancement import RadarSupport, RadarCompensation
from pairing import PairedDataset, targets
from __main__ import CONFIG, OUT, build, loader, make_model, components, update, bind, engine


def tests():
    coords = torch.tensor([[0,0,1,1], [0,0,1,2], [1,0,1,1], [1,0,7,7]], device='cuda')
    xyz = torch.zeros(4, 32, 4, device='cuda')
    xyz[:, 0, :2] = coords[:, 2:].flip(-1)*.4
    batch = dict(voxel_coords=coords, voxel_features=xyz,
                 voxel_num_points=torch.ones(4, device='cuda', dtype=torch.long))
    a = RadarSupport([.4,.4,4], (8,8)).cuda()
    f = torch.randn(4,64,device='cuda')
    g, report = a(f, batch)
    assert torch.equal(f, g)
    assert report['neighbors'].tolist() == [1,1,0,0]
    with torch.no_grad():
        a.project.weight.copy_(torch.eye(64, device='cuda'))
    g, _ = a(f, batch)
    assert torch.equal(g[2:], f[2:]) and not torch.equal(g[:2], f[:2])
    assert torch.equal(a(f, batch, False)[0], f)
    assert a(f[:0], {k:v[:0] for k,v in batch.items()})[0].shape == (0,64)
    x = np.array([[0,0.],[4,.2],[4,.2],[4,.2]])
    y = np.array([[0,0.],[0,0.],[2,.1],[4,.2]])
    got = targets(x, y, True)
    assert np.array_equal(got[:,1], [0,1,1,1]) and np.allclose(got[:,0], [0,1,.5,0])
    assert np.array_equal(got[:,2], [0,0,0,1])
    normal = RadarCompensation().cuda()
    ordinary = RadarCompensation(ordinary=True).cuda()
    assert sum(p.numel() for p in normal.parameters()) == sum(p.numel() for p in ordinary.parameters())
    with torch.no_grad():
        normal.project.weight.copy_(torch.eye(64, device='cuda'))
    delta, _ = normal(f, f, f[:, :10], f.new_zeros(4))
    assert not torch.count_nonzero(delta)


def preflight(settings, contract):
    path = OUT / 'PREFLIGHT.json'
    if path.is_file():
        marker = read_json(path)
        if marker['contract'] != contract or marker['status'] != 'complete':
            raise ValueError('Preflight contract changed')
        return
    seed(20261017)
    tests()
    dataset, cfg = build(CONFIG, condition='clean')
    batch = next(iter(loader(dataset, evaluation=True, limit=1)))
    ego = train_utils.to_device(batch['ego'], torch.device('cuda:0'))
    seed(20261017)
    reference = LR1Model(cfg['model']['args'], 'LR1').cuda().eval()
    preserve_empty_vehicle_slots(reference)
    ref = {k:tensor_digest(v) for k,v in reference.state_dict().items()}
    ref_rng = tensor_digest(rng())
    with torch.no_grad():
        baseline = reference(ego)
    del reference
    initial, previous = {}, {}
    for name, group in settings.items():
        seed(20261017)
        model = make_model(cfg, group).eval()
        shared = {k:tensor_digest(v) for k,v in model.state_dict().items()}
        if any(shared[k] != v for k,v in ref.items()) or tensor_digest(rng()) != ref_rng:
            raise ValueError(f'{name} changed LR1 shared tensors or data RNG')
        if name == 'F2':
            previous = shared
        if name == 'F3' and shared != previous:
            raise ValueError('F2/F3 initialization differs')
        initial[name] = dict(lr1_shared=tensor_digest(ref), all_tensors=tensor_digest(shared), rng=ref_rng,
            total=sum(p.numel() for p in model.parameters()),
            ra=sum(p.numel() for p in model.ra.parameters()) if model.ra else 0,
            rb=sum(p.numel() for p in model.rb.parameters()) if model.rb else 0)
        with torch.no_grad():
            for enabled in (True, False):
                model.ra_on, model.rb_on = enabled and group['ra'], enabled and group['rb']
                pred = model(ego)
                if any(not torch.equal(pred[k], baseline[k]) for k in baseline):
                    raise ValueError(f'{name} identity/disabled forward differs from native LR1')
            if name == 'F3':
                model.ra_on = model.rb_on = True
                for sensors in (('lidar',), ('radar',), ('lidar','radar')):
                    test = dict(ego)
                    for sensor in sensors:
                        name_key = f'processed_{sensor}'
                        test[name_key] = {k:v[:0] for k,v in ego[name_key].items()}
                    result = model(test)
                    if any(not torch.isfinite(v).all() for v in result.values()):
                        raise ValueError(f'Nonfinite empty modality {sensors}')
                    if 'radar' in sensors and model.aux is not None:
                        raise ValueError('Empty radar created compensation')
        print('IDENTITY', name, initial[name], flush=True)
        del model
    torch.cuda.empty_cache()
    atomic_json(path, dict(status='complete', contract=contract, initial=initial))


def smoke(settings, contract):
    path = OUT / 'SMOKE.json'
    smoke_root = OUT / 'smoke'
    if path.is_file():
        report = read_json(path)
        if report['contract'] != contract or any(sha256(smoke_root / k) != v for k,v in report['files'].items()):
            raise ValueError('Smoke evidence differs')
        return
    seed(20261017)
    native, cfg = build(CONFIG, split='train')
    paired_native, _ = build(CONFIG, split='train')
    nx = cfg['model']['args']['point_pillar_scatter']['grid_size'][0]
    paired = PairedDataset(paired_native, int(nx))
    seed(20261017)
    first = native[4]
    expected_rng = tensor_digest(rng())
    seed(20261017)
    second = paired[4]
    if (tensor_digest(first['ego']) != tensor_digest(second['ego'])
            or expected_rng != tensor_digest(rng())):
        raise ValueError('Paired read changed the native sample, weather RNG, or geometry')
    print('PAIR IDENTITY AND RNG PASSED', flush=True)
    seed(20261017)
    model, loss, optim, scheduler = components(cfg, 2, settings)
    weights, resume = smoke_root / 'weights.pth', smoke_root / 'resume.pth'
    marker = smoke_root / 'steps.json'
    if marker.is_file():
        record = read_json(marker)
        if record['contract'] != contract or any(sha256(smoke_root / k) != v for k,v in record['files'].items()):
            raise ValueError('Two-step checkpoint changed')
        restore(resume, model, optim, scheduler, loss, contract, 1)
    else:
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for i, batch in enumerate(loader(paired, training=True, limit=16), 1):
            started = time.monotonic()
            value, _ = update(model, loss, optim, batch, 0, inspect=True)
            if model.extra_used != {'ra','rb'} or loss.last_terms.get('supervised', 0) <= 0:
                raise ValueError('Smoke did not exercise both modules and F3 supervision')
            row = dict(loss=value, seconds=time.monotonic()-started, terms=dict(loss.last_terms))
            rows.append(row)
            print('SMOKE UPDATE', i, row, flush=True)
        if len(rows) != 2:
            raise ValueError('Expected exactly two real batch-8 smoke steps')
        scheduler.step(0)
        save(weights, model.state_dict())
        save(resume, state(model, optim, scheduler, loss, 1, contract))
        expected = tensor_digest([model.state_dict(), optim.state_dict(), scheduler.state_dict(), rng()])
        seed(20261018)
        restore(resume, model, optim, scheduler, loss, contract, 1)
        if expected != tensor_digest([model.state_dict(), optim.state_dict(), scheduler.state_dict(), rng()]):
            raise ValueError('Complete checkpoint restore failed')
        record = dict(status='complete', contract=contract, steps=rows,
            peak_allocated=torch.cuda.max_memory_allocated(),
            files={p.name:sha256(p) for p in (weights, resume)})
        bind(marker, record)
    optim.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    # Strict eval mode accepts only current weather and native same-frame radar.
    evaluation = engine.evaluate(model, loss, CONFIG, smoke_root/'evaluation',
                                 ('clean','fog','snow'), 12, contract, sha256(weights), loss=True)
    if loss.pair is not None or evaluation['frames'] != 12:
        raise ValueError('Unexpected smoke evaluation context')
    atomic_json(path, dict(status='complete', contract=contract, steps=record['steps'],
        peak_allocated=record['peak_allocated'], gpu=evaluation['performance'],
        files={**record['files'], 'evaluation/summary.json':sha256(smoke_root/'evaluation/summary.json')}))
    print('SMOKE COMPLETE', flush=True)

