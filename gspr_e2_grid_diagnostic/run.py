"""Read-only fixed-weight E2 destination ablation; no optimizer or training entry."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import types

MODES = ('on', 'off', 'observed_only', 'empty_only')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default='/home/cjm/OpenCOOD-main/cjmnet/gspr_e1e2_v5')
    parser.add_argument('--run-root', default='/home/cjm/datasets/logs/gspr-e1e2-20261101-v5')
    parser.add_argument('--output', default='/home/cjm/datasets/logs/e2-grid-diagnostic-v1')
    parser.add_argument('--smoke', action='store_true')
    opts = parser.parse_args()
    sys.path.insert(0, str(Path(opts.engine).resolve()))
    from auditlib import (DEFAULTS, sources, paths, gpu, server_only, separate,
                          sha256, identity, bound_json, atomic_json, read_json, lock, runtime, frozen_c)
    server_only()
    out = Path(opts.output).resolve()
    old = Path(opts.run_root).resolve()
    separate(out, old)
    args = argparse.Namespace(**DEFAULTS)
    args.output_root = args.experiment_root = str(out)
    args.variant = 'E2'
    paths(args)
    sources()
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / '.lock'):
        try:
            execute(opts, args, out, old)
        except BaseException as e:
            import traceback
            atomic_json(out / 'failure.json', dict(error=repr(e), traceback=traceback.format_exc()))
            raise


def execute(opts, args, out, old):
    from auditlib import sources, sha256, identity, bound_json, atomic_json, read_json, gpu, runtime, frozen_c
    gpu()
    from provenance import verify_inputs
    evidence = verify_inputs(args)
    frozen_c(args)
    runtime(args, evidence['a'])
    import torch
    from torch.utils.data import DataLoader, Subset
    from opencood.hypes_yaml import yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.utils import eval_utils_opv2v as ev
    from models import create, OBSERVATIONS
    from checkpoint import seed, committed
    from data import tensor_digest
    from evaluate import official_match_trace

    selected = read_json(old / 'E2/selected.json')
    tc = read_json(old / 'E2/contract.json')
    if identity({k: v for k, v in tc.items() if k != 'identity'}) != tc['identity']:
        raise ValueError('Original contract identity changed')
    if tc['sources'] != sources() or sha256(old/'E2/contract.json') != selected['training_contract_sha256']:
        raise ValueError('Source or selection contract changed')
    epochs = committed(old/'E2', 20, tc['identity'])
    if len(epochs) != 20:
        raise ValueError('E2 training is incomplete')
    best = min(epochs, key=lambda x: (-x['AP70'], x['epoch']))
    if (best['checkpoint_sha256'] != selected['checkpoint_sha256'] or
            sha256(selected['checkpoint']) != selected['checkpoint_sha256']):
        raise ValueError('Not the original Clean-selected E2 checkpoint')
    indices = [0, 828] if opts.smoke else list(range(829))
    contract = dict(source=sources(), diagnostic_sha256=sha256(Path(__file__)), selected=selected,
                    gt_sha256=sha256(Path(args.gt_audit_root)/'per_frame.jsonl'),
                    input_manifest=evidence['input_manifest'], indices=indices, modes=MODES,
                    mask='original pre-C occupancy; sources unchanged; destination residual masked only',
                    protocols=['O', 'O*', 'F'], smoke=opts.smoke)
    contract['identity'] = identity(contract)
    bound_json(out/'contract.json', contract)
    gt = [json.loads(x) for x in (Path(args.gt_audit_root)/'per_frame.jsonl').read_text().splitlines()]
    h0 = yaml_utils.load_yaml(str(Path(args.runtime_root)/'opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'))
    model = create(h0, 'E2').cuda().eval()
    model.load_state_dict(torch.load(selected['checkpoint'], map_location='cpu', weights_only=False), strict=True)
    before = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
    cache, active = {}, ['on']
    original = model.repe_c.forward

    def choose(x, changed, occupied, mode):
        if mode == 'on': return changed
        if mode == 'off': return x
        return torch.where(occupied if mode == 'observed_only' else ~occupied, changed, x)

    # Actual GPU tensor mask checks, executed only on the server.
    x = torch.randn(2, 3, 4, 5, device='cuda')
    changed = x + torch.randn_like(x)
    occ = torch.rand(2, 1, 4, 5, device='cuda') > .5
    assert torch.equal(choose(x, changed, occ, 'off'), x)
    for mode, mask in [('observed_only', occ), ('empty_only', ~occ)]:
        y = choose(x, changed, occ, mode)
        assert torch.equal(y[mask.expand_as(x)], changed[mask.expand_as(x)])
        assert torch.equal(y[(~mask).expand_as(x)], x[(~mask).expand_as(x)])
    atomic_json(out/'mask_test.json', dict(status='complete', on_off_and_destination_masks_exact=True))

    def replay(self, x, obs, occupied, **kw):
        if not cache:
            y = original(x, obs, occupied, **kw)
            cache.update(x=x.detach().clone(), y=y.detach().clone(), occupied=occupied.clone(),
                         strength=self.last['strength'].clone(), mass=self.last['source_mass'].clone())
        elif not torch.equal(occupied, cache['occupied']):
            raise ValueError('Original occupancy changed across switches')
        self.last = dict(strength=cache['strength'], source_mass=cache['mass'])
        return choose(cache['x'], cache['y'], cache['occupied'], active[0])
    model.repe_c.forward = types.MethodType(replay, model.repe_c)
    all_results, per_target, strata, frame_rows = {}, {}, {}, []
    with torch.inference_mode():
        for weather, suffix in [('clean',''), ('fog','_fog_0.060'), ('snow','_snow_2.5_2.0')]:
            seed(20261016)
            h = copy.deepcopy(h0)
            h.update(validate_dir=evidence['input_manifest']['overlay'], train_sim=False,
                     eval_sim=weather!='clean', sim_weather=suffix)
            dataset = build_dataset(h, visualize=False, train=False)
            keys = []
            for scene in dataset.scenario_database.values():
                ego = next(c for c in scene.values() if c['ego'])
                keys.extend([[Path(a['yaml']).parent.parent.name, t] for t,a in ego.items() if t!='ego'])
            if len(dataset)!=829 or keys!=evidence['b']['selected_frame_keys']:
                raise ValueError('Validation frame identities differ')
            loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0,
                                collate_fn=dataset.collate_batch_test, shuffle=False)
            stats = {m:{p:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.3,.5,.7)}
                        for p in ('O','O*','F')} for m in MODES}
            for index, batch in zip(indices, loader):
                data = train_utils.to_device(batch, torch.device('cuda:0'))
                digest = tensor_digest(data['ego']['processed_lidar'])
                dest = out/'frames'/weather/f'{index:06d}.json'
                if dest.exists():
                    record=read_json(dest)
                    payload={k:v for k,v in record.items() if k!='payload_sha256'}
                    if (record['contract']!=contract['identity'] or record['input_sha256']!=digest or
                            record['payload_sha256']!=identity(payload)):
                        raise ValueError('Cached frame identity/hash changed')
                else:
                    cache.clear()
                    record=dict(contract=contract['identity'], input_sha256=digest,
                                index=index, weather=weather, frame_key=keys[index], predictions={})
                    for mode in MODES:
                        active[0]=mode
                        current=copy.deepcopy(data)
                        pred=model(current['ego'])
                        boxes,scores,_=dataset.post_process(current, {'ego':pred})
                        if boxes is not None and (not torch.isfinite(boxes).all() or not torch.isfinite(scores).all()):
                            raise ValueError('Nonfinite predictions')
                        y=choose(cache['x'], cache['y'], cache['occupied'], mode)
                        delta=(y-cache['x']).square().mean(1,keepdim=True).sqrt()
                        measures={}
                        for name,mask in [('observed',cache['occupied']),('empty',~cache['occupied'])]:
                            measures[name]=dict(cells=int(mask.sum()),nonzero=int((mask&(delta>0)).sum()),
                                                delta_rms_sum=float(delta[mask].sum()))
                        record['predictions'][mode]=dict(boxes=None if boxes is None else boxes.cpu().tolist(),
                            scores=None if scores is None else scores.cpu().tolist(), propagation=measures)
                        OBSERVATIONS.clear()
                    record['payload_sha256']=identity(record)
                    atomic_json(dest,record)
                fixed=gt[index]['conditions']['clean']['final_gt']
                official=gt[index]['conditions'][weather]['final_gt']
                for mode,r in record['predictions'].items():
                    boxes=None if r['boxes'] is None else torch.tensor(r['boxes'],dtype=torch.float32).reshape(-1,8,3)
                    scores=None if r['scores'] is None else torch.tensor(r['scores'],dtype=torch.float32)
                    matches=[]
                    for p in ('O','O*','F'):
                        targets=fixed if p=='F' else official
                        corners=torch.tensor([a['corners'] for a in targets],dtype=torch.float32).reshape(-1,8,3)
                        if p=='O' and boxes is None:continue
                        for t in (.3,.5,.7):
                            if p=='F' and t==.7:
                                matches,_=official_match_trace(boxes,scores,corners,[str(a['object_id']) for a in targets],stats[mode][p],t)
                            else:ev.caluclate_tp_fp(boxes,scores,corners,stats[mode][p],t)
                    matched={a['object_id']:a['iou'] for a in matches}
                    per_target[weather,index,mode]=matched
                    for a in fixed:
                        oid=str(a['object_id']);n=0
                        for cav in gt[index]['conditions'][weather]['cavs']:
                            ids=[str(i) for i in cav['before_point_filter_ids']]
                            if oid in ids:n+=cav['official_points_in_box'][ids.index(oid)]
                        distance=math.hypot(*(sum(c[k] for c in a['corners'])/8 for k in (0,1)))
                        pb='0' if n==0 else '1-5' if n<=5 else '6-20' if n<=20 else '21+'
                        db='0-30' if distance<30 else '30-60' if distance<60 else '60+'
                        for group,label in [('target_points',pb),('distance_m',db),('distance_x_points',db+'|'+pb)]:
                            z=strata.setdefault((weather,mode,group,label),dict(gt=0,tp=0));z['gt']+=1;z['tp']+=oid in matched
                frame_rows.append(record)
                print(weather,index+1,'/829',flush=True)
            all_results[weather]={m:{p:dict(AP={f'ap{int(t*100)}':float(ev.calculate_ap(copy.deepcopy(stats[m][p]),t,False)[0]) for t in (.3,.5,.7)},
                tp=sum(stats[m][p][.7]['tp']),fp=sum(stats[m][p][.7]['fp']),fn=stats[m][p][.7]['gt']-sum(stats[m][p][.7]['tp']),gt=stats[m][p][.7]['gt']) for p in ('O','O*','F')} for m in MODES}
    if any(not torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items()):
        raise ValueError('Weights or BN state changed')
    if sha256(selected['checkpoint'])!=selected['checkpoint_sha256']:
        raise ValueError('Original weight file changed')
    changes=[]
    for w in all_results:
        for i in indices:
            on=per_target[w,i,'on']
            for mode in MODES[1:]:
                other=per_target[w,i,mode]
                changes.append(dict(weather=w,index=i,mode=mode,frame_key=gt[i]['frame_key'],
                    recovered_vs_on=sorted(other.keys()-on.keys()),lost_vs_on=sorted(on.keys()-other.keys()),
                    common_iou_delta={k:other[k]-on[k] for k in other.keys()&on.keys()}))
    # Existing E0 uses a separately trained A1: arithmetic contrasts are descriptive.
    e0=read_json(Path(read_json(old/'baseline_reference.json')['evaluation_root'])/'summary.json')
    contrasts={w:{p:{k:dict(on_minus_off=100*(all_results[w]['on'][p]['AP'][k]-all_results[w]['off'][p]['AP'][k]),
                    off_minus_E0=100*(all_results[w]['off'][p]['AP'][k]-e0['results'][w]['AP'][p][k]))
                  for k in ('ap30','ap50','ap70')} for p in ('O','O*','F')} for w in all_results}
    result=dict(status='complete',contract=contract,results=all_results,contrasts_pp=contrasts,
                strata=[dict(weather=k[0],mode=k[1],group=k[2],bin=k[3],**v) for k,v in strata.items()],
                target_changes=changes,frames=frame_rows,weights_BN_unchanged=True,optimization_steps=0,
                interpretation='on-off is the direct inference contribution conditional on E2-trained weights; off-E0 compares separately trained non-C paths, not a pure causal backbone effect',
                definitions='near=distance<30m; dense=target native points summed across CAVs>=21; sparse=1-5 points; original occupancy is pre-C grid occupancy, not target point count')
    atomic_json(out/'RESULTS.json',result)
    lines=['# E2固定权重网格诊断','','完整829帧/天气' if not opts.smoke else '工程smoke：每天气仅0/828两帧，不是正式成绩','',
           '| 天气 | 模式 | 协议 | AP30% | AP50% | AP70% | TP70 | FP70 | FN70 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for w,ms in all_results.items():
        for m,ps in ms.items():
            for p,a in ps.items():lines.append('| '+w+' | '+m+' | '+p+' | '+' | '.join(f'{100*a["AP"][k]:.6f}' for k in ('ap30','ap50','ap70'))+f" | {a['tp']} | {a['fp']} | {a['fn']} |")
    lines+=['','Snow距离×目标点数召回：','','| 模式 | 距离m及目标点数 | TP/GT | 召回% |','|---|---|---:|---:|']
    for k,v in strata.items():
        if k[0]=='snow' and k[2]=='distance_x_points':lines.append(f'| {k[1]} | {k[3]} | {v["tp"]}/{v["gt"]} | {100*v["tp"]/v["gt"]:.6f} |')
    lines+=['','on=全部；off=全部关闭；observed_only=仅原本有点网格；empty_only=仅空网格。',
            '各模式保持相同原始来源掩码、key/value、相似度及残差门，只屏蔽目标位置的增量。',
            'on−off衡量固定E2权重下推理增强的条件贡献；off−E0包括联合训练造成的整个非C路径变化，不单独归因为主干。',
            '两种目标增强可能经后续网络发生非线性交互，不能假定各部分效果可相加。',
            '完整AP差值、逐目标恢复/丢失、共同IoU差、传播量及分层见RESULTS.json。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
