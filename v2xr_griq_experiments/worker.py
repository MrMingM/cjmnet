"""Train one independent experiment, lock policies, then score held-out scenes."""
import gzip
import importlib
import json
import sys

import numpy as np
from scipy.special import expit
import torch

from v2xr_griq_experiments.dataset import F0, OUT, WEATHERS, bind, frame_rows, import_runtime, load, manifest, read, save, sha, smoke_root
from v2xr_griq_experiments.evaluate import frame_result
from v2xr_griq_experiments.features import FEATURES
from v2xr_griq_experiments.metrics import binary, combine, scene_delta, target_changes, equal_fp_frontier
from v2xr_griq_experiments.native import processor
from v2xr_griq_experiments.training import prepare

GROUPS = {'M1': ('m1_incremental','q1'), 'M2': ('m2_quality','dual'), 'M3': ('m3_ranking','gated')}
THRESHOLDS = (.10,.20,.30,.45,.60,.75)
INTERVAL_ARMS = {'F0','score_threshold','q0','q1','clip','logistic_q1','base_radar','point_counts',
                 'dual','single','bce','random','hard','gated','gated_f0_fallback','additive'}


def cut_trace(stats, threshold):
    row = stats[.7]
    n = sum(s > np.float32(threshold) for s in row['score'])
    return {t:({k:(v if k=='gt' else v[:n]) for k,v in r.items()} if t==.7 else r)
            for t,r in stats.items()}


def calibrate(frames, specs, pp, smoke):
    cache = {}
    base = [frame_result(d, pp, d['logit'], d['native_threshold'].item(), False, False) for d in frames]
    target_fp = combine([r['F'] for r in base])['0.7']['fp']
    for name,(fn,options) in specs.items():
        attempts = []
        for opt in options:
            z = [fn(d['x'],opt) for d in frames]
            traces = [frame_result(d,pp,s,min(THRESHOLDS),False,False)['F'] for d,s in zip(frames,z)]
            if smoke:
                for d,s,r in zip(frames,z,traces):
                    if cut_trace(r,.45)[.7] != frame_result(d,pp,s,.45,False,False)['F'][.7]:
                        raise ValueError('Calibration prefix and native threshold replay differ')
            # With fixed scores and greedy NMS, raising the cutoff removes a suffix.
            # Validate against direct native replays on smoke below.
            for prob in THRESHOLDS:
                rows = [cut_trace(r,prob) for r in traces]
                metric = combine(rows)['0.7']
                attempts.append(dict(options=opt,prob=prob,**metric))
        best = max(attempts,key=lambda r:(r['ap'] if r['ap'] is not None else -1,
                                         -abs(r['fp']-target_fp), r['prob']))
        budget = max(attempts,key=lambda r:(r['fp']<=target_fp, r['tp'] if r['fp']<=target_fp else -r['fp'],
                                            r['ap'] if r['ap'] is not None else -1))
        cache[name] = dict(primary=best,budget=budget,trials=attempts)
    return dict(native_fp_budget=target_fp, arms=cache)


def candidate_summary(frames, models, group, gate, calibration):
    x = np.concatenate([d['x'] for d in frames])
    iou = np.concatenate([d['iou'] for d in frames])
    y = iou >= .7
    score = x[:, FEATURES.index('score')]
    r = x[:, FEATURES.index('r_mean')]
    radar = x[:, FEATURES.index('radar_count')] > 0
    points = x[:, FEATURES.index('lidar_count')]
    distance = x[:, FEATURES.index('distance')]
    near_threshold = (score < .3) & (score > .03)
    masks = {'all': np.ones(len(y),bool), 'low': score<=.2, 'hard':near_threshold,
             'radar_sparse_lidar':radar & (points<=5), 'near':distance<30,
             'mid':(distance>=30)&(distance<60),'far':distance>=60,
             'lidar_absent':points==0,'lidar_sparse':(points>0)&(points<=5),
             'lidar_observed':points>5}
    results = {'F0': {k: binary(y[m], score[m]) for k,m in masks.items()}}
    for name,m in models.items():
        if name.startswith('fold'):
            continue
        z, heads = m(x)
        p = expit(z)
        results[name] = {k:binary(y[v], p[v]) for k,v in masks.items()}
        if heads is not None:
            obj = iou >= .3
            results[name].update(wide=binary(obj,heads[:,0]), strict=binary(y[obj],heads[obj,1]),
                high_but_imprecise=int(((p>.5)&obj&~y).sum()),
                high_but_imprecise_at_calibrated_threshold=int(((p>calibration['arms'][name]['primary']['prob'])&obj&~y).sum()),
                low_but_precise_above_calibrated_threshold=int(((score<=.2)&(p>calibration['arms'][name]['primary']['prob'])&y).sum()))
    if group in ('M1','M3'):
        z0=models['q0'](x)[0]
        d=models['q1' if group=='M1' else 'hard'](x)[0]-z0
        ok = np.ones(len(d),bool)
        if gate is not None:
            _,_,ok = gate(x,models,calibration['arms']['gated']['primary']['options'])
        masks.update(gspr_lower=np.isfinite(r)&(r<.75),gspr_higher=np.isfinite(r)&(r>=.75),
                     gspr_unknown=~np.isfinite(r))
        results['increment'] = {k:dict(n=int(v.sum()), applied=int((ok&v).sum()),
                mean=float(d[v].mean()), positive=int((v&(d>0)).sum()),
                positive_and_y70=int((v&(d>0)&y).sum()), negative_and_y70=int((v&(d<0)&y).sum()),
                applied_positive_and_y70=int((v&ok&(d>0)&y).sum()),
                applied_negative_and_y70=int((v&ok&(d<0)&y).sum())) for k,v in masks.items() if v.any()}
    return results


def official_archive_check(stats, weather):
    expected = read(F0/'final_evaluation/summary.json')['results'][weather]
    for proto in ('O','O*','F'):
        ap = expected['AP'][proto]
        for t in ('0.3','0.5','0.7'):
            observed = stats[proto][t]
            target = ap['ap'+str(int(100*float(t)))]
            if abs(observed['ap']-target) > .001:
                raise ValueError(f'Official archive AP differs {weather}/{proto}/{t}: {observed["ap"]} vs {target}')
            counts = expected['matches'][proto][t]
            if any(observed[k] != counts[k] for k in ('tp','fp','fn','gt')):
                raise ValueError(f'Official archive matching counts differ {weather}/{proto}/{t}')


def evaluate(frames, specs, calibration, pp, smoke):
    def native(x,p): return x[:,0]
    all_specs = dict(F0=(native,[{}]),**specs)
    base = None
    results, evidence = {}, {}
    for name, (fn,_) in all_specs.items():
        if name=='F0':
            policy = dict(primary=dict(options={},prob=.2),budget=dict(options={},prob=.2))
        else:
            policy = calibration['arms'][name]
        rows_by_mode = {}
        for mode in ('primary','budget'):
            opt, prob = policy[mode]['options'], policy[mode]['prob']
            if mode == 'budget' and policy[mode] == policy['primary']:
                rows = rows_by_mode['primary']
            else:
                rows = [frame_result(d,pp,fn(d['x'],opt),prob, mode=='primary')
                        for d in frames]
            rows_by_mode[mode] = rows
        rows = rows_by_mode['primary']
        if base is None:
            base = rows
        result = {p:combine([r[p] for r in rows]) for p in ('F','O','O*')}
        result.update(none=sum(r['none'] for r in rows),empty=sum(r['empty'] for r in rows),
                      global_ranked_F=combine([r['F'] for r in rows],True),
                      targets=target_changes(rows,base),
                      budget=dict(F=combine([r['F'] for r in rows_by_mode['budget']]),
                                  targets=target_changes(rows_by_mode['budget'],base)))
        if name in INTERVAL_ARMS:
            low_cut_rows = rows if policy['primary']['prob'] == min(THRESHOLDS) else [
                frame_result(d,pp,fn(d['x'],policy['primary']['options']),min(THRESHOLDS),False)
                for d in frames]
            result['equal_fp_offline'] = equal_fp_frontier(low_cut_rows,base)
        if name != 'F0' and not smoke and name in INTERVAL_ARMS:
            result['paired'] = scene_delta(rows,base)
        # Store exact per-frame records for main mechanistic arms and all target outcomes.
        if name in INTERVAL_ARMS:
            evidence[name] = [{k:v for k,v in r.items() if k!='observations'} for r in rows]
        results[name] = result
    return results,evidence


def write_report(path, group, primary, result):
    lines = [f'# {group} 候选质量实验','',
        f'主对照为 {primary}。候选池、F0 权重及验证身份由 DATA_AUDIT.json 和 CONTRACT.json 固定。',
        'F 使用固定 Clean 真值，O 按原规则跳过 None，O* 计入 None。原协议按帧顺序累计 AP。', '',
        '| 天气 | 方法 | F AP30 | F AP50 | F AP70 | O AP70 | TP / FP / FN | 恢复 / 丢失 |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for w,rows in result['weather'].items():
        for name,r in rows.items():
            f = r['F']
            a = f['0.7']
            hits = r['targets'].get('all',{})
            lines.append(f'| {w} | {name} | {f["0.3"]["ap"]:.6f} | {f["0.5"]["ap"]:.6f} | {a["ap"]:.6f} | '
                         f'{format(r["O"]["0.7"]["ap"], ".6f") if r["O"]["0.7"]["ap"] is not None else "—"} | {a["tp"]} / {a["fp"]} / {a["fn"]} | '
                         f'{hits.get("recovered",0)} / {hits.get("lost",0)} |')
    lines += ['', '候选预测、概率校准、场景配对区间及预算对照见 metrics.json。']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(group, smoke):
    root = smoke_root() if smoke else OUT
    out = root/group
    out.mkdir(parents=True, exist_ok=True)
    index = manifest(root) if not smoke else read(root/'DATA_AUDIT.json')
    source, primary = GROUPS[group]
    code = importlib.import_module('v2xr_griq_experiments.'+source)
    import_runtime()
    pp = processor(F0/'config.yaml')
    torch.set_num_threads(1)
    checkpoint = out/'best.pt'
    signature = sha(root/'DATA_AUDIT.json')
    data = prepare(index)
    if checkpoint.exists():
        stored = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if stored['data'] != signature or stored['group'] != group:
            raise ValueError('Checkpoint and frozen cache differ')
        fitted = stored['fitted']
    else:
        fitted = code.train(data['fit'],data['select'],smoke)
        tmp = checkpoint.with_suffix('.tmp')
        torch.save(dict(data=signature,group=group,fitted=fitted),tmp)
        tmp.replace(checkpoint)
        print('TRAINED',group,len(fitted),flush=True)
    models,specs = code.policies(fitted)
    specs['score_threshold'] = (lambda x,p:x[:,0],[{}])
    calibration_path = out/'CALIBRATION.json'
    if not calibration_path.exists():
        bind(calibration_path, calibrate(data['calibrate'], specs, pp, smoke))
    calibration = read(calibration_path)
    del data
    print('LOCKED',group,sha(calibration_path),flush=True)
    result = dict(group=group,primary=primary,data=signature,weather={},candidates={},
                  training={k:{a:b for a,b in v.items() if a!='state'} for k,v in fitted.items()},
                  calibration=calibration)
    for weather in WEATHERS:
        complete = out/f'{weather}.json'
        sidecar = out/f'{weather}_frames.json.gz'
        if not complete.exists():
            frames = [load(row,index['policy']) for row in frame_rows(index,'validate') if row['weather']==weather]
            rows,evidence = evaluate(frames,specs,calibration,pp,smoke)
            if not smoke:
                official_archive_check(rows['F0'],weather)
            gate = getattr(code,'gate_details',None)
            candidates = candidate_summary(frames,models,group,gate,calibration)
            tmp = sidecar.with_suffix('.tmp')
            with gzip.open(tmp,'wt',encoding='utf-8') as f:
                json.dump(evidence,f,allow_nan=False,separators=(',',':'))
            tmp.replace(sidecar)
            bind(complete,dict(rows=rows,candidates=candidates,sidecar=sha(sidecar)))
        saved = read(complete)
        if saved['sidecar'] != sha(sidecar):
            raise ValueError('Evaluation frames differ from completion record')
        result['weather'][weather],result['candidates'][weather] = saved['rows'],saved['candidates']
        print('EVALUATED',group,weather,flush=True)
    save(out/'metrics.json',result)
    write_report(out/'RESULTS.md',group,primary,result)
    artifacts = [p for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json' and p.suffix!='.tmp']
    bind(out/'COMPLETE.json',dict(data=signature,artifacts={p.name:sha(p) for p in artifacts}))
    print('COMPLETE',group,flush=True)


if __name__=='__main__':
    main(sys.argv[1],'--smoke' in sys.argv)
