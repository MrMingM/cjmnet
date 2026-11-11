"""Aggregate full-frame predictions and paired fixed-GT outcome changes."""
from collections import Counter, defaultdict
import copy
import json
import numpy as np
from diag_bootstrap import OUT,WEATHERS,atomic_json,identity,read_json,sha256
from diag_eval import finish,merge,score
from evaluate import statistics
from support import atomic_jsonl


def relative(triple):
    return (triple[0]/triple[1])**.5 if triple[1]>1e-18 else None


def distribution(values):
    if not values:
        return dict(n=0)
    q=np.quantile(values,[.1,.25,.5,.75,.9]).tolist()
    return dict(n=len(values),p10=q[0],p25=q[1],median=q[2],p75=q[3],p90=q[4])


def bootstrap(values):
    data=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261111)
    draws=rng.choice(data,(2000,len(data)),replace=True).sum(1)
    return [float(x) for x in np.quantile(draws,[.025,.975])]


def add_changes(count,before,after):
    count['kept']+=before and after
    count['recovered']+=not before and after
    count['lost']+=before and not after
    count['missed_both']+=not before and not after


def score_frame(data,weather):
    same={}
    for name,prediction in data['prediction'].items():
        key=identity(prediction)
        if key not in same:
            same[key]=score(prediction,data['official_gt'],data['fixed_gt'],weather=='clean')
        yield name,same[key]
    for name,original in data['aliases'].items():
        yield name,same[identity(data['prediction'][original])]


def summarize(group,run_id):
    root=OUT/group
    final_path=root/'FINAL_RESULTS.json'
    if final_path.is_file():
        result=read_json(final_path)
        if (result['status']!='complete' or result['contract']!=run_id or
                any(sha256(root/p)!=h for p,h in result['artifacts'].items())):
            raise ValueError('Completed result evidence changed')
        return
    summary,artifacts={},[]
    for weather in WEATHERS:
        totals=defaultdict(lambda:{p:statistics() for p in ('O','O*','F')})
        frames_per_mode=Counter()
        changes=defaultdict(lambda:defaultdict(Counter))
        target_records=[]
        regions=defaultdict(list)
        target_stats=defaultdict(list)
        cohort_controls=defaultdict(Counter)
        changes_per_frame=defaultdict(list)
        diagnostic=Counter()
        counterexamples=defaultdict(dict)
        full_ids=0
        for index in range(829):
            stored=read_json(root/'frames'/weather/f'{index:06d}.json')
            data=stored['data']
            if stored['contract']!=run_id or stored['sha256']!=identity(data):
                raise ValueError('Frame data hash differs during evaluation')
            all_targets=set(data['targets'])
            full_ids+=len(all_targets)
            hits={}
            for mode,(stats,ids) in score_frame(data,weather):
                hits[mode]=set(ids)
                if not (group=='D1' and mode.rsplit('_',1)[-1] in ('s0','s1','s2')):
                    merge(totals[mode],stats)
                    frames_per_mode[mode]+=1
            base=hits['base']
            diagnostic['frames']+=1
            diagnostic['single_vehicle_frames']+=data['vehicle_count']==1
            if group=='D2':
                diagnostic['layer_single_nonclose_frames']+=not data['checks']['layer_single_close']
                diagnostic['layer_single_max_head_error']=max(diagnostic['layer_single_max_head_error'],
                    data['checks']['layer_single_vs_true_single_max'])
                diagnostic['layer_single_target_disagreement']+=len(hits['single_all']^hits['true_single'])
            for scale,masks in data['spatial'].items():
                for region,measures in masks.items():
                    for name,triple in measures.items():
                        value=relative(triple)
                        if value is not None:
                            regions[f'{scale}/{region}/{name}'].append(value)
                            kind='single' if data['vehicle_count']==1 else 'multi'
                            regions[f'{scale}/{kind}/{region}/{name}'].append(value)
                        if group=='D1' and name=='combined':
                            before=relative(measures['neighbor_input'])
                            if before and value is not None:
                                regions[f'{scale}/{region}/relative_gain'].append(value/before)
            for name,ids in hits.items():
                if name in totals:
                    changes_per_frame[name].append(len(ids)-len(base))
            for oid,obj in data['targets'].items():
                row=dict(index=index,frame=data['frame'],object_id=oid,**obj,
                         detected={name:oid in ids for name,ids in hits.items()})
                target_records.append(row)
                original=oid in base
                cohort=obj['cohort']
                labels=('all', 'cohort' if cohort else 'normal_detected' if original else 'other_missed',
                        'distance/'+obj['distance'])
                for name,ids in hits.items():
                    for label in labels:
                        add_changes(changes[name][label],original,oid in ids)
                if group=='D3':
                    for scope in ('ego','neighbor','both'):
                        score_dd=(int(row['detected'][scope+'_r1'])-int(original)
                                  -int(row['detected'][scope+'_r0'])+int(row['detected']['weather_r0']))
                        for label in labels:
                            controls=cohort_controls[f'{scope}/{label}']
                            controls['positive']+=score_dd>0
                            controls['negative']+=score_dd<0
                            controls['zero']+=score_dd==0
                            controls['net']+=score_dd
                if group=='D2':
                    for family in ('single','context'):
                        all_outcome=row['detected'][family+'_all']
                        scale_deltas=[int(row['detected'][f'{family}_s{i}'])-int(original) for i in range(3)]
                        for label in labels:
                            counts=cohort_controls[f'{family}/{label}']
                            counts['all_only_recovered']+=all_outcome and not original and not any(x>0 for x in scale_deltas)
                            counts['all_only_lost']+=not all_outcome and original and not any(x<0 for x in scale_deltas)
                            counts['nonadditive_outcome']+=int(all_outcome)-int(original)!=sum(scale_deltas)
                comparison={'D1':('combined','combined'),'D2':('context_all','context_fused'),
                            'D3':('both_r1','both_post_interaction')}[group]
                changed=original!=row['detected'][comparison[0]]
                for scale,measurements in obj['scales'].items():
                    for name,triple in measurements.items():
                        value=relative(triple)
                        if value is not None:
                            for label in labels:
                                target_stats[f'{scale}/{label}/{name}'].append(value)
                                if cohort:
                                    target_stats[f'{scale}/cohort_{"changed" if changed else "same"}/{name}'].append(value)
                            if name==comparison[1]:
                                bucket=f'{scale}/{"small_changed" if changed else "large_unchanged"}'
                                score_key=-value if changed else value
                                previous=counterexamples.get(bucket)
                                if not previous or score_key>previous['rank']:
                                    counterexamples[bucket]=dict(frame=data['frame'],object_id=oid,
                                        relative_change=value,rank=score_key,detected=row['detected'])
                    if group=='D1':
                        inp=relative(measurements['neighbor_input'])
                        post=relative(measurements['combined'])
                        if inp and post is not None:
                            for label in labels:
                                target_stats[f'{scale}/{label}/relative_gain'].append(post/inp)
                            if cohort:
                                target_stats[f'{scale}/cohort_{"changed" if changed else "same"}/relative_gain'].append(post/inp)
            if (index+1)%100==0:
                print('SCORE',group,weather,index+1,flush=True)
        if full_ids!=6235 or set(frames_per_mode.values())!={829}:
            raise ValueError('Incomplete fixed evaluation')
        atomic_jsonl(root/f'TARGETS_{weather}.jsonl',target_records)
        artifacts.append(root/f'TARGETS_{weather}.jsonl')
        report=dict(AP={m:finish(v) for m,v in totals.items()}, changes=changes,
                    full_mode_frames=frames_per_mode, controls=cohort_controls,checks=diagnostic,
                    target_feature_quantiles={k:distribution(v) for k,v in target_stats.items()},
                    spatial_quantiles={k:distribution(v) for k,v in regions.items()},
                    net_change_ci95={k:bootstrap(v) for k,v in changes_per_frame.items()},examples=counterexamples)
        summary[weather]=report
        print('SCORE COMPLETE',group,weather,flush=True)
    expected={'fog':103,'snow':224}
    for w,n in expected.items():
        if sum(summary[w]['changes']['base']['cohort'].values())!=n:
            raise ValueError(f'Historical cohort count differs for {w}')
    from diag_report import write_group
    report_path=root/'MECHANISM_ANALYSIS.md'
    finding=write_group(group,summary,report_path)
    artifacts.append(report_path)
    atomic_json(final_path,dict(status='complete',contract=run_id,group=group,
        device=read_json(root/'PREFLIGHT.json')['device'],frames=829,
        target_counts=expected,finding=finding,weather=summary,
        artifacts={p.relative_to(root).as_posix():sha256(p) for p in artifacts}))
