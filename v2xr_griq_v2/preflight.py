"""Freeze the official F0 identity, prior cache and v2 source versions."""
import shutil
from pathlib import Path
from v2xr_griq_v2.dataset import DATA,F0,HERE,OUT,REPO,V1,WEIGHT,bind,read,sha
from v2xr_griq_experiments.features import FEATURES


def main():
    selected = read(F0/'selected.json')
    if selected['epoch']!=20 or selected['sha256']!=WEIGHT or sha(selected['checkpoint'])!=WEIGHT:
        raise ValueError('Formal F0 weight differs')
    old = read(V1/'CONTRACT.json')
    if old['weight']!=WEIGHT or old['config']!=sha(F0/'config.yaml'):
        raise ValueError('Historical F0 contract differs')
    audit = read(V1/'DATA_AUDIT.json')
    previous = F0.parent/'CONTRACT.json'
    pinned = read(previous)
    for name,h in pinned['frozen'].items():
        if sha(REPO/name)!=h:
            raise ValueError(f'Frozen F0 dependency differs: {name}')
    for name,h in pinned['source'].items():
        if sha(REPO/'v2xr_radar_ra_rb'/name)!=h:
            raise ValueError(f'Frozen F0 source differs: {name}')
    for name,h in old['inputs'].items():
        if sha(name)!=h:
            raise ValueError(f'Frozen input differs: {name}')
    complete = read(V1/'COMPLETE.json')
    if sha(V1/'CONTRACT.json')!=audit['contract'] or sha(V1/'DATA_AUDIT.json')!=complete['data']:
        raise ValueError('Historical manifests changed')
    if tuple(audit['feature_schema'])!=FEATURES:
        raise ValueError('Unexpected global feature schema')
    if shutil.disk_usage(OUT).free<10*2**30:
        raise ValueError('Need 10 GiB free for candidate sidecars and results')
    bind(OUT/'CONTRACT.json',dict(weight=WEIGHT,config=old['config'],prior=sha(V1/'DATA_AUDIT.json'),
         previous=sha(previous),source={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
         if p.suffix in ('.py','.sh')},frozen_inputs=old['inputs'],
         loss_weights=[[1,1,1],[1,1,2],[2,2,1]],thresholds=[.05,.10,.15,.20,.30,.45,.60,.75],
         extraction='per-frame RNG from weather and original index; archived global values are read-only',
         selection='min mean marginal BCE; E1-3 weight chosen by selection scene MacroAP',
         calibration='Pareto first within 0.5 pp per metric of F0 if feasible; then MacroAP, least TP loss by tie',
         sampling='same split; fixed update budget and convergence for E2; same convergence for E1/E3/E4'))


if __name__=='__main__':
    main()
