"""Independent rank models with scene-fold radar increment consistency."""
import numpy as np

from v2xr_griq_experiments.features import BASE, GSPR, RADAR
from v2xr_griq_experiments.training import Scorer, fit, tune


def train(fit_set, dev, smoke):
    full = BASE+GSPR+RADAR
    result = dict(
        q0=fit(fit_set,dev,BASE+GSPR,smoke=smoke),
        bce=fit(fit_set,dev,full,smoke=smoke),
        random=tune(fit_set,dev,full,pair='random',weights=(.1,.3,1.),smoke=smoke),
        hard=tune(fit_set,dev,full,pair='hard',weights=(.1,.3,1.),smoke=smoke),
        no_gspr=tune(fit_set,dev,BASE+RADAR,pair='hard',weights=(.1,.3,1.),smoke=smoke),
    )
    scenes = sorted({r['meta']['key'][0] for r in fit_set})
    for k in range(3):
        held = set(scenes[k::3])
        subset = fit_set if smoke else [r for r in fit_set if r['meta']['key'][0] not in held]
        result[f'fold{k}_base'] = fit(subset,dev,BASE+GSPR,smoke=smoke)
        result[f'fold{k}_radar'] = fit(subset,dev,full,pair='hard',weight=result['hard']['weight'],smoke=smoke)
        for kind in ('base','radar'):
            result[f'fold{k}_{kind}']['fit_scenes'] = sorted({r['meta']['key'][0] for r in subset})
    return result


def gate_details(x, models, cfg):
    z0 = models['q0'](x)[0]
    z1 = models['hard'](x)[0]
    delta = z1-z0
    fold = np.stack([models[f'fold{i}_radar'](x)[0]-models[f'fold{i}_base'](x)[0] for i in range(3)])
    votes = (np.sign(fold) == np.sign(delta)).sum(0)
    gate = (votes >= cfg['votes']) & (np.abs(delta) >= cfg['minimum'])
    return z0, delta, gate


def policies(checkpoints):
    models = {name:Scorer(ck) for name,ck in checkpoints.items()}
    def clip(x,p):
        z0,z1 = models['q0'](x)[0],models['hard'](x)[0]
        return z0 + np.clip(z1-z0,-p['cap'],p['cap'])
    def gated(x,p):
        z0,d,g = gate_details(x,models,p)
        base = x[:,0] if p.get('fallback_f0') else z0
        return base+g*np.clip(d,-p['cap'],p['cap'])
    def bce_clip(x,p):
        z0,z1 = models['q0'](x)[0],models['bce'](x)[0]
        return z0 + np.clip(z1-z0,-p['cap'],p['cap'])
    specs = {n:(lambda x,p,m=models[n]: m(x)[0], [{}]) for n in ('q0','bce','random','hard','no_gspr')}
    specs['clip'] = (clip,[dict(cap=c) for c in (.5,1.,2.)])
    specs['bce_clip'] = (bce_clip,[dict(cap=c) for c in (.5,1.,2.)])
    specs['gated'] = (gated,[dict(cap=c,minimum=m,votes=v) for c,m,v in ((.5,.1,3),(1.,.25,3),(2.,.5,2))])
    specs['gated_f0_fallback'] = (gated,[dict(cap=c,minimum=m,votes=v,fallback_f0=True)
                                      for c,m,v in ((.5,.1,3),(1.,.25,3),(2.,.5,2))])
    return models,specs
