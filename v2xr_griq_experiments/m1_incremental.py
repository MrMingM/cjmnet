"""Independent nested quality models and radar increment ablations."""
import numpy as np

from v2xr_griq_experiments.features import BASE, GSPR, RADAR
from v2xr_griq_experiments.training import Scorer, fit


def train(fit_set, dev, smoke):
    variants = {
        'base': BASE, 'base_radar': BASE+RADAR, 'q0': BASE+GSPR, 'q1': BASE+GSPR+RADAR,
        'shuffled_radar': BASE+GSPR+RADAR, 'additive': BASE+GSPR+RADAR,
        'point_counts': BASE+('radar_count', 'radar_empty'),
        'logistic_q0': BASE+GSPR, 'logistic_q1': BASE+GSPR+RADAR,
    }
    return {name: fit(fit_set,dev,fields,kind='logistic' if name.startswith('logistic') else ('additive' if name=='additive' else 'single'),
                      smoke=smoke,shuffle=name=='shuffled_radar') for name,fields in variants.items()}


def policies(checkpoints):
    models = {name:Scorer(ck) for name,ck in checkpoints.items()}
    def clipped(x, p):
        z0,_ = models['q0'](x)
        z1,_ = models['q1'](x)
        return z0+np.clip(z1-z0,-p['cap'],p['cap'])
    specs = {n:(lambda x,p,m=m: m(x)[0], [{}]) for n,m in models.items()}
    specs['clip'] = (clipped,[dict(cap=c) for c in (.5,1.,2.)])
    return models, specs
