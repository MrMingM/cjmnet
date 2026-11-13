"""Matched-architecture single and conditional two-head quality models."""
from v2xr_griq_experiments.features import BASE, GSPR, RADAR
from v2xr_griq_experiments.training import Scorer, fit, tune


def train(fit_set, dev, smoke):
    result = {'single': fit(fit_set,dev,BASE+GSPR+RADAR,smoke=smoke)}
    for name, fields in [('dual',BASE+GSPR+RADAR),('no_radar',BASE+GSPR),('no_gspr',BASE+RADAR)]:
        result[name] = tune(fit_set,dev,fields,'dual',weights=(.5,1.,2.),smoke=smoke)
    return result


def policies(checkpoints):
    models = {name:Scorer(ck) for name,ck in checkpoints.items()}
    return models,{n:(lambda x,p,m=m:m(x)[0],[{}]) for n,m in models.items()}
