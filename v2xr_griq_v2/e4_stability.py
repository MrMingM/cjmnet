from v2xr_griq_v2.dataset import SEEDS

VARIANTS = ('additive','cross','joint_local_capacity','independent')


def arms(smoke=False):
    return [(f'{v}-{s}','s128','ordinal',v,'converge',s) for s in (SEEDS[:1] if smoke else SEEDS)
            for v in (('cross',) if smoke else VARIANTS)]


def folds(split):
    # Only rotate the 22 historical fitting scenes. Preserve the five calibration scenes.
    s = split['fit']
    for k in range(3):
        test = s[k::3]
        train = sorted(set(s)-set(test))
        yield dict(fit=train,select=split['select'],test=test,calibrate=split['calibrate'])
