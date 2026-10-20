"""NumPy-only pre-registered A/B/C/D quality diagnosis; no fabricated target or AUROC."""
import numpy as np


def shuffled_features(q, counts, weather, seed):
    result = q.copy(); rng = np.random.RandomState(seed)
    bins = np.floor(np.log2(np.maximum(counts, 1))).astype(int)
    for w in np.unique(weather):
        for b in np.unique(bins[weather == w]):
            ids = np.flatnonzero((weather == w) & (bins == b))
            result[ids] = q[rng.permutation(ids)]
    return result


def ridge(train, target, test):
    # All feature scaling is fitted on train_core; fixed ridge, no validation search.
    mean = train.mean(0); std = np.maximum(train.std(0), 1e-6)
    x = np.c_[np.ones(len(train)), (train-mean)/std]
    y = np.c_[np.ones(len(test)), (test-mean)/std]
    penalty = np.eye(x.shape[1])*1e-3; penalty[0, 0] = 0
    beta = np.linalg.solve(x.T@x + penalty, x.T@target)
    return np.clip(y@beta, 0, 1), dict(coefficients=beta.tolist(), mean=mean.tolist(), std=std.tolist())


def scene_interval(values, scenes, repetitions, seed):
    names = np.unique(scenes)
    if len(names) < 2:
        return None
    means = np.array([np.mean(values[scenes == s]) for s in names])
    r = np.random.RandomState(seed)
    samples = means[r.randint(len(means), size=(repetitions, len(means)))].mean(1)
    return dict(mean=float(means.mean()), lower=float(np.quantile(samples, .025)),
                upper=float(np.quantile(samples, .975)), scenes=len(names), aggregation='equal scene weight')


def contrasts(fit, diagnostic, cfg, seed):
    # Columns: log1p count, density coverage, mean r, low fraction, uncertainty, evidence.
    enough = (len(fit['y']) >= cfg['minimum_labeled_regions_per_split']
              and len(diagnostic['y']) >= cfg['minimum_labeled_regions_per_split']
              and len(np.unique(diagnostic['scene'])) >= cfg['minimum_diagnostic_scenes'])
    if not enough:
        return dict(conclusion='INCONCLUSIVE', reason='Insufficient provenance-bound region labels or diagnostic scenes',
            A=None, B=None, C=None, D=None, fit_regions=len(fit['y']), diagnostic_regions=len(diagnostic['y']))
    f, d = fit['x'], diagnostic['x']
    fs = shuffled_features(f[:, 2:], np.expm1(f[:, 0]), fit['weather'], seed+1)
    ds = shuffled_features(d[:, 2:], np.expm1(d[:, 0]), diagnostic['weather'], seed+2)
    features = dict(A=(f[:, :2], d[:, :2]), B=(f[:, 2:], d[:, 2:]), C=(f, d),
                    D=(np.c_[f[:, :2], fs], np.c_[d[:, :2], ds]))
    report, errors = {}, {}
    for name, (train, test) in features.items():
        prediction, model = ridge(train, fit['y'], test)
        error = (prediction-diagnostic['y'])**2; errors[name] = error
        report[name] = dict(brier=float(error.mean()), model=model,
            scene_interval=scene_interval(error, diagnostic['scene'], cfg['bootstrap_repetitions'], seed),
            features=train.shape[1], fitted_regions=len(train))
    gain = errors['A']-errors['C']; shuffle_gain = errors['A']-errors['D']
    interval = scene_interval(gain, diagnostic['scene'], cfg['bootstrap_repetitions'], seed)
    shuffle_interval = scene_interval(shuffle_gain, diagnostic['scene'], cfg['bootstrap_repetitions'], seed)
    if (interval['lower'] > 0 and interval['mean'] >= cfg['minimum_brier_gain']
            and shuffle_interval['mean'] <= cfg['maximum_shuffle_gain']):
        conclusion = 'SUPPORTED'
    elif interval['upper'] <= 0:
        conclusion = 'NOT_SUPPORTED'
    else:
        conclusion = 'INCONCLUSIVE'
    strata = []
    bins = np.floor(np.log2(np.maximum(np.expm1(d[:, 0]), 1))).astype(int)
    for kind, grouping in (('weather', diagnostic['weather']), ('log2_count_bin', bins), ('scene', diagnostic['scene'])):
        for group in np.unique(grouping):
            mask = grouping == group
            strata.append(dict(kind=kind, group=str(group), n=int(mask.sum()),
                brier={k: float(v[mask].mean()) for k, v in errors.items()},
                gain_interval=scene_interval(gain[mask], diagnostic['scene'][mask], cfg['bootstrap_repetitions'], seed)))
    return dict(conclusion=conclusion, **report, additional_gain=interval, randomized_gain=shuffle_interval,
        strata=strata, target='fraction of reliable simulator-origin points among known labels, squared probability error',
        caveat='scene bootstrap for this quality target only; no detector AP significance claim')


def distribution(x):
    if not len(x):
        return dict(n=0, mean=None, std=None, quantiles=None)
    return dict(n=len(x), mean=np.mean(x, 0).tolist(), std=np.std(x, 0).tolist(),
                quantiles=np.quantile(x, [.05, .5, .95], axis=0).tolist())
