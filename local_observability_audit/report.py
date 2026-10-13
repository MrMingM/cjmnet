"""Readable reports plus full structured results; no AP claims."""
from .common import atomic_json, atomic_text, digest

def number(x):
    return 'NA' if x is None else f'{x:.4f}'

def assoc(a):
    ci=a.get('ci95')
    return number(a.get('estimate'))+(' ['+', '.join(number(v) for v in ci)+']' if ci else ' [CI unavailable]')

def write_reports(run,result,protocol):
    cfg=protocol['config']; d=result['decision']; reports={}
    reports['O0_SUMMARY.md']='\n'.join([
        '# Observability premise audit', '', '**Decision: '+d['verdict']+'**','',
        'H1: different state directions have different local geometric support.',
        'H2: direction-specific support explains frozen-detector marginal localization utility beyond scalar quality.',
        'H3: complementary peer directions explain gains over the better individual peer.', '',
        'GT defines ROI and reference faces; these features are oracle-assisted diagnostic quantities, never deployment inputs.',
        'I = J^T W J is a geometric information surrogate; weights are not calibrated inverse noise variances.',
        'Singular directions are retained. Two adjacent single-face views generally have rank <= 3 in this five-state model.',
        'Raw diagonal information conditions on other parameters; it is not marginalized parameter precision.',
        'No detector training, no AP threshold, no formal test, no S4 RUN/artifact dependencies.', '',
        'A directionality: '+str(d['A_directionality']),
        'BC supported parameters: '+str(d['BC_parameters']),
        'D pair complementarity: '+str(d['D_complementarity']),
        'Cohort counts: '+str(result['cohorts']),
        'Failed frames: '+str(d['failed_frames']), '',
        'Scene-cluster bootstrap preserves within-scene/frame/object/source/weather dependence.',
        'Ridge uses leave-one-scene-out folds, fixed alpha and training-fold scaling. Increment CI bootstraps fixed OOF predictions.',
        'Thresholds are exploratory preregistered decision rules, not confirmatory multiple-testing-adjusted claims.',
        'All invalid observations and unmatched predictions remain in artifacts and denominators.',
        'Insufficient scenes or frame failures => INCONCLUSIVE, never evidence of no effect.',
        'STOP_V1_PREMISE rejects this operational surrogate/detector route, not every possible observability method.',
        '', 'Detailed numeric effects, controls and failures: ../final_results.json'])+'\n'
    h1=['# H1 Directionality','','Anisotropy alone is insufficient: fixed-face Jacobians structurally create it.','',
        '|Matrix|Weather|Rows|Median anisotropy|Rank histogram|','|---|---|---:|---:|---|']
    for matrix,values in result['H1'].items():
        for w,v in values['strata'].items():
            h1.append(f"|{matrix}|{w}|{v['rows']}|{number(v['median_anisotropy'])}|{v['rank_histogram']}|")
        h1.extend(['','View relation (scene-separated): '+str(values['view_relation']),''])
    reports['H1_DIRECTIONALITY.md']='\n'.join(h1)+'\n'
    h2=['# H2 Directional utility','','Positive utility means absolute error decreases after adding the peer.',
        'Recovery/loss is excluded from the common-match regression cohort.',
        'Scalar baselines: raw/prevoxel count, retained count, reliability, uncertainty, trace and regularized logdet.',
        'Additional controls: distance, weather, GT reference length/width/yaw (shared oracle nuisance controls, not deployable inputs). Unregularized singular logdet remains null in source artifacts.','',
        '|Matrix/variant|Parameter|Pooled Spearman [scene CI]|Delta R2 [CI]|MAE reduction|','|---|---|---|---|---|']
    for key,params in result['H2'].items():
        for p,e in params.items():
            inc=e['incremental']
            h2.append(f"|{key}|{p}|{assoc(e['pooled']['directional'])}|{assoc(inc.get('delta_r2',{}))}|{number(inc.get('mae_reduction'))}|")
            for w in cfg['weathers']:
                h2.append(f"|{key}/{w}|{p}|{assoc(e[w]['directional'])}|—|—|")
    reports['H2_DIRECTIONAL_UTILITY.md']='\n'.join(h2)+'\n'
    h3=['# H3 Complementarity','','Primary outcome: best single-peer mean scaled error minus pair mean scaled error.',
        'All three subsets must match the GT. Best is an evaluation oracle, never an inference selection.',
        'C_logdet and C_weak use normalized matrices. Raw, count-normalized and trace-normalized shape versions are reported.',
        'C_weak can be identically zero due to structural nullspaces; this is reported without changing the definition.','',
        '|Matrix/metric|Pooled rho [CI]|Delta R2 [CI]|Zero fraction|','|---|---|---|---|']
    for key,e in result['H3'].items():
        if 'associations' in e:
            h3.append(f"|{key}|{assoc(e['associations']['pooled'])}|{assoc(e['incremental'].get('delta_r2',{}))}|{number(e['zero_fraction'])}|")
    reports['H3_COMPLEMENTARITY.md']='\n'.join(h3)+'\n'
    nc=['# Negative controls','','View permutation: exchange observability within fixed weather/distance bins.',
        'Eigenvector control: conjugate information by random coordinate permutation, preserving eigenvalues.',
        'Matched controls: within-weather fixed retained-point-count and reliability bins.',
        'Geometry-only and GSPR-weighted analyses share the exact retained slots.',
        'Controls are diagnostic null distributions; scene bootstrap supplies uncertainty.','',
        '|Control|Observed rho|Null median|Null q95|Above q95|','|---|---|---|---|---|']
    for group in ('h2','h3'):
        for key,values in result['negative_controls'][group].items():
            for name,v in values.items():
                nc.append(f"|{group}/{key}/{name}|{number(v['observed'])}|{number(v['median'])}|{number(v['q95'])}|{v['above_q95']}|")
    reports['NEGATIVE_CONTROLS.md']='\n'.join(nc)+'\n'
    for name,text in reports.items():
        atomic_text(run/'reports'/name,text)
    atomic_json(run/'final_results.json', {'identity':protocol['identity'],'payload':result})
    return {str(p.relative_to(run)):digest(p) for p in [run/'final_results.json',*[run/'reports'/k for k in reports]]}
