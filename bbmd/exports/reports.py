from copy import deepcopy
import json

from .utils import check_valid_report_format
from ..utils import get_summary_stats


report_template = """BBMD REPORT
===========

Report name: {name}
Time generated: {report_time}

Pystan version: {pystan_version}

Dataset:
--------
{dataset}

Trend test p-value: {pvalue}
Trend test z-score: {zscore}

MCMC SETTINGS:
--------------

MCMC iterations: {mcmc_iterations}
Number of chains: {mcmc_num_chains}
Warmup fraction: {mcmc_warmup_fraction}
Random seed: {seed}

MODELS:
-------
{models}

BMRS:
-----
{bmrs}
"""

model_template = """{name}
--------------------
Model class: {class_name}
LaTeX equation: {equation}
{model_settings}
{fit_summary}

Posterior predictive p-value for model fit: {predicted_pvalue:5.4f}

Model weight: {weight:5.4f}

Parameter correlations:
{parameter_correlations}

Parameter distributions:
{parameter_distributions}
"""

bmr_template = """{name}
--------------------
BMR class: {class_name}

BMR value: {bmr}
Adversity value: {adversity_value}

Calculated BMR estimate:
{bmr_table}
"""


def get_dataset_txt(session, ds):
    lines = []
    if session.dataset_type in session.DICHOTOMOUS_TYPES:
        lines.append('      Dose     N   Inc')
        for i, d in enumerate(ds['d']):
            txt = '{:10.4f}{:6d}{:6d}'.format(
                ds['d'][i], ds['n'][i], ds['y'][i])
            lines.append(txt)
    else:
        if session.dataset_type == 'C':
            lines.append('       Dose     N   Response      Stdev')
            for i, d in enumerate(ds['d']):
                txt = '{:11.4f}{:6d}{:11.4f}{:11.4f}'.format(
                    ds['d'][i], ds['n'][i], ds['resp'][i], ds['stdev'][i])
                lines.append(txt)
        else:
            lines.append('       Dose   Response')
            for i, d in enumerate(ds['d']):
                txt = '{:11.4f}{:11.4f}'.format(
                    ds['d'][i], ds['y'][i])
                lines.append(txt)

    return '\n'.join(lines)


def processing_for_report(session, d):
    # replace values to make ready for report imports
    d['pvalue'] = '{:9.8f}'.format(d['pvalue']) \
        if isinstance(d['pvalue'], float) else d['pvalue']
    d['zscore'] = '{:0.3f}'.format(d['zscore']) \
        if isinstance(d['zscore'], float) else d['zscore']
    d['dataset'] = get_dataset_txt(session, d['dataset'])
    d['models'] = '\n'.join([
        prepped_model_to_text(m)
        for m in d['models']
    ])
    d['bmrs'] = '\n'.join([
        prepped_bmr_to_text(b)
        for b in d['bmrs']
    ])
    return d


def prep_dataset(ds):
    d = {}
    for k, v in ds.iteritems():
        if hasattr(v, 'tolist'):
            v = v.tolist()
        d[k] = v
    return d


def get_correlation_tbl(d):
    rows = []
    params = d['parameters']
    corrs = d['parameter_correlations']

    header = ['{:>10}'.format(p) for p in params]
    header.insert(0, '{:>10}'.format(''))
    rows.append(''.join(header))

    for i, param in enumerate(params):
        row = ['{:10.4f}'.format(c) for c in corrs[i]]
        row.insert(0, '{:>10}'.format(param))
        rows.append(''.join(row))

    return '\n'.join(rows)


def prepped_model_to_text(d):
    d['parameter_correlations'] = get_correlation_tbl(d)
    d['parameter_distributions'] = get_param_dists_text(d)
    d['model_settings'] = get_model_settings_text(d)
    return model_template.format(**d)


def get_model_settings_text(d):
    settings = d['settings']
    txt = []

    if not settings:
        return ''

    val = settings.get('pwr_lbound')
    if val:
        txt.append('Power lower-bound: {}'.format(val))

    return '\n'.join(txt) + '\n'


def get_param_dists_text(d):
    rows = []
    params = d['parameters']
    corrs = d['parameter_distributions']

    header = ['{:>10}'.format(p) for p in params]
    header.insert(0, '{:>10}'.format(''))
    rows.append(''.join(header))

    stats = [
        'mean',
        'std',
        'p5',
        'p25',
        'p50',
        'p75',
        'p95',
    ]
    for stat in stats:
        row = ['{:10.4f}'.format(corrs[param][stat]) for param in params]
        row.insert(0, '{:>10}'.format(stat))
        rows.append(''.join(row))

    return '\n'.join(rows)


def get_param_dists(model):
    d = {}
    for param in model.PARAMETERS:
        d[param] = get_summary_stats(model.parameters[param])
    return d


def noop():
    pass


def prep_model(model):
    return {
        'name': model.name,
        'class_name': model.__class__.__name__,
        'pystan_version': model.pystan_version,
        'fit_summary': model.fit_summary,
        'fit_summary_dict': model.fit_summary_dict,
        'predicted_pvalue': model.predicted_pvalue,
        'weight': model.model_weight_scaler,
        'equation': model.LATEX_EQUATION,
        'parameters': model.PARAMETERS,
        'parameter_correlations': model.parameter_correlation,
        'parameter_distributions': get_param_dists(model),
        'settings': getattr(model, 'get_settings', noop)()
    }


def get_bmr_table_text(d):
    rows = []
    models = d['models']
    model_average = d['model_average']

    header = ['{:>12}'.format(m['name']) for m in models]
    header.insert(0, '{:>25}'.format('Statistic'))
    header.insert(1, '{:>12}'.format('Model avg.'))
    rows.append(''.join(header))

    stats = [
        ('prior_weight', 'prior model weight'),
        ('weight', 'posterior model weight'),
        ('p50', 'BMD (median)'),
        ('p5', 'BMD (5%)'),
        ('p25', '25%'),
        ('mean', 'Mean'),
        ('std', 'SD'),
        ('p75', '75%'),
        ('p95', '95%'),
    ]
    for stat, header in stats:
        row = ['{:12.4f}'.format(m[stat]) for m in models]
        row.insert(0, '{:>25}'.format(header))
        if stat in model_average:
            row.insert(1, '{:12.4f}'.format(model_average[stat]))
        else:
            row.insert(1, '{:>12}'.format('N/A'))
        rows.append(''.join(row))

    return '\n'.join(rows)


def prepped_bmr_to_text(d):
    d['bmr_table'] = get_bmr_table_text(d)
    return bmr_template.format(**d)


def prep_bmr_model(session, bmr):
    arr = []
    for i, b in enumerate(bmr.results):
        d = deepcopy(b['stats'])
        d['name'] = session.models[i].name
        d['prior_weight'] = bmr.priors[i]
        d['weight'] = bmr.model_posterior_weights[i]
        d['n_total'] = b['n_total']
        d['n_non_nans'] = b['n_non_nans']
        arr.append(d)
    return arr


def prep_bmr(session, bmr):
    return {
        'name': bmr.name,
        'class_name': bmr.__class__.__name__,
        'model_average': bmr.model_average['stats'],
        'model_average_n_total': bmr.model_average['n_total'],
        'model_average_n_non_nans': bmr.model_average['n_non_nans'],
        'models': prep_bmr_model(session, bmr),
        'bmr': getattr(bmr, 'bmr', None),
        'adversity_value': getattr(bmr, 'adversity_value', 'N/A')
    }


def export_report(session, output, format='txt'):
    check_valid_report_format(format)

    d = {
        'name': session.name,
        'created': session.created,
        'report_time': session._get_timestamp().isoformat(),
        'dataset': prep_dataset(session.dataset),
        'pvalue': session.trend_p_value or 'N/A',
        'zscore': session.trend_z_test or 'N/A',
        'mcmc_iterations': session.mcmc_iterations,
        'mcmc_num_chains': session.mcmc_num_chains,
        'mcmc_warmup_fraction': session.mcmc_warmup_fraction,
        'seed': session.seed,
        'models': [prep_model(m) for m in session.models],
        'bmrs': [prep_bmr(session, b) for b in session.bmrs],
        'pystan_version': session.get_pystan_version(),
    }

    if format == 'txt':
        d = processing_for_report(session, d)
        txt = report_template.format(**d)
    elif format == 'json':
        txt = json.dumps(d, indent=4, separators=(',', ': '))

    if isinstance(output, basestring):
        with open(output, 'w') as f:
            f.write(txt)
    elif hasattr(output, 'write'):
        output.write(txt)
    else:
        raise ValueError('Unknown output type')
