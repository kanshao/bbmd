import numpy as np
import pandas as pd

from .utils import check_valid_flat_format


def export_bmds(session, output, format='txt'):
    check_valid_flat_format(format)

    export = pd.DataFrame()
    for bmr in session.bmrs:
        df = pd.DataFrame()
        mdl = bmr.model_average
        nrows = mdl['n_total']
        df['bmr'] = np.repeat(bmr.name, nrows)
        df['index'] = np.arange(1, nrows + 1)
        df['model_average'] = np.concatenate((
            mdl['bmd'],
            np.repeat(np.nan, mdl['n_total'] - mdl['n_non_nans'])))
        for i, model in enumerate(session.models):
            mdl = bmr.results[i]
            df[model.name] = np.concatenate((
                mdl['bmd'],
                np.repeat(np.nan, mdl['n_total'] - mdl['n_non_nans'])))

        export = export.append(df, ignore_index=True)

    if format == 'txt':
        export.to_csv(output, sep='\t', index=False)
    elif format == 'xlsx':
        export.to_excel(output, index=False)
