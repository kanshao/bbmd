import numpy as np
import pandas as pd

from .utils import check_valid_flat_format


def export_parameters(session, output, format='txt'):
    check_valid_flat_format(format)

    params = []
    for model in session.models:
        params.extend(model.PARAMETERS)
    params = set(params)

    export = pd.DataFrame()
    for model in session.models:
        df = pd.DataFrame()
        nrows = model.model_weight_vector.size
        df['model'] = np.repeat(model.name, nrows)
        df['index'] = np.arange(1, nrows + 1)
        for param in params:
            val = model.parameters[param] \
                if param in model.parameters \
                else np.repeat(np.nan, nrows)
            df[param] = val

        export = export.append(df, ignore_index=True)

    if format == 'txt':
        export.to_csv(output, sep='\t', index=False)
    elif format == 'xlsx':
        export.to_excel(output, index=False)
