import collections
import multiprocessing
import logging
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas as pd
from datetime import datetime

from scipy import stats

from .bmr.base import BMRBase
from .models.base import DoseResponseModel
from .models.dichotomous import Dichotomous
from . import exports


class Session(object):
    """
    Modeling session. Composed of a dataset, modeling inputs, some model
    specifications, and BMRs.
    """

    ZEROISH = 1e-8
    TIME_FORMAT = '%b %d %Y, %I:%M %p'

    def __init__(self, mcmc_iterations=20000, mcmc_num_chains=2,
                 mcmc_warmup_fraction=0.5, seed=12345, **kwargs):
        self.mcmc_iterations = mcmc_iterations
        self.mcmc_num_chains = mcmc_num_chains
        self.mcmc_warmup_fraction = mcmc_warmup_fraction
        self.seed = seed
        self._unset_dataset()
        self.models = []
        self.bmrs = []
        self.name = kwargs.get('name', self._get_default_name())
        self.created = self._get_timestamp().isoformat()

    def add_models(self, *args):
        for model in args:
            if not isinstance(model, DoseResponseModel):
                raise ValueError('Not a DoseResponseModel')
            model._set_session(self)
            self.models.append(model)

    def _unset_dataset(self):
        self.dataset_type = None
        self.dataset = None

    def _get_default_name(self):
        return 'New run {}'.format(self._get_timestamp().strftime(self.TIME_FORMAT))

    def _get_timestamp(self):
        return datetime.now()

    DICHOTOMOUS_SUMMARY = 'D'
    DICHOTOMOUS_INDIVIDUAL = 'E'
    CONTINUOUS_SUMMARY = 'C'
    CONTINUOUS_INDIVIDUAL = 'I'
    DICHOTOMOUS_TYPES = ['D', 'E']
    CONTINUOUS_TYPES = ['C', 'I']

    def add_dichotomous_data(self, dose, n, incidence):
        if not isinstance(dose, collections.Iterable) or \
           not isinstance(n, collections.Iterable) or \
           not isinstance(incidence, collections.Iterable):
            raise ValueError('Must be iterables')
        if any([len(dose) != len(n), len(n) != len(incidence)]):
            raise ValueError('All arrays must be same length')
        if (len(dose) < 2):
            raise ValueError('Must have at least 2 values')

        incidence = np.array(incidence, dtype=np.int64)
        n = np.array(n, dtype=np.int64)
        doses = np.array(dose, dtype=np.float64)

        # more checks in numpy form
        if np.any(incidence > n):
            raise ValueError('Incidence must be <= N')
        if np.any(doses < 0) or np.any(n <= 0) or np.any(incidence < 0):
            raise ValueError('Values out of bound')

        dno0 = np.copy(doses)  # for log-models
        if dno0[0] == 0:
            if dno0[1] > 1e-6:
                dno0[0] = 1e-6
            else:
                dno0[0] = dno0[1] * 0.01

        self.dataset_type = self.DICHOTOMOUS_INDIVIDUAL
        self.dataset = {
            'd': doses,
            'dnorm': doses/doses.max(),
            'dno0': dno0,
            'dno0norm': dno0/dno0.max(),
            'y': incidence,
            'n': n,
            'len': doses.size,
        }

    INDIVIDUAL = 1
    SUMMARY = 0

    def add_continuous_summary_data(self, dose, n, response, stdev):
        if not isinstance(dose, collections.Iterable) or \
           not isinstance(n, collections.Iterable) or \
           not isinstance(response, collections.Iterable) or \
           not isinstance(stdev, collections.Iterable):
            raise ValueError('Must be iterables')
        if any([len(dose) != len(n),
                len(n) != len(response),
                len(response) != len(stdev)]):
            raise ValueError('All arrays must be same length')
        if (len(dose) < 2):
            raise ValueError('Must have at least 2 values')

        doses = np.array(dose, dtype=np.float64)
        ns = np.array(n, dtype=np.int64)
        resps = np.array(response, dtype=np.float64)
        stdevs = np.array(stdev, dtype=np.float64)

        dnorm = doses / doses.max()
        resp_ln = np.log(resps) - 0.5 * np.log((stdevs / resps) ** 2 + 1)
        stdev_ln = np.sqrt(np.log((stdevs / resps) ** 2 + 1))

        self.dataset_type = self.CONTINUOUS_SUMMARY
        self.dataset = {
            'individual': self.SUMMARY,

            # original data
            'd': doses,
            'n': ns,
            'resp': resps,
            'stdev': stdevs,

            # processed
            'len': len(doses),
            'dnorm': dnorm,
            'ym': resp_ln,
            'ysd': stdev_ln,
        }

    def add_continuous_individual_data(self, dose, response):
        if not isinstance(dose, collections.Iterable) or \
           not isinstance(response, collections.Iterable):
            raise ValueError('Must be iterables')
        if (len(dose) != len(response)):
            raise ValueError('All arrays must be same length')
        if (len(dose) < 3):
            raise ValueError('Must have at least 3 values')

        doses = np.array(dose, dtype=np.float64)
        resps = np.array(response, dtype=np.float64)
        resps[resps == 0] = self.ZEROISH
        dnorm = doses / doses.max()

        self.dataset_type = self.CONTINUOUS_INDIVIDUAL
        self.dataset = {
            'individual': self.INDIVIDUAL,
            'd': doses,
            'dnorm': dnorm,
            'y': resps,
            'len': doses.size,
        }

    def get_trend_test(self):
        if self.dataset_type in self.DICHOTOMOUS_TYPES:
            model = Dichotomous()
            model._set_session(self)
            z, p = model.get_trend_test()
        else:
            z, p = None, None
        return self.set_trend_test(z, p)

    def set_trend_test(self, z, p):
        self.trend_z_test = z
        self.trend_p_value = p
        return z, p

    def _execute_models(self, pythreads=False):
        if pythreads:
            logging.debug('Using Pythreads')
            pool = ThreadPool(processes=multiprocessing.cpu_count())
            for model in self.models:
                model._set_session(self)
                pool.apply_async(
                    model.execute,
                    kwds={'n_jobs': -1}
                )
            pool.close()
            pool.join()
        else:
            logging.debug('Using STAN multiprocessing')
            for model in self.models:
                model._set_session(self)
                model.execute()

    def execute(self, pythreads=False):
        # Pythreads use a python-based threadpool to run models in
        # parallel; if pythreads is false then models are run serially,
        # and STAN runs each chain in a separate thread.
        if len(self.models) == 0:
            raise ValueError('At least one model must be included.')
        self.get_trend_test()
        self._execute_models(pythreads)
        self._calc_model_weights()

    def _calc_model_weights(self):
        m_matrix = np.empty((
            len(self.models),
            self.models[0].model_weights.size,
        ))

        for i, model in enumerate(self.models):
            m_matrix[i, :] = model.model_weights

        m_matrix = np.exp(m_matrix - m_matrix.min(axis=0))
        weights = np.divide(m_matrix, m_matrix.sum(axis=0))

        for i, model in enumerate(self.models):
            model.set_model_averaged_weight(weights[i, :])

    def add_bmrs(self, *args):
        for bmr in args:
            if not isinstance(bmr, BMRBase):
                raise ValueError('Not a BMR')
            self.bmrs.append(bmr)

    def calculate_bmrs(self):
        for bmr in self.bmrs:
            bmr.calculate(self)

    def get_bmr_adversity_value_domains(self):
        ds = self.dataset

        if self.dataset_type in self.DICHOTOMOUS_TYPES:
            rates = ds['y'].astype('float64') / ds['n'].astype('float64')
            bgrate = rates[0]
            suggested = rates.max()
            domain = (1e-4, min((1. - bgrate, suggested * 3.)))

            if np.isnan(suggested):
                suggested = None

            return {
                'bmr_max_suggested': suggested,
                'bmr_domain': domain
            }

        doses = ds['d']
        is_increasing = self.is_increasing()

        if self.dataset_type == self.CONTINUOUS_SUMMARY:
            resps = ds['resp']
            stdevs = ds['stdev']
            mean_control = resps[doses == doses.min()].mean()
            resp_ln = ds['ym']
            stdev_ln = ds['ysd']
            resp_ln_min = resp_ln[doses == doses.min()].mean()
            stdev_ln_min = stdev_ln[doses == doses.min()].mean()

            if is_increasing:
                max_resp = resps.max()
                max_resp_sd = stdevs[resps == max_resp][0]

                cutoff_domain = (
                    mean_control,
                    (max_resp + 2.5 * max_resp_sd) * 2.0,
                )

                ac_upper = (max_resp + 2.5 * max_resp_sd) * 2.0 - mean_control
                abs_change_domain = (0., ac_upper)

                rc_upper = ((max_resp + 2.5 * max_resp_sd) * 2.0 - mean_control) / mean_control
                rel_change_domain = (0., rc_upper)

            else:
                min_resp = resps.min()
                min_resp_sd = stdevs[resps == min_resp][0]

                cutoff_domain = (
                    np.max([0.0, (min_resp - 2.5 * min_resp_sd) * 0.5]),
                    mean_control,
                )

                ac_upper = (mean_control - min_resp + 2.5 * min_resp_sd) * 2.0
                abs_change_domain = (0., np.min([mean_control, ac_upper]))

                rc_upper = (mean_control - min_resp + 2.5 * min_resp_sd) * 2.0 / mean_control
                rel_change_domain = (0., np.min([1.0, rc_upper]))

        elif self.dataset_type == self.CONTINUOUS_INDIVIDUAL:
            resps = ds['y']
            min_dose_resps = resps[doses == doses.min()]
            mean_control = min_dose_resps.mean()
            resp_ln_min = np.log(min_dose_resps).mean()
            stdev_ln_min = np.log(min_dose_resps).std()

            if is_increasing:
                max_resp = resps.max()
                cutoff_domain = (mean_control, max_resp * 2.)
                ac_upper = max_resp * 2.0 - mean_control
                rc_upper = (max_resp * 2.0 - mean_control) / mean_control

            else:
                min_resp = resps.min()
                cutoff_domain = (mean_control, min_resp * 0.5)
                ac_upper = np.min([mean_control, mean_control - min_resp * 2.0])
                rc_upper = np.min([1.0, (mean_control - min_resp) * 2.0 / mean_control])

            abs_change_domain = [0., ac_upper]
            rel_change_domain = [0., rc_upper]

        if is_increasing:
            cutoff_domain_hybrid = (
                np.exp(stats.norm.ppf(0.5001, resp_ln_min, stdev_ln_min)),
                np.exp(stats.norm.ppf(0.9999, resp_ln_min, stdev_ln_min)),
            )
        else:
            cutoff_domain_hybrid = (
                np.exp(stats.norm.ppf(0.0001, resp_ln_min, stdev_ln_min)),
                np.exp(stats.norm.ppf(0.4999, resp_ln_min, stdev_ln_min)),
            )

        bmr_domain = (1e-4, 0.5)
        quantile_control_range = (0.5001, 0.9999) \
            if is_increasing else \
            (1e-4, 0.4999)

        return {
            'is_increasing': is_increasing,
            'bmr_domain': bmr_domain,
            'quantile_domain': quantile_control_range,
            'cutoff_domain': cutoff_domain,
            'cutoff_domain_hybrid': cutoff_domain_hybrid,
            'absolute_change_domain': abs_change_domain,
            'relative_change_domain': rel_change_domain,
        }

    def is_increasing(self):
        if self.dataset_type in self.DICHOTOMOUS_TYPES:
            return True

        doses = self.dataset['d']
        if self.dataset_type == self.CONTINUOUS_SUMMARY:
            resps = self.dataset['resp']
        else:
            resps = self.dataset['y']

        avg_min_resp = resps[np.where(doses == doses.min())].mean()
        avg_max_resp = resps[np.where(doses == doses.max())].mean()
        return bool(avg_min_resp < avg_max_resp)  # convert np -> python

    EXPORT_FORMATS = [
        'txt',
        'xlsx',
    ]

    def get_pystan_version(self):
        return self.models[0].pystan_version if \
            len(self.models) > 0 else None

    def dr_to_string(self):
        rows = []
        ds = self.dataset
        if self.dataset_type == self.DICHOTOMOUS_SUMMARY:
            for d, n, y in zip(ds['d'], ds['n'], ds['y']):
                rows.append('{}\t{}\t{}'.format(d, n, y))
        elif self.dataset_type == self.DICHOTOMOUS_INDIVIDUAL:
            raise NotImplementedError('not implemented via command-line')
        elif self.dataset_type == self.CONTINUOUS_SUMMARY:
            for d, n, resp, stdev in zip(ds['d'], ds['n'], ds['resp'], ds['stdev']):
                rows.append('{}\t{}\t{}\t{}'.format(d, n, resp, stdev))
        elif self.dataset_type == self.CONTINUOUS_INDIVIDUAL:
            for d, y in zip(ds['d'], ds['y']):
                rows.append('{}\t{}'.format(d, y))
        return '\n'.join(rows)

    def export_report(self, file, format='txt'):
        exports.export_report(self, file, format)

    def export_parameters(self, file, format='txt'):
        # expects a string or BytesIO object
        isString = True
        if not isinstance(file, basestring):
            isString = False
            file = pd.ExcelWriter(file)
        exports.export_parameters(self, file, format)
        if not isString:
            file.save()

    def export_bmds(self, file, format='txt'):
        # expects a string or BytesIO object
        isString = True
        if not isinstance(file, basestring):
            isString = False
            file = pd.ExcelWriter(file)
        exports.export_bmds(self, file, format)
        if not isString:
            file.save()

    def export_word_report(self, file):
        # expects a filename or file-type object
        exports.WordReportFactory(self, file)

    def copy_as_template(self):
        self._unset_dataset()
        for model in self.models:
            model._clear_results()
        for bmr in self.bmrs:
            bmr.clear_results()
