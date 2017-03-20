import cPickle
import os
import logging
from copy import deepcopy
import re

import numpy as np
import scipy as sp
import pystan

from ..utils import get1Dkernel


class DoseResponseModel(object):

    ZEROISH = 1e-8
    MODEL_WEIGHT_COLNAMES = (
        'logpriors', 'loglikelihoods', 'logposteriors', 'weights'
    )
    NUM_PLOT_VALUES = 131
    MAX_PLOT_VALUE_RATIO = 1.3  # from max dose
    SUMMARY_REGEX = ur'^([\w]+)\s+([\d\-\.e]+)\s+([\d\-\.e]+)\s+([\d\-\.e]+)\s+([\d\-\.e]+)\s+([\d\-\.e]+)\s+([\d\-\.e]+)\s+([\d\-\.e]+)\s+([\d\-\.e]+)\s+([\d\-\.e]+)\s+([\d\-\.e]+)\s*$'  # noqa

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = kwargs.get('name', self.get_name())

    def get_name(self):
        if hasattr(self, 'name'):
            return self.name
        return str(self.__class__.__name__)

    def extract_summary_values(self, txt):
        extracted = {}
        regex = re.compile(self.SUMMARY_REGEX, re.MULTILINE)
        for ext in re.findall(regex, txt):
            extracted[ext[0]] = {
                'mean': float(ext[1]),
                'se_mean': float(ext[2]),
                'sd': float(ext[3]),
                '2.5%': float(ext[4]),
                '25%': float(ext[5]),
                '50%': float(ext[6]),
                '75%': float(ext[7]),
                '97.5%': float(ext[8]),
                'n_eff': float(ext[9]),
                'Rhat': float(ext[10]),
            }
        return extracted

    def _set_session(self, session):
        self.session = session
        self._set_data()

    def _set_data(self):
        self.data = {}
        if hasattr(self.session, 'dataset') and self.session.dataset is not None:
            self.data = deepcopy(self.session.dataset)
        self.data.update(self.get_priors())
        if hasattr(self, 'get_settings'):
            self.data.update(self.get_settings())

        self.iterations = self.session.mcmc_iterations
        self.warmup_fraction = self.session.mcmc_warmup_fraction
        self.warmup = int(self.iterations * self.warmup_fraction)
        self.chains = self.session.mcmc_num_chains
        self.seed = self.session.seed

    def execute(self, n_jobs=-1):
        self._set_data()
        self.set_pystan_version()

        if 'individual' in self.data:
            path = self.get_precompiled_path(self.data['individual'])
        else:
            path = self.get_precompiled_path()

        if os.path.exists(path):
            logging.info(u'Using pre-compiled pystan model: {}'.format(path))
            model = cPickle.load(open(path, 'rb'))
            self.fit = model.sampling(
                data=self.data,
                iter=self.iterations,
                warmup=self.warmup,
                chains=self.chains,
                seed=self.seed,
                n_jobs=n_jobs)
        else:
            logging.info('Pystan pre-compiled not-found; compilation required')
            self.fit = pystan.stan(
                model_code=self.get_stan_model(),
                data=self.data,
                iter=self.iterations,
                warmup=self.warmup,
                chains=self.chains,
                seed=self.seed,
                n_jobs=n_jobs)

        self.post_execution_processing()

    def post_execution_processing(self):
        self.parameters = self.fit.extract()
        self.get_response_vectors()
        self.get_fit_summary()
        self.correlation_coefficient()
        self.get_kernels()
        self.get_model_weights()
        self.plot_models()

    def _clear_results(self):
        # todo: add
        pass

    @classmethod
    def get_precompiled_path(cls, *args):
        return os.path.join(
            os.path.dirname(__file__),
            'compiled',
            cls.__name__.lower() + '.pkl'
        )

    @property
    def PARAMETERS(self):
        raise NotImplementedError('Abstract property')

    @property
    def STAN_MODEL(self):
        raise NotImplementedError('Abstract property')

    def set_pystan_version(self):
        self.pystan_version = pystan.__version__

    def set_trend_test(self):
        raise NotImplementedError('Abstract method')

    def get_response_vectors(self):
        predicted, observed = self.get_predicted_response_vector()
        self.predicted = predicted
        self.observed = observed
        self.test_stat = predicted - observed
        self.predicted_pvalue = \
            self.test_stat[self.test_stat > 0].size / \
            float(self.test_stat.size)

    def get_response_values(self, x, **kw):
        raise NotImplementedError('Abstract method')

    def get_loglikelihood(self, samples):
        raise NotImplementedError('Abstract method')

    def risk_at_dose(self, dose):
        # return a vector of risks at specified dose
        raise NotImplementedError('Abstract method')

    def get_priors(self):
        return dict()

    def get_input_count(self):
        raise NotImplementedError('Abstract method')

    def get_stan_model(self):
        raise NotImplementedError('Abstract method')

    def get_plot_bounds(self, xs, vectors):
        raise NotImplementedError('Abstract method')

    def correlation_coefficient(self):
        self.parameter_correlation = np.corrcoef([
            self.parameters[param] for param in self.PARAMETERS
        ]).tolist()

    def get_fit_summary(self):
        self.fit_summary = self.fit.__unicode__()
        self.fit_summary_dict = self.extract_summary_values(self.fit_summary)

    def get_kernels(self):
        self.kernels = {
            'test_stat': self.test_stat,
            'test_stat_kernel': get1Dkernel(self.test_stat)
        }
        for param in self.PARAMETERS:
            self.kernels[param] = \
                get1Dkernel(self.parameters[param], steps=100j)

    def get_max_dose(self):
        return self.data['d'].max()

    def get_num_samples(self):
        return self.test_stat.size

    def get_model_weights(self):
        # get log-likelihood
        samples = np.empty((len(self.PARAMETERS), self.get_num_samples()), dtype=np.float64)
        for j, param in enumerate(self.PARAMETERS):
            samples[j, :] = self.parameters[param]
        loglikelihoods = self.get_loglikelihood(samples)

        # get model-weight vectors
        numDoses = float(self.get_input_count())
        numParameters = float(len(self.PARAMETERS))
        self.model_weights = loglikelihoods - \
            numParameters * np.log(numDoses) / 2.0

    def set_model_averaged_weight(self, vals):
        self.model_weight_vector = vals
        self.model_weight_scaler = vals.mean()

    def _getLogPDF(self, prior, arr):
        distType = prior[0]
        if distType == 'u':
            return sp.stats.uniform.logpdf(arr, *prior[1:])
        elif distType == 'n':
            return sp.stats.norm.logpdf(arr, *prior[1:])
        else:
            raise ValueError('Unknown distribution: {0}'.format(distType))

    def plot_models(self):
        xs = np.linspace(
            self.ZEROISH,
            self.MAX_PLOT_VALUE_RATIO,
            num=self.NUM_PLOT_VALUES)
        vectors = np.zeros((xs.size, 4), dtype=np.float64)
        vectors = self.get_plot_bounds(xs, vectors)
        vectors[0, 0] = 0.  # change 1e-8 to 0 for clean plots
        vectors[:, 0] *= self.get_max_dose()
        self.plotting = vectors
