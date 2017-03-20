import re

import numpy as np
import scipy.special as special
from scipy import stats

from ..utils import get1Dkernel, get_summary_stats


class BMRBase(object):

    def clear_results(self):
        self.results = None
        self.model_average = None

    def _set_priors(self, priors):
        if priors is None:
            priors = np.repeat(1, len(self.session.models))
        else:
            if len(priors) != len(self.session.models):
                raise ValueError('Unknown number of priors')

        priors = np.array(priors, dtype=np.float64)
        priors = priors / priors.sum()
        self.priors = priors

    def validate_inputs(self):
        # check and set priors
        self._set_priors(self._priors)
        domains = self.session.get_bmr_adversity_value_domains()

        # check bmr
        if hasattr(self, 'bmr'):
            domain = domains['bmr_domain']
            if not domain[0] < self.bmr < domain[1]:
                raise ValueError(
                    'BMR not in allowable domain: {} ({} - {})'
                    .format(self.bmr, domain[0], domain[1]))

        # check adversity value
        if hasattr(self, 'adversity_value'):
            domain = self.get_adversity_domain(domains)
            if not domain[0] < self.adversity_value < domain[1]:
                raise ValueError(
                    'Adversity value not in allowable domain: {} ({} - {})'
                    .format(self.adversity_value, domain[0], domain[1]))

    def calculate(self, session):
        self.session = session
        self.validate_inputs()
        self.results = [
            self.calculate_for_model(model)
            for model in self.session.models
        ]
        self.model_average = self.calc_model_average()

    def get_adversity_domain(self, domains):
        raise NotImplementedError('Abstract method')

    def get_bmr_vector(self, model):
        raise NotImplementedError('Abstract method')

    NAME_REGEX = re.compile(r'([a-z])([A-Z])')

    def __str__(self):
        if hasattr(self, 'name'):
            return self.name
        params = []
        if hasattr(self, 'bmr'):
            params.append('bmr={}'.format(self.bmr))
        if hasattr(self, 'adversity_value'):
            params.append('{}={}'.format(
                self.ADVERSITY_VERBOSE_NAME, self.adversity_value))
        name = re.sub(self.NAME_REGEX, '\g<1> \g<2>', self.__class__.__name__)
        return '{} ({})'.format(name, ', '.join(params))

    def calculate_for_model(self, model):
        bmd_vector = self.get_bmr_vector(model) * model.get_max_dose()
        n_total = bmd_vector.size
        non_nans = bmd_vector[~np.isnan(bmd_vector)]
        n_non_nans = non_nans.size
        return {
            'n_total': n_total,
            'n_non_nans': n_non_nans,
            'bmd': non_nans,
            'kernel': get1Dkernel(non_nans, steps=100j),
            'stats': get_summary_stats(bmd_vector),
        }

    def get_model_posterior_weights(self, models):
        # same method as session._calc_model_weights except we in include
        # a prior vector; the session assumes an equal prior for all models
        # but here we drop that assumption in a case a user wants to remove
        # a model.
        matrix = np.empty((
            len(models),
            models[0].model_weight_vector.size
        ), dtype=np.float64)
        priors = self.priors

        # build inputs
        for i, model in enumerate(models):
            matrix[i, :] = model.model_weights

        matrix = np.exp(matrix - matrix.min(axis=0))
        matrix = (matrix.T * priors).T
        weights = np.divide(matrix, matrix.sum(axis=0)).mean(axis=1)
        return weights

    def calc_model_average(self):
        model_posterior_weights = \
            self.get_model_posterior_weights(self.session.models)

        vsize = min([d['n_non_nans'] for d in self.results])

        bmd = np.empty(shape=(model_posterior_weights.size, vsize))

        for i in xrange(bmd.shape[0]):
            bmd[i, :] = self.results[i]['bmd'][:vsize] * \
                model_posterior_weights[i]

        bmd = bmd.sum(axis=0)

        return dict(
            model_posterior_weights=model_posterior_weights,
            n_total=self.results[0]['n_total'],
            n_non_nans=vsize,
            bmd=bmd,
            kernel=get1Dkernel(bmd, steps=100j),
            stats=get_summary_stats(bmd),
        )

    @classmethod
    def get_related_models(cls, bmrs):
        # group together extra/added bmr models if possible
        related = []
        skip_index = None
        for i in range(len(bmrs) - 1):

            if i == skip_index:
                continue

            a = bmrs[i]
            b = bmrs[i + 1]
            cnames = [
                getattr(a, 'DUAL_TYPE', None),
                getattr(b, 'DUAL_TYPE', None)
            ]
            if a.__class__.__base__ == b.__class__.__base__ and \
               a.DUAL and b.DUAL and \
               'Extra' in cnames and 'Added' in cnames and \
               a.bmr == b.bmr:
                    related.append([a, b])
                    skip_index = i + 1
            else:
                related.append([a])

        return related


class DichotomousBase(BMRBase):

    DUAL = True

    def __init__(self, bmr, priors=None, **kwargs):
        self.bmr = bmr
        self._priors = priors
        self.name = kwargs.get('name', str(self))


class CentralTendencyBase(BMRBase):

    DUAL = False

    def __init__(self, adversity_value, priors=None, **kwargs):
        self.adversity_value = adversity_value
        self._priors = priors
        self.name = kwargs.get('name', str(self))


class HybridBase(BMRBase):

    DUAL = True

    def __init__(self, adversity_value, bmr, priors=None, **kwargs):
        self.adversity_value = adversity_value
        self.bmr = bmr
        self._priors = priors
        self.name = kwargs.get('name', str(self))

    def calc_bmd_quantile_hybrid(self, model, isExtra=True):
        # Adversity defined based on a quantile of the control, such as 99th
        # percentile of control
        sigma = model.parameters['sigma']
        cutoff_log = stats.norm.ppf(
            self.adversity_value,
            np.log(model.get_control_vector()),
            sigma
        )
        fn = self.quantile_at_bmd_extra if (isExtra) else \
            self.quantile_at_bmd_added
        quantile = fn(model, self.adversity_value)
        mean_log = self.cutoff_quantile_to_mean(cutoff_log, quantile, sigma)
        return model.calc_central_tendency(mean_log)

    def calc_bmd_cutoff_hybrid(self, model, isExtra=True):
        # Adversity defined based on an absolute cutoff value
        sigma = model.parameters['sigma']
        log_cutoff = np.log(self.adversity_value)
        quantal_cutoff = stats.norm.cdf(
            log_cutoff,
            np.log(model.get_control_vector()),
            sigma
        )
        fn = self.quantile_at_bmd_extra if (isExtra) else \
            self.quantile_at_bmd_added
        quantile = fn(model, quantal_cutoff)
        mean_log = self.cutoff_quantile_to_mean(log_cutoff, quantile, sigma)
        return model.calc_central_tendency(mean_log)

    def quantile_at_bmd_added(self, model, quantile_at_control):
        return quantile_at_control - self.bmr \
            if (model.response_direction == 1) \
            else quantile_at_control + self.bmr

    def quantile_at_bmd_extra(self, model, quantile_at_control):
        return (1. - self.bmr) * quantile_at_control \
            if (model.response_direction == 1) \
            else self.bmr + (1. - self.bmr) * quantile_at_control

    def cutoff_quantile_to_mean(self, cutoff, quantile, sigma):
        # Calculate mean value (on the log-scale) using cutoff and quantile,
        # output is the median value (on regular scale) of DR model
        return np.exp(cutoff - sigma * np.sqrt(2.) *
                      special.erfinv(2. * quantile - 1.))
