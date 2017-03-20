import os

import numpy as np
from scipy import stats

from . import base


class Continuous(base.DoseResponseModel):

    INDIVIDUAL = 1
    SUMMARY = 0

    @classmethod
    def get_precompiled_path(cls, data_type):
        fn = '{}.individual.pkl'.format(cls.__name__.lower())\
            if data_type == cls.INDIVIDUAL \
            else '{}.summary.pkl'.format(cls.__name__.lower())
        return os.path.join(os.path.dirname(__file__), 'compiled', fn)

    def get_input_count(self):
        return self.data['len']

    @property
    def response_direction(self):
        if not hasattr(self, '_response_direction'):

            if self.is_individual_dataset:
                doses = self.data['dnorm']
                resps = self.data['y']
                avg_min_dose = np.mean(resps[np.where(doses == doses.min())])
                avg_max_dose = np.mean(resps[np.where(doses == doses.max())])
                self._response_direction = \
                    1 if (avg_min_dose < avg_max_dose) else -1
            else:
                dnorm = self.data['dnorm']
                resp = self.data['resp']
                self._response_direction = 1 if \
                    resp[dnorm.argmin()] < resp[dnorm.argmax()] \
                    else -1
        return self._response_direction

    @property
    def is_individual_dataset(self):
        return self.data['individual'] == self.INDIVIDUAL

    def get_stan_model(self):
        return self.STAN_INDIVIDUAL \
            if self.is_individual_dataset \
            else self.STAN_SUMMARY

    def get_prior_upper(self):
        if self.is_individual_dataset:
            return self.data['y'].max() * 2.
        else:
            return (
                self.data['resp'].max() +
                2. * self.data['stdev'][np.argmax(self.data['resp'])]
            ) * 2.

    def get_prior_slope(self):
        if self.is_individual_dataset:
            y = self.data['y']
            dnorm = self.data['dnorm']

            slope = (y.max() - y.min()) /\
                    (dnorm[y.argmax()] - dnorm[y.argmin()])

        else:
            dose = self.data['d']
            resp = self.data['resp']
            stdev = self.data['stdev']
            dnorm = self.data['dnorm']

            mean_dmax = resp[dose == dose.max()]
            std_dmax = stdev[dose == dose.max()]
            mean_dmin = resp[dose == dose.min()]
            std_dmin = stdev[dose == dose.min()]
            slope = (mean_dmax + std_dmax * 2 - mean_dmin - std_dmin * 2) /\
                    (dnorm.max() - dnorm.min())

        b = np.array([0., slope * 5.])
        if self.response_direction == -1:
            b = b[::-1]

        return b

    def likelihoodI(self, resplog, meanlog, sdlog):
        return np.sum(np.log(stats.norm.pdf(resplog, meanlog, sdlog)))

    def likelihoodC(self, resplog, sdlog, iresplog, isdlog, ins):
        return (
            -0.5 * np.sum(ins) * np.log(np.pi * 2.) -
            np.sum(0.5 * ins * np.log(sdlog ** 2.) +
                   0.5 * (ins * (iresplog - resplog) ** 2. +
                          (ins - 1.) * isdlog ** 2.) / sdlog ** 2.))

    def get_plot_bounds(self, xs, vectors):
        sigma = np.percentile(self.parameters['sigma'], 50.)
        for i in xrange(xs.size):
            resps = self.get_response_values(xs[i], **self.parameters)
            resp = np.percentile(resps, 50.)
            vectors[i, :] = (
                xs[i],
                np.exp(stats.norm.ppf(0.05, np.log(resp), sigma)),
                resp,
                np.exp(stats.norm.ppf(0.95, np.log(resp), sigma)),
            )
        return vectors


class Exponential2(Continuous):

    PARAMETERS = ('a', 'b', 'sigma')

    STAN_INDIVIDUAL = """
        data{
            int <lower=0> len;          // number of dose points
            real <lower=0> dnorm[len];  // dose levels
            real <lower=0> y[len];      // observed responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len)
                log(y[i]) ~ normal(log(a*exp(b*dnorm[i])), sigma);
        }
    """

    STAN_SUMMARY = """
        data{
            int <lower=0> len;          // number of dose groups
            int <lower=0> n[len];       // number of subjects in each dose group
            real <lower=0> dnorm[len];  // dose levels
            real ym[len];               // observed mean of responses
            real <lower=0> ysd[len];    // observed stdev of responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len){
                target += (-n[i]*log(sigma^2)*0.5-(n[i]*(ym[i]-log(a*exp(b*dnorm[i])))^2+(n[i]-1)*ysd[i]^2)/(2*sigma^2));
            }
        }
    """

    LATEX_EQUATION = r'$f(dose) = a \times e^{b \times dose}$'  # noqa

    def get_priors(self):
        if self.response_direction == 1:
            b_prior = np.array([0., 50.])
        else:
            b_prior = np.array([-50., 0.])

        return {
            'p_a': np.array([0., self.get_prior_upper()]),
            'p_b': b_prior,
            'p_sig': np.array([0., 2.5]),
        }

    def get_response_vector(self, a, b, doses):
        return a * np.exp(b * doses)

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        sigma = self.parameters['sigma']

        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], doses))
                y_post_pred = np.random.normal(mean_posterior, sigma[i])
                predicted[i] = -2. * self.likelihoodI(mean_posterior, y_post_pred, sigma[i])
                observed[i] = -2. * self.likelihoodI(mean_posterior, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], dnorm))
                mean_pred = np.empty(dnorm.size)
                std_pred = np.empty(dnorm.size)
                for j in xrange(dnorm.size):
                    resp_ind_pred = np.random.normal(mean_posterior[j], sigma[i], ns[j])
                    mean_pred[j] = np.average(resp_ind_pred)
                    std_pred[j] = np.std(resp_ind_pred)
                predicted[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    mean_pred, std_pred, ns)
                observed[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        sigma = samples[2, :]

        predicted = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], doses))
                predicted[i] = self.likelihoodI(resp, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], dnorm))
                predicted[i] = self.likelihoodC(
                    resp, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return self.get_response_vector(kw['a'], kw['b'], x)

    def get_control_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        return self.get_response_vector(a, b, 0.)

    def calc_central_tendency(self, cutoff):
        a = self.parameters['a']
        b = self.parameters['b']
        return np.log(cutoff / a) / b

    def added_risk(self, bmr):
        return 1.


class Exponential3(Continuous):

    PARAMETERS = ('a', 'b', 'g', 'sigma')

    STAN_INDIVIDUAL = """
        data{
            int <lower=0> len;          // number of dose points
            real pwr_lbound;            // restraint value
            real <lower=0> dnorm[len];  // dose levels
            real <lower=0> y[len];      // observed responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_g[2];                // prior for g
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=pwr_lbound> g;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            g ~ uniform(p_g[1], p_g[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len)
                log(y[i]) ~ normal(log(a*exp(b*dnorm[i]^g)), sigma);
        }
    """

    STAN_SUMMARY = """
        data{
            int <lower=0> len;          // number of dose groups
            real pwr_lbound;            // restraint value
            int <lower=0> n[len];       // number of subjects in each dose group
            real <lower=0> dnorm[len];  // dose levels
            real ym[len];               // observed mean of responses
            real <lower=0> ysd[len];    // observed stdev of responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_g[2];                // prior for g
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=pwr_lbound> g;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            g ~ uniform(p_g[1], p_g[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len){
                target += (-n[i]*log(sigma^2)*0.5-(n[i]*(ym[i]-log(a*exp(b*dnorm[i]^g)))^2+(n[i]-1)*ysd[i]^2)/(2*sigma^2));
            }
        }
    """

    LATEX_EQUATION = r'$f(dose) = a \times e^{b \times dose^g}$'  # noqa

    def get_priors(self):
        if self.response_direction == 1:
            b_prior = np.array([0., 50.])
        else:
            b_prior = np.array([-50., 0.])

        return {
            'p_a': np.array([0., self.get_prior_upper()]),
            'p_b': b_prior,
            'p_g': np.array([0., 15.]),
            'p_sig': np.array([0., 2.5]),
        }

    def get_settings(self):
        pwr_lbound = self.kwargs.get('pwr_lbound', 1.)
        if pwr_lbound < 0. or pwr_lbound > 1.:
            raise ValueError('Invalid pwr_lbound: {}'.format(pwr_lbound))
        return {
            'pwr_lbound': pwr_lbound,
        }

    def get_response_vector(self, a, b, g, doses):
        return a * np.exp(b * doses ** g)

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        g = self.parameters['g']
        sigma = self.parameters['sigma']

        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], g[i], doses))
                y_post_pred = np.random.normal(mean_posterior, sigma[i])
                predicted[i] = -2. * self.likelihoodI(mean_posterior, y_post_pred, sigma[i])
                observed[i] = -2. * self.likelihoodI(mean_posterior, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], g[i], dnorm))
                mean_pred = np.empty(dnorm.size)
                std_pred = np.empty(dnorm.size)
                for j in xrange(dnorm.size):
                    resp_ind_pred = np.random.normal(mean_posterior[j], sigma[i], ns[j])
                    mean_pred[j] = np.average(resp_ind_pred)
                    std_pred[j] = np.std(resp_ind_pred)
                predicted[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    mean_pred, std_pred, ns)
                observed[i] = -2. * self.likelihoodC(
                        mean_posterior, sigma[i],
                        resp_mean_ln, resp_std_ln, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        g = samples[2, :]
        sigma = samples[3, :]

        predicted = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], g[i], doses))
                predicted[i] = self.likelihoodI(resp, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], g[i], dnorm))
                predicted[i] = self.likelihoodC(
                    resp, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return self.get_response_vector(kw['a'], kw['b'], kw['g'], x)

    def get_control_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        g = self.parameters['g']
        return self.get_response_vector(a, b, g, 0.)

    def calc_central_tendency(self, cutoff):
        a = self.parameters['a']
        b = self.parameters['b']
        g = self.parameters['g']
        return np.exp(np.log(np.log(cutoff / a) / b) / g)

    def added_risk(self, bmr):
        return 1.


class Exponential4(Continuous):

    PARAMETERS = ('a', 'b', 'c', 'sigma')

    STAN_INDIVIDUAL = """
        data{
            int <lower=0> len;          // number of dose points
            real <lower=0> dnorm[len];  // dose levels
            real <lower=0> y[len];      // observed responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_c[2];                // prior for c
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real <lower=0> b;
            real <lower=0> c;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            c ~ uniform(p_c[1], p_c[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len)
                log(y[i]) ~ normal(log(a*(c-(c-1)*exp(-1*b*dnorm[i]))), sigma);
        }
    """

    STAN_SUMMARY = """
        data{
            int <lower=0> len;          // number of dose groups
            int <lower=0> n[len];       // number of subjects in each dose group
            real <lower=0> dnorm[len];  // dose levels
            real ym[len];               // observed mean of responses
            real <lower=0> ysd[len];    // observed stdev of responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_c[2];                // prior for c
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real <lower=0> b;
            real <lower=0> c;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            c ~ uniform(p_c[1], p_c[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len){
                target += (-n[i]*log(sigma^2)*0.5-(n[i]*(ym[i]-log(a*(c-(c-1)*exp(-1*b*dnorm[i]))))^2+(n[i]-1)*ysd[i]^2)/(2*sigma^2));
            }
        }
    """

    LATEX_EQUATION = r'$f(dose) = a \times [c-(c-1) \times e^{-b \times dose}]$'  # noqa

    def get_priors(self):
        c_prior = np.array([1., 15.]) \
            if self.response_direction == 1 \
            else np.array([0., 1.])

        return {
            'p_a': np.array([0., self.get_prior_upper()]),
            'p_b': np.array([0., 100.]),
            'p_c': c_prior,
            'p_sig': np.array([0., 2.5]),
        }

    def get_response_vector(self, a, b, c, doses):
        return a * (c - (c - 1) * np.exp(-1. * b * doses))

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        sigma = self.parameters['sigma']

        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], c[i], doses))
                y_post_pred = np.random.normal(mean_posterior, sigma[i])
                predicted[i] = -2. * self.likelihoodI(mean_posterior, y_post_pred, sigma[i])
                observed[i] = -2. * self.likelihoodI(mean_posterior, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], c[i], dnorm))
                mean_pred = np.empty(dnorm.size)
                std_pred = np.empty(dnorm.size)
                for j in xrange(dnorm.size):
                    resp_ind_pred = np.random.normal(mean_posterior[j], sigma[i], ns[j])
                    mean_pred[j] = np.average(resp_ind_pred)
                    std_pred[j] = np.std(resp_ind_pred)
                predicted[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    mean_pred, std_pred, ns)
                observed[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]
        sigma = samples[3, :]

        predicted = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], c[i], doses))
                predicted[i] = self.likelihoodI(resp, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], c[i], dnorm))
                predicted[i] = self.likelihoodC(
                    resp, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return self.get_response_vector(kw['a'], kw['b'], kw['c'], x)

    def get_control_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return self.get_response_vector(a, b, c, 0.)

    def calc_central_tendency(self, cutoff):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return -1. * np.log((c - cutoff / a)/(c - 1)) / b

    def added_risk(self, bmr):
        return 1.


class Exponential5(Continuous):

    PARAMETERS = ('a', 'b', 'c', 'g', 'sigma')

    STAN_INDIVIDUAL = """
        data{
            int <lower=0> len;          // number of dose points
            real pwr_lbound;            // restraint value
            real <lower=0> dnorm[len];  // dose levels
            real <lower=0> y[len];      // observed responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_c[2];                // prior for c
            real p_g[2];                // prior for g
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real <lower=0> b;
            real <lower=0> c;
            real <lower=pwr_lbound> g;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            c ~ uniform(p_c[1], p_c[2]);
            g ~ uniform(p_g[1], p_g[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len)
              log(y[i]) ~ normal(log(a*(c-(c-1)*exp(-1*(b*dnorm[i])^g))), sigma);
        }
    """

    STAN_SUMMARY = """
        data{
            int <lower=0> len;          // number of dose groups
            real pwr_lbound;            // restraint value
            int <lower=0> n[len];       // number of subjects in each dose group
            real <lower=0> dnorm[len];  // dose levels
            real ym[len];               // observed mean of responses
            real <lower=0> ysd[len];    // observed stdev of responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_c[2];                // prior for c
            real p_g[2];                // prior for g
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real <lower=0> b;
            real <lower=0> c;
            real <lower=pwr_lbound> g;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            c ~ uniform(p_c[1], p_c[2]);
            g ~ uniform(p_g[1], p_g[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len){
               target += (-n[i]*log(sigma^2)*0.5-(n[i]*(ym[i]-log(a*(c-(c-1)*exp(-1*(b*dnorm[i])^g))))^2+(n[i]-1)*ysd[i]^2)/(2*sigma^2));
            }
        }
    """

    LATEX_EQUATION = r'$f(dose) = a \times [c-(c-1) \times e^{-(b \times dose)^g}]$'  # noqa

    def get_priors(self):
        return {
            'p_a': np.array([0., self.get_prior_upper()]),
            'p_b': np.array([0., 100.]),
            'p_c': np.array([0., 15.]),
            'p_g': np.array([0., 15.]),
            'p_sig': np.array([0., 2.5]),
        }

    def get_settings(self):
        pwr_lbound = self.kwargs.get('pwr_lbound', 1.)
        if pwr_lbound < 0. or pwr_lbound > 1.:
            raise ValueError('Invalid pwr_lbound: {}'.format(pwr_lbound))
        return {
            'pwr_lbound': pwr_lbound,
        }

    def get_response_vector(self, a, b, c, g, doses):
        return a * (c - (c - 1) * np.exp(-1. * (b * doses) ** g))

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']
        sigma = self.parameters['sigma']

        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], c[i], g[i], doses))
                y_post_pred = np.random.normal(mean_posterior, sigma[i])
                predicted[i] = -2. * self.likelihoodI(mean_posterior, y_post_pred, sigma[i])
                observed[i] = -2. * self.likelihoodI(mean_posterior, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], c[i], g[i], dnorm))
                mean_pred = np.empty(dnorm.size)
                std_pred = np.empty(dnorm.size)
                for j in xrange(dnorm.size):
                    resp_ind_pred = np.random.normal(mean_posterior[j], sigma[i], ns[j])
                    mean_pred[j] = np.average(resp_ind_pred)
                    std_pred[j] = np.std(resp_ind_pred)
                predicted[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    mean_pred, std_pred, ns)
                observed[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]
        g = samples[3, :]
        sigma = samples[4, :]

        predicted = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], c[i], g[i], doses))
                predicted[i] = self.likelihoodI(resp, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], c[i], g[i], dnorm))
                predicted[i] = self.likelihoodC(
                    resp, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return self.get_response_vector(kw['a'], kw['b'], kw['c'], kw['g'], x)

    def get_control_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']
        return self.get_response_vector(a, b, c, g, 0.)

    def calc_central_tendency(self, cutoff):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']
        return np.exp(np.log(-1 * np.log((c - cutoff / a) / (c - 1))) / g) / b

    def added_risk(self, bmr):
        return 1.


class Hill(Continuous):

    PARAMETERS = ('a', 'b', 'c', 'g', 'sigma')

    STAN_INDIVIDUAL = """
        data{
            int <lower=0> len;          // number of dose points
            real pwr_lbound;            // restraint value
            real <lower=0> dnorm[len];  // dose levels
            real <lower=0> y[len];      // observed responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_c[2];                // prior for c
            real p_g[2];                // prior for g
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=0> c;
            real <lower=pwr_lbound> g;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            c ~ uniform(p_c[1], p_c[2]);
            g ~ uniform(p_g[1], p_g[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len)
                log(y[i]) ~ normal(log(a+b*dnorm[i]^g/(c^g+dnorm[i]^g)), sigma);
        }
    """

    STAN_SUMMARY = """
        data{
            int <lower=0> len;          // number of dose groups
            real pwr_lbound;            // restraint value
            int <lower=0> n[len];       // number of subjects in each dose group
            real <lower=0> dnorm[len];  // dose levels
            real ym[len];               // observed mean of responses
            real <lower=0> ysd[len];    // observed stdev of responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_c[2];                // prior for c
            real p_g[2];                // prior for g
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=0> c;
            real <lower=pwr_lbound> g;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            c ~ uniform(p_c[1], p_c[2]);
            g ~ uniform(p_g[1], p_g[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len){
               target += (-n[i]*log(sigma^2)*0.5-(n[i]*(ym[i]-log(a+b*dnorm[i]^g/(c^g+dnorm[i]^g)))^2+(n[i]-1)*ysd[i]^2)/(2*sigma^2));
            }
        }
    """

    LATEX_EQUATION = r'$f(dose) = a + \frac{b \times dose^g}{c^g + dose^g}$'  # noqa

    def get_priors(self):
        return {
            'p_a': np.array([0., self.get_prior_upper()]),
            'p_b': self.get_prior_slope(),
            'p_c': np.array([0., 15.]),
            'p_g': np.array([0., 15.]),
            'p_sig': np.array([0., 2.5]),
        }

    def get_settings(self):
        pwr_lbound = self.kwargs.get('pwr_lbound', 1.)
        if pwr_lbound < 0. or pwr_lbound > 1.:
            raise ValueError('Invalid pwr_lbound: {}'.format(pwr_lbound))
        return {
            'pwr_lbound': pwr_lbound,
        }

    def get_response_vector(self, a, b, c, g, doses):
        return np.array(a + b * doses ** g / (c ** g + doses ** g))

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']
        sigma = self.parameters['sigma']

        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], c[i], g[i], doses))
                y_post_pred = np.random.normal(mean_posterior, sigma[i])
                predicted[i] = -2. * self.likelihoodI(mean_posterior, y_post_pred, sigma[i])
                observed[i] = -2. * self.likelihoodI(mean_posterior, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], c[i], g[i], dnorm))
                mean_pred = np.empty(dnorm.size)
                std_pred = np.empty(dnorm.size)
                for j in xrange(dnorm.size):
                    resp_ind_pred = np.random.normal(mean_posterior[j], sigma[i], ns[j])
                    mean_pred[j] = np.average(resp_ind_pred)
                    std_pred[j] = np.std(resp_ind_pred)
                predicted[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    mean_pred, std_pred, ns)
                observed[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]
        g = samples[3, :]
        sigma = samples[4, :]

        predicted = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], c[i], g[i], doses))
                predicted[i] = self.likelihoodI(resp, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], c[i], g[i], dnorm))
                predicted[i] = self.likelihoodC(resp, sigma[i], resp_mean_ln, resp_std_ln, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return self.get_response_vector(kw['a'], kw['b'], kw['c'], kw['g'], x)

    def get_control_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']
        return self.get_response_vector(a, b, c, g, 0.)

    def calc_central_tendency(self, cutoff):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']
        return np.exp(np.log(((cutoff - a) * c ** g) / (a + b - cutoff)) / g)

    def added_risk(self, bmr):
        return 1.


class Power(Continuous):

    PARAMETERS = ('a', 'b', 'g', 'sigma')

    STAN_INDIVIDUAL = """
        data{
            int <lower=0> len;          // number of dose points
            real pwr_lbound;            // restraint value
            real <lower=0> dnorm[len];  // dose levels
            real <lower=0> y[len];      // observed responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_g[2];                // prior for g
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=pwr_lbound> g;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            g ~ uniform(p_g[1], p_g[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len)
                log(y[i]) ~ normal(log(a+b*dnorm[i]^g), sigma);
        }
    """

    STAN_SUMMARY = """
        data{
            int <lower=0> len;          // number of dose groups
            real pwr_lbound;            // restraint value
            int <lower=0> n[len];       // number of subjects in each dose group
            real <lower=0> dnorm[len];  // dose levels
            real ym[len];               // observed mean of responses
            real <lower=0> ysd[len];    // observed stdev of responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_g[2];                // prior for g
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=pwr_lbound> g;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            g ~ uniform(p_g[1], p_g[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len){
               target += (-n[i]*log(sigma^2)*0.5-(n[i]*(ym[i]-log(a+b*dnorm[i]^g))^2+(n[i]-1)*ysd[i]^2)/(2*sigma^2));
            }
        }
    """

    LATEX_EQUATION = r'$f(dose) = a + b \times dose^g$'  # noqa

    def get_priors(self):
        return {
            'p_a': np.array([0., self.get_prior_upper()]),
            'p_b': self.get_prior_slope(),
            'p_g': np.array([0., 15.]),
            'p_sig': np.array([0., 2.5]),
        }

    def get_settings(self):
        pwr_lbound = self.kwargs.get('pwr_lbound', 1.)
        if pwr_lbound < 0. or pwr_lbound > 1.:
            raise ValueError('Invalid pwr_lbound: {}'.format(pwr_lbound))
        return {
            'pwr_lbound': pwr_lbound,
        }

    def get_response_vector(self, a, b, g, doses):
        return np.array(a + b * doses ** g)

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        g = self.parameters['g']
        sigma = self.parameters['sigma']

        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], g[i], doses))
                y_post_pred = np.random.normal(mean_posterior, sigma[i])
                predicted[i] = -2. * self.likelihoodI(mean_posterior, y_post_pred, sigma[i])
                observed[i] = -2. * self.likelihoodI(mean_posterior, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], g[i], dnorm))
                mean_pred = np.empty(dnorm.size)
                std_pred = np.empty(dnorm.size)
                for j in xrange(dnorm.size):
                    resp_ind_pred = np.random.normal(mean_posterior[j], sigma[i], ns[j])
                    mean_pred[j] = np.average(resp_ind_pred)
                    std_pred[j] = np.std(resp_ind_pred)
                predicted[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    mean_pred, std_pred, ns)
                observed[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        g = samples[2, :]
        sigma = samples[3, :]

        predicted = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], g[i], doses))
                predicted[i] = self.likelihoodI(resp, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], g[i], dnorm))
                predicted[i] = self.likelihoodC(resp, sigma[i], resp_mean_ln, resp_std_ln, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return self.get_response_vector(kw['a'], kw['b'], kw['g'], x)

    def get_control_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        g = self.parameters['g']
        return self.get_response_vector(a, b, g, 0.)

    def calc_central_tendency(self, cutoff):
        a = self.parameters['a']
        b = self.parameters['b']
        g = self.parameters['g']
        return np.exp(np.log((cutoff - a) / b) / g)


class MichaelisMenten(Continuous):

    PARAMETERS = ('a', 'b', 'c', 'sigma')

    STAN_INDIVIDUAL = """
        data{
            int <lower=0> len;          // number of dose points
            real <lower=0> dnorm[len];  // dose levels
            real <lower=0> y[len];      // observed responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_c[2];                // prior for c
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=0> c;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            c ~ uniform(p_c[1], p_c[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len)
                log(y[i]) ~ normal(log(a+b*dnorm[i]/(c+dnorm[i])), sigma);
        }
    """

    STAN_SUMMARY = """
        data{
            int <lower=0> len;          // number of dose groups
            int <lower=0> n[len];       // number of subjects in each dose group
            real <lower=0> dnorm[len];  // dose levels
            real ym[len];               // observed mean of responses
            real <lower=0> ysd[len];    // observed stdev of responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_c[2];                // prior for c
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=0> c;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            c ~ uniform(p_c[1], p_c[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len){
              target += (-n[i]*log(sigma^2)*0.5-(n[i]*(ym[i]-log(a+b*dnorm[i]/(c+dnorm[i])))^2+(n[i]-1)*ysd[i]^2)/(2*sigma^2));
            }
        }
    """

    LATEX_EQUATION = r'$f(dose) = a + \frac{b \times dose}{c + dose}$'  # noqa

    def get_priors(self):
        return {
            'p_a': np.array([0., self.get_prior_upper()]),
            'p_b': self.get_prior_slope(),
            'p_c': np.array([0., 15.]),
            'p_sig': np.array([0., 2.5]),
        }

    def get_response_vector(self, a, b, c, doses):
        return np.array(a + b * doses / (c + doses))

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        sigma = self.parameters['sigma']

        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], c[i], doses))
                y_post_pred = np.random.normal(mean_posterior, sigma[i])
                predicted[i] = -2 * self.likelihoodI(mean_posterior, y_post_pred, sigma[i])
                observed[i] = -2 * self.likelihoodI(mean_posterior, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], c[i], dnorm))
                mean_pred = np.empty(dnorm.size)
                std_pred = np.empty(dnorm.size)
                for j in xrange(dnorm.size):
                    resp_ind_pred = np.random.normal(mean_posterior[j], sigma[i], ns[j])
                    mean_pred[j] = np.average(resp_ind_pred)
                    std_pred[j] = np.std(resp_ind_pred)
                predicted[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    mean_pred, std_pred, ns)
                observed[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]
        sigma = samples[3, :]

        predicted = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], c[i], doses))
                predicted[i] = self.likelihoodI(resp, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], c[i], dnorm))
                predicted[i] = self.likelihoodC(resp, sigma[i], resp_mean_ln, resp_std_ln, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return self.get_response_vector(kw['a'], kw['b'], kw['c'], x)

    def get_control_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return self.get_response_vector(a, b, c, 0.)

    def calc_central_tendency(self, cutoff):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return (cutoff - a) * c / (a + b - cutoff)


class Linear(Continuous):

    PARAMETERS = ('a', 'b', 'sigma')

    STAN_INDIVIDUAL = """
        data{
            int <lower=0> len;          // number of dose points
            real <lower=0> dnorm[len];  // dose levels
            real <lower=0> y[len];      // observed responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len)
                log(y[i]) ~ normal(log(a+b*dnorm[i]), sigma);
        }
    """

    STAN_SUMMARY = """
        data{
            int <lower=0> len;          // number of dose groups
            int <lower=0> n[len];       // number of subjects in each dose group
            real <lower=0> dnorm[len];  // dose levels
            real ym[len];               // observed mean of responses
            real <lower=0> ysd[len];    // observed stdev of responses
            real p_a[2];                // prior for a
            real p_b[2];                // prior for b
            real p_sig[2];              // prior for sig
        }
        parameters{
            real <lower=0> a;
            real b;
            real <lower=0> sigma;
        }
        model{
            a ~ uniform(p_a[1], p_a[2]);
            b ~ uniform(p_b[1], p_b[2]);
            sigma ~ cauchy(p_sig[1], p_sig[2]);
            for (i in 1:len){
                target += (-n[i]*log(sigma^2)*0.5-(n[i]*(ym[i]-log(a+b*dnorm[i]))^2+(n[i]-1)*ysd[i]^2)/(2*sigma^2));
            }
        }
    """

    LATEX_EQUATION = r'$f(dose) = a + b \times dose$'  # noqa

    def get_priors(self):
        return {
            'p_a': np.array([0., self.get_prior_upper()]),
            'p_b': self.get_prior_slope(),
            'p_sig': np.array([0., 2.5]),
        }

    def get_response_vector(self, a, b, doses):
        return np.array(a + b * doses)

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        sigma = self.parameters['sigma']

        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], doses))
                y_post_pred = np.random.normal(mean_posterior, sigma[i])
                predicted[i] = -2 * self.likelihoodI(mean_posterior, y_post_pred, sigma[i])
                observed[i] = -2 * self.likelihoodI(mean_posterior, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                mean_posterior = np.log(self.get_response_vector(a[i], b[i], dnorm))
                mean_pred = np.empty(dnorm.size)
                std_pred = np.empty(dnorm.size)
                for j in xrange(dnorm.size):
                    resp_ind_pred = np.random.normal(mean_posterior[j], sigma[i], ns[j])
                    mean_pred[j] = np.average(resp_ind_pred)
                    std_pred[j] = np.std(resp_ind_pred)
                predicted[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    mean_pred, std_pred, ns)
                observed[i] = -2. * self.likelihoodC(
                    mean_posterior, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        sigma = samples[2, :]

        predicted = np.zeros(a.size, dtype=np.float64)

        if self.is_individual_dataset:
            doses = self.data['dnorm']
            resps = self.data['y']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], doses))
                predicted[i] = self.likelihoodI(resp, np.log(resps), sigma[i])
        else:
            dnorm = self.data['dnorm']
            ns = self.data['n']
            resp_mean_ln = self.data['ym']
            resp_std_ln = self.data['ysd']

            for i in xrange(a.size):
                resp = np.log(self.get_response_vector(a[i], b[i], dnorm))
                predicted[i] = self.likelihoodC(
                    resp, sigma[i],
                    resp_mean_ln, resp_std_ln, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return self.get_response_vector(kw['a'], kw['b'], x)

    def get_control_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        return self.get_response_vector(a, b, 0.)

    def calc_central_tendency(self, cutoff):
        a = self.parameters['a']
        b = self.parameters['b']
        return (cutoff - a) / b
