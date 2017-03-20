import numpy as np
import logging
from scipy import stats

from . import base


class Dichotomous(base.DoseResponseModel):

    def extra_risk(self, bmr):
        raise NotImplementedError('Abstract method')

    def added_risk(self, bmr):
        raise NotImplementedError('Abstract method')

    def get_input_count(self):
        return self.data['len']

    def likelihood(self, ps, ys, ns):
        ys2 = ys.copy()
        ys2[ys2 == 0] = self.ZEROISH
        ys2[ys2 == 1] = 1. - self.ZEROISH
        return np.sum(ys2 * np.log(ps) + (ns - ys2) * np.log(1. - ps))

    def get_plot_bounds(self, xs, vectors):
        for i in xrange(xs.size):
            resps = self.get_response_values(xs[i], **self.parameters)
            vectors[i, :] = (
                xs[i],
                np.percentile(resps, 5.),
                np.percentile(resps, 50.),
                np.percentile(resps, 95.),
            )
        return vectors

    def get_predicted_response_vector(self):
        raise NotImplementedError('Abstract method')

    def get_trend_test(self):
        if not hasattr(self, '_trend_z'):
            ns = self.data['n']
            cases = self.data['y']
            doses = self.data['dnorm']

            ns_sum = ns.sum()
            cases_sum = cases.sum()
            expect_case = ns * cases_sum / ns_sum
            prod_nd = doses * ns
            prod_nd2 = (doses ** 2) * ns
            test_v = (ns_sum-cases_sum) * cases_sum * \
                (ns_sum * prod_nd2.sum() - prod_nd.sum() ** 2) / \
                (ns_sum ** 3)
            prod_d_diffoe = (cases - expect_case) * doses
            test_z = prod_d_diffoe.sum() / test_v ** 0.5

            self._trend_z = test_z
            self._trend_p_value = 1 - stats.norm.cdf(test_z)

        return [self._trend_z, self._trend_p_value]

    def get_stan_model(self):
        return self.STAN_MODEL


class Logistic(Dichotomous):

    PARAMETERS = ('a', 'b')

    STAN_MODEL = """
       data {
         int<lower=0> len;          // number of dose groups
         int<lower=0> y[len];       // observed number of cases
         int<lower=0> n[len];       // number of subjects
         real<lower=0> dnorm[len];  // dose levels
         real p_a[2];               // prior for a
         real p_b[2];               // prior for b
       }
       parameters {
         real a;
         real<lower=0> b;
       }
       model {
         a ~ uniform (p_a[1], p_a[2]);
         b ~ uniform (p_b[1], p_b[2]);
         for (i in 1:len)
           y[i] ~ binomial(n[i],1/(1+exp(-a-b*dnorm[i])));
       }
    """

    LATEX_EQUATION = r'$f(dose) = \frac{1}{1+e^{-a-b \times dose}}$'  # noqa

    def get_priors(self):
        return {
            'p_a': [-50, 50],
            'p_b': [0, 100],
        }

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']

        doses = self.data['dnorm']
        ys = self.data['y']
        ns = self.data['n']
        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = (1. / (1. + np.exp(-a[i] - b[i] * doses)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            y_post_pred = np.random.binomial(ns, resp)
            predicted[i] = -2. * self.likelihood(resp, y_post_pred, ns)
            observed[i] = -2. * self.likelihood(resp, ys, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]

        doses = self.data['dnorm']
        ns = self.data['n']
        ys = self.data['y']
        predicted = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = (1. / (1. + np.exp(-a[i] - b[i] * doses)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            predicted[i] = self.likelihood(resp, ys, ns)
        return predicted

    def get_response_values(self, x, **kw):
        return 1. / (1. + np.exp(-kw['a'] - kw['b'] * x))

    def extra_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        return np.log((1-bmr)/(1+bmr*np.exp(-a)))/(-b)

    def added_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        return np.log((1-bmr-bmr/np.exp(-a))/(1+bmr+bmr*np.exp(-a)))/(-b)

    def risk_at_dose(self, dose):
        a = self.parameters['a']
        b = self.parameters['b']
        return (1. / (1. + np.exp(-a - b * dose)))


class LogLogistic(Dichotomous):

    PARAMETERS = ('a', 'b', 'c')

    STAN_MODEL = """
       data {
         int<lower=0> len;              // number of dose groups
         int<lower=0> y[len];           // observed number of cases
         int<lower=0> n[len];           // number of subjects
         real<lower=0> dno0norm[len];   // dose levels
         real pwr_lbound;               // restraint value
         real p_a[2];                   // prior for a
         real p_b[2];                   // prior for b
         real p_c[2];                   // prior for c
       }
       parameters {
         real <lower=0, upper=1> a;
         real <lower=pwr_lbound> b;
         real c;
       }
       model {
         a ~ uniform (p_a[1], p_a[2]);
         b ~ uniform (p_b[1], p_b[2]);
         c ~ uniform (p_c[1], p_c[2]);
         for (i in 1:len)
           y[i] ~ binomial(n[i],a+(1-a)/(1+exp(-c-b*log(dno0norm[i]))));
       }
    """

    LATEX_EQUATION = r'$f(dose) = a+\frac{(1-a)}{1+e^{-c-b \times \log(dose)}}$'  # noqa

    def get_priors(self):
        return {
            'p_a': [0, 1],
            'p_b': [0, 15],
            'p_c': [-5, 15],
        }

    def get_settings(self):
        pwr_lbound = self.kwargs.get('pwr_lbound', 1.)
        if pwr_lbound < 0. or pwr_lbound > 1.:
            raise ValueError('Invalid pwr_lbound: {}'.format(pwr_lbound))
        return {
            'pwr_lbound': pwr_lbound,
        }

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']

        doses = self.data['dno0norm']
        ys = self.data['y']
        ns = self.data['n']
        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i]+(1-a[i])/(1+np.exp(-c[i]-b[i]*np.log(doses))))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            y_post_pred = np.random.binomial(ns, resp)
            predicted[i] = -2. * self.likelihood(resp, y_post_pred, ns)
            observed[i] = -2. * self.likelihood(resp, ys, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        # TODO; refactor to not duplicate get_predicted_response_vector
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]

        doses = self.data['dno0norm']
        ns = self.data['n']
        ys = self.data['y']
        predicted = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i]+(1-a[i])/(1+np.exp(-c[i]-b[i]*np.log(doses))))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            predicted[i] = self.likelihood(resp, ys, ns)

        return predicted

    def get_response_values(self, x, **kw):
        if x == 0:
            x = self.ZEROISH
        return kw['a'] + (1 - kw['a']) / (1 + np.exp(-kw['c'] - kw['b'] * np.log(x)))

    def extra_risk(self, bmr):
        b = self.parameters['b']
        c = self.parameters['c']
        return np.exp((np.log(bmr / (1. - bmr)) - c) / b)

    def added_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return np.exp((np.log(bmr / (1. - a - bmr)) - c) / b)

    def risk_at_dose(self, dose):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return (a + (1 - a) / (1 + np.exp(-c - b * np.log(dose))))


class LogProbit(Dichotomous):

    PARAMETERS = ('a', 'b', 'c')

    STAN_MODEL = """
       data {
         int<lower=0> len;              // number of dose groups
         int<lower=0> y[len];           // observed number of cases
         int<lower=0> n[len];           // number of subjects
         real<lower=0> dno0norm[len];   // dose levels
         real pwr_lbound;               // restraint value
         real p_a[2];                   // prior for a
         real p_b[2];                   // prior for b
         real p_c[2];                   // prior for c
       }
       parameters {
         real <lower=0, upper=1> a;
         real <lower=pwr_lbound> b;
         real c;
       }
       model {
         a ~ uniform (p_a[1], p_a[2]);
         b ~ uniform (p_b[1], p_b[2]);
         c ~ uniform (p_c[1], p_c[2]);
         for (i in 1:len)
           y[i] ~ binomial(n[i], a + (1-a) * normal_cdf(c + b * log(dno0norm[i]), 0, 1));
       }
    """

    LATEX_EQUATION = r'$f(dose) = a + (1 - a) \times \Phi(c+b \times \log(dose))$'  # noqa

    def get_priors(self):
        return {
            'p_a': [0, 1],
            'p_b': [0, 15],
            'p_c': [-5, 15],
        }

    def get_settings(self):
        pwr_lbound = self.kwargs.get('pwr_lbound', 1.)
        if pwr_lbound < 0. or pwr_lbound > 1.:
            raise ValueError('Invalid pwr_lbound: {}'.format(pwr_lbound))
        return {
            'pwr_lbound': pwr_lbound,
        }

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']

        doses = self.data['dno0norm']
        ys = self.data['y']
        ns = self.data['n']
        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i]+(1.-a[i])*stats.norm.cdf(c[i]+b[i]*np.log(doses)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            y_post_pred = np.random.binomial(ns, resp)
            predicted[i] = -2. * self.likelihood(resp, y_post_pred, ns)
            observed[i] = -2. * self.likelihood(resp, ys, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        # TODO; refactor to not duplicate get_predicted_response_vector
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]

        doses = self.data['dno0norm']
        ns = self.data['n']
        ys = self.data['y']
        predicted = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i]+(1.-a[i])*stats.norm.cdf(c[i]+b[i]*np.log(doses)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            predicted[i] = self.likelihood(resp, ys, ns)

        return predicted

    def get_response_values(self, x, **kw):
        if x == 0:
            x = self.ZEROISH
        return kw['a'] + (1 - kw['a']) * stats.norm.cdf(kw['c'] + kw['b'] * np.log(x))

    def extra_risk(self, bmr):
        b = self.parameters['b']
        c = self.parameters['c']
        return np.exp((stats.norm.ppf(bmr) - c) / b)

    def added_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return np.exp((stats.norm.ppf(bmr / (1. - a)) - c) / b)

    def risk_at_dose(self, dose):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return (a + (1.-a) * stats.norm.cdf(c + b * np.log(dose)))


class Probit(Dichotomous):

    PARAMETERS = ('a', 'b')

    STAN_MODEL = """
       data {
         int<lower=0> len;          // number of dose groups
         int<lower=0> y[len];       // observed number of cases
         int<lower=0> n[len];       // number of subjects
         real<lower=0> dnorm[len];  // dose levels
         real p_a[2];               // prior for a
         real p_b[2];               // prior for b
       }
       parameters {
         real a;
         real<lower=0> b;
       }
       model {
         a ~ uniform (p_a[1], p_a[2]);
         b ~ uniform (p_b[1], p_b[2]);
         for (i in 1:len)
           y[i] ~ binomial(n[i],normal_cdf(a+b*dnorm[i],0,1));
       }
    """

    LATEX_EQUATION = r'$f(dose) = \Phi(a+b \times dose)$'  # noqa

    def get_priors(self):
        return {
            'p_a': [-50, 50],
            'p_b': [0, 100],
        }

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']

        doses = self.data['dnorm']
        ys = self.data['y']
        ns = self.data['n']
        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = stats.norm.cdf(a[i] + b[i] * doses)
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            y_post_pred = np.random.binomial(ns, resp)
            predicted[i] = -2. * self.likelihood(resp, y_post_pred, ns)
            observed[i] = -2. * self.likelihood(resp, ys, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]

        doses = self.data['dnorm']
        ns = self.data['n']
        ys = self.data['y']
        predicted = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = stats.norm.cdf(a[i] + b[i] * doses)
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            predicted[i] = self.likelihood(resp, ys, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return stats.norm.cdf(kw['a'] + kw['b'] * x)

    def extra_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        return (stats.norm.ppf((bmr + (1 - bmr) * stats.norm.cdf(a))) - a) / b

    def added_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        return (stats.norm.ppf(bmr + stats.norm.cdf(a)) - a) / b

    def risk_at_dose(self, dose):
        a = self.parameters['a']
        b = self.parameters['b']
        return stats.norm.cdf(a + b * dose)


class QuantalLinear(Dichotomous):

    PARAMETERS = ('a', 'b')

    STAN_MODEL = """
       data {
         int<lower=0> len;          // number of dose groups
         int<lower=0> y[len];       // observed number of cases
         int<lower=0> n[len];       // number of subjects
         real<lower=0> dnorm[len];  // dose levels
         real p_a[2];               // prior for a
         real p_b[2];               // prior for b
       }
       parameters {
         real <lower=0, upper=1> a;
         real <lower=0> b;
       }
       model {
         a ~ uniform (p_a[1], p_a[2]);
         b ~ uniform (p_b[1], p_b[2]);
         for (i in 1:len)
           y[i] ~ binomial(n[i],a+(1-a)*(1-exp(-b*dnorm[i])));
       }
    """

    LATEX_EQUATION = r'$f(dose) = a + (1 - a) \times (1 - e^{-b \times dose})$'  # noqa

    def get_priors(self):
        return {
            'p_a': [0, 1],
            'p_b': [0, 100],
        }

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']

        doses = self.data['dnorm']
        ys = self.data['y']
        ns = self.data['n']
        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i] + (1 - a[i]) * (1 - np.exp(-b[i] * doses)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            y_post_pred = np.random.binomial(ns, resp)
            predicted[i] = -2. * self.likelihood(resp, y_post_pred, ns)
            observed[i] = -2. * self.likelihood(resp, ys, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]

        doses = self.data['dnorm']
        ns = self.data['n']
        ys = self.data['y']
        predicted = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i] + (1 - a[i]) * (1 - np.exp(-b[i] * doses)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            predicted[i] = self.likelihood(resp, ys, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return kw['a'] + (1 - kw['a'])*(1 - np.exp(- kw['b'] * x))

    def extra_risk(self, bmr):
        b = self.parameters['b']
        return np.log(1-bmr)/(-b)

    def added_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        return np.log(1-bmr/(1-a))/(-b)

    def risk_at_dose(self, dose):
        a = self.parameters['a']
        b = self.parameters['b']
        return a+(1-a)*(1-np.exp(-b*dose))


class Multistage2(Dichotomous):

    PARAMETERS = ('a', 'b', 'c')

    STAN_MODEL = """
       data {
         int<lower=0> len;          // number of dose groups
         int<lower=0> y[len];       // observed number of cases
         int<lower=0> n[len];       // number of subjects
         real<lower=0> dnorm[len];  // dose levels
         real p_a[2];               // prior for a
         real p_b[2];               // prior for b
         real p_c[2];               // prior for c
       }
       parameters {
         real <lower=0, upper=1> a;
         real <lower=0> b;
         real <lower=0> c;
       }
       model {
         a ~ uniform (p_a[1], p_a[2]);
         b ~ uniform (p_b[1], p_b[2]);
         c ~ uniform (p_c[1], p_c[2]);
         for (i in 1:len)
           y[i] ~ binomial(n[i],a+(1-a)*(1-exp(-b*dnorm[i]-c*(dnorm[i]^2))));
       }
    """

    LATEX_EQUATION = r'$f(dose) = a + (1 - a) \times (1 - e^{-b \times dose -c \times dose^{2}})$'  # noqa

    def get_priors(self):
        return {
            'p_a': [0, 1],
            'p_b': [0, 100],
            'p_c': [0, 100],
        }

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']

        doses = self.data['dnorm']
        ys = self.data['y']
        ns = self.data['n']
        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i]+(1-a[i])*(1-np.exp(-b[i]*doses-c[i]*doses**2)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            y_post_pred = np.random.binomial(ns, resp)
            predicted[i] = -2. * self.likelihood(resp, y_post_pred, ns)
            observed[i] = -2. * self.likelihood(resp, ys, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]

        doses = self.data['dnorm']
        ns = self.data['n']
        ys = self.data['y']
        predicted = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i]+(1-a[i])*(1-np.exp(-b[i]*doses-c[i]*doses**2)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            predicted[i] = self.likelihood(resp, ys, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return kw['a'] + (1 - kw['a'])*(1 - np.exp(- kw['b'] * x - kw['c'] * x**2))

    def extra_risk(self, bmr):
        b = self.parameters['b']
        c = self.parameters['c']
        return (-b+np.sqrt(b**2-4*c*np.log(1-bmr)))/(2*c)

    def added_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return (-b+np.sqrt(b**2-4*c*np.log(1-bmr/(1-a))))/(2*c)

    def risk_at_dose(self, dose):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return a+(1-a)*(1-np.exp(-b*dose-c*dose**2))


class Weibull(Dichotomous):

    PARAMETERS = ('a', 'b', 'c')

    STAN_MODEL = """
       data {
         int<lower=0> len;          // number of dose groups
         int<lower=0> y[len];       // observed number of cases
         int<lower=0> n[len];       // number of subjects
         real<lower=0> dnorm[len];  // dose levels
         real pwr_lbound;           // restraint value
         real p_a[2];               // prior for a
         real p_b[2];               // prior for b
         real p_c[2];               // prior for c
       }
       parameters {
         real <lower=0, upper=1> a;
         real <lower=pwr_lbound> b;
         real <lower=0> c;
       }
       model {
         a ~ uniform (p_a[1], p_a[2]);
         b ~ uniform (p_b[1], p_b[2]);
         c ~ uniform (p_c[1], p_c[2]);
         for (i in 1:len)
            y[i] ~ binomial(n[i], a+(1-a)*(1-exp(-c*(dnorm[i])^b)));
       }
    """

    LATEX_EQUATION = r'$f(dose) = a + (1 - a) \times (1 - e^{-c \times dose^{b}})$'  # noqa

    def get_priors(self):
        return {
            'p_a': [0, 1],
            'p_b': [0, 15],
            'p_c': [0, 50],
        }

    def get_settings(self):
        pwr_lbound = self.kwargs.get('pwr_lbound', 1.)
        if pwr_lbound < 0. or pwr_lbound > 1.:
            raise ValueError('Invalid pwr_lbound: {}'.format(pwr_lbound))
        return {
            'pwr_lbound': pwr_lbound,
        }

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']

        doses = self.data['dnorm']
        ys = self.data['y']
        ns = self.data['n']
        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i]+(1-a[i])*(1-np.exp(-c[i]*(doses**b[i]))))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            y_post_pred = np.random.binomial(ns, resp)
            predicted[i] = -2. * self.likelihood(resp, y_post_pred, ns)
            observed[i] = -2. * self.likelihood(resp, ys, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]

        doses = self.data['dnorm']
        ns = self.data['n']
        ys = self.data['y']
        predicted = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i]+(1-a[i])*(1-np.exp(-c[i]*(doses**b[i]))))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            predicted[i] = self.likelihood(resp, ys, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return kw['a'] + (1 - kw['a']) * (1 - np.exp(- kw['c'] * (x**kw['b'])))

    def extra_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return np.exp(np.log(np.log((1-bmr*(1-a)-a)/(1-a))/(-c))/b)

    def added_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return np.exp(np.log(np.log((1-bmr-a)/(1-a))/(-c))/b)

    def risk_at_dose(self, dose):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return a+(1-a)*(1-np.exp(-c*(dose**b)))


class Gamma(Dichotomous):

    PARAMETERS = ('a', 'b', 'c')

    STAN_MODEL = """
       data {
         int<lower=0> len;          // number of dose groups
         int<lower=0> y[len];       // observed number of cases
         int<lower=0> n[len];       // number of subjects
         real<lower=0> dnorm[len];  // dose levels
         real pwr_lbound;           // restraint value
         real p_a[2];               // prior for a
         real p_b[2];               // prior for b
         real p_c[2];               // prior for c
       }
       parameters {
         real <lower=0,upper=1> a;
         real <lower=pwr_lbound> b;
         real <lower=0> c;
       }
       model {
         a ~ uniform (p_a[1], p_a[2]);
         b ~ uniform (p_b[1], p_b[2]);
         c ~ uniform (p_c[1], p_c[2]);
         for (i in 1:len)
           y[i] ~ binomial(n[i],a+(1-a)*gamma_cdf(c*dnorm[i],b,1));
       }
    """

    LATEX_EQUATION = r'$f(dose) = a + (1 - a) \times CumGamma(c \times dose, b)$'  # noqa

    def get_priors(self):
        return {
            'p_a': [0, 1],
            'p_b': [0, 15],
            'p_c': [0, 100],
        }

    def get_settings(self):
        pwr_lbound = self.kwargs.get('pwr_lbound', 1.)
        if pwr_lbound < 0. or pwr_lbound > 1.:
            raise ValueError('Invalid pwr_lbound: {}'.format(pwr_lbound))
        return {
            'pwr_lbound': pwr_lbound,
        }

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']

        doses = self.data['dnorm']
        ys = self.data['y']
        ns = self.data['n']
        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i] + (1 - a[i]) * stats.gamma.cdf(c[i] * doses, b[i]))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            y_post_pred = np.random.binomial(ns, resp)
            predicted[i] = -2. * self.likelihood(resp, y_post_pred, ns)
            observed[i] = -2. * self.likelihood(resp, ys, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]

        doses = self.data['dnorm']
        ns = self.data['n']
        ys = self.data['y']
        predicted = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = np.array(a[i] + (1 - a[i]) * stats.gamma.cdf(c[i] * doses, b[i]))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            predicted[i] = self.likelihood(resp, ys, ns)

        return predicted

    def get_response_values(self, x, **kw):
        return kw['a'] + (1 - kw['a']) * stats.gamma.cdf(kw['c'] * x, kw['b'])

    def extra_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return stats.gamma.ppf(bmr, b) / c

    def added_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return stats.gamma.ppf(bmr / (1 - a), b) / c

    def risk_at_dose(self, dose):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        return np.array(a + (1 - a) * stats.gamma.cdf(c * dose, b))


class DichotomousHill(Dichotomous):

    RESAMPLE_MAX_THRESHOLD = 0.05

    PARAMETERS = ('a', 'b', 'c', 'g')

    STAN_MODEL = """
       data {
         int<lower=0> len;              // number of dose groups
         int<lower=0> y[len];           // observed number of cases
         int<lower=0> n[len];           // number of subjects
         real<lower=0> dno0norm[len];   // dose levels
         real pwr_lbound;               // restraint value
         real p_a[2];                   // prior for a
         real p_b[2];                   // prior for b
         real p_c[2];                   // prior for c
         real p_g[2];                   // prior for g
       }
       parameters {
         real <lower=0, upper=1> a;
         real <lower=pwr_lbound> b;
         real c;
         real <lower=0, upper=1> g;
       }
       model {
         a ~ uniform (p_a[1], p_a[2]);
         b ~ uniform (p_b[1], p_b[2]);
         c ~ uniform (p_c[1], p_c[2]);
         g ~ uniform (p_g[1], p_g[2]);
         for (i in 1:len)
           y[i] ~ binomial(n[i], a * g + (a - a * g)/(1 + exp(-c - b * log(dno0norm[i]))));
       }
    """

    LATEX_EQUATION = r'$f(dose) = a \times g + \frac{a - a \times g}{1 + e^{-c - b \times \log(dose)}}$'  # noqa

    def get_priors(self):
        return {
            'p_a': [0, 1],
            'p_b': [0, 15],
            'p_c': [-5, 15],
            'p_g': [0, 1],
        }

    def get_settings(self):
        pwr_lbound = self.kwargs.get('pwr_lbound', 1.)
        if pwr_lbound < 0. or pwr_lbound > 1.:
            raise ValueError('Invalid pwr_lbound: {}'.format(pwr_lbound))
        return {
            'pwr_lbound': pwr_lbound,
        }

    def get_predicted_response_vector(self):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']

        doses = self.data['dno0norm']
        ys = self.data['y']
        ns = self.data['n']
        predicted = np.zeros(a.size, dtype=np.float64)
        observed = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = a[i] * g[i] + (a[i] - a[i] * g[i]) / (1 + np.exp(-c[i] - b[i] * np.log(doses)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            y_post_pred = np.random.binomial(ns, resp)
            predicted[i] = -2. * self.likelihood(resp, y_post_pred, ns)
            observed[i] = -2. * self.likelihood(resp, ys, ns)

        return predicted, observed

    def get_loglikelihood(self, samples):
        a = samples[0, :]
        b = samples[1, :]
        c = samples[2, :]
        g = samples[3, :]

        doses = self.data['dno0norm']
        ns = self.data['n']
        ys = self.data['y']
        predicted = np.zeros(a.size, dtype=np.float64)

        for i in xrange(a.size):
            resp = a[i] * g[i] + (a[i] - a[i] * g[i]) / (1 + np.exp(-c[i] - b[i] * np.log(doses)))
            resp[resp == 0] = self.ZEROISH
            resp[resp == 1] = 1. - self.ZEROISH
            predicted[i] = self.likelihood(resp, ys, ns)

        return predicted

    def get_response_values(self, x, **kw):
        if x == 0:
            x = self.ZEROISH
        return kw['a'] * kw['g'] + \
            (kw['a'] - kw['a'] * kw['g']) / \
            (1 + np.exp(-kw['c'] - kw['b'] * np.log(x)))

    def extra_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']
        return np.exp((np.log(
            (bmr - a + a * g - bmr * a * g) /
            (bmr * (a * g - 1.))) + c) / (-b))

    def added_risk(self, bmr):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']
        return np.exp((np.log((bmr - a + a * g) / (-bmr)) + c) / (-b))

    def risk_at_dose(self, dose):
        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']
        g = self.parameters['g']
        return a * g + (a - a * g) / (1 + np.exp(-c - b * np.log(dose)))
