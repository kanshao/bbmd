import pytest

from bbmd import Session
from bbmd.models import continuous as cmodels
from bbmd.models import dichotomous as dmodels


def test_pwrlbound_settings():
    Models = [
        dmodels.LogLogistic,
        dmodels.LogProbit,
        dmodels.Weibull,
        dmodels.Gamma,
        dmodels.DichotomousHill,
        cmodels.Exponential3,
        cmodels.Exponential5,
        cmodels.Power,
        cmodels.Hill,
    ]

    for Model in Models:
        with pytest.raises(ValueError):
            m = Model(pwr_lbound=-0.01)
            m.get_settings()

        with pytest.raises(ValueError):
            m = Model(pwr_lbound=1.01)
            m.get_settings()

        m = Model()
        m.get_settings()['pwr_lbound'] == 1.

        m = Model(pwr_lbound=0.)
        m.get_settings()['pwr_lbound'] == 0.


def test_name():
    m = dmodels.LogProbit(name='foo')
    assert m.get_name() == 'foo'

    m = dmodels.LogProbit()
    assert m.get_name() == 'LogProbit'


def test_extract_summary_value():
    model = dmodels.Logistic()
    summary_text = 'Inference for Stan model: anon_model_6ec9f268cf3fd9261d734b7a1232b68a.\n2 chains, each with iter=20000; warmup=10000; thin=1; \npost-warmup draws per chain=10000, total post-warmup draws=20000.\n\n       mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat\na     -1.72  2.7e-3   0.17  -2.06  -1.83  -1.71  -1.61  -1.41 3818.0    1.0\nb      1.13  4.2e-3   0.26   0.63   0.96   1.13    1.3   1.65 3856.0    1.0\nlp__ -65.81    0.02   1.01  -68.5  -66.2  -65.5 -65.08 -64.84 4253.0    1.0\n\nSamples were drawn using NUTS at Sun Aug  7 19:14:37 2016.\nFor each parameter, n_eff is a crude measure of effective sample size,\nand Rhat is the potential scale reduction factor on split chains (at \nconvergence, Rhat=1).'  # noqa
    result = model.extract_summary_values(summary_text)
    expected = {
        'a': {'25%': -1.83, '97.5%': -1.41, 'Rhat': 1.0, '50%': -1.71, '75%': -1.61, '2.5%': -2.06, 'sd': 0.17, 'n_eff': 3818.0, 'se_mean': 0.0027, 'mean': -1.72},  # noqa
        'b': {'25%': 0.96, '97.5%': 1.65, 'Rhat': 1.0, '50%': 1.13, '75%': 1.3, '2.5%': 0.63, 'sd': 0.26, 'n_eff': 3856.0, 'se_mean': 0.0042, 'mean': 1.13},  # noqa
        'lp__': {'25%': -66.2, '97.5%': -64.84, 'Rhat': 1.0, '50%': -65.5, '75%': -65.08, '2.5%': -68.5, 'sd': 1.01, 'n_eff': 4253.0, 'se_mean': 0.02, 'mean': -65.81},  # noqa
    }
    assert expected == result


@pytest.fixture
def bad_initial_value_dataset():
    return dict(
        dose=[0, 1., 2.5, 5., 10., 25., 50., 100., 200.],
        n=[11, 10, 10, 10, 10, 10, 10, 10, 10],
        response=[0.136, 0.148, 0.142, 0.141, 0.139, 0.144, 0.122, 0.101, 0.072],
        stdev=[0.009, 0.008, 0.01, 0.006, 0.005, 0.01, 0.005, 0.007, 0.007],
    )


def test_bad_initial_values(bad_initial_value_dataset):

    def assert_failure(session):
        # A runtime error is expected. This isn't ideal, but shows that
        # this test does successfully recreate a server error
        with pytest.raises(RuntimeError) as e:
            session.execute()
        assert 'Log probability evaluates to log(0), i.e. negative infinity.' in str(e.value)

    # check hill failure
    session = Session(
        mcmc_iterations=20000,
        mcmc_num_chains=2,
        mcmc_warmup_fraction=0.3,
        seed=57917,
    )
    session.add_continuous_summary_data(**bad_initial_value_dataset)
    session.add_models(
        cmodels.Hill(),
    )
    assert_failure(session)

    # check linear failure
    session = Session(
        mcmc_iterations=30000,
        mcmc_num_chains=1,
        mcmc_warmup_fraction=0.5,
        seed=80121,
    )
    session.add_continuous_summary_data(**bad_initial_value_dataset)
    session.add_models(
        cmodels.Linear(),
    )
    assert_failure(session)
