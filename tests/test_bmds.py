from bbmd import Session
from bbmd.models import dichotomous as dmodels, continuous as cmodels
from bbmd.bmr import dichotomous as dbmr, continuous as cbmr

import numpy as np


def test_default_bmr_names():
    session = Session()
    session.add_bmrs(
        dbmr.Added(bmr=0.5),
        dbmr.Extra(bmr=0.5),
        cbmr.CentralTendencyRelativeChange(adversity_value=0.5),
        cbmr.CentralTendencyAbsoluteChange(adversity_value=0.5),
        cbmr.CentralTendencyCutoff(adversity_value=0.5),
        cbmr.HybridControlPercentileExtra(bmr=0.1, adversity_value=0.5),
        cbmr.HybridControlPercentileAdded(bmr=0.1, adversity_value=0.5),
        cbmr.HybridAbsoluteCutoffExtra(bmr=0.1, adversity_value=0.5),
        cbmr.HybridAbsoluteCutoffAdded(bmr=0.1, adversity_value=0.5),
    )
    names = [
        'Added (bmr=0.5)',
        'Extra (bmr=0.5)',
        'Central Tendency Relative Change (relative change=0.5)',
        'Central Tendency Absolute Change (absolute change=0.5)',
        'Central Tendency Cutoff (cutoff=0.5)',
        'Hybrid Control Percentile Extra (bmr=0.1, percentile=0.5)',
        'Hybrid Control Percentile Added (bmr=0.1, percentile=0.5)',
        'Hybrid Absolute Cutoff Extra (bmr=0.1, cutoff=0.5)',
        'Hybrid Absolute Cutoff Added (bmr=0.1, cutoff=0.5)',
    ]

    for i, bmr in enumerate(session.bmrs):
        assert names[i] == bmr.name


def test_dichotomous_bmrs():
    # not checking if numbers are correct, but that functionality works
    session = Session(
        mcmc_iterations=2000,
        mcmc_num_chains=4,
        mcmc_warmup_fraction=0.25,
    )
    session.add_dichotomous_data(
        dose=[0, 1.96, 5.69, 29.75],
        n=[75, 49, 50, 49],
        incidence=[5, 1, 3, 14]
    )
    session.add_models(
        dmodels.Logistic(),
    )
    session.add_bmrs(
        dbmr.Extra(bmr=0.1, priors=[1, ]),
        dbmr.Added(bmr=0.1, priors=[1, ])
    )

    session.execute()
    session.calculate_bmrs()

    # check scaler weight
    assert session.models[0].model_weight_scaler == 1.

    # check bmd medians are calculated
    medians1 = [b.model_average['stats']['p50'] for b in session.bmrs]
    medians2 = [17.887098904857172, 18.403145098571546]
    assert np.all(np.isclose(medians1, medians2))


def test_bmr_priors():
    # Check that modifying BMR priors give expected results.
    session = Session(
        mcmc_iterations=2000,
        mcmc_num_chains=2,
        mcmc_warmup_fraction=0.25,
    )
    session.add_dichotomous_data(
        dose=[0, 1.96, 5.69, 29.75],
        n=[75, 49, 50, 49],
        incidence=[5, 1, 3, 14]
    )
    session.add_models(
        dmodels.Logistic(),
        dmodels.Probit(),
    )
    session.add_bmrs(
        dbmr.Extra(bmr=0.1, priors=[0, 1]),
        dbmr.Extra(bmr=0.1, priors=[0.5, 0.5]),
        dbmr.Extra(bmr=0.1, priors=[3, 1]),
    )

    session.execute()
    session.calculate_bmrs()

    # get model weights using a non-informative prior
    expected_weights = [0.51190026721838455, 0.48809973278161539]
    model_weights = [m.model_weight_scaler for m in session.models]
    assert np.isclose(model_weights, expected_weights).all()

    # when using a zero weight, posterior should also be zero
    bmr = session.bmrs[0]
    assert np.isclose(bmr.priors, [0., 1.]).all()
    assert np.isclose(bmr.get_model_posterior_weights(session.models),
                      [0., 1.]).all()

    # when using equal weights, posterior should be same as model weights
    bmr = session.bmrs[1]
    assert np.isclose(bmr.get_model_posterior_weights(session.models),
                      model_weights).all()

    # when using non-equal weights, ensure posterior are different and move
    # in the expected direction using the weights. Also check to make sure the
    # priors are normalized correctly.
    bmr = session.bmrs[2]
    assert np.isclose(bmr.priors, [0.75, 0.25]).all()
    assert np.isclose(bmr.get_model_posterior_weights(session.models),
                      [0.70457014, 0.29542986]).all()


def test_continuous_bmrs():
    # not checking if numbers are correct, but that functionality works
    session = Session(
        mcmc_iterations=2000,
        mcmc_num_chains=4,
        mcmc_warmup_fraction=0.25,
    )
    session.add_continuous_summary_data(
        dose=[0, 10, 50, 150, 400],
        n=[111, 142, 143, 93, 42],
        response=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdev=[0.235, 0.209, 0.231, 0.263, 0.159]
    )
    session.add_models(
        cmodels.Exponential2(),
        cmodels.Exponential3(),
        cmodels.Exponential4(),
        cmodels.Exponential5(),
        cmodels.Hill(),
        cmodels.Power(),
        cmodels.MichaelisMenten(),
        cmodels.Linear(),
    )
    session.add_bmrs(
        cbmr.CentralTendencyRelativeChange(
            adversity_value=0.1,
        ),
        cbmr.CentralTendencyAbsoluteChange(
            adversity_value=0.2,
        ),
        cbmr.CentralTendencyCutoff(
            adversity_value=1.9,
        ),
        cbmr.HybridControlPercentileExtra(
            bmr=0.1,
            adversity_value=0.1,
        ),
        cbmr.HybridControlPercentileAdded(
            bmr=0.1,
            adversity_value=0.1,
        ),
        cbmr.HybridAbsoluteCutoffExtra(
            bmr=0.1,
            adversity_value=1.5,
        ),
        cbmr.HybridAbsoluteCutoffAdded(
            bmr=0.1,
            adversity_value=1.5,
        ),
    )

    session.execute()
    session.calculate_bmrs()

    # check scaler weights
    wts1 = [m.model_weight_scaler for m in session.models]
    wts2 = [
        9.7323145637911399e-09, 2.2165285739061686e-09,
        0.088963321920941252, 0.46427866596343953,
        0.42157708167497254, 4.4816125123407728e-16,
        0.025180918491801055, 2.0973462667512396e-15
    ]
    assert np.all(np.isclose(wts1, wts2))

    # check bmd medians are calculated
    medians1 = [b.model_average['stats']['p50'] for b in session.bmrs]
    medians2 = [
        60.285956373191098, 57.794779267663003,
        58.821657133451481, 34.035772527993053,
        36.252246542471724, 95.948042006071034,
        96.043001941323453
    ]
    assert np.all(np.isclose(medians1, medians2))


def test_nans():
    # assert that a BMR and model-average can be calculated
    # when there are nans in the BMR estimations
    session = Session(
        mcmc_iterations=2000,
        mcmc_num_chains=4,
        mcmc_warmup_fraction=0.25,
        seed=12345
    )
    session.add_continuous_summary_data(
        dose=[0, 1340, 2820, 5600, 11125, 23000],
        n=[10, 10, 10, 10, 10, 8],
        response=[29.3, 28.4, 27.2, 26, 25.8, 24.5],
        stdev=[2.53, 1.9, 2.53, 2.53, 2.21, 1.58],
    )
    session.add_models(
        cmodels.Linear(),
        cmodels.Hill(),
    )
    session.add_bmrs(
        cbmr.CentralTendencyRelativeChange(
            adversity_value=0.1,
        ),
    )

    session.execute()
    session.calculate_bmrs()

    # assert nans exist and errors exist
    assert session.bmrs[0].model_average['n_non_nans'] == 5614
    assert session.bmrs[0].model_average['n_total'] == 6000
