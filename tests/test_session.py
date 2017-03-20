import pytest
import json
from datetime import datetime
from bbmd import Session
from bbmd.models import dichotomous as dmodels

import numpy as np


def test_defaults():
    session = Session()
    assert session.mcmc_iterations == 20000
    assert session.mcmc_num_chains == 2
    assert session.mcmc_warmup_fraction == 0.5
    assert session.seed == 12345
    now = datetime.now()
    assert str(now.year) in session.name and \
        str(now.day) in session.name


def test_name():
    name = 'my name'
    session = Session(name=name)
    assert session.name == name


def test_dichotomous():
    session = Session()
    session.add_dichotomous_data(
        dose=[0, 1, 2],
        n=[2, 2, 2],
        incidence=[1, 2, 2])
    assert session.dataset['len'] == 3

    with pytest.raises(ValueError):
        session.add_dichotomous_data(
            dose=1,
            n=1,
            incidence=1)

    with pytest.raises(ValueError):
        session.add_dichotomous_data(
            dose=[1, 2],
            n=[1, 2],
            incidence=[1, 2, 3])

    with pytest.raises(ValueError):
        session.add_dichotomous_data(
            dose=[1, 2],
            n=[1, 2],
            incidence=[1, 3])


def test_continuous_summary():
    session = Session()
    session.add_continuous_summary_data(
        dose=[0, 10, 50, 150, 400],
        n=[111, 142, 143, 93, 42],
        response=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdev=[0.235, 0.209, 0.231, 0.263, 0.159]
    )
    assert session.dataset['individual'] == session.SUMMARY


def test_continuous_individual():
    session = Session()
    session.add_continuous_individual_data(
        dose=[0, 0, 0, 100, 100, 100],
        response=[0.4, 0.7, 0.2, 13.1, 16.2, 18.5],
    )
    assert session.dataset['individual'] == session.INDIVIDUAL


def test_bmr_ranges():
    session = Session()
    session.add_continuous_summary_data(
        dose=[0, 23000],
        n=[10, 8],
        response=[29.3, 24.5],
        stdev=[2.53, 1.58],
    )
    domains = session.get_bmr_adversity_value_domains()
    expected = """{"absolute_change_domain": [0.0, 17.5], "bmr_domain": [0.0001, 0.5], "cutoff_domain": [10.275, 29.300000000000001], "cutoff_domain_hybrid": [21.185975603682031, 29.190746024001704], "is_increasing": false, "quantile_domain": [0.0001, 0.4999], "relative_change_domain": [0.0, 0.59726962457337884]}"""
    assert json.dumps(domains, sort_keys=True) == expected

    session = Session()
    session.add_continuous_summary_data(
        dose=[0, 23000],
        n=[10, 8],
        response=[10., 15.],
        stdev=[2., 3.],
    )
    domains = session.get_bmr_adversity_value_domains()
    expected = """{"absolute_change_domain": [0.0, 35.0], "bmr_domain": [0.0001, 0.5], "cutoff_domain": [10.0, 45.0], "cutoff_domain_hybrid": [9.806293547070343, 20.480986022709423], "is_increasing": true, "quantile_domain": [0.5001, 0.9999], "relative_change_domain": [0.0, 3.5]}"""
    assert json.dumps(domains, sort_keys=True) == expected


def test_empty_execute():
    session = Session(mcmc_iterations=2000)
    session.add_dichotomous_data(
        dose=[0, 1.96, 5.69, 29.75],
        n=[75, 49, 50, 49],
        incidence=[5, 1, 3, 14]
    )
    with pytest.raises(ValueError):
        session.execute()


def test_basic_dichotomous():
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
        dmodels.LogLogistic(),
    )
    session.execute()
    weights = [m.model_weight_scaler for m in session.models]
    expected = [0.62553842799869863, 0.37446157200130137]
    assert np.all(np.isclose(weights, expected))
    assert np.isclose(sum(weights), 1.)


def test_threaded():
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
        dmodels.LogLogistic(),
        dmodels.Logistic(),
        dmodels.LogLogistic(),
    )
    # check that both run
    session.execute(pythreads=True)
    session.execute(pythreads=False)
