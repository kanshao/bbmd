import os
from io import BytesIO
from bbmd import Session
from bbmd.models import dichotomous as dmodels, \
                        continuous as cmodels
from bbmd.bmr import dichotomous as dbmr, \
                     continuous as cbmr


def setup_dich_session():
    session = Session(mcmc_iterations=2000)
    session.add_dichotomous_data(
        dose=[0, 1.96, 5.69, 29.75],
        n=[75, 49, 50, 49],
        incidence=[5, 1, 3, 14]
    )
    session.add_models(
        dmodels.Logistic(),
        dmodels.LogLogistic(),
    )
    session.add_bmrs(
        dbmr.Extra(bmr=0.1),
        dbmr.Added(bmr=0.1)
    )

    session.execute()
    session.calculate_bmrs()
    return session


def setup_cs_session():
    session = Session(mcmc_iterations=2000)
    session.add_continuous_summary_data(
        dose=[0, 10, 50, 150, 400],
        n=[111, 142, 143, 93, 42],
        response=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdev=[0.235, 0.209, 0.231, 0.263, 0.159]
    )
    session.add_models(
        cmodels.Linear(),
        cmodels.MichaelisMenten()
    )
    session.add_bmrs(
        cbmr.HybridControlPercentileExtra(bmr=0.1, adversity_value=0.1),
    )

    session.execute()
    session.calculate_bmrs()
    return session


def setup_ci_session():
    session = Session(mcmc_iterations=2000)
    session.add_continuous_individual_data(
        dose=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600,],
        response=[14.71, 13.19, 14.11, 14.12, 13.8, 11.59, 12.65, 14.21, 13.69, 13.21, 15.22, 14.07, 15.18, 15.31, 15.61, 14.38, 14.1, 13.38, 14.08, 13.6, 14.03, 12.75, 13.34, 12.81, 13.35, 16.32, 14.8, 14.3, 13.53, 13.54, 15.36, 15.06, 14.35, 14.46, 16.32, 12.45, 13.91, 13.48, 14.38, 13.45, 15.24, 16.21, 17.19, 19.4, 18.76, 22.74, 16.95, 20.06, 22.93, 19.74, 25.69, 26.58, 23.73, 28.53, 30.25, 29.7, 22.63, 29.43, 25.22, 30.51,],
    )
    session.add_models(
        cmodels.Linear(),
        cmodels.MichaelisMenten()
    )
    session.add_bmrs(
        cbmr.HybridControlPercentileExtra(bmr=0.1, adversity_value=0.6),
    )

    session.execute()
    session.calculate_bmrs()
    return session


def test_dich_export_report():
    session = setup_dich_session()
    export_report_checks(session)


def test_cs_export_report():
    session = setup_cs_session()
    export_report_checks(session)


def test_ci_export_report():
    session = setup_ci_session()
    export_report_checks(session)


def export_report_checks(session):
    fn = os.path.expanduser('~/Desktop/report.txt')
    session.export_report(fn)
    assert os.path.exists(fn)
    os.remove(fn)

    fn = os.path.expanduser('~/Desktop/report.json')
    session.export_report(fn, format='json')
    assert os.path.exists(fn)
    os.remove(fn)


def test_export_flatfiles():
    session = setup_dich_session()

    fn = os.path.expanduser('~/Desktop/parameters.txt')
    session.export_parameters(fn)
    assert os.path.exists(fn)
    os.remove(fn)

    fn = os.path.expanduser('~/Desktop/bmd.txt')
    session.export_bmds(fn)
    assert os.path.exists(fn)
    os.remove(fn)


def test_word_report():
    session = Session(mcmc_iterations=2000)
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

    # ensure file-object can be created w/ report
    f = BytesIO()
    session.export_word_report(f)

    # ensure file can also be created
    fn = os.path.expanduser('~/Desktop/_test_report.docx')
    if os.path.exists(fn):
        os.remove(fn)
    session.export_word_report(fn)
    assert os.path.exists(fn)
    os.remove(fn)
