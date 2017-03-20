import cPickle
import inspect
import os
import multiprocessing
import shutil

from pystan import StanModel

import models.dichotomous as dmodels
import models.continuous as cmodels


def compile_stan(silent=False, parallel=True):
    # Compile each PyStan model and save output as pickle binary serialization.
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, 'models', 'compiled')

    if not silent and os.path.exists(path) and len(os.listdir(path)) > 0:
        resp = raw_input('Delete pre-compiled models and re-compile [y or n]? ')  # noqa
        if resp.lower() != 'y':
            return

    # delete path and re-recreate
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)

    models = []

    # get dichotomous models for compilation
    parent = dmodels.Dichotomous
    for name in dir(dmodels):
        obj = dmodels.__dict__[name]
        if inspect.isclass(obj) and issubclass(obj, parent) and obj != parent:
            models.append((obj.STAN_MODEL, obj.get_precompiled_path()))

    # get continuous models for compilation
    parent = cmodels.Continuous
    for name in dir(cmodels):
        obj = cmodels.__dict__[name]
        if inspect.isclass(obj) and issubclass(obj, parent) and obj != parent:
            models.append((obj.STAN_INDIVIDUAL, obj.get_precompiled_path(parent.INDIVIDUAL)))
            models.append((obj.STAN_SUMMARY, obj.get_precompiled_path(parent.SUMMARY)))

    # compile each model
    if parallel:
        ncores = max(multiprocessing.cpu_count()-1, 1)
        p = multiprocessing.Pool(ncores)
        p.map(compile_code, models)
    else:
        for model in models:
            compile_code(model)


def compile_code(data):
    source, output = data
    compiled = StanModel(model_code=source)
    with open(output, 'wb') as f:
        cPickle.dump(compiled, f, protocol=2)
    return True
