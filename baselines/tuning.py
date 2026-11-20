"""Fitting and tuning one baseline on one fold matrix; shared by experiments/tabular/run_baselines.py and by the
worker that runs CLASSY in its own environment (this module needs only numpy and scikit-learn; Optuna and the data
helpers are imported when tuning starts).

Tuning is the same for every method: Optuna's TPE sampler draws `trials` configurations from
`baselines.tabular.space`; each is fitted on 80% of the training rows and scored by macro-F1 on the other 20%
(`tuning_split`, stratified); `timeout` seconds stop the search early. A configuration that cannot be fitted scores 0.
"""
import time

import numpy as np

from baselines import tabular


def macro_f1(y, pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, labels=np.unique(y), average='macro', zero_division=0))


def fit_predict(name, params, seed, X, y, Xs):
    """Fit `name` with `params` on (X, y): predictions on each matrix of Xs, the fitted model and the fit time (s)."""
    start = time.time()
    model = tabular.build(name, params, seed, len(np.unique(y))).fit(X, y)
    seconds = time.time() - start
    return [model.predict(Z) for Z in Xs], model, seconds


def tune(name, seed, X, y, trials, timeout):
    """(best configuration of `name`, trials done, its validation macro-F1). `seed` seeds the split, the sampler and
    the fits of the trials."""
    import optuna
    from experiments.data.openml import tuning_split
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr, va = tuning_split(np.arange(len(y)), y, seed)

    def objective(trial):
        params = tabular.space(name, trial)
        try:
            (pred,), _, _ = fit_predict(name, params, seed, X[tr], y[tr], [X[va]])
            return macro_f1(y[va], pred)
        except Exception as error:                      # a configuration the method cannot fit scores 0
            trial.set_user_attr('error', f'{type(error).__name__}: {error}'[:200])
            return 0.0
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, timeout=timeout)
    best = study.best_trial
    params = tabular.space(name, optuna.trial.FixedTrial(best.params))
    return params, len(study.trials), best.value
