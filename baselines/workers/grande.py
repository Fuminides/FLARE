'''GRANDE (Marton et al., ICLR 2024, `GRANDE` 0.1.6) on one fold inside its own environment.

GRANDE is a TensorFlow package, so experiments/tabular/run_baselines.py runs it in a subprocess of the interpreter
named in $FLARE_GRANDE_PYTHON (baselines/README.md lists the packages). Reads <dir>/in.npz (X_train, y_train, X_val,
y_val, X_full, X_test), fits at the package's defaults or tunes with Optuna on the validation rows (GRANDE stops early
on them, so the 80 / 20 split is kept for the final fit), and writes <dir>/out.npz (predictions on X_full and X_test,
class probabilities on X_test, fit seconds, the configuration, trials done, validation macro-F1).

    $FLARE_GRANDE_PYTHON baselines/workers/grande.py --dir D --mode tuned --seed 33
'''
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import sklearn.preprocessing  # noqa: F401  GRANDE 0.1.6 calls sklearn.preprocessing without importing it
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repository root
from baselines import tabular as M  # noqa: E402

# GRANDE's default configuration, with class weights on.
DEFAULT = dict(depth=5, n_estimators=2048, learning_rate_weights=0.005, learning_rate_index=0.01, learning_rate_values=0.01,
               learning_rate_leaf=0.01, dropout=0.0, selected_variables=0.8, use_class_weights=True)


def fit(params, seed, X, y, X_val, y_val, n_classes):
    from GRANDE import GRANDE
    p = dict(params)
    config = dict(depth=p['depth'], n_estimators=p['n_estimators'], learning_rate_weights=p['learning_rate_weights'],
                  learning_rate_index=p['learning_rate_index'], learning_rate_values=p['learning_rate_values'],
                  learning_rate_leaf=p['learning_rate_leaf'], optimizer='adam', cosine_decay_steps=0, loss='crossentropy',
                  focal_loss=False, temperature=0.0, from_logits=True, use_class_weights=p['use_class_weights'],
                  dropout=p['dropout'], selected_variables=p['selected_variables'], data_subset_fraction=1.0)
    args = dict(epochs=1000, early_stopping_epochs=25, batch_size=64, cat_idx=[], random_seed=seed, verbose=0,
                objective='binary' if n_classes == 2 else 'classification')
    model = GRANDE(params=config, args=args)
    cols = [f'x{j}' for j in range(X.shape[1])]
    model.fit(X_train=pd.DataFrame(X, columns=cols), y_train=pd.Series(y), X_val=pd.DataFrame(X_val, columns=cols), y_val=pd.Series(y_val))

    def proba(Z):
        P = np.asarray(model.predict(pd.DataFrame(Z, columns=cols)), dtype=float)
        return P if P.ndim == 2 else np.column_stack([1 - P, P])

    def predict(Z):
        return proba(Z).argmax(axis=1)
    predict.proba = proba
    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--mode', choices=['default', 'tuned'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=3600)
    a = parser.parse_args()
    d = np.load(os.path.join(a.dir, 'in.npz'))
    classes, y = np.unique(np.concatenate([d['y_train'], d['y_val']]), return_inverse=True)
    y_train, y_val = y[:len(d['y_train'])], y[len(d['y_train']):]
    K, n_trials, params = len(classes), 0, DEFAULT
    if a.mode == 'tuned':
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            try:
                predict = fit(M.space('GRANDE', trial), a.seed, d['X_train'], y_train, d['X_val'], y_val, K)
                return float(f1_score(y_val, predict(d['X_val']), labels=np.unique(y_val), average='macro', zero_division=0))
            except Exception:
                return 0.0
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=a.seed))
        study.optimize(objective, n_trials=a.trials, timeout=a.timeout)
        params, n_trials = M.space('GRANDE', optuna.trial.FixedTrial(study.best_trial.params)), len(study.trials)
    start = time.time()
    predict = fit(params, a.seed, d['X_train'], y_train, d['X_val'], y_val, K)
    seconds = time.time() - start
    val = float(f1_score(y_val, predict(d['X_val']), labels=np.unique(y_val), average='macro', zero_division=0))
    P = np.zeros((len(d['X_test']), int(classes.max()) + 1))               # probabilities of the original labels
    P[:, classes] = predict.proba(d['X_test'])
    np.savez(os.path.join(a.dir, 'out.npz'), pred_full=classes[predict(d['X_full'])], pred_test=classes[predict(d['X_test'])], proba_test=P,
             fit_seconds=seconds, params=json.dumps(params), n_trials=n_trials, val_f1=val)


if __name__ == '__main__':
    main()
