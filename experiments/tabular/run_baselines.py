"""The compared methods on the tabular benchmarks (CC18 and TabArena), at their defaults and tuned with an equal budget.

One line = one method, one mode, one dataset and one seed, run over every outer fold of the benchmark (CC18: 5
stratified folds shuffled with the seed; TabArena: the task's own splits, seed 33), on exactly the data FLARE sees
(flare.preprocessing.preprocess_fold: the 80% training part of the fold, real columns standardised; categorical columns
one-hot encoded on top, fitted on those rows; TabArena categoricals with more than 32 categories replaced by their
effect on the class, as for FLARE's baselines).

Tuned mode: Optuna TPE, `--trials` configurations per fold (30), each fitted on 80% of the training part and scored by
macro-F1 on the other 20% (seeded with seed + fold); at most `--tune_timeout` seconds per fold; the best configuration
is refitted on the whole training part. See baselines/tuning.py.

    python -m experiments.tabular.run_baselines --benchmark cc18 --method XGBoost --mode tuned --dataset diabetes --seed 33 --out results/baselines
    python -m experiments.tabular.run_baselines manifest --benchmark cc18 --out results/baselines     # writes the job list
    python -m experiments.tabular.run_baselines --list results/baselines/jobs/cc18.txt --index 7 --out results/baselines

Output: <out>/rows/<method>_<mode>_<dataset>_seed<S>.csv, one row per fold (TabArena datasets are prefixed with
"tabarena-"); a line whose file is complete and error-free is not run again. The method column is <method> for the
defaults and <method>_tuned for tuned runs. A line that cannot run here (the authors' code not cloned, an environment
variable of CLASSY / GRANDE unset, a package missing) is skipped with a message and writes nothing.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

from baselines import tabular, workers
from baselines.external import repo_path
from baselines.tuning import fit_predict, macro_f1, tune
from experiments.data import openml as data
from flare.preprocessing import preprocess_fold

TABARENA_SEED = 33          # TabArena's repeats already resample the splits
EFFECT_ORDER = 32           # TabArena: categoricals with more than 32 training categories are effect-ordered
ALL_METHODS = tabular.METHODS + tabular.INTERPRETABLE + tabular.REFERENCES
# method -> clone of the authors' code it needs / module it imports in this environment
CLONES = {'RRL': 'rrl', 'RL-Net': 'RLNet', 'DR-Net': 'DRNet', 'NeuRules': 'neurules'}
MODULES = {'XGBoost': 'xgboost', 'LightGBM': 'lightgbm', 'CatBoost': 'catboost', 'RIPPER': 'wittgenstein', 'CORELS': 'corels',
           'FIGS': 'imodels', 'RuleFit': 'imodels', 'EBM': 'interpret'}


def unavailable(name):
    """Why `name` cannot run here, or None."""
    if name in CLONES:
        try:
            repo_path(CLONES[name])
        except ImportError as error:
            return str(error)
    if name in workers.INTERPRETER and workers.interpreter(name) is None:
        return f'set {workers.INTERPRETER[name]} to the Python of its environment (baselines/README.md)'
    if name in MODULES and importlib.util.find_spec(MODULES[name]) is None:
        return f'the package {MODULES[name]} is not installed (pip install -e ".[baselines]")'
    return None


def one_hot(X_train, X_test, categorical):
    """One-hot encode the categorical columns, fitted on the training rows; real columns first, then the indicators."""
    from sklearn.preprocessing import OneHotEncoder
    if not categorical.any():
        return X_train, X_test
    encoder = OneHotEncoder(handle_unknown='ignore').fit(X_train[:, categorical])
    return (np.hstack([X_train[:, ~categorical], encoder.transform(X_train[:, categorical]).toarray()]),
            np.hstack([X_test[:, ~categorical], encoder.transform(X_test[:, categorical]).toarray()]))


def fold_arrays(X, y, categorical, train_idx, test_idx, seed, effect_order=0):
    """X_train, y_train, X_test, y_test of one outer fold, as every baseline receives them."""
    X_train, X_test = data.impute(X.iloc[train_idx], X.iloc[test_idx], categorical)
    fold = preprocess_fold(X_train, y.iloc[train_idx], X_test, y.iloc[test_idx], categorical, seed, effect_order=effect_order)
    # Rounded to float32, FLARE's own precision, so the baselines see the same numbers as FLARE (this matters for
    # thresholds and ties in trees and rule learners).
    X_train, X_test = one_hot(fold.X_train.astype(np.float32), fold.X_test.astype(np.float32), fold.categorical)
    return (np.asarray(X_train, dtype=np.float64), np.asarray(fold.y_train).astype(int),
            np.asarray(X_test, dtype=np.float64), np.asarray(fold.y_test).astype(int))


def scores(y_train, p_train, y_test, p_test):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    return dict(train_acc=accuracy_score(y_train, p_train), train_f1=macro_f1(y_train, p_train),
                test_acc=accuracy_score(y_test, p_test), test_f1=macro_f1(y_test, p_test),
                test_bal_acc=balanced_accuracy_score(y_test, p_test))


def probability_scores(y, P):
    """ROC AUC (binary: class 1; multiclass: one-vs-rest macro over the classes present) and log-loss, TabArena's metrics."""
    from sklearn.metrics import log_loss, roc_auc_score
    P = np.clip(np.asarray(P, dtype=float), 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    present = np.unique(y)
    if P.shape[1] == 2:
        auc = roc_auc_score(y, P[:, 1]) if len(present) == 2 else np.nan
    else:
        Q = P[:, present] / P[:, present].sum(axis=1, keepdims=True)
        auc = roc_auc_score(y, Q, multi_class='ovr', average='macro', labels=present) if len(present) > 2 else np.nan
    return dict(test_roc_auc=float(auc), test_logloss=float(log_loss(y, P, labels=list(range(P.shape[1])))))


def grande(mode, seed, X, y, X_test, trials, timeout):
    """GRANDE (its own environment): trained on the 80% of the rows of the tuning split, which it also uses to stop."""
    tr, va = data.tuning_split(np.arange(len(y)), y, seed)
    with tempfile.TemporaryDirectory() as tmp:
        r = workers.run('GRANDE', tmp, dict(X_train=X[tr], y_train=y[tr], X_val=X[va], y_val=y[va], X_full=X, X_test=X_test),
                        mode=mode, seed=seed, trials=trials, timeout=timeout)
        return (r['pred_full'], r['pred_test'], float(r['fit_seconds']), json.loads(str(r['params'])), int(r['n_trials']),
                float(r['val_f1']), r['proba_test'])


def classy(mode, seed, tune_seed, X, y, X_test, trials, timeout):
    """CLASSY (its own environment) on one fold."""
    with tempfile.TemporaryDirectory() as tmp:
        r = workers.run('CLASSY', tmp, dict(X=X, y=y, X_test=X_test), mode=mode, seed=seed, tune_seed=tune_seed,
                        trials=trials, timeout=timeout)
        return (r['pred_full'], r['pred_test'], float(r['fit_seconds']), json.loads(str(r['params'])), int(r['n_trials']),
                float(r['val_f1']), float(r['complexity']))


def run_one(benchmark, name, mode, dataset, seed, trials=30, timeout=3600, out='results/baselines', max_folds=None):
    """One line of the job list: the rows of every outer fold, written to <out>/rows and returned (None if skipped)."""
    tabarena = benchmark == 'tabarena'
    reason = unavailable(name)
    if reason:
        print(f'[skipped] {name}: {reason}', flush=True)
        return None              # main() exits with code 3, so a job list does not count the line as done
    X, y, categorical, _, folds = data.load(benchmark, dataset)
    folds = folds(seed)[:max_folds]
    # A tuning budget other than the default (30 trials, one hour) is part of the file name, so a smoke test never
    # stands in for a real run.
    budget = '' if mode == 'default' else ('' if trials == 30 else f'_trials{trials}') + ('' if timeout == 3600 else f'_timeout{timeout}')
    path = os.path.join(out, 'rows', f"{name}_{mode}_{'tabarena-' if tabarena else ''}{dataset}_seed{seed}{budget}.csv")
    if os.path.exists(path):
        done = pd.read_csv(path)
        if len(done) == len(folds) and ('error' not in done or done['error'].isna().all()):
            print(f'[reused] {path}', flush=True)
            return done
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_classes = int(y.max()) + 1
    rows = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        row = dict(dataset=dataset, collection=benchmark, fold=fold, seed=seed, method=name if mode == 'default' else f'{name}_tuned',
                   base_method=name, mode=mode, n_trials=0, tune_seconds=0.0, val_f1=np.nan, params='{}', complexity=np.nan)
        start = time.time()
        try:
            X_train, y_train, X_test, y_test = fold_arrays(X, y, categorical, train_idx, test_idx, seed,
                                                           EFFECT_ORDER if tabarena else 0)
            if name == 'CLASSY':
                p_train, p_test, fit_s, params, n, val, size = classy(mode, seed, seed + fold, X_train, y_train, X_test, trials, timeout)
                row.update(fit_time=fit_s, n_trials=n, val_f1=val, params=json.dumps(params), complexity=size,
                           tune_seconds=time.time() - start - fit_s if mode == 'tuned' else 0.0)
            elif name == 'GRANDE':
                p_train, p_test, fit_s, params, n, val, P = grande(mode, seed + fold, X_train, y_train, X_test, trials, timeout)
                row.update(fit_time=fit_s, n_trials=n, val_f1=val, params=json.dumps(params), tune_seconds=time.time() - start - fit_s)
                if tabarena:                                # GRANDE's own probabilities, padded to every class
                    Q = np.zeros((len(P), n_classes))
                    Q[:, :P.shape[1]] = P
                    row.update(probability_scores(y_test, Q), own_probabilities=True)
            else:
                params = dict(tabular.DEFAULTS[name])
                if mode == 'tuned':
                    params, n, val = tune(name, seed + fold, X_train, y_train, trials, timeout)
                    row.update(n_trials=n, val_f1=val, tune_seconds=time.time() - start)
                (p_train, p_test), model, fit_s = fit_predict(name, params, seed, X_train, y_train, [X_train, X_test])
                row.update(fit_time=fit_s, params=json.dumps(params, default=str), complexity=tabular.n_rules(name, model))
                if tabarena:
                    P, own = tabular.probabilities(model, X_test, n_classes)
                    row.update(probability_scores(y_test, P), own_probabilities=own)
            if tabarena and 'test_logloss' not in row:      # CLASSY's worker returns labels: one-hot of them
                row.update(probability_scores(y_test, np.eye(n_classes)[np.asarray(p_test).astype(int)]), own_probabilities=False)
            row.update(scores(y_train, p_train, y_test, p_test))
            print(f"[{fold + 1}/{len(folds)}] {name} {mode} {dataset} seed={seed} test_f1={row['test_f1']:.4f} "
                  f"trials={row['n_trials']} seconds={time.time() - start:.0f}", flush=True)
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'[:300]
            print(f"[{fold + 1}/{len(folds)}] FAILED {name} {mode} {dataset} seed={seed}: {row['error']}", flush=True)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def datasets(benchmark):
    """CC18: smallest first (cc18.json); TabArena: by training size times features, smallest first."""
    if benchmark == 'cc18':
        return data.cc18_names()
    tasks = data.tabarena_manifest()
    return sorted(tasks, key=lambda d: (tasks[d]['n_samples_train_per_fold'] * tasks[d]['n_features'], d))


def manifest(benchmark, methods, modes, out):
    """Write the job list <out>/jobs/<benchmark>.txt: 'benchmark method mode dataset seed' per line."""
    seeds = data.SEEDS if benchmark == 'cc18' else (TABARENA_SEED,)
    lines = [f'{benchmark} {m} {mode} {d} {s}\n' for s in seeds for d in datasets(benchmark) for m in methods for mode in modes]
    os.makedirs(os.path.join(out, 'jobs'), exist_ok=True)
    path = os.path.join(out, 'jobs', f'{benchmark}.txt')
    with open(path, 'w') as f:
        f.writelines(lines)
    print(f'{len(lines)} lines ({len(methods)} methods x {len(modes)} modes x {len(datasets(benchmark))} datasets x '
          f'{len(seeds)} seeds) -> {path}')


def main(argv=None):
    warnings.filterwarnings('ignore')
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='results/baselines')
    if argv[:1] == ['manifest']:
        parser.add_argument('command', choices=['manifest'])
        parser.add_argument('--benchmark', required=True, choices=['cc18', 'tabarena'])
        parser.add_argument('--methods', nargs='+', default=None, choices=ALL_METHODS,
                            help='default: every method for CC18; for TabArena the rule learners and trees it lacks')
        parser.add_argument('--modes', nargs='+', default=None, choices=['default', 'tuned'],
                            help='default: both for CC18, default only for TabArena')
        args = parser.parse_args(argv)
        tabarena = args.benchmark == 'tabarena'
        return manifest(args.benchmark, args.methods or (tabular.TABARENA_METHODS if tabarena else ALL_METHODS),
                        args.modes or (['default'] if tabarena else ['default', 'tuned']), args.out)
    parser.add_argument('--benchmark', choices=['cc18', 'tabarena'])
    parser.add_argument('--method', choices=ALL_METHODS)
    parser.add_argument('--mode', choices=['default', 'tuned'])
    parser.add_argument('--dataset')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--list', help='job list written by `manifest`; with --index runs that line (1-based)')
    parser.add_argument('--index', type=int)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--tune_timeout', type=int, default=3600, help='seconds of tuning per fold')
    parser.add_argument('--max_folds', type=int, default=None, help='only the first folds (smoke tests)')
    args = parser.parse_args(argv)
    if args.list:
        with open(args.list) as f:
            benchmark, method, mode, dataset, seed = f.readlines()[args.index - 1].split()
        args.benchmark, args.method, args.mode, args.dataset, args.seed = benchmark, method, mode, dataset, int(seed)
    if None in (args.benchmark, args.method, args.mode, args.dataset, args.seed):
        parser.error('give --benchmark --method --mode --dataset --seed, or --list with --index')
    frame = run_one(args.benchmark, args.method, args.mode, args.dataset, args.seed, args.trials, args.tune_timeout, args.out,
                    args.max_folds)
    if frame is None:
        sys.exit(3)                                         # skipped: the method cannot run here
    sys.exit(0 if 'error' not in frame or frame['error'].isna().all() else 1)


if __name__ == '__main__':
    main()
