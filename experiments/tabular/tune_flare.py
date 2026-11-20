"""Tuned FLARE: the options of flare.config.tuning_space searched per fold, with the same budget as the tuned baselines
(baselines/tuning.py).

Per outer fold: Optuna's TPE sampler (seeded with seed + fold) proposes up to `--trials` configurations (30) or runs for
`--tune_timeout` seconds (3600), whichever comes first. Each configuration trains FLARE on 80% of the training fold and
is scored by macro-F1 on the other 20% (`tuning_split`: stratified, seeded with seed + fold; the test rows of the fold are
never read). A configuration that fails scores 0. The best one is then trained on the whole training fold, as FLARE's
default run is (its own 80 / 20 split of the training rows) and scored on the test rows. The form of FLARE is fixed
(label runs, learned crossings, averaged models); the search covers its step sizes, temperature, resolution, rule
budget, number of averaged models and class weighting (flare.config.tuning_space). The selectors keep gradient grafting.

    python -m experiments.tabular.tune_flare --dataset diabetes --seed 33 --out results/flare_tuned
    python -m experiments.tabular.tune_flare --dataset diabetes --seed 33 --fold 0 --trials 3     # a smoke test

Output: <out>/rows/FLARE_tuned_<dataset>_seed<S>.csv, one row per fold: the refit's row plus n_trials, val_f1 (the best
holdout F1), tune_seconds and params (the chosen options as JSON); the trials of each fold in
<out>/trials/FLARE_tuned_<dataset>_seed<S>_fold<k>.csv. Needs Optuna (pip install -e ".[experiments]").
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

from experiments.data import openml as data
from experiments.tabular.run_flare import make_config, run_fold
from flare.config import TUNED_NAMES, tuning_space

METHOD = 'FLARE_tuned'


def tune_fold(X, y, categorical, names, train_idx, test_idx, seed, fold, base, trials=30, timeout=3600, log=None):
    """The tuned row of one outer fold, and the table of the trials."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inner_train, inner_val = data.tuning_split(train_idx, y.iloc[train_idx].to_numpy(), seed + fold)
    start = time.time()

    def objective(trial):
        params = tuning_space(trial)
        try:
            row, _, _ = run_fold(X, y, categorical, names, inner_train, inner_val, base.replace(**params), seed, fold)
        except Exception as error:                              # noqa: BLE001 - a configuration that fails scores 0
            trial.set_user_attr('error', f'{type(error).__name__}: {error}'[:300])
            return 0.0
        trial.set_user_attr('seconds', float(row['time2train']))
        return float(row['f1'])

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed + fold))
    study.optimize(objective, n_trials=trials, timeout=timeout)
    tune_seconds = time.time() - start
    params = tuning_space(optuna.trial.FixedTrial(study.best_trial.params))
    row, _, _ = run_fold(X, y, categorical, names, train_idx, test_idx, base.replace(**params), seed, fold)
    row.update(n_trials=len(study.trials), val_f1=study.best_value, tune_seconds=tune_seconds,
               params=json.dumps({TUNED_NAMES.get(k, k): v for k, v in params.items()}))
    return row, study.trials_dataframe()


def run_one(benchmark, dataset, seed, out, trials=30, timeout=3600, only_fold=None, overrides=None):
    base = make_config('flare', benchmark, overrides=overrides)
    tabarena = benchmark == 'tabarena'
    # A budget other than the default (30 trials, one hour) is part of the file name, so a smoke test never stands in for
    # a real run.
    budget = ('' if trials == 30 else f'_trials{trials}') + ('' if timeout == 3600 else f'_timeout{timeout}')
    tag = f"{METHOD}_{'tabarena-' if tabarena else ''}{dataset}_seed{seed}{budget}"
    path = os.path.join(out, 'rows', tag + ('' if only_fold is None else f'_fold{only_fold}') + '.csv')
    X, y, categorical, names, folds = data.load(benchmark, dataset)
    folds = folds(seed)
    wanted = list(range(len(folds))) if only_fold is None else [only_fold]
    if os.path.exists(path) and len(pd.read_csv(path)) == len(wanted):
        print(f'[reused] {path}', flush=True)
        return pd.read_csv(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(os.path.join(out, 'trials'), exist_ok=True)
    rows = []
    for done, fold in enumerate(wanted, 1):
        train_idx, test_idx = folds[fold]
        row, table = tune_fold(X, y, categorical, names, train_idx, test_idx, seed, fold, base, trials, timeout)
        table.to_csv(os.path.join(out, 'trials', tag + f'_fold{fold}.csv'), index=False)
        rows.append({'dataset': dataset, 'collection': benchmark, 'seed': seed, 'fold': fold, 'config': 'tuned', 'method': METHOD,
                     'base_method': 'FLARE', 'mode': 'tuned', **row, 'test_f1': row['f1']})
        print(f"[{done}/{len(wanted)}] {METHOD} {dataset} seed={seed} fold={fold} test_f1={row['f1']:.4f} holdout_f1={row['val_f1']:.4f} "
              f"trials={row['n_trials']} seconds={row['tune_seconds']:.0f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--benchmark', default='cc18', choices=['cc18', 'tabarena'])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, default=data.SEEDS[0])
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--tune_timeout', type=int, default=3600, help='seconds of tuning per fold')
    parser.add_argument('--out', default='results/flare_tuned')
    args = parser.parse_args(argv)
    run_one(args.benchmark, args.dataset, args.seed, args.out, args.trials, args.tune_timeout, args.fold)


if __name__ == '__main__':
    main(sys.argv[1:])
