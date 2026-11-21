"""FLARE on the tabular benchmarks (CC18 and TabArena): one dataset, one configuration and one seed, over every outer fold.

CC18: 5 stratified folds shuffled with the seed (seeds 33, 34 and 35). TabArena: the task's own outer splits (seed 33). Per fold: the training rows are imputed and split 80 / 20 into training
and validation rows, the real variables are standardised (flare.preprocessing), FLARE is trained, and the deployed
(discrete) model is scored on the test rows. On TabArena (--benchmark tabarena) FLARE also learns category spaces
for categoricals of more than 32 categories and reports probability metrics recalibrated on the validation rows.

    python -m experiments.tabular.run_flare --benchmark cc18 --dataset diabetes --config flare --seed 33 --out results/flare
    python -m experiments.tabular.run_flare --benchmark tabarena --dataset anneal --config flare-explainability --out results/flare
    python -m experiments.tabular.run_flare ... --gumbel         # Gumbel-softmax selectors instead of gradient grafting
    python -m experiments.tabular.run_flare ... --fold 0         # only that fold (a smoke test)
    python -m experiments.tabular.run_flare ... --config flare-explainability --set rules=20 single_condition_rules=15 \
        --name "FLARE-20 rules"                                  # another size

Output: <out>/rows/<method>_<dataset>_seed<S>.csv, one row per fold (TabArena datasets are prefixed with "tabarena-"),
and the printed rules of every fold in <out>/rules/<method>_<dataset>_seed<S>/rules_fold<k>.txt. `method` is FLARE or
FLARE-explainability (plus _gumbel). A file that is complete is not run again.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from experiments.data import openml as data
from flare.config import PRESETS, parse_options
from flare.preprocessing import preprocess_fold
from flare.printing import rules_text
from flare.train import fit_fold, score_fold

METHODS = {'flare': 'FLARE', 'flare-explainability': 'FLARE-explainability'}


def make_config(config, benchmark, gumbel=False, overrides=None):
    return PRESETS[config](tabarena=benchmark == 'tabarena', gumbel=gumbel, **(overrides or {}))


def run_fold(X, y, categorical, names, train_idx, test_idx, config, seed, fold, **fit_options):
    """Train FLARE on the rows `train_idx` of the dataset and score it on `test_idx`. `seed` is the seed of the run: it
    fixes the split of the training rows; the initial values and the batch order use seed + fold. Returns (the results
    row, the printed rules, the train.FitResult). `fit_options` go to train.fit_fold."""
    X_train, X_test = data.impute(X.iloc[train_idx], X.iloc[test_idx], categorical)
    fold_data = preprocess_fold(X_train, y.iloc[train_idx], X_test, y.iloc[test_idx], categorical, seed,
                                category_space=config.category_space_min if config.category_space > 0 else 0)
    # The number of categories comes from the whole dataset: it sets the size of the selectors.
    cardinalities = [int(X.iloc[:, j].nunique()) for j in np.flatnonzero(fold_data.categorical)]
    result = fit_fold(fold_data, config, cardinalities, random_seed=seed + fold, **fit_options)
    row = score_fold(result)
    row['route'] = result.route
    categories = names.get('categories', {})
    category_names = {j: categories[X.columns[j]] for j in fold_data.category_space if X.columns[j] in categories}
    return row, rules_text(result.model, feat_names=list(X.columns), category_names=category_names), result


def run_one(benchmark, dataset, config_name, seed, out, gumbel=False, only_fold=None, overrides=None, name=None):
    """The rows of every outer fold (or of `only_fold`), written to <out>/rows and returned."""
    config = make_config(config_name, benchmark, gumbel, overrides)
    method = name or METHODS[config_name] + ('_gumbel' if gumbel else '')
    tabarena = benchmark == 'tabarena'
    tag = f"{method.replace(' ', '_')}_{'tabarena-' if tabarena else ''}{dataset}_seed{seed}"
    path = os.path.join(out, 'rows', tag + ('' if only_fold is None else f'_fold{only_fold}') + '.csv')
    X, y, categorical, names, folds = data.load(benchmark, dataset)
    folds = folds(seed)
    wanted = list(range(len(folds))) if only_fold is None else [only_fold]
    if os.path.exists(path) and len(pd.read_csv(path)) == len(wanted):
        print(f'[reused] {path}', flush=True)
        return pd.read_csv(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(os.path.join(out, 'rules', tag), exist_ok=True)
    rows = []
    for done, fold in enumerate(wanted, 1):
        train_idx, test_idx = folds[fold]
        start = time.time()
        row, rules, _ = run_fold(X, y, categorical, names, train_idx, test_idx, config, seed, fold)
        with open(os.path.join(out, 'rules', tag, f'rules_fold{fold}.txt'), 'w') as f:
            f.write(rules + '\n')
        rows.append({'dataset': dataset, 'collection': benchmark, 'seed': seed, 'fold': fold, 'config': config_name, 'method': method,
                     **row, 'test_f1': row['f1']})
        print(f"[{done}/{len(wanted)}] {method} {dataset} seed={seed} fold={fold} f1={row['f1']:.4f} acc={row['test_acc']:.4f} "
              f"rules={row['n_rules']} seconds={time.time() - start:.0f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--benchmark', required=True, choices=['cc18', 'tabarena'])
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', required=True, choices=sorted(PRESETS))
    parser.add_argument('--gumbel', action='store_true')
    parser.add_argument('--seed', type=int, default=data.SEEDS[0])
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='change options of the configuration (flare.config.FLAREConfig), e.g. rules=20')
    parser.add_argument('--name', default=None, help='the method name of the rows (default: FLARE or FLARE-explainability)')
    parser.add_argument('--out', default='results/flare')
    args = parser.parse_args(argv)
    if args.set and not args.name:
        parser.error('--set changes the model: give it a --name, so that its rows do not replace those of the preset')
    run_one(args.benchmark, args.dataset, args.config, args.seed, args.out, args.gumbel, args.fold,
            parse_overrides(args.set), args.name)


def parse_overrides(items):
    try:
        return parse_options(items)
    except ValueError as error:
        raise SystemExit(str(error))


if __name__ == '__main__':
    main(sys.argv[1:])
