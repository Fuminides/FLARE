"""Table 1 (results on the two tabular benchmarks) and the paired comparisons.

    python -m experiments.paper.table1 --results results --out results/tables

reads every `rows/*.csv` below the folders given to --results (the output of experiments/tabular/run_flare.py,
tune_flare.py and run_baselines.py) and writes, in --out:

    table1.csv        one row per method: CC18 macro-F1 (%), its std over the seeds, mean rank and its interval;
                      TabArena ROC AUC (30 binary tasks), log-loss (8 multi-class tasks), mean rank and its interval
    cc18_paired.csv   FLARE against each method on CC18: datasets both finished, W/T/L (tie margin 0.5 points), Wilcoxon p
    tabarena_paired.csv  the same on TabArena, on the error of each task (1 - ROC AUC or log-loss), W/L
    cc18_per_dataset.csv, tabarena_per_task_error.csv   the per-dataset scores behind them

Which configuration of each method is shown:
* CC18: the tuned one for every method (30 configurations or one hour per fold, see baselines/tuning.py and
  experiments/tabular/tune_flare.py), except FLARE-explainability, which keeps its fixed configuration.
* TabArena: every method at its defaults, seed 33, the task's own splits. FLARE reports its probabilities recalibrated
  on its validation rows. The gradient-boosted trees, random forest, EBM and logistic regression are the results that
  TabArena published for its own default configurations (`_c1_BAG_L1`), downloaded from TabArena's public artifacts
  and cached in $FLARE_DATA_DIR/tabarena_results.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.data import openml as data
from experiments.paper.common import (complete_mean, expected_cc18, expected_tabarena, rank_interval, rank_matrix,
                                      read_rows, seed_std)

# Display name -> the method name in the result rows (CC18: the shown configuration).
METHODS = ['FLARE', 'FLARE-explainability', 'GRANDE', 'CART', 'FIGS', 'RuleFit', 'RRL', 'CLASSY', 'RIPPER', 'DR-Net',
           'RL-Net', 'CORELS', 'NeuRules', 'GradTree', 'CatBoost', 'LightGBM', 'XGBoost', 'Random forest', 'EBM',
           'Logistic regression']
ROW_NAME = {'Random forest': 'RF', 'Logistic regression': 'LR'}
CC18_NAME = {m: ('FLARE-explainability' if m == 'FLARE-explainability' else
                 'FLARE_tuned' if m == 'FLARE' else ROW_NAME.get(m, m) + '_tuned') for m in METHODS}
# TabArena's own results for these methods (its names), default configuration, bagged.
PUBLISHED = {'CatBoost': 'CatBoost', 'LightGBM': 'LightGBM', 'XGBoost': 'XGBoost', 'Random forest': 'RandomForest',
             'EBM': 'ExplainableBM', 'Logistic regression': 'LinearModel'}
PUBLISHED_URL = ('https://tabarena.s3.amazonaws.com/cache/artifacts/tabarena-2025-06-12/methods/{}/results/'
                 'model_results.parquet')


def published(name):
    """TabArena's published per-split results of one of its methods (default configuration): dataset, fold, error."""
    import requests
    folder = os.path.join(data.DATA_DIR, 'tabarena_results')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + '.parquet')
    if not os.path.exists(path):
        response = requests.get(PUBLISHED_URL.format(name), timeout=120)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)
    d = pd.read_parquet(path)
    return d[d.method.str.endswith('_c1_BAG_L1')]     # `fold` already combines TabArena's repeat and fold


def cc18_table(rows, flare='FLARE_tuned'):
    """Per-dataset macro-F1 (%) of the 20 methods on CC18 and the std over the seeds. `flare`: the rows shown as FLARE
    (FLARE_tuned in Table 1; FLARE, its default configuration, in the scalability figure)."""
    rows = rows[rows.collection == 'cc18'].assign(score=lambda r: r.test_f1 * 100)
    wanted = {v: k for k, v in {**CC18_NAME, 'FLARE': flare}.items()}
    rows = rows[rows.method.isin(wanted)].assign(method=lambda r: r.method.map(wanted))
    per = complete_mean(rows, 'score', expected_cc18()).reindex(columns=METHODS)
    return per, seed_std(rows, 'score', per)


def tabarena_errors(rows):
    """Per task and method, the error that TabArena ranks with: 1 - ROC AUC on binary tasks, log-loss on the others."""
    tasks = data.tabarena_manifest()
    rows = rows[rows.collection == 'tabarena'].copy()
    flare = rows.method.isin(['FLARE', 'FLARE-explainability'])
    # FLARE is scored with its probabilities recalibrated on the validation rows (the predicted classes do not change).
    column = lambda name: rows[name] if name in rows else np.nan
    rows['auc'] = np.where(flare, column('test_roc_auc_cal'), column('test_roc_auc'))
    rows['logloss'] = np.where(flare, column('test_logloss_cal'), column('test_logloss'))
    ours = {ROW_NAME.get(m, m): m for m in METHODS if m not in PUBLISHED}
    rows = rows[rows.method.isin(ours)].assign(method=lambda r: r.method.map(ours))
    frames = [rows[['dataset', 'seed', 'fold', 'method', 'auc', 'logloss']]]
    for display, name in PUBLISHED.items():
        d = published(name)
        frames.append(pd.DataFrame({'dataset': d.dataset, 'seed': 33, 'fold': d['fold'], 'method': display,
                                    'auc': 1 - d.metric_error, 'logloss': d.metric_error}))
    d = pd.concat(frames, ignore_index=True)
    d = d[d.dataset.isin(tasks)]
    binary = d.dataset.map(lambda t: tasks[t]['problem_type'] == 'binary')
    d['error'] = np.where(binary, 1 - d.auc, d.logloss)
    return complete_mean(d, 'error', expected_tabarena()).reindex(columns=METHODS)


def paired(per, lower_is_better, margin):
    out = []
    for m in METHODS[1:]:
        diff = (per[m] - per.FLARE if lower_is_better else per.FLARE - per[m]).dropna()   # > 0: FLARE is better
        if len(diff) == 0:
            continue
        out.append(dict(method=m, n=len(diff), W=int((diff > margin).sum()), T=int((diff.abs() <= margin).sum()),
                        L=int((diff < -margin).sum()), p=wilcoxon(diff).pvalue if (diff != 0).any() else 1.0))
    return pd.DataFrame(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', nargs='+', default=['results'])
    parser.add_argument('--out', default='results/tables')
    parser.add_argument('--no_tabarena', action='store_true', help='only the CC18 columns')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = read_rows(args.results)

    cc, std = cc18_table(rows)
    table = pd.DataFrame(index=METHODS)
    table['CC18 macro-F1'] = cc.mean()
    table['CC18 std'] = std
    table['CC18 datasets'] = cc.notna().sum()
    table['CC18 rank'] = rank_matrix(cc, ascending=False).mean()
    table['CC18 rank low'], table['CC18 rank high'] = rank_interval(cc, ascending=False)
    cc.to_csv(os.path.join(args.out, 'cc18_per_dataset.csv'))
    paired(cc, False, 0.5).to_csv(os.path.join(args.out, 'cc18_paired.csv'), index=False)

    if not args.no_tabarena:
        errors = tabarena_errors(rows)
        binary = [t for t, v in data.tabarena_manifest().items() if v['problem_type'] == 'binary']
        table['ROC AUC'] = 1 - errors.reindex(binary).mean()
        table['Log-loss'] = errors.drop(index=binary).mean()
        table['TabArena tasks'] = errors.notna().sum()
        table['TabArena rank'] = rank_matrix(errors, ascending=True).mean()
        table['TabArena rank low'], table['TabArena rank high'] = rank_interval(errors, ascending=True)
        errors.to_csv(os.path.join(args.out, 'tabarena_per_task_error.csv'))
        paired(errors, True, 0.0).drop(columns='T').to_csv(os.path.join(args.out, 'tabarena_paired.csv'), index=False)

    table.to_csv(os.path.join(args.out, 'table1.csv'))
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(table.round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1:])
