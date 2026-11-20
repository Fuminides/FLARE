"""What the table and figure scripts share: reading the result rows, the per-dataset means and the mean ranks.

Every runner of experiments/ writes CSV rows with one row per outer fold and at least the columns `dataset`,
`collection` (cc18 or tabarena), `seed`, `fold`, `method` and the scores. `read_rows` collects them from one or more
result folders (any `rows/*.csv` below them).

Conventions:

* A method's score on a dataset is the mean over its folds and seeds. A dataset only counts for a method when every
  expected (seed, fold) pair finished; the benchmark mean of a method uses the datasets it finished.
* The mean rank compares one configuration per method over every dataset: a dataset a method did not finish gets the
  worst rank (the number of methods), and exact ties get their average rank.
* The interval of a mean rank is a 95% percentile bootstrap over the datasets (10,000 resamples, one draw shared by all
  the methods).
"""
import glob
import os

import numpy as np
import pandas as pd

from experiments.data import openml as data


def read_rows(folders):
    """All the result rows below the given folders (`<folder>/**/rows/*.csv`), in one DataFrame."""
    paths = sorted({p for folder in folders for p in glob.glob(os.path.join(folder, '**', 'rows', '*.csv'), recursive=True)})
    if not paths:
        raise FileNotFoundError(f'no rows/*.csv below {folders}')
    rows = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if 'error' in rows:
        rows = rows[rows['error'].isna()]
    return rows


def expected_cc18(seeds=data.SEEDS, n_folds=data.N_FOLDS):
    return {d: {(s, f) for s in seeds for f in range(n_folds)} for d in data.cc18_names()}


def expected_tabarena(seed=33):
    return {name: {(seed, f) for f in range(t['n_folds'] * t['n_repeats'])} for name, t in data.tabarena_manifest().items()}


def complete_mean(rows, metric, expected):
    """{method: {dataset: mean}} as a DataFrame (datasets x methods), only for datasets with every expected fold."""
    rows = rows[np.isfinite(rows[metric])]
    if rows.duplicated(['method', 'dataset', 'seed', 'fold']).any():
        raise ValueError('duplicated rows: the same method, dataset, seed and fold appear twice')
    per = {}
    for (method, dataset), group in rows.groupby(['method', 'dataset']):
        if dataset in expected and set(zip(group.seed.astype(int), group.fold.astype(int))) == expected[dataset]:
            per[method, dataset] = group[metric].mean()
    return pd.Series(per, dtype=float).unstack(0).reindex(list(expected))


def seed_std(rows, metric, per_dataset):
    """Std over the seeds of each method's benchmark mean (a seed's value is the mean, over the datasets the method
    finished, of that seed's mean over folds)."""
    out = {}
    for method in per_dataset.columns:
        finished = per_dataset.index[per_dataset[method].notna()]
        r = rows[(rows.method == method) & rows.dataset.isin(finished)]
        out[method] = r.groupby(['seed', 'dataset'])[metric].mean().groupby('seed').mean().std(ddof=1)
    return pd.Series(out)


def rank_matrix(scores, ascending):
    """Rank of each method (columns) on each dataset (rows); unfinished entries get the worst rank."""
    n = scores.shape[1]
    r = scores.rank(axis=1, ascending=ascending, method='average').fillna(n)
    complete = scores.notna().all(axis=1)
    assert np.allclose(r[complete].sum(axis=1), n * (n + 1) / 2)
    return r


def rank_interval(scores, ascending, resamples=10_000, seed=0, level=0.95):
    """95% percentile bootstrap interval of each mean rank over the datasets (the same draws for every method)."""
    r = rank_matrix(scores, ascending).to_numpy()
    draws = np.random.default_rng(seed).integers(0, len(r), size=(resamples, len(r)))
    means = r[draws].mean(axis=1)
    low, high = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2], axis=0)
    return pd.Series(low, index=scores.columns), pd.Series(high, index=scores.columns)
