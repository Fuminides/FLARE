"""The two tabular benchmarks, downloaded from OpenML on first use.

* **OpenML-CC18**: the 42 tabular classification datasets listed in `cc18.json` (name, OpenML dataset id, size,
  classes and rows with missing values). The folds are a stratified 5-fold split, shuffled with the seed of the run
  (the runs use seeds 33, 34 and 35), see `cc18_folds`.
* **TabArena v0.1**: the 38 classification tasks listed in `tabarena.json` (OpenML task ids). Each task brings its own
  outer splits (`tabarena_splits`), so every split pairs with the results TabArena published for its own methods.

Nothing is stored in this repository. OpenML caches what it downloads in `$FLARE_DATA_DIR/openml`
(default `~/.cache/flare/openml`); only the first use of a dataset needs the network.

Every loader returns the same four things:

    X            pandas DataFrame; categorical columns as integer codes 0..K-1 (NaN when missing), the rest as floats
    y            pandas Series of class codes 0..C-1
    categorical  boolean numpy array, True for the categorical columns
    names        dict with the class names ('classes') and the category names of each categorical column ('categories'),
                 used only to print the rules with the dataset's own words

Missing values stay as NaN here and are filled per fold by `impute`, from the training rows only.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('FLARE_DATA_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'flare'))
SEEDS = (33, 34, 35)               # the three seeds of the CC18 runs
N_FOLDS = 5


def cc18_manifest():
    """The 42 CC18 datasets, smallest first: [{'name', 'did', 'rows', 'features', 'classes', 'missing'}]."""
    with open(os.path.join(HERE, 'cc18.json')) as f:
        return json.load(f)


def cc18_names():
    return [entry['name'] for entry in cc18_manifest()]


def tabarena_manifest():
    """The 38 TabArena classification tasks: {name: {'tid', 'problem_type', 'n_folds', 'n_repeats', ...}}."""
    with open(os.path.join(HERE, 'tabarena.json')) as f:
        return {entry['dataset']: entry for entry in json.load(f)}


def _openml():
    import openml
    cache = os.path.join(DATA_DIR, 'openml')
    os.makedirs(cache, exist_ok=True)
    openml.config.set_root_cache_directory(cache)
    return openml


def load_cc18(name):
    """One CC18 dataset (see the module docstring for what is returned)."""
    ids = {entry['name']: entry['did'] for entry in cc18_manifest()}
    if name not in ids:
        raise KeyError(f'{name} is not one of the 42 CC18 datasets of the paper (experiments/data/cc18.json)')
    dataset = _openml().datasets.get_dataset(ids[name])
    X, y, categorical, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format='dataframe')
    return encode(X, y, categorical)


def _tabarena_task(name):
    return _openml().tasks.get_task(tabarena_manifest()[name]['tid'], download_splits=True)


def load_tabarena(name):
    """The data of one TabArena task, with the task's own target."""
    task = _tabarena_task(name)
    X, y, categorical, _ = task.get_dataset().get_data(target=task.target_name, dataset_format='dataframe')
    return encode(X, y, categorical, strings=True)


def tabarena_splits(name):
    """The task's outer splits in TabArena's order (split index = n_folds * repeat + fold): [(train_idx, test_idx)]."""
    task, entry = _tabarena_task(name), tabarena_manifest()[name]
    return [tuple(np.asarray(i) for i in task.get_train_test_split_indices(fold=fold, repeat=repeat))
            for repeat in range(entry['n_repeats']) for fold in range(entry['n_folds'])]


def encode(X, y, categorical, strings=False):
    """Categorical columns as codes and the rest as floats.

    strings=True (TabArena): text columns that the dataset does not flag as categorical become categorical, and text
    columns that are dates become days since 1970 (pandas gives text a string dtype, not object)."""
    X = X.copy()
    categorical = np.array(categorical, dtype=bool)
    if strings:
        for j, column in enumerate(X.columns):
            if categorical[j] or pd.api.types.is_numeric_dtype(X[column]) or isinstance(X[column].dtype, pd.CategoricalDtype):
                continue
            dates = pd.to_datetime(X[column], errors='coerce', format='mixed')
            if dates.notna().sum() == X[column].notna().sum() > 0:
                X[column] = (dates - pd.Timestamp('1970-01-01')).dt.days.astype(float)
            else:
                categorical[j] = True
    categories = {}
    for column, is_categorical in zip(X.columns, categorical):
        if is_categorical or X[column].dtype == object or isinstance(X[column].dtype, pd.CategoricalDtype):
            codes, uniques = pd.factorize(X[column])
            X[column] = np.where(codes < 0, np.nan, codes)          # factorize codes a missing value as -1
            categories[column] = [str(u) for u in uniques]
        else:
            X[column] = X[column].astype(float)
    target = pd.Categorical(y)
    y = pd.Series(target.codes, index=X.index, name='class')
    return X, y, categorical, {'classes': [str(c) for c in target.categories], 'categories': categories}


def impute(X_train, X_test, categorical):
    """Fill missing values from the training rows only: a categorical column's most frequent code, else the median."""
    if not (X_train.isna().any().any() or X_test.isna().any().any()):
        return X_train, X_test
    fill = {}
    for column, is_categorical in zip(X_train.columns, categorical):
        if is_categorical:
            modes = X_train[column].mode()
            stat = modes.iloc[0] if not modes.empty else np.nan
        else:
            stat = X_train[column].median()
        fill[column] = 0.0 if pd.isna(stat) else stat                # a column missing in every training row
    return X_train.fillna(fill), X_test.fillna(fill)


def cc18_folds(y, seed, n_folds=N_FOLDS):
    """The CC18 folds of a run: stratified, shuffled with the run's seed. [(train_idx, test_idx)]."""
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def tuning_split(train_idx, y_train, seed):
    """80 / 20 split of a training fold for hyperparameter tuning (every tuned method uses it, FLARE included),
    stratified unless a class has a single row. The test rows of the fold are never read while tuning."""
    from sklearn.model_selection import train_test_split
    _, counts = np.unique(y_train, return_counts=True)
    inner_tr, inner_va = train_test_split(np.arange(len(train_idx)), test_size=0.2, random_state=seed,
                                          stratify=y_train if counts.min() >= 2 else None)
    return np.asarray(train_idx)[inner_tr], np.asarray(train_idx)[inner_va]


def load(benchmark, name):
    """('cc18' | 'tabarena', name) -> X, y, categorical, names, and the folds: a function of the seed for CC18, the
    task's fixed splits for TabArena (which is run with seed 33 only)."""
    if benchmark == 'cc18':
        X, y, categorical, names = load_cc18(name)
        return X, y, categorical, names, lambda seed: cc18_folds(y, seed)
    if benchmark == 'tabarena':
        X, y, categorical, names = load_tabarena(name)
        splits = tabarena_splits(name)
        return X, y, categorical, names, lambda seed: splits
    raise ValueError(f'unknown benchmark {benchmark!r} (cc18 or tabarena)')
