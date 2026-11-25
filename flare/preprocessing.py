"""Preprocessing of one training fold.

FLARE and every compared method see the same data: the training rows of a fold are split into 80% for training and
20% for validation (stratified), the real variables are standardised with statistics of the training part only, and
the test rows are transformed with them. FLARE uses the validation part to choose its epoch; the baselines simply
leave it out, so all methods are fitted on the same rows.

Categorical variables with many categories get special treatment, also fitted on the training part only:

* `category_space=K` (FLARE on TabArena): a categorical with more than K training categories is replaced by the rank
  of each category by frequency (0 = most frequent, unseen categories get the next rank). The rank is only an index:
  FLARE learns a point for each category in a small space (flare/category_space.py) and its labels are regions there.
* `effect_order=K` (the baselines on TabArena): the same columns are replaced by a smoothed estimate of each
  category's effect on the class (`category_effect`), a single real number, so methods that cannot handle thousands of
  categories (one-hot) still see them.

Everything here is numpy; flare/data.py turns the result into PyTorch batches.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Fold:
    X_train: np.ndarray            # float64, real columns standardised
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    categorical: np.ndarray        # the fold's mask: category-space and effect-ordered columns are no longer categorical
    scaler: object = None          # sklearn StandardScaler of the real columns (None when there are none)
    X_test_raw: np.ndarray = None  # the test rows in the dataset's own units
    category_space: dict = field(default_factory=dict)    # {column: {'counts', 'codes', 'effects'}} (frequency order)
    category_effects: dict = field(default_factory=dict)  # {column: (scores, unseen score, training counts)}
    n_classes: int = 0


def category_effect(codes, y, n_classes, m=10.0):
    """Each category's effect on the class, from the training rows: {category: score} and the score of an unseen one.

    The class frequencies of a category are shrunk towards the overall ones with m pseudo-rows (an m-estimate). With two
    classes the score is the shrunk frequency of class 1; with more, it is the projection of the frequencies on their
    first principal direction (weighted by the category counts), so one number orders the categories."""
    cats, inv = np.unique(codes, return_inverse=True)
    counts = np.bincount(inv, minlength=len(cats)).astype(float)
    C = np.zeros((len(cats), n_classes))
    np.add.at(C, (inv, np.asarray(y, dtype=int)), 1.0)
    prior = C.sum(axis=0) / C.sum()
    P = (C + m * prior) / (counts[:, None] + m)
    if n_classes == 2:
        return dict(zip(cats.tolist(), P[:, 1].tolist())), float(prior[1])
    mean = (counts[:, None] * P).sum(axis=0) / counts.sum()
    D = P - mean
    _, vectors = np.linalg.eigh((counts[:, None] * D).T @ D)
    v = vectors[:, -1]
    v = v if v[np.argmax(np.abs(v))] > 0 else -v                    # a fixed sign, so the order is reproducible
    return dict(zip(cats.tolist(), (D @ v).tolist())), float((prior - mean) @ v)


def _effect_order(X_tr, y_tr, others, categorical, threshold, n_classes):
    """Replace, in place, each categorical of more than `threshold` training values by its effect score."""
    categorical = np.array(categorical, dtype=bool).copy()
    fitted = {}
    for j in np.flatnonzero(categorical):
        if len(np.unique(X_tr[:, j])) <= threshold:
            continue
        scores, unseen = category_effect(X_tr[:, j], y_tr, n_classes)
        values, n = np.unique(X_tr[:, j], return_counts=True)
        for X in [X_tr, *others]:
            X[:, j] = np.array([scores.get(c, unseen) for c in X[:, j].tolist()], dtype=float)
        categorical[j] = False
        fitted[int(j)] = (scores, unseen, dict(zip(values.tolist(), n.tolist())))
    return categorical, fitted


def preprocess_fold(X_train, y_train, X_test, y_test, categorical, seed, val_proportion=0.2, effect_order=0,
                    category_space=0):
    """Split, scale and encode one fold. X_* are DataFrames (already imputed, see experiments/data), y_* Series.

    `seed` is the seed of the run (not seed + fold): it fixes the 80/20 split of the training rows."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if effect_order > 0 and category_space > 0:
        raise ValueError('effect_order and category_space are alternatives for the same columns')
    categorical = np.array(categorical, dtype=bool)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=val_proportion, random_state=seed,
                                                stratify=y_train)
    # Cast to float and copy: for an all-integer frame .values is an int array and the scaled values would be truncated,
    # and with pandas 3 (copy-on-write) .values is read-only.
    X_tr, X_val, X_te = (np.array(X.values, dtype=float) for X in (X_tr, X_val, X_test))
    y_tr, y_val, y_te = (np.asarray(y).astype(int) for y in (y_tr, y_val, y_test))
    raw_test = X_te.copy()
    # The number of classes: the largest code seen in the fold, so a class that is missing from a fold keeps its index.
    n_classes = int(max(np.max(y_train), np.max(y_test))) + 1

    effects = {}
    if effect_order > 0:
        categorical, effects = _effect_order(X_tr, y_tr, [X_val, X_te], categorical, effect_order, n_classes)

    scaler = None
    if (~categorical).any():
        scaler = StandardScaler()
        X_tr[:, ~categorical] = scaler.fit_transform(X_tr[:, ~categorical])
        X_val[:, ~categorical] = scaler.transform(X_val[:, ~categorical])
        X_te[:, ~categorical] = scaler.transform(X_te[:, ~categorical])

    spaces = {}
    if category_space > 0:
        mask = categorical.copy()
        for j in np.flatnonzero(mask):
            values, counts = np.unique(X_tr[:, j], return_counts=True)
            if len(values) <= category_space:
                continue
            order = np.argsort(-counts, kind='stable')           # most frequent first
            # Each category's effect (from its codes, before they become ranks) starts its point in the space.
            scores, _ = category_effect(X_tr[:, j], y_tr, n_classes)
            rank = {v: r for r, v in enumerate(values[order].tolist())}
            for X in (X_tr, X_val, X_te):
                X[:, j] = [rank.get(v, len(values)) for v in X[:, j].tolist()]   # unseen categories: rank K
            mask[j] = False
            spaces[int(j)] = dict(counts=counts[order], codes=values[order],
                                  effects=np.array([scores[v] for v in values[order].tolist()]))
        categorical = mask

    return Fold(X_tr, y_tr, X_val, y_val, X_te, y_te, categorical, scaler, raw_test, spaces, effects, n_classes)


def preprocess_split(X_tr, y_tr, X_val, y_val, X_test, y_test, categorical, category_space=0):
    """The same scaling and category spaces as `preprocess_fold`, for a split that is already made (numpy arrays). Used
    by `FLAREClassifier`, whose user may bring their own validation rows. `preprocess_fold` on the same split gives the
    same arrays (tests/test_preprocessing.py)."""
    categorical = np.array(categorical, dtype=bool)
    X_tr, X_val, X_te = (np.array(X, dtype=float) for X in (X_tr, X_val, X_test))
    y_tr, y_val, y_te = (np.asarray(y).astype(int) for y in (y_tr, y_val, y_test))
    raw_test = X_te.copy()
    n_classes = int(max(np.max(y_tr), np.max(y_val), np.max(y_te))) + 1
    scaler = None
    if (~categorical).any():
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr[:, ~categorical] = scaler.fit_transform(X_tr[:, ~categorical])
        X_val[:, ~categorical] = scaler.transform(X_val[:, ~categorical])
        X_te[:, ~categorical] = scaler.transform(X_te[:, ~categorical])
    spaces = {}
    if category_space > 0:
        mask = categorical.copy()
        for j in np.flatnonzero(mask):
            values, counts = np.unique(X_tr[:, j], return_counts=True)
            if len(values) <= category_space:
                continue
            order = np.argsort(-counts, kind='stable')
            scores, _ = category_effect(X_tr[:, j], y_tr, n_classes)
            rank = {v: r for r, v in enumerate(values[order].tolist())}
            for X in (X_tr, X_val, X_te):
                X[:, j] = [rank.get(v, len(values)) for v in X[:, j].tolist()]
            mask[j] = False
            spaces[int(j)] = dict(counts=counts[order], codes=values[order],
                                  effects=np.array([scores[v] for v in values[order].tolist()]))
        categorical = mask
    return Fold(X_tr, y_tr, X_val, y_val, X_te, y_te, categorical, scaler, raw_test, spaces, {}, n_classes)


def transform_rows(fold, X, categorical):
    """Scale and encode new rows (a float array, already imputed) with what `fold` learned: `categorical` is the mask of
    the original data (before the category spaces), whose complement the scaler was fitted on."""
    X = np.array(X, dtype=float)
    categorical = np.array(categorical, dtype=bool)
    if fold.scaler is not None:
        X[:, ~categorical] = fold.scaler.transform(X[:, ~categorical])
    for j, info in fold.category_space.items():
        rank = {v: r for r, v in enumerate(info['codes'].tolist())}
        X[:, j] = [rank.get(v, len(info['codes'])) for v in X[:, j].tolist()]      # unseen categories: rank K
    return X
