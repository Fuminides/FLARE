"""Small builders shared by the tests."""
import numpy as np
import torch

from flare.config import FLAREConfig
from flare.inputs import TabularInputs
from flare.model import FLAREModel
from flare.partitions import quantile_partitions
from flare.preprocessing import preprocess_split
from flare.selectors import GradMode


def toy_partitions(n_labels, n_variables=3, rows=200, seed=0):
    X = np.random.default_rng(seed).normal(size=(rows, n_variables)).astype(np.float32)
    return X, quantile_partitions(X, n_labels)


def toy_model(n_labels=7, n_variables=3, n_rules=6, n_slots=2, label_runs='shoulders', mode=GradMode.GRAFT, temp=1.0,
              joint_bags=1, categorical_sizes=(), seed=3, rows=60, device='cpu'):
    """A small FLARE model on random real variables (plus categorical ones with the given numbers of categories)."""
    X, partitions = toy_partitions(n_labels, n_variables, rows, seed=2)
    torch.manual_seed(seed)
    inputs = TabularInputs(partitions, label_runs, None, None, device)
    mask = np.array([False] * n_variables + [True] * len(categorical_sizes))
    model = FLAREModel(inputs, 2, mask, list(categorical_sizes), n_rules * joint_bags, n_slots, temp, mode, joint_bags,
                      device=device)
    return model, torch.tensor(X, device=device)


def toy_fold(rows=160, n_real=3, categorical_sizes=(), seed=0, n_classes=2, category_space=0):
    """A preprocessed fold of random rows whose class depends on the first variable (and the first categorical)."""
    rng = np.random.default_rng(seed)
    real = rng.normal(size=(rows, n_real))
    cats = [rng.integers(0, size, size=rows) for size in categorical_sizes]
    X = np.column_stack([real] + cats) if cats else real
    score = (real[:, 0] if n_real else 0) + sum(c % 2 for c in cats[:2]) + 0.5 * rng.normal(size=rows)
    y = np.digitize(score, np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]))
    categorical = np.array([False] * n_real + [True] * len(cats))
    train, val, test = slice(0, 100), slice(100, 130), slice(130, rows)
    fold = preprocess_split(X[train], y[train], X[val], y[val], X[test], y[test], categorical, category_space=category_space)
    # the categories of the columns that stay categorical (a category-space column is real in the fold)
    return fold, [int(c.max()) + 1 for c, j in zip(cats, range(n_real, n_real + len(cats))) if fold.categorical[j]]


def small_config(**changes):
    """A configuration small enough to train in a second or two."""
    base = dict(rules=8, single_condition_rules=5, slots=2, labels=5, bags=1, parallel_bags=False, epochs=4, steps_per_epoch=0,
                batch_size=32, cuda_graph=False)
    return FLAREConfig(**{**base, **changes})
