"""The arrays the baselines receive: SHA-256 of the fold matrices (float64, after the one-hot encoding of the
categorical columns) against stored hashes. Needs the OpenML datasets (skipped when they cannot be downloaded)."""
import hashlib
import json
import os

import numpy as np
import pytest

from experiments.data import openml as data
from experiments.tabular.run_baselines import EFFECT_ORDER, fold_arrays

HASHES = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'baseline_fold_hashes.json')))


@pytest.mark.parametrize('key', sorted(HASHES))
def test_fold_arrays_match_the_paper_runs(key):
    benchmark, dataset, seed, fold = key.split('/')
    try:
        X, y, categorical, _, folds = data.load(benchmark, dataset)
    except Exception as error:                                  # no network and nothing cached
        pytest.skip(f'{dataset} not available: {error}')
    train_idx, test_idx = folds(int(seed))[int(fold)]
    arrays = fold_arrays(X, y, categorical, train_idx, test_idx, int(seed), EFFECT_ORDER if benchmark == 'tabarena' else 0)
    got = {k: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for k, a in zip(('X', 'y', 'Xt', 'yt'), arrays)}
    assert got == HASHES[key]
