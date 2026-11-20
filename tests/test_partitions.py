"""The vendored quantile trapezoids are exactly those of ex_fuzzy (which FLARE does not depend on)."""
import numpy as np
import pandas as pd
import pytest

from flare.partitions import label_names, quantile_partitions

ex_fuzzy = pytest.importorskip('ex_fuzzy')


def _data(dtype, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=300), rng.integers(0, 4, size=300), rng.exponential(size=300),
                         rng.integers(0, 3, size=300), rng.normal(size=300) * 100])
    return pd.DataFrame(X.astype(dtype), columns=list('abcde'))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('n_labels', [3, 5, 7, 9, 11])
def test_matches_ex_fuzzy_exactly(dtype, n_labels):
    frame = _data(dtype)
    categorical = np.array([False, False, False, True, False])
    reference = ex_fuzzy.utils.construct_partitions(frame, ex_fuzzy.fuzzy_sets.FUZZY_SETS.t1, categorical_mask=categorical,
                                                    n_partitions=n_labels, shape='trapezoid')
    real = [i for i in range(5) if not categorical[i]]
    ours = quantile_partitions(frame.values[:, real], n_labels)
    assert ours.shape == (4, n_labels, 4)
    for v, column in enumerate(real):
        sets = reference[column]
        assert [s.name for s in sets] == label_names(n_labels)
        for label, s in enumerate(sets):
            np.testing.assert_array_equal(ours[v, label], s.membership_parameters)


def test_the_ends_are_a_shoulder_that_covers_the_whole_range():
    parameters = quantile_partitions(np.random.default_rng(1).normal(size=(100, 2)).astype(np.float32), 7)
    assert np.all(parameters[:, 0, 0] == parameters[:, 0, 1]) and np.all(parameters[:, -1, 2] == parameters[:, -1, 3])
    assert np.all(parameters[:, 0, 0] < parameters[:, 0, 2])
