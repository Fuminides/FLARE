"""Initial fuzzy partitions of the real variables.

Every real variable is covered by L trapezoidal fuzzy sets whose breakpoints sit at quantiles of the training rows, so
that every plateau and every ramp holds about the same share of the data. The construction follows
`ex_fuzzy.utils.construct_partitions` (`shape="trapezoid"`) and returns plain arrays; `tests/test_partitions.py`
checks that both give the same numbers. The partition is only the starting point: `labels.py` turns it into a learned
Ruspini partition.
"""
import numpy as np

# Width of the outer half-ramps beyond the first and last quantile.
_END_OFFSET = 0.0001


def label_names(n_labels):
    """Names of the labels of a variable: Low / Medium / High, the five-label version, or just the position."""
    if n_labels == 3:
        return ['Low', 'Medium', 'High']
    if n_labels == 5:
        return ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    return [str(ix) for ix in range(n_labels)]


def quantile_partitions(X, n_labels):
    """Trapezoid parameters [variables, labels, 4] (a, b, c, d: foot, shoulder, shoulder, foot) of each column of X.

    The 4 + 2 (L - 2) quantiles of a column, equally spaced between its minimum and maximum, are shared out as follows:
    the first label is a shoulder that is 1 up to the first quantile boundary and falls to 0 over the next one; the last
    is its mirror image; every inner label rises over two consecutive quantiles, stays at 1 over the next gap and falls
    over the next. Neighbouring labels thus overlap on one ramp. The dtype of X matters: the quantiles are computed on
    the float32 matrix that the model trains on (see `fit_fold` in train.py), so float32 rounding enters the
    breakpoints."""
    X = np.asarray(X)
    n_variables = X.shape[1]
    quantiles = np.nanpercentile(X, np.linspace(0, 100, 4 + (n_labels - 2) * 2), axis=0)
    parameters = np.zeros((n_variables, n_labels, 4))
    for label in range(n_labels):
        if label == 0:
            parameters[:, label, 0] = quantiles[0, :] - _END_OFFSET
            parameters[:, label, 1] = quantiles[0, :] - _END_OFFSET
            parameters[:, label, 2] = quantiles[1, :]
            parameters[:, label, 3] = quantiles[2, :]
        elif label == n_labels - 1:
            parameters[:, label, 0] = quantiles[-3, :]
            parameters[:, label, 1] = quantiles[-2, :]
            parameters[:, label, 2] = quantiles[-1, :] + _END_OFFSET
            parameters[:, label, 3] = quantiles[-1, :] + _END_OFFSET
        else:
            first = 1 + 2 * (label - 1)
            for corner in range(4):
                parameters[:, label, corner] = quantiles[first + corner, :]
    return parameters
