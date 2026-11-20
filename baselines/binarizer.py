"""Threshold binarisation of real columns, the input of the rule-network baselines (RL-Net, DR-Net)."""
import numpy as np


class QuantileBinarizer:
    """One 0/1 column "x_j <= t" per threshold t: for a column with at most n_bins + 1 distinct values every value but
    the largest, otherwise its n_bins evenly spaced quantiles (duplicates merged). Fitted on the training rows."""

    def __init__(self, n_bins=9):
        self.n_bins = n_bins
        self.thresholds_ = []

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        probs = np.linspace(1.0 / (self.n_bins + 1), self.n_bins / (self.n_bins + 1), self.n_bins)
        self.thresholds_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            vals = np.unique(col[~np.isnan(col)])
            if vals.size <= 1:
                cuts = np.array([], dtype=float)
            elif vals.size <= self.n_bins + 1:
                cuts = vals[:-1].astype(float)
            else:
                cuts = np.unique(np.nanquantile(col, probs)).astype(float)
            self.thresholds_.append(cuts)
        self.n_features_out_ = int(sum(len(t) for t in self.thresholds_))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        cols = []
        for j, cuts in enumerate(self.thresholds_):
            if len(cuts):
                cols.append((X[:, [j]] <= cuts[None, :]).astype(np.float32))
        if not cols:
            return np.zeros((X.shape[0], 1), dtype=np.float32)
        return np.concatenate(cols, axis=1)
