"""A scikit-learn style interface to FLARE.

    clf = FLAREClassifier(flare(), seed=33).fit(X, y, categorical=[False, True, ...])
    clf.predict(X_new); clf.predict_proba(X_new); clf.print_rules()

It is a thin layer over `train.fit_fold`: it imputes missing values (median, or the most frequent category) from the
training rows, splits off 20% of them for validation (or uses the `X_val`, `y_val` you give), standardises the real
variables (preprocessing.py), trains FLARE and keeps the model. Categorical columns must be coded 0..K-1.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from .calibration import calibrate_probabilities
from .config import FLAREConfig
from .metrics import eval_chunk_rows
from .preprocessing import preprocess_split, transform_rows
from .printing import print_rules
from .selectors import GradMode
from .train import fit_fold


class FLAREClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, config=None, seed=33, device=None, verbose=False):
        self.config = config
        self.seed = seed
        self.device = device
        self.verbose = verbose

    # --- data ------------------------------------------------------------------------------------------------------

    def _frame(self, X):
        """A DataFrame with the columns seen in fit (an array gets them by position)."""
        if isinstance(X, pd.DataFrame):
            return X
        X = pd.DataFrame(np.asarray(X))
        if hasattr(self, 'columns_'):
            if X.shape[1] != len(self.columns_):
                raise ValueError(f'X has {X.shape[1]} columns; the model was fitted on {len(self.columns_)}')
            X.columns = self.columns_
        return X

    def _impute(self, X):
        X = self._frame(X).fillna(self.fill_)
        if hasattr(self, 'fold_'):
            # A categorical column only knows the codes 0..K-1 seen in fit (K: its largest code + 1); a column with a
            # category space (fold_.categorical is False there) gives unseen categories a place of their own.
            mask = self.categorical_ & np.asarray(self.fold_.categorical, dtype=bool)
            codes = X.values[:, mask]
            unseen = (codes >= self.n_categories_[mask]) | (codes < 0)
            if unseen.any():
                column = self.columns_[np.flatnonzero(mask)[np.flatnonzero(unseen.any(axis=0))[0]]]
                raise ValueError(f'column {column!r} has a category code that was not seen in fit')
        return X

    def fit(self, X, y, categorical=None, X_val=None, y_val=None):
        """categorical: a boolean mask over the columns (default: all real). X_val, y_val: validation rows used to
        choose the epoch (default: a stratified 20% of the training rows)."""
        from sklearn.model_selection import train_test_split
        config = self.config if self.config is not None else FLAREConfig()
        for attribute in ('columns_', 'n_categories_', 'fold_'):      # a new fit forgets the previous one
            self.__dict__.pop(attribute, None)
        X = self._frame(X)
        self.columns_ = X.columns
        self.feature_names_ = [str(c) for c in X.columns]
        self.categorical_ = np.zeros(X.shape[1], dtype=bool) if categorical is None else np.asarray(categorical, dtype=bool)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        # Missing values: the median of a real column, the most frequent code of a categorical one (training rows).
        fill = {}
        for column, is_categorical in zip(X.columns, self.categorical_):
            statistic = X[column].mode().iloc[0] if is_categorical and X[column].notna().any() else X[column].median()
            fill[column] = 0.0 if pd.isna(statistic) else statistic
        self.fill_ = fill
        X = self._impute(X)
        if X_val is None:
            X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=self.seed, stratify=y)
        else:
            y_val = np.asarray(y_val)
            if not np.isin(y_val, self.classes_).all():
                raise ValueError('y_val has a class that is not in y')
            X_tr, y_tr, X_va, y_va = X, y, self._impute(X_val), np.searchsorted(self.classes_, y_val)
        n_categories = np.maximum(X_tr.values.max(axis=0), X_va.values.max(axis=0)).astype(int) + 1
        self.n_categories_ = n_categories
        fold = preprocess_split(X_tr.values, y_tr, X_va.values, y_va, X_va.values, y_va, self.categorical_,
                                category_space=config.category_space_min if config.category_space > 0 else 0)
        cardinalities = [int(n_categories[j]) for j in np.flatnonzero(fold.categorical)]
        self.result_ = fit_fold(fold, config, cardinalities, random_seed=self.seed, device=self.device, verbose=self.verbose)
        self.model_, self.fold_, self.config_ = self.result_.model, fold, config
        self.calibration_ = None
        if config.calibrate:
            self.calibration_ = calibrate_probabilities(self.model_, self.result_.val_loader, config.calibration_min_temperature,
                                                        config.calibration_bias_penalty)
        return self

    # --- prediction -----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def decision_function(self, X):
        """Class scores s_c(x) = b_c + sum_r w_rc T_r(x) of the deployed (discrete) model."""
        X = transform_rows(self.fold_, self._impute(X).values, self.categorical_)
        device = next(self.model_.parameters()).device
        self.model_.set_inference_mode(GradMode.PURE, 0.0)
        X = torch.tensor(X, dtype=torch.float32, device=device)
        return torch.cat([self.model_(chunk)[0].float().cpu() for chunk in X.split(eval_chunk_rows(self.model_))]).double().numpy()

    def predict_proba(self, X):
        scores = torch.as_tensor(self.decision_function(X))
        if self.calibration_ is not None:
            scores = scores / self.calibration_[0] + torch.as_tensor(self.calibration_[1], dtype=scores.dtype)
        return torch.softmax(scores, dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    # --- the rules ----------------------------------------------------------------------------------------------------

    @property
    def rules_(self):
        """(rules, points, consequents): rules[r, j] is the condition that rule r tests on column j (-1: none)."""
        return self.model_.extract_rules()

    def print_rules(self, class_names=None, category_names=None):
        """The rules as text (also returned). `category_names`: {column index: [name of each category code]}."""
        return print_rules(self.model_, feat_names=self.feature_names_,
                          class_names=class_names if class_names is not None else [f'Class {c}' for c in self.classes_],
                          category_names=category_names)
