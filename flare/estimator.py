"""A scikit-learn style interface to FLARE.

    model = FLAREClassifier(flare()).fit(X, y)            # X: a DataFrame (or an array), y: the classes
    model.predict(X_new); model.predict_proba(X_new)
    print(model.print_rules())
    model.save('model.pt'); model = FLAREClassifier.load('model.pt')

It is a thin layer over `train.fit_fold`. Categorical columns may hold any values (text, numbers): they are found from
the column types unless a mask or a list of names is given, and coded internally. Missing values are filled from the
training rows (the median of a real column, the most frequent category of a categorical one), 20% of the training
rows are held out to choose the epoch (or the `X_val`, `y_val` you give), and the real variables are standardised.
The printed rules use the names of the categories and the units of the data.
"""
import dataclasses

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from .calibration import calibrate_probabilities
from .config import FLAREConfig
from .metrics import eval_chunk_rows
from .model import BagEnsemble
from .preprocessing import preprocess_split, transform_rows
from .printing import rules_text
from .selectors import GradMode
from .train import fit_fold

# Names of the labels of a real variable in the printed rules.
LABEL_NAMES = {3: ['low', 'medium', 'high'],
               5: ['very low', 'low', 'medium', 'high', 'very high'],
               7: ['very low', 'low', 'fairly low', 'medium', 'fairly high', 'high', 'very high']}


class FLAREClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, config=None, seed=33, device=None, verbose=False):
        self.config = config
        self.seed = seed
        self.device = device
        self.verbose = verbose

    # --- data ------------------------------------------------------------------------------------------------------

    def _frame(self, X):
        """A DataFrame with the columns seen in fit (an array gets them by position)."""
        frame = isinstance(X, pd.DataFrame)
        X = X.copy() if frame else pd.DataFrame(np.asarray(X))
        if hasattr(self, 'columns_'):
            if frame and set(self.columns_) <= set(X.columns):
                return X[list(self.columns_)]                         # by name, whatever the order
            if X.shape[1] != len(self.columns_):
                raise ValueError(f'X has {X.shape[1]} columns; the model was fitted on {len(self.columns_)}')
            if frame:
                raise ValueError(f'X lacks columns the model was fitted on: {sorted(set(map(str, self.columns_)) - set(map(str, X.columns)))}')
            X.columns = self.columns_                                 # an array: by position
        return X

    def _encode(self, X):
        """Categories as codes 0..K-1 (in the order of `categories_`), everything as floats, missing values filled."""
        X = self._frame(X)
        space = np.zeros(len(self.columns_), dtype=bool)
        if hasattr(self, 'fold_'):            # a column with a category space gives unseen categories a place of their own
            space = self.categorical_ & ~np.asarray(self.fold_.categorical, dtype=bool)
        for j in np.flatnonzero(self.categorical_):
            column = self.columns_[j]
            codes = {value: code for code, value in enumerate(self.categories_[column])}
            values = X[column]
            coded = values.map(lambda v: codes.get(v, -1) if not pd.isna(v) else np.nan).astype(float)
            unseen = coded == -1
            if unseen.any() and not space[j]:
                raise ValueError(f'column {column!r} has a category that was not seen in fit: {values[unseen].iloc[0]!r}')
            X[column] = coded
        X = X.apply(pd.to_numeric, errors='raise').astype(float)
        return X.fillna(self.fill_) if hasattr(self, 'fill_') else X

    def fit(self, X, y, categorical=None, X_val=None, y_val=None):
        """categorical: a boolean mask over the columns or a list of column names (default: the columns whose type is
        not numeric, plus boolean ones). X_val, y_val: validation rows used to choose the epoch (default: a stratified
        20% of the training rows)."""
        from sklearn.model_selection import train_test_split
        config = self.config if self.config is not None else FLAREConfig()
        for attribute in ('columns_', 'fold_', 'fill_'):              # a new fit forgets the previous one
            self.__dict__.pop(attribute, None)
        X = self._frame(X)
        self.columns_ = X.columns
        self.feature_names_ = [str(c) for c in X.columns]
        if categorical is None:
            mask = [not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]) for c in X.columns]
        elif len(categorical) == 0:
            mask = [False] * X.shape[1]
        elif all(isinstance(c, str) for c in categorical) and set(categorical) <= set(map(str, X.columns)):
            mask = [str(c) in set(categorical) for c in X.columns]
        else:
            mask = categorical
        self.categorical_ = np.asarray(mask, dtype=bool)
        if len(self.categorical_) != X.shape[1]:
            raise ValueError(f'categorical has {len(self.categorical_)} entries for {X.shape[1]} columns')
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)

        # The categories of each categorical column: those of the training (and validation) rows, sorted.
        seen = [X] + ([self._frame(X_val)] if X_val is not None else [])
        self.categories_ = {}
        for j in np.flatnonzero(self.categorical_):
            column = self.columns_[j]
            values = pd.unique(pd.concat([frame[column] for frame in seen]).dropna())
            try:
                values = sorted(values)
            except TypeError:                                        # mixed types: keep the order of appearance
                values = list(values)
            self.categories_[column] = list(values)
        X = self._encode(X)
        # Missing values: the median of a real column, the most frequent category of a categorical one.
        fill = {}
        for column, is_categorical in zip(X.columns, self.categorical_):
            statistic = X[column].mode().iloc[0] if is_categorical and X[column].notna().any() else X[column].median()
            fill[column] = 0.0 if pd.isna(statistic) else statistic
        self.fill_ = fill
        X = X.fillna(fill)
        if X_val is None:
            X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=self.seed, stratify=y)
        else:
            y_val = np.asarray(y_val)
            if not np.isin(y_val, self.classes_).all():
                raise ValueError('y_val has a class that is not in y')
            X_tr, y_tr, X_va, y_va = X, y, self._encode(X_val), np.searchsorted(self.classes_, y_val)
        fold = preprocess_split(X_tr.values, y_tr, X_va.values, y_va, X_va.values, y_va, self.categorical_,
                                category_space=config.category_space_min if config.category_space > 0 else 0)
        cardinalities = [len(self.categories_[self.columns_[j]]) for j in np.flatnonzero(fold.categorical)]
        result = fit_fold(fold, config, cardinalities, random_seed=self.seed, device=self.device, verbose=self.verbose)
        self.model_, self.config_ = result.model, config
        self.calibration_ = None
        if config.calibrate:
            self.calibration_ = calibrate_probabilities(self.model_, result.val_loader, config.calibration_min_temperature,
                                                        config.calibration_bias_penalty)
        # Keep what new rows need (the scaler, the category spaces, the column types), not the training rows.
        self.fold_ = dataclasses.replace(fold, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None,
                                         y_test=None, X_test_raw=None)
        self._readable_rules()
        return self

    def _readable_rules(self):
        """Print the cuts of the real variables in the data's units and give their labels names."""
        real = np.flatnonzero(~np.asarray(self.fold_.categorical, dtype=bool))        # the model's real variables
        scaled = list(np.flatnonzero(~self.categorical_))                            # the columns the scaler saw
        mean, std = np.zeros(len(real)), np.ones(len(real))
        if self.fold_.scaler is not None:
            for i, column in enumerate(real):
                if column in scaled:
                    mean[i] = self.fold_.scaler.mean_[scaled.index(column)]
                    std[i] = self.fold_.scaler.scale_[scaled.index(column)]
        for model in (self.model_.members if isinstance(self.model_, BagEnsemble) else [self.model_]):
            if model.inputs is not None:
                model.inputs.cut_scale = (mean, std)
                n_labels = len(model.inputs.label_names)
                if n_labels in LABEL_NAMES:
                    model.inputs.label_names = LABEL_NAMES[n_labels]

    # --- prediction -----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def decision_function(self, X):
        """Class scores s_c(x) = b_c + sum_r w_rc T_r(x) of the deployed (discrete) model."""
        X = transform_rows(self.fold_, self._encode(X).values, self.categorical_)
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

    def rules_text(self):
        """The rules as text, one per line."""
        names = {j: [str(v) for v in self.categories_[self.columns_[j]]] for j in np.flatnonzero(self.categorical_)}
        return rules_text(self.model_, feat_names=self.feature_names_, class_names=[str(c) for c in self.classes_],
                          category_names=names)

    def print_rules(self):
        """Print the rules (and return them as text)."""
        text = self.rules_text()
        print(text)
        return text

    # --- saving ---------------------------------------------------------------------------------------------------------

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path, device='cpu'):
        model = torch.load(path, map_location=device, weights_only=False)
        model.model_.to(device)
        return model
