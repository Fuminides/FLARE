"""NeuRules (Xu, Walter and Vreeken, NeurIPS 2025, Neural Rule Lists) through the authors' code
(github.com/Schascha1/neurules, cloned by `fetch_external.py`; the repository has no licence file, so nothing of it is
copied here).

From the clone only `neurules.py` (the network, `NeuralRuleList`) is used, loaded by its path. Its training loop lives
in `training.py`, which imports the authors' dataset loaders (pmlb, torchvision, imodels, ...), so the loop is written
out here following their `train_model`: two Adam steps per batch, a penalty on rules that cover less than `min_support`
of the batch, the weights of the rule-to-class layer frozen for the first half of the epochs, and the temperatures of
the discretiser and of the rule selector lowered linearly (to a fifth) during the second half. Defaults are those of
their `run_method.py`. The fitted list is evaluated as a hard rule list (their `HardRuleList`, vectorised): the first
rule, in decreasing priority, whose active conditions all hold decides the class.
"""
import importlib.util
import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.external import repo_path


@lru_cache(maxsize=None)
def _upstream():
    """The authors' neurules.py as a module of its own name (the clone's other files have generic names, utils and
    configs, that must not enter sys.path)."""
    spec = importlib.util.spec_from_file_location('neurules_authors', os.path.join(repo_path('neurules'), 'neurules.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SmallCategoricalExpander:
    """The authors' preprocessing of their experiments: columns with 3 or 4 distinct values are one-hot encoded."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.categories_ = []
        for column in X.T:
            values = np.unique(column)
            self.categories_.append(values if 2 < len(values) < 5 else None)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        columns = []
        for index, values in enumerate(self.categories_):
            column = X[:, index]
            columns.append(column[:, None] if values is None else (column[:, None] == values[None, :]).astype(float))
        return np.concatenate(columns, axis=1)


class _Schedule:
    """Linear from start to end over `steps` batches, then constant."""

    def __init__(self, start, end, steps):
        self.start, self.end, self.steps = start, end, max(steps, 1)

    def value(self, step):
        return self.start + min(max(step, 0), self.steps) / self.steps * (self.end - self.start)


class NeuRules(BaseEstimator, ClassifierMixin):
    """`n_rules` defaults to 10 (binary) or 5 per class; `coverage_weight` is the authors' `lambd`."""

    def __init__(self, n_rules=None, epochs=500, batch_size=2048, learning_rate=0.01, min_support=0.2,
                 coverage_weight=0.4, predicate_temperature=0.2, selector_temperature=1.0, random_state=0):
        self.n_rules = n_rules
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_support = min_support
        self.coverage_weight = coverage_weight
        self.predicate_temperature = predicate_temperature
        self.selector_temperature = selector_temperature
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        self.imputer_ = SimpleImputer(strategy='median').fit(X)
        X = self.imputer_.transform(X)
        self.expander_ = _SmallCategoricalExpander().fit(X)
        X = self.expander_.transform(X)
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X).astype(np.float32, copy=False)
        n_rules = self.n_rules or (10 if n_classes == 2 else 5 * n_classes)

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)

        tensor_X = torch.from_numpy(X)
        tensor_y = torch.from_numpy(encoded.astype(np.int64, copy=False))
        limits = torch.stack((tensor_X.min(dim=0).values, tensor_X.max(dim=0).values), dim=1)
        config = SimpleNamespace(n_features=X.shape[1], n_classes=n_classes, n_rules=n_rules, random_cutpoints=False,
                                 predicate_temperature=self.predicate_temperature,
                                 selector_temperature=self.selector_temperature)
        network = self.network_ = _upstream().NeuralRuleList(config, limits)
        # Initial rule-to-class weights: 0.5 everywhere and 10 on the block of rules assigned to each class (the
        # authors' `init="const"`; their network class ignores that setting and would keep nn.Linear's initialisation).
        with torch.no_grad():
            network.rule_weights.weight.fill_(0.5)
            part = n_rules // n_classes
            for c in range(n_classes):
                network.rule_weights.weight[c, c * part:(n_rules if c == n_classes - 1 else (c + 1) * part)] = 10.0

        loader = DataLoader(TensorDataset(tensor_X, tensor_y), batch_size=min(self.batch_size, len(tensor_X)), shuffle=True)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        half = self.epochs // 2
        selector = _Schedule(self.selector_temperature, self.selector_temperature / 5, half)
        predicate = _Schedule(self.predicate_temperature, self.predicate_temperature / 5, half)
        step = 0
        network.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits, _, selection = network(batch_X, return_rules=True)
                coverage = torch.relu(self.min_support - selection.mean(dim=0)).mean()
                loss = criterion(logits, batch_y) + self.coverage_weight * coverage
                loss.backward()
                if epoch < half:
                    network.rule_weights.weight.grad = None
                if torch.isnan(loss):                       # skip the update of a batch with a NaN loss
                    continue
                optimizer.step()
                optimizer.step()
                network.fix_parameters()
                if epoch >= half:
                    step += 1
                    network.selector_temperature = selector.value(step)
                    network.discretizer.temperature = predicate.value(step)
        network.eval()

        self.cut_points_ = network.discretizer.cut_points.detach().cpu().numpy().copy()      # [features, 2, rules]
        self.and_weights_ = network.rules.and_weights.detach().cpu().numpy().copy()          # [rules, features]
        self.rule_logits_ = network.rule_weights.weight.detach().cpu().numpy().T.copy()      # [rules, classes]
        self.rule_indices_ = np.argsort(network.rule_order.detach().cpu().numpy())[::-1]     # decreasing priority
        self.active_features_ = self.and_weights_ > 0
        self.complexity_ = float(self.active_features_.any(axis=1).sum())                     # rules with a condition
        return self

    def _hard_logits(self, X):
        X = self.scaler_.transform(self.expander_.transform(self.imputer_.transform(np.asarray(X, dtype=float))))
        X = X.astype(np.float32, copy=False)
        logits = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        uncovered = np.ones(len(X), dtype=bool)
        for index in self.rule_indices_:
            active = self.active_features_[index]
            if not active.any():
                continue
            match = ((X[:, active] > self.cut_points_[active, 0, index])
                     & (X[:, active] < self.cut_points_[active, 1, index])).all(axis=1)
            selected = uncovered & match
            logits[selected] = self.rule_logits_[index]
            uncovered[selected] = False
            if not uncovered.any():
                break
        return logits

    def predict(self, X):
        return self.classes_[np.argmax(self._hard_logits(X), axis=1)]
