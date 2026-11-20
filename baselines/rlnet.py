"""RL-Net (Dierckx et al., PAKDD 2023, Rule List Network) through the authors' code (github.com/luciledierckx/RLNet).

The authors' `networkTorch_multiClass.Network` is trained here with a plain loop that keeps the weights of the epoch
with the lowest loss on a held-out 20% of the training rows. CPU, sklearn interface (fit, predict).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from baselines.binarizer import QuantileBinarizer
from baselines.external import prepend_sys_path, repo_path

EPS = 1e-12


def _l2(parameters):
    """The penalty on the output layer: the sum of the spectral norms torch.linalg.norm(p, 2). When the float32 SVD
    fails to converge the norm is taken in float64 instead."""
    import torch
    total = 0
    for p in parameters:
        try:
            total = total + torch.linalg.norm(p, 2)
        except torch.linalg.LinAlgError:
            total = total + torch.linalg.norm(p.double(), 2).to(p.dtype)
    return total


class RLNetExternal(BaseEstimator, ClassifierMixin):
    """Defaults: 20 rules, 9 quantile thresholds per real column, 3000 epochs, Adam at 0.01, batch 5% of the training
    part, AND-layer sparsity 1e-3, no L2 on the output layer."""

    def __init__(self, n_rules=20, n_bins=9, epochs=3000, batch_size=None, learning_rate=0.01, lambda_and=1e-3,
                 validation_fraction=0.2, l2_lambda=0.0, random_state=0):
        self.n_rules = n_rules
        self.n_bins = n_bins
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_and = lambda_and
        self.validation_fraction = validation_fraction
        self.l2_lambda = l2_lambda
        self.random_state = random_state

    def fit(self, X, y):
        with prepend_sys_path(repo_path('RLNet')):
            import torch
            from networkTorch_multiClass import Network

        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.imputer_ = SimpleImputer(strategy='median').fit(X)
        Xi = self.imputer_.transform(X)
        self.binarizer_ = QuantileBinarizer(self.n_bins).fit(Xi)
        Xb = self.binarizer_.transform(Xi)

        indices = np.arange(len(yi))
        if 0.0 < self.validation_fraction < 1.0 and len(yi) > len(self.classes_):
            try:
                train_idx, val_idx = train_test_split(indices, test_size=self.validation_fraction,
                                                      random_state=self.random_state, stratify=yi)
            except ValueError:                              # a class with a single row cannot be stratified
                train_idx, val_idx = train_test_split(indices, test_size=self.validation_fraction,
                                                      random_state=self.random_state)
        else:
            train_idx = val_idx = indices
        batch_size = self.batch_size or max(1, int(len(train_idx) * 0.05))

        self.model_ = Network(Xb.shape[1], self.n_rules, len(self.classes_))
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        ds = torch.utils.data.TensorDataset(torch.tensor(Xb[train_idx], dtype=torch.float32),
                                            torch.tensor(yi[train_idx], dtype=torch.long))
        loader = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
        Xval = torch.tensor(Xb[val_idx], dtype=torch.float32)
        yval = torch.tensor(yi[val_idx], dtype=torch.long)

        best_loss = np.inf
        best_state = None
        for _ in range(self.epochs):
            self.model_.train()
            for xb, yb in loader:
                opt.zero_grad()
                proba, _ = self.model_(xb)
                loss = (loss_fn(proba, yb) + self.lambda_and * self.model_.regularization()
                        + self.l2_lambda * _l2(self.model_.output.parameters()))
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_proba, _ = self.model_(Xval)
                val_loss = (loss_fn(val_proba, yval) + self.lambda_and * self.model_.regularization()
                            + self.l2_lambda * _l2(self.model_.output.parameters()))
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        import torch
        xb = torch.tensor(self.binarizer_.transform(self.imputer_.transform(np.asarray(X, dtype=float))), dtype=torch.float32)
        with torch.no_grad():
            p, _ = self.model_(xb)
            p = p.cpu().numpy()
        p = np.clip(p, EPS, None)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
