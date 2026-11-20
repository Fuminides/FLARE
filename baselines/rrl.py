"""RRL (Wang et al., NeurIPS 2021, Rule-based Representation Learner) through the authors' code (github.com/12wang3/rrl).

The authors' code is a training script; this adapter builds their `Net` and trains it with a plain loop, with the
sklearn interface the harness needs (fit, predict). It runs on the CPU.
"""
from types import SimpleNamespace

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from baselines.external import prepend_sys_path, repo_path

EPS = 1e-12


class RRLExternal(BaseEstimator, ClassifierMixin):
    """Defaults: 80 epochs, batch 128, 5 binarisation nodes, 64 logical nodes, Adam at 0.002, L2 penalty 1e-4 on the
    logical layers. Real inputs are median-imputed and standardised (their column ranges bound the learnt thresholds)."""

    def __init__(self, epochs=80, batch_size=128, binarization_nodes=5, hidden_width=64, learning_rate=0.002,
                 weight_decay=1e-4, random_state=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.binarization_nodes = binarization_nodes
        self.hidden_width = hidden_width
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        with prepend_sys_path(repo_path('rrl')):
            import torch
            from rrl import components as rrl_components
            from rrl.models import Net

        self._patch_binarize_layer(rrl_components, torch)
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.imputer_ = SimpleImputer(strategy='median').fit(X)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(X))
        Xp = self.scaler_.transform(self.imputer_.transform(X)).astype(np.float32)
        left = torch.tensor(np.nanmin(Xp, axis=0), dtype=torch.float32)
        right = torch.tensor(np.nanmax(Xp, axis=0), dtype=torch.float32)

        self.net_ = Net([(0, Xp.shape[1]), self.binarization_nodes, self.hidden_width, len(self.classes_)],
                        left=left, right=right, use_nlaf=True, use_skip=False, temperature=0.01)
        # The authors store each layer's connection metadata on a local lambda, which cannot be pickled; a plain
        # namespace with the same attributes does the same job in the forward pass.
        for layer in self.net_.layer_list:
            connection = layer.conn
            layer.conn = SimpleNamespace(prev_layer=connection.prev_layer, is_skip_to_layer=connection.is_skip_to_layer,
                                         skip_from_layer=connection.skip_from_layer)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        ds = torch.utils.data.TensorDataset(torch.tensor(Xp, dtype=torch.float32), torch.tensor(yi, dtype=torch.long))
        loader = torch.utils.data.DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True)

        self.net_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.net_(xb) / torch.exp(self.net_.t)
                loss = loss_fn(logits, yb)
                if self.weight_decay:
                    penalty = torch.zeros((), dtype=logits.dtype)
                    for layer in self.net_.layer_list[1:]:
                        if hasattr(layer, 'l2_norm'):
                            penalty = penalty + layer.l2_norm()
                    loss = loss + self.weight_decay * penalty
                loss.backward()
                opt.step()
                with torch.no_grad():                       # keep the thresholds and the weights inside their ranges
                    for layer in self.net_.layer_list:
                        if hasattr(layer, 'clip'):
                            layer.clip()
        self.net_.eval()
        return self

    @staticmethod
    def _patch_binarize_layer(rrl_components, torch):
        """The authors' BinarizeLayer.forward reshapes with `.view`, which fails on the non-contiguous tensor that
        current PyTorch returns for the broadcast subtraction; this is the same forward with `.reshape`."""
        if getattr(rrl_components.BinarizeLayer, '_flare_reshape_patch', False):
            return

        def forward(layer, x):
            if layer.input_dim[1] > 0:
                x_disc, x_cont = x[:, 0: layer.input_dim[0]], x[:, layer.input_dim[0]:]
                x_cont = x_cont.unsqueeze(-1)
                if layer.use_not:
                    x_disc = torch.cat((x_disc, 1 - x_disc), dim=1)
                binarize_res = rrl_components.Binarize.apply(x_cont - layer.cl.t()).reshape(x_cont.shape[0], -1)
                return torch.cat((x_disc, binarize_res, 1.0 - binarize_res), dim=1)
            if layer.use_not:
                x = torch.cat((x, 1 - x), dim=1)
            return x

        rrl_components.BinarizeLayer.forward = forward
        rrl_components.BinarizeLayer._flare_reshape_patch = True

    def predict_proba(self, X):
        import torch
        Xp = torch.tensor(self.scaler_.transform(self.imputer_.transform(np.asarray(X, dtype=float))).astype(np.float32))
        with torch.no_grad():
            logits = self.net_(Xp) / torch.exp(self.net_.t)
            p = torch.softmax(logits, dim=1).cpu().numpy()
        p = np.clip(p, EPS, None)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
