'''GradTree (Marton et al., AAAI 2024) in PyTorch: one hard, axis-aligned decision tree of fixed depth trained by
gradient descent.

Every internal node has a feature-choice vector and one threshold per feature (the dense representation). The forward
pass is the hard tree: entmax-1.5 then hardmax picks one feature (straight-through: the backward pass uses the entmax
distribution), and the split is the rounded sigmoid of feature minus threshold (straight-through: the backward pass uses
the sigmoid). Leaf indicators are products of split tests along the paths and leaves hold real-valued class scores.
Adam with separate learning rates for the feature choices, the thresholds and the leaves, class-balanced cross-entropy,
early stopping on the validation loss (best weights restored), inputs quantile-transformed to a normal distribution.

This is a PyTorch reimplementation of the official TensorFlow code (github.com/s-marton/GradTree), whose defaults it
uses: depth 6, learning rates 0.01, batch 64, patience 25 with min_delta 1e-3, random-normal initialisation, class
balancing on. Categorical features are one-hot encoded rather than leave-one-out encoded, since leave-one-out values
leak each training row's label and the quantile transform makes them easy to split on.
'''
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer


class _Entmax15(torch.autograd.Function):
    '''entmax with alpha = 1.5 over the last dimension (Peters, Niculae and Martins, 2019), exact forward and backward.'''

    @staticmethod
    def forward(ctx, X):
        X = (X - X.max(dim=-1, keepdim=True).values) / 2
        Xs = torch.sort(X, dim=-1, descending=True).values
        rho = torch.arange(1, X.shape[-1] + 1, device=X.device, dtype=X.dtype)
        mean = Xs.cumsum(-1) / rho
        mean_sq = (Xs ** 2).cumsum(-1) / rho
        delta = (1 - rho * (mean_sq - mean ** 2)) / rho
        tau = mean - torch.sqrt(delta.clamp(min=0))
        support = (tau <= Xs).sum(dim=-1, keepdim=True)
        Y = (X - tau.gather(-1, support - 1)).clamp(min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        Y, = ctx.saved_tensors
        g = Y.sqrt()
        dX = dY * g
        return dX - (dX.sum(-1, keepdim=True) / g.sum(-1, keepdim=True)) * g


def paths(depth):
    '''(node [L, d], direction [L, d]) of every leaf: breadth-first node index and branch (1 = right, x >= threshold).'''
    leaves = 2 ** depth
    node = np.zeros((leaves, depth), dtype=np.int64)
    direction = np.zeros((leaves, depth), dtype=np.float32)
    for leaf in range(leaves):
        n = 0
        for j in range(depth):
            bit = (leaf >> (depth - 1 - j)) & 1
            node[leaf, j], direction[leaf, j] = n, bit
            n = 2 * n + 1 + bit
    return node, direction


class GradTreeClassifier:

    def __init__(self, depth=6, lr_index=0.01, lr_values=0.01, lr_leaf=0.01, batch_size=64, max_epochs=1000, patience=25,
                 min_delta=1e-3, val_fraction=0.2, class_balance=True, random_state=0, device='cpu'):
        self.depth, self.lr_index, self.lr_values, self.lr_leaf = depth, lr_index, lr_values, lr_leaf
        self.batch_size, self.max_epochs, self.patience, self.val_fraction = batch_size, max_epochs, patience, val_fraction
        self.min_delta, self.class_balance, self.random_state, self.device = min_delta, class_balance, random_state, device

    def _forward(self, X):
        soft = _Entmax15.apply(self.index_)                                          # [M, F]
        hard = torch.nn.functional.one_hot(soft.argmax(dim=1), soft.shape[1]).to(soft.dtype)
        choice = soft + (hard - soft).detach()
        z = X @ choice.T - (choice * self.threshold_).sum(dim=1)                    # [B, M]: chosen feature minus its threshold
        s = torch.sigmoid(z)
        s = s + (torch.round(s) - s).detach()                                       # 1 = right branch
        s = s[:, self.node_]                                                         # [B, L, d]
        leaves = torch.prod(self.direction_ * s + (1 - self.direction_) * (1 - s), dim=2)
        return leaves @ self.leaf_

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.quantile_ = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X)),
                                             random_state=self.random_state).fit(X)
        Xq = self.quantile_.transform(X).astype(np.float32)
        try:
            tr, va = train_test_split(np.arange(len(yi)), test_size=self.val_fraction, random_state=self.random_state, stratify=yi)
        except ValueError:
            tr, va = train_test_split(np.arange(len(yi)), test_size=self.val_fraction, random_state=self.random_state)
        dev = torch.device(self.device)
        Xt, yt = torch.tensor(Xq[tr], device=dev), torch.tensor(yi[tr], device=dev)
        Xv, yv = torch.tensor(Xq[va], device=dev), torch.tensor(yi[va], device=dev)
        M, F, L, C = 2 ** self.depth - 1, Xq.shape[1], 2 ** self.depth, len(self.classes_)
        node, direction = paths(self.depth)
        self.node_ = torch.tensor(node, device=dev)
        self.direction_ = torch.tensor(direction, device=dev)
        self.index_ = torch.nn.Parameter(torch.randn(M, F, device=dev))
        self.threshold_ = torch.nn.Parameter(torch.randn(M, F, device=dev))         # inputs are standard normal after the transform
        self.leaf_ = torch.nn.Parameter(0.1 * torch.randn(L, C, device=dev))
        weight = None
        if self.class_balance:                                                      # sklearn's 'balanced' weights
            counts = np.bincount(yi[tr], minlength=C).astype(np.float64)
            weight = torch.tensor(len(tr) / (C * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
        opt = torch.optim.Adam([{'params': [self.index_], 'lr': self.lr_index}, {'params': [self.threshold_], 'lr': self.lr_values},
                                {'params': [self.leaf_], 'lr': self.lr_leaf}])
        best, best_state, waited = np.inf, None, 0
        generator = torch.Generator(device='cpu').manual_seed(self.random_state)
        for epoch in range(self.max_epochs):
            order = torch.randperm(len(yt), generator=generator).to(dev)
            for start in range(0, len(yt), self.batch_size):
                batch = order[start:start + self.batch_size]
                loss = torch.nn.functional.cross_entropy(self._forward(Xt[batch]), yt[batch], weight=weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                val = float(torch.nn.functional.cross_entropy(self._forward(Xv), yv, weight=weight))
            if val < best - self.min_delta:
                best, waited = val, 0
                best_state = [p.detach().clone() for p in (self.index_, self.threshold_, self.leaf_)]
            else:
                waited += 1
                if waited >= self.patience:
                    break
        with torch.no_grad():
            for p, v in zip((self.index_, self.threshold_, self.leaf_), best_state):
                p.copy_(v)
        self.epochs_ = epoch + 1
        return self

    def transform(self, X):
        return self.quantile_.transform(X).astype(np.float32)

    def predict(self, X):
        with torch.no_grad():
            out = self._forward(torch.tensor(self.transform(X), device=torch.device(self.device)))
        return self.classes_[out.argmax(dim=1).cpu().numpy()]
