"""DR-Net (Qiao, Wang and Lin, AAAI 2021, Decision Rules Network) through the authors' code
(github.com/Joeyonng/decision-rules-network).

The authors' `DRNet.DRNet` is a binary classifier, so beyond two classes one network is trained per class and the
largest OR output decides. The defaults are the settings of the authors' tutorial notebook (50 rules, Adam at 1e-2,
2000 epochs, batch 400, AND / OR sparsity 1e-2 / 1e-5, the AND and OR layers optimised in turns of 500 epochs). Their
loop drops incomplete batches, so the batch is capped at the training size. CPU, sklearn-like interface.
"""
import numpy as np
import torch

from baselines.binarizer import QuantileBinarizer
from baselines.external import prepend_sys_path, repo_path

LITERAL = 1e-6      # the rule neurons fire above 0.999999, so smaller weights never block a rule


class DRNetModel:

    def __init__(self, seed, rules=50, epochs=2000, batch=400, lr=1e-2, and_lam=1e-2, or_lam=1e-5, alter=500, bins=9):
        self.seed, self.n_rules, self.epochs, self.batch, self.lr = seed, rules, epochs, batch, lr
        self.lams, self.alter, self.bins = (and_lam, or_lam), alter, bins

    def fit(self, X, y):
        with prepend_sys_path(repo_path('DRNet')):
            import DRNet as drnet
        self.drnet = drnet
        self.classes_ = np.unique(y)
        self.binarizer = QuantileBinarizer(self.bins).fit(X)
        Xb = torch.tensor(self.binarizer.transform(X), dtype=torch.float32)
        targets = [self.classes_[1]] if len(self.classes_) == 2 else list(self.classes_)
        self.nets = [self._train(Xb, torch.tensor((y == c).astype(np.float32))) for c in targets]
        return self

    def _train(self, Xb, t):
        torch.manual_seed(self.seed)
        net = self.drnet.DRNet(Xb.shape[1], self.n_rules, 1)
        optimizers = [torch.optim.Adam(net.and_layer.parameters(), lr=self.lr), torch.optim.Adam(net.or_layer.parameters(), lr=self.lr)]
        criterion = torch.nn.BCEWithLogitsLoss()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xb, t), batch_size=min(self.batch, len(t)), shuffle=True)
        net.train()
        for epoch in range(self.epochs):
            phase = int((epoch / self.alter) % 2)
            for xb, tb in loader:
                out = net(xb)
                loss = criterion(out, tb.reshape(out.size())) + self.lams[phase] * net.regularization()
                optimizers[phase].zero_grad()
                loss.backward()
                optimizers[phase].step()
        net.eval()
        return net

    def scores(self, X):
        Xb = torch.tensor(self.binarizer.transform(X), dtype=torch.float32)
        with torch.no_grad():
            return np.column_stack([net(Xb).numpy().ravel() for net in self.nets])

    def predict(self, X):
        s = self.scores(X)
        if len(self.classes_) == 2:
            return np.where(s[:, 0] >= 0, self.classes_[1], self.classes_[0])
        return self.classes_[s.argmax(axis=1)]

    def rules(self):
        """(positive literals, negative literals, 'and') of every AND neuron some OR neuron uses, over the binary columns."""
        out = []
        for net in self.nets:
            W = net.and_layer.masked_weight().detach().numpy()
            used = np.flatnonzero(net.or_layer.masked_weight().detach().numpy().ravel() != 0)
            out += [(np.flatnonzero(W[r] > LITERAL), np.flatnonzero(W[r] < -LITERAL), 'and') for r in used]
        return out
