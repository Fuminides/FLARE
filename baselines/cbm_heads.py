"""The compared heads of a concept bottleneck model: a classifier that reads the concepts (truth degrees in [0, 1]) and
predicts the class (experiments/cbm/run_heads.py).

Most heads are the adapters of the tabular experiments (RRL, RL-Net, DR-Net, NeuRules, GradTree, logistic regression);
`build_head` gives each of them the settings of the concept experiments. Two heads of the literature have no adapter
elsewhere and are written here:

* LogicCBM (Vemuri et al., WACV 2026): logic neurons over pairs of concepts followed by a linear layer;
* the Deep Concept Reasoner (DCR, Barbiero et al., ICML 2023): a rule per sample and class, evaluated on the concepts.

All heads have the interface `fit(C, y, C_val, y_val)` / `predict(C)`; the validation rows are used by the heads that
choose their epoch or temperature with them (LogicCBM, DCR) and ignored by the others.
"""
import copy
import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# --- LogicCBM ---------------------------------------------------------------------------------------------------------

def difflogic_gates(a, b):
    """The 16 real-valued binary logic operations of Petersen et al. (2022), stacked on the last axis."""
    ab = a * b
    return torch.stack([torch.zeros_like(a), ab, a - ab, a, b - ab, b, a + b - 2 * ab, a + b - ab,
                        1 - (a + b - ab), 1 - (a + b - 2 * ab), 1 - b, 1 - b + ab, 1 - a, 1 - a + ab, 1 - ab,
                        torch.ones_like(a)], dim=-1)


class LogicModule(nn.Module):
    """LogicCBM head: p two-input logic neurons, then a linear predicate -> class layer.

    Each neuron takes the top-2 concepts of its row of CP and the max of the softmax(G)-weighted 16 gates. To give CP a
    gradient through the top-2 choice, the chosen activations are scaled by w / w.detach() with w = softmax(CP row),
    which leaves the forward pass unchanged and lets the gradient reach CP.
    """

    def __init__(self, n_concepts, n_classes, n_neurons):
        super().__init__()
        self.CP = nn.Parameter(torch.randn(n_neurons, n_concepts) * 0.01)
        self.G = nn.Parameter(torch.randn(n_neurons, 16) * 0.01)
        self.V = nn.Linear(n_neurons, n_classes)

    def pairs(self):
        return self.CP.topk(2, dim=1).indices

    def forward(self, c):
        w = self.CP.softmax(dim=1)
        idx = self.pairs()
        scale = (w.gather(1, idx) / w.gather(1, idx).detach())
        a, b = c[:, idx[:, 0]] * scale[:, 0], c[:, idx[:, 1]] * scale[:, 1]
        z = (self.G.softmax(dim=1) * difflogic_gates(a, b)).max(dim=-1).values
        return self.V(z)


class LogicCBMHead:
    """Trains LogicModule with cross-entropy (Adam, weight decay 1e-4); keeps the best validation state."""

    def __init__(self, n_neurons=250, epochs=100, batch_size=64, lr=1e-2, weight_decay=1e-4, random_state=0, device=None):
        self.n_neurons, self.epochs, self.batch_size = n_neurons, epochs, batch_size
        self.lr, self.weight_decay, self.random_state = lr, weight_decay, random_state
        self.device = device or default_device()

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.random_state)
        device = self.device
        self.classes_, yi = np.unique(y, return_inverse=True)
        X, yi = torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(yi, device=device)
        if X_val is None:
            X_val, yv = X, yi
        else:
            X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
            yv = torch.tensor(np.searchsorted(self.classes_, y_val), device=device)
        self.model_ = LogicModule(X.shape[1], len(self.classes_), self.n_neurons).to(device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator(device=device).manual_seed(self.random_state)
        best = (-1.0, None)
        for _ in range(self.epochs):
            self.model_.train()
            for batch in torch.randperm(len(X), device=device, generator=generator).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.model_(X[batch]), yi[batch]).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                acc = (self.model_(X_val).argmax(1) == yv).float().mean().item()
            if acc > best[0]:
                best = (acc, copy.deepcopy(self.model_.state_dict()))
        self.model_.load_state_dict(best[1])
        self.complexity_ = float(self.n_neurons)
        return self

    def predict(self, X):
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, dtype=torch.float32, device=next(self.model_.parameters()).device))
        return self.classes_[logits.argmax(1).cpu().numpy()]


# --- Deep Concept Reasoner --------------------------------------------------------------------------------------------

def softselect(values, temperature):
    """DCR's relevance activation, as torch_explain.nn.concepts.softselect: a log-softmax over the concepts, recentred
    by temperature x its mean."""
    scores = torch.log_softmax(values, dim=1)
    return torch.sigmoid(scores - temperature * scores.mean(dim=1, keepdim=True))


def godel_iff(a, b):
    return torch.minimum(torch.maximum(1 - a, b), torch.maximum(a, 1 - b))


class ConceptReasoner(nn.Module):
    """Deep Concept Reasoner on concept truth degrees c.

    DCR reads concept embeddings to build a rule per sample and class, and evaluates the rule on the truth degrees only.
    A head on given concepts has no image features, so the embeddings come from torch_explain's ConceptEmbedding with
    the concept vector as its input: per concept, a context Linear(k, 2m) + LeakyReLU split into c+ and c-, mixed as
    c_i c+ + (1 - c_i) c- with the given truth degree in place of the CEM's own concept prediction (the CEM under full
    intervention). The k context layers are one Linear(k, 2mk): the same functions and the same initial distribution.
    The reasoning layer is torch_explain's ConceptReasoningLayer with Goedel semantics: role (sign) and relevance (filter)
    networks Linear(m, m) + LeakyReLU + Linear(m, classes) per concept embedding, literal = sign <-> c, relevance
    implication max(literal, 1 - relevance), class truth = min over concepts.
    """

    def __init__(self, n_concepts, n_classes, emb_size, temperature):
        super().__init__()
        self.emb_size, self.temperature = emb_size, temperature
        self.context = nn.Sequential(nn.Linear(n_concepts, 2 * emb_size * n_concepts), nn.LeakyReLU())
        self.filter_nn = nn.Sequential(nn.Linear(emb_size, emb_size), nn.LeakyReLU(), nn.Linear(emb_size, n_classes))
        self.sign_nn = nn.Sequential(nn.Linear(emb_size, emb_size), nn.LeakyReLU(), nn.Linear(emb_size, n_classes))

    def forward(self, c, return_attn=False):
        context = self.context(c).view(len(c), c.shape[1], 2 * self.emb_size)
        emb = context[..., :self.emb_size] * c.unsqueeze(-1) + context[..., self.emb_size:] * (1 - c.unsqueeze(-1))
        values = c.unsqueeze(-1).expand(-1, -1, self.sign_nn[-1].out_features)
        sign = torch.sigmoid(self.sign_nn(emb))
        relevance = softselect(self.filter_nn(emb), self.temperature)
        preds = torch.maximum(godel_iff(sign, values), 1 - relevance).min(dim=1).values
        return (preds, sign, relevance) if return_attn else preds


class DCRHead:
    """Trains ConceptReasoner as the DCR authors do: BCE on one-hot labels, Adam lr 1e-2, weight decay 4e-5, batch 256, up
    to 500 epochs, learning rate x 0.1 after 10 epochs without a better validation loss, early stopping after 15; keeps
    the best validation-loss state. Temperature 100 and embedding size 16 are the settings DCR uses on CelebA (128 on its
    small synthetic tasks). Predicts the class with the largest truth (ties: the first class).

    Global rules as torch_explain's explain(mode='global'): per training row, each class with truth > 0.5 contributes the
    Booleanised rule (a concept enters when its literal, not 1 - relevance, attains the disjunction; negated when
    sign < 0.5); rules_ maps (class, literals) to its count, literals a tuple of +1 / -1 / 0 per concept.
    """

    def __init__(self, emb_size=16, temperature=100.0, epochs=500, batch_size=256, lr=1e-2, weight_decay=4e-5,
                 patience=15, lr_patience=10, random_state=0, device=None):
        self.emb_size, self.temperature, self.epochs, self.batch_size = emb_size, temperature, epochs, batch_size
        self.lr, self.weight_decay, self.patience, self.lr_patience = lr, weight_decay, patience, lr_patience
        self.random_state = random_state
        self.device = device or default_device()

    def _tensor(self, X):
        return torch.tensor(np.asarray(X), dtype=torch.float32, device=next(self.model_.parameters()).device)

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.random_state)
        device = self.device
        self.classes_, yi = np.unique(y, return_inverse=True)
        onehot = lambda labels: nn.functional.one_hot(torch.tensor(labels, device=device), len(self.classes_)).float()  # noqa: E731
        X, Y = torch.tensor(X, dtype=torch.float32, device=device), onehot(yi)
        if X_val is None:
            X_val, Y_val = X, Y
        else:
            X_val, Y_val = torch.tensor(X_val, dtype=torch.float32, device=device), onehot(np.searchsorted(self.classes_, y_val))
        self.model_ = ConceptReasoner(X.shape[1], len(self.classes_), self.emb_size, self.temperature).to(device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=self.lr_patience)
        loss_fn = nn.BCELoss()
        generator = torch.Generator(device=device).manual_seed(self.random_state)
        best, waited = (np.inf, None), 0
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            for batch in torch.randperm(len(X), device=device, generator=generator).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.model_(X[batch]), Y[batch]).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val), Y_val).item()
            scheduler.step(val_loss)
            if val_loss < best[0]:
                best, waited = (val_loss, copy.deepcopy(self.model_.state_dict())), 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.epochs_run_ = epoch
        self.model_.load_state_dict(best[1])
        self.rules_, self.covered_ = self.global_rules(X)
        self.n_rules_ = len(self.rules_)
        self.complexity_ = float(sum(np.count_nonzero(literals) for _, literals in self.rules_))
        return self

    def truths(self, X, batch_size=1024):
        with torch.no_grad():
            C = self._tensor(X)
            return torch.cat([self.model_(C[i:i + batch_size]) for i in range(0, len(C), batch_size)]).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.truths(X).argmax(1)]

    def global_rules(self, X, batch_size=256):
        """(Counter {(class index, literals): rows}, share of rows with at least one class above 0.5)."""
        rules, covered = Counter(), 0
        self.model_.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                c = X[i:i + batch_size] if torch.is_tensor(X) else self._tensor(X[i:i + batch_size])
                preds, sign, relevance = self.model_(c, return_attn=True)
                used = (1 - relevance) < godel_iff(sign, c.unsqueeze(-1).expand_as(sign))
                literals = torch.where(used, torch.where(sign >= 0.5, 1, -1), 0).to(torch.int8)   # [B, k, classes]
                active = preds > 0.5
                covered += int(active.any(1).sum())
                for row, k in active.nonzero().tolist():
                    rules[(k, tuple(literals[row, :, k].tolist()))] += 1
        return rules, covered / len(X)


class DCRValidated:
    """DCRHead with the temperature chosen on validation macro-F1 among 100, 10 and 1 (1 is no temperature); a tie keeps
    the earlier one. The chosen head answers for every attribute."""

    def __init__(self, temperatures=(100.0, 10.0, 1.0), random_state=0, **kwargs):
        self.temperatures, self.random_state, self.kwargs = temperatures, random_state, kwargs

    def fit(self, X, y, X_val, y_val):
        best = (-1.0, None)
        self.validation_f1_ = {}
        for temperature in self.temperatures:
            head = DCRHead(temperature=temperature, random_state=self.random_state, **self.kwargs).fit(X, y, X_val, y_val)
            self.validation_f1_[temperature] = float(f1_score(y_val, head.predict(X_val), average='macro'))
            if self.validation_f1_[temperature] > best[0]:
                best = (self.validation_f1_[temperature], head)
        self.head_ = best[1]
        return self

    def __getattr__(self, name):
        if name == 'head_':
            raise AttributeError(name)
        return getattr(self.head_, name)


# --- the heads of the concept experiments -----------------------------------------------------------------------------

HEADS = ['LR', 'RRL', 'RL-Net', 'DR-Net', 'GradTree', 'NeuRules', 'LogicCBM', 'DCR']
CRISP = ('RRL', 'RL-Net', 'DR-Net', 'GradTree')     # rule learners over binary inputs: the concepts are cut at 0.5
MAX_CLASSES = {'DR-Net': 20}                        # its network is binary: one per class, so not run beyond this many


def rule_count(n_classes):
    """RRL's width and the rules of RL-Net, NeuRules and LogicCBM: at least one per class, and 250 at least."""
    return max(250, n_classes + 50)


def runs_on(head, n_classes):
    return n_classes <= MAX_CLASSES.get(head, np.inf)


class ConceptHead:
    """One head of the concept experiments.

    `name` is one of HEADS ('LR' is the class-balanced logistic regression). The rule learners of CRISP read the concepts
    cut at 0.5 in training and in prediction (RL-Net and DR-Net need binary inputs); the others read them as they are.
    RRL's width, the rules of RL-Net, NeuRules and LogicCBM are `rule_count`; GradTree's depth is
    max(6, ceil(log2 classes) + 1), so that it has leaves for every class; DCR is the variant that chooses its
    temperature among 100, 10 and 1 on the validation rows.
    """

    def __init__(self, name, n_classes, seed, device=None):
        if name not in HEADS:
            raise KeyError(f'{name}: not one of {HEADS}')
        self.name, self.n_classes, self.seed, self.device = name, n_classes, seed, device or default_device()

    def _build(self):
        name, rules, seed = self.name, rule_count(self.n_classes), self.seed
        if name == 'LR':
            return LogisticRegression(max_iter=3000, class_weight='balanced')
        if name == 'RRL':
            from baselines.rrl import RRLExternal
            return RRLExternal(hidden_width=rules, random_state=seed)
        if name == 'RL-Net':
            from baselines.rlnet import RLNetExternal
            return RLNetExternal(n_rules=rules, random_state=seed)
        if name == 'DR-Net':
            from baselines.drnet import DRNetModel
            return DRNetModel(seed)
        if name == 'GradTree':
            from baselines.gradtree import GradTreeClassifier
            return GradTreeClassifier(depth=max(6, math.ceil(math.log2(self.n_classes)) + 1), random_state=seed, device=self.device)
        if name == 'NeuRules':
            from baselines.neurules import NeuRules
            return NeuRules(n_rules=rules, random_state=seed)
        if name == 'LogicCBM':
            return LogicCBMHead(n_neurons=rules, random_state=seed, device=self.device)
        return DCRValidated(random_state=seed, device=self.device)

    def _inputs(self, C):
        C = np.asarray(C)
        return (C > 0.5).astype(np.float32) if self.name in CRISP else C

    def fit(self, C, y, C_val=None, y_val=None):
        self.model = self._build()
        if self.name in ('LogicCBM', 'DCR'):
            self.model.fit(self._inputs(C), y, self._inputs(C_val), y_val)
        else:
            self.model.fit(self._inputs(C), y)
        return self

    def predict(self, C):
        return np.asarray(self.model.predict(self._inputs(C)))

    def macro_f1(self, y_true, y_pred):
        """Macro-F1 for the heads that read crisp concepts and for the logistic regression is taken over the classes
        present in the test labels; for the others, over the classes present in the labels or in the predictions
        (scikit-learn's default)."""
        if self.name in CRISP or self.name == 'LR':
            return float(f1_score(y_true, y_pred, labels=np.unique(y_true), average='macro', zero_division=0))
        return float(f1_score(y_true, y_pred, average='macro'))
