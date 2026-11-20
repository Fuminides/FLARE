"""FLARE on graphs: node classification with quantified neighbourhood conditions.

Each node is a sample. Besides its own variables, a rule can read, for every condition "x_j is l" of the node's own
variables, the share of the node's neighbours for which that condition holds, computed with the truth degree of the
label l. A fuzzy quantifier partition of [0, 1] with three labels (SOME, MANY, MOST) turns each share into a new
variable, so a rule can say "MANY neighbours HAVE word_12 IS present". The shares are constants of the graph, so they
are computed once, from partitions fitted on the training nodes, and every node becomes one ordinary row:

    [ own variables (real ones standardised with the training nodes) | shares of the neighbours (one block per hop) ]

`neighbourhood_features` builds these rows and `GraphInputs` is the inputs module (see inputs.py) that turns the real
columns into truths: the quantile partitions of the own variables and the fixed quantifier partitions of the shares.
`fit_graph` trains FLARE on the labelled nodes of a split with `train.fit_fold`.

Differences from the tabular inputs:

* The labels of a variable are the trapezoids of a partition whose breakpoints are described by the 4 L - 3 gaps between
  them (`DeltaPartitions`), clipped at zero and rescaled to the range of the variable. With `learn_sets=True` (the
  default) the gaps are trained like the rest of the rule structure, with the step of the selectors, and every averaged
  model learns its own; the partitions of the tabular FLARE are Ruspini partitions that the first model learns and the
  others read frozen (labels.py).
* One step size for the points, the selectors and the interaction rules, and a floor of 0.1 on the uniform share k.
* The averaged models are always trained one after another.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .config import FLAREConfig
from .labels import trapezoid_constants, trapezoid_memberships
from .metrics import eval_chunk_rows
from .partitions import label_names, quantile_partitions
from .preprocessing import Fold
from .selectors import GradMode
from .train import fit_fold

QUANTIFIERS = ('SOME', 'MANY', 'MOST')          # the labels of the quantifier partition (three labels, like the variables)
N_LABELS = len(QUANTIFIERS)
HOP_NAMES = ['partners', 'partners-of-partners', 'partners-at-three-hops']


# --- the fixed partition of a share --------------------------------------------------------------------------------

def quantifier_partition(n_labels=N_LABELS):
    """[labels, 4] trapezoids of a Ruspini partition of [0, 1] that classify the share of the neighbours that satisfy a
    condition. The domain is cut into 2 L - 1 equal segments, alternating plateau and overlap, so with three labels SOME
    ends at 0.2, MANY is centred on half of the neighbours and MOST starts at 0.8."""
    width = 1.0 / (2 * n_labels - 1)
    sets = []
    for k in range(n_labels):
        a, b, c, d = (2 * k - 1) * width, 2 * k * width, (2 * k + 1) * width, (2 * k + 2) * width
        if k == 0:
            a = b = 0.0
        if k == n_labels - 1:
            c = d = 1.0
        sets.append([a, b, c, d])
    return np.array(sets)


# --- partitions described by the gaps between breakpoints ---------------------------------------------------------

def delta_chain(n_labels):
    """The breakpoints of L interleaved trapezoids in increasing order, as (label, parameter) pairs after a_0 = b_0 =
    the start of the domain: c_0, then a_k, d_{k-1}, b_k, c_k for each later label k (d_{L-1} is c_{L-1}). The gaps
    between consecutive breakpoints are 4 L - 3 numbers."""
    chain = [(0, 2)]
    for k in range(1, n_labels):
        chain += [(k, 0), (k - 1, 3), (k, 1), (k, 2)]
    return chain


def delta_trapezoids(starts, ends, deltas):
    """Trapezoid parameters [variables, labels, 4] from the gaps: negative gaps are clipped to zero, the gaps are rescaled
    to sum to the range of each variable, and the breakpoints are accumulated one after another from its start."""
    n_labels = (deltas.shape[1] + 3) // 4
    deltas = torch.maximum(deltas, torch.zeros_like(deltas))
    deltas = deltas / torch.sum(deltas, dim=1, keepdim=True) * (ends - starts).unsqueeze(1)
    parameters = [[None] * 4 for _ in range(n_labels)]
    parameters[0][0] = parameters[0][1] = starts
    point = starts
    for j, (label, index) in enumerate(delta_chain(n_labels)):
        point = point + deltas[:, j]
        parameters[label][index] = point
    parameters[n_labels - 1][3] = parameters[n_labels - 1][2]
    return torch.stack([torch.stack(label_parameters, dim=1) for label_parameters in parameters], dim=1)


class DeltaPartitions(nn.Module):
    """The partitions of several variables, [N, V] -> [N, V, L] memberships. The gaps are a parameter when `learn` is
    True and a constant otherwise (the memberships are then rebuilt only when the gaps change)."""

    def __init__(self, partitions, learn, device='cpu'):
        """partitions: [V, L, 4] trapezoids (partitions.py: quantiles of the training nodes, or `quantifier_partition`)."""
        super().__init__()
        partitions = np.asarray(partitions)
        chain = delta_chain(partitions.shape[1])
        gaps = np.zeros((partitions.shape[0], len(chain)))
        for i in range(partitions.shape[0]):
            previous = partitions[i, 0, 0]
            for j, (label, index) in enumerate(chain):
                gaps[i, j] = partitions[i, label, index] - previous
                previous = partitions[i, label, index]
        gaps = torch.tensor(gaps.astype(np.float32), device=device)
        if learn:
            self.deltas = nn.Parameter(gaps)
        else:
            self.register_buffer('deltas', gaps, persistent=False)
        self.register_buffer('begin_points', torch.tensor(partitions[:, 0, 0].astype(np.float32), device=device), persistent=False)
        self.register_buffer('end_points', torch.tensor(partitions[:, -1, -1].astype(np.float32), device=device), persistent=False)

    def forward(self, x):
        key = (self.deltas._version, self.deltas.device)
        if self.deltas.requires_grad or getattr(self, '_frozen_key', None) != key:
            constants = trapezoid_constants(delta_trapezoids(self.begin_points, self.end_points, self.deltas).unsqueeze(0))
            self._frozen_key, self._frozen_constants = (None, None) if self.deltas.requires_grad else (key, constants)
        else:
            constants = self._frozen_constants
        return trapezoid_memberships(x, constants)


# --- the inputs module ---------------------------------------------------------------------------------------------

class GraphInputs(nn.Module):
    """Inputs module of FLARE for graphs (the contract is in inputs.py). The real columns of a node's row are the own
    real variables followed by the shares of each block; every block has its own partitions."""

    def __init__(self, own_partitions, share_blocks, learn, device='cpu'):
        """own_partitions: [V, L, 4] quantile trapezoids of the own real variables (None when there are none).
        share_blocks: [(number of shares in the block, [L, 4] quantifier partition)]."""
        super().__init__()
        blocks = []
        if own_partitions is not None and len(own_partitions) > 0:
            blocks.append(DeltaPartitions(own_partitions, learn, device))
        self.n_own = len(own_partitions) if own_partitions is not None else 0
        for n, partition in share_blocks:
            if n > 0:
                blocks.append(DeltaPartitions(np.repeat(np.asarray(partition)[None], n, axis=0), learn, device))
        self.fuzzifiers = nn.ModuleList(blocks)
        self.widths = [block.deltas.shape[0] for block in blocks]
        self.offsets = np.cumsum([0] + self.widths).tolist()
        self.n_variables = sum(self.widths)
        self.n_conditions = N_LABELS
        self.label_names = label_names(N_LABELS)

    def forward(self, x):
        return torch.cat([block(x[:, self.offsets[i]:self.offsets[i + 1]]) for i, block in enumerate(self.fuzzifiers)], dim=1)

    def crossing_parameters(self):
        return []                           # the gaps are trained with the step of the selectors

    def space_parameters(self):
        return []

    def inherit(self, first):
        """Nothing is shared: every averaged model keeps the partitions it learns itself."""

    def condition_text(self, variable, condition, name, category_names=None):
        if variable < self.n_own:
            return f'{name} IS {self.label_names[condition]}'
        return f'{QUANTIFIERS[condition]} {name}'


# --- the rows of the nodes -----------------------------------------------------------------------------------------

def row_normalise(adjacency):
    """The adjacency with every row summing to one (isolated rows stay zero)."""
    import scipy.sparse as sp
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.where(degree > 0, 1.0 / np.maximum(degree, 1e-12), 0.0)
    return sp.diags(inverse).dot(adjacency).tocsr()


def hop_matrices(adjacency, hops):
    """[A_1, ..., A_hops]: the row-normalised adjacency, then the row-normalised binary reachability in two steps (paths
    of length two, without self loops), and so on. Multiplying by A_h averages a node's neighbours at h hops."""
    matrices, reach = [], adjacency
    for hop in range(hops):
        if hop > 0:
            reach = reach @ adjacency
            reach = (reach > 0).astype(np.float32)
            reach.setdiag(0.0)
            reach.eliminate_zeros()
        matrices.append(row_normalise(reach))
    return matrices


@dataclass
class GraphFeatures:
    """The rows of all the nodes of a graph and what FLARE needs to read them."""
    X: np.ndarray                  # [nodes, columns] float32: the own columns, then the shares of every block
    categorical: np.ndarray        # [columns] bool: the own binary / categorical columns
    cardinalities: list            # number of categories of each categorical column
    names: list                    # a readable name of each column
    own_partitions: object         # [own real variables, L, 4] or None
    share_blocks: list = field(default_factory=list)

    def inputs_factory(self, learn_sets=True):
        return lambda device: GraphInputs(self.own_partitions, self.share_blocks, learn_sets, device).to(device)


def neighbourhood_features(X, categorical, adjacency, fit_index, hops=1, hop_names=None, category_names=None,
                           quantify_categorical='all'):
    """The row of every node: its own variables and, for each hop up to `hops`, the share of its neighbours at that
    hop that satisfy each condition of the own variables.

    X: DataFrame of the own variables of all the nodes (categorical columns as integer codes). adjacency: SciPy sparse
    matrix, symmetric, binary, without self loops. fit_index: the training nodes, the only ones whose rows fit the
    standardisation of the real variables and the quantile partitions. quantify_categorical: 'all' counts the
    neighbours in every category; 'nonzero' only in the categories other than the first (for 0 / 1 indicators, where
    "absent" is the complement of "present"). category_names: {column: [name of each category]}."""
    from sklearn.preprocessing import StandardScaler
    categorical = np.asarray(categorical, dtype=bool)
    real_columns, categorical_columns = np.flatnonzero(~categorical), np.flatnonzero(categorical)
    columns = list(X.columns)
    values = X.values.astype(float)
    if len(real_columns) > 0:
        scaler = StandardScaler().fit(values[np.asarray(fit_index)][:, real_columns])
        values[:, real_columns] = scaler.transform(values[:, real_columns])
    own = values.astype(np.float32)             # the precision of the model; the partitions are fitted on these numbers
    cardinalities = [int(np.max(X.values[:, c])) + 1 for c in categorical_columns]
    category_names = category_names or {}
    hop_names = hop_names or HOP_NAMES

    own_partitions = None
    if len(real_columns) > 0:
        own_partitions = quantile_partitions(own[np.asarray(fit_index)][:, real_columns], N_LABELS)

    # Every condition of the own variables (real variables first, as in the model) and its truth at every node.
    # Categorical conditions are crisp indicators, because a share should count the neighbours that are in the category.
    X_all = torch.as_tensor(own)
    conditions, truths = [], []
    if len(real_columns) > 0:
        memberships = DeltaPartitions(own_partitions, learn=False)(X_all[:, real_columns])       # [nodes, real, L]
        conditions += [(v, label) for v in range(len(real_columns)) for label in range(N_LABELS)]
        truths.append(memberships.flatten(1))
    for i, cardinality in enumerate(cardinalities):
        truths.append(torch.nn.functional.one_hot(X_all[:, categorical_columns[i]].long(), cardinality).to(X_all.dtype))
        conditions += [(len(real_columns) + i, c) for c in range(cardinality)]
    truths = torch.cat(truths, dim=1)
    if quantify_categorical == 'nonzero':
        keep = [i for i, (variable, value) in enumerate(conditions) if variable < len(real_columns) or value != 0]
    elif quantify_categorical == 'all':
        keep = list(range(len(conditions)))
    else:
        raise ValueError(f"quantify_categorical must be 'all' or 'nonzero', got {quantify_categorical!r}")
    texts = []
    for i in keep:
        variable, value = conditions[i]
        if variable < len(real_columns):
            column, label = int(real_columns[variable]), label_names(N_LABELS)[value]
        else:
            column = int(categorical_columns[variable - len(real_columns)])
            label = category_names[column][value] if column in category_names else str(value)
        texts.append(f'{columns[column]} IS {label}')
    truths = truths[:, torch.as_tensor(keep)].numpy()

    names, parts, share_blocks = list(columns), [], []
    for hop, matrix in enumerate(hop_matrices(adjacency, hops)):
        parts.append(np.asarray(matrix @ truths, dtype=np.float32))
        share_blocks.append((len(texts), quantifier_partition()))
        names += [f'{hop_names[hop]} HAVE {text}' for text in texts]
    return GraphFeatures(np.concatenate([own] + parts, axis=1), np.concatenate([categorical, np.zeros(sum(n for n, _ in share_blocks), dtype=bool)]),
                         cardinalities, names, own_partitions, share_blocks)


# --- training ------------------------------------------------------------------------------------------------------

def graph_config(balanced=False, epochs=100, **overrides):
    """The configuration of FLARE for graphs: FLARE with labels L = 3, two slots per interaction rule, no runs of labels, the
    same step for the points, the selectors and the interaction rules, and k never below 0.1 during the training.
    `balanced` weights the classes by their inverse frequency."""
    config = FLAREConfig(labels=N_LABELS, slots=2, label_runs='none', lr_structure=1.0, lr_interactions=1.0, k_lower_train=0.1,
                         class_weight='balanced' if balanced else 'none', epochs=epochs, parallel_bags=False, cuda_graph=False)
    return config.replace(**overrides)


def graph_fold(features, y, train, val, test):
    """The training, validation and test nodes of a split as a preprocessing.Fold of their rows."""
    train, val, test = (np.asarray(index) for index in (train, val, test))
    X_train, X_val, X_test = (features.X[index].astype(np.float64) for index in (train, val, test))
    return Fold(X_train, y[train], X_val, y[val], X_test, y[test], features.categorical, n_classes=int(y.max()) + 1)


def fit_graph(features, y, train, val, test, config, random_seed, learn_sets=True, device=None, verbose=False):
    """Train FLARE on the nodes `train` (validation nodes: `val`) and return a train.FitResult whose test loader holds the
    nodes `test`. y holds the class of every node (unlabelled nodes are never read). The batch is a fiftieth of the
    training nodes when that is larger than config.batch_size (steps_per_epoch = 50)."""
    if config.steps_per_epoch > 0:
        config = config.replace(batch_size=max(config.batch_size, math.ceil(len(train) / config.steps_per_epoch)), steps_per_epoch=0)
    fold = graph_fold(features, y, train, val, test)
    # Sequential route only: the averaged models do not share their partitions.
    return fit_fold(fold, config, features.cardinalities, random_seed, device=device, verbose=verbose, max_parallel_elements=0,
                    inputs_factory=features.inputs_factory(learn_sets))


@torch.no_grad()
def predict_proba(result):
    """[test nodes, classes] class probabilities of the deployed (discrete) model on the nodes of the test loader, in the
    order of the `test` index that was given to fit_graph."""
    model = result.model
    model.set_inference_mode(GradMode.PURE, 0.0)
    X = result.test_loader.dataset.X
    scores = torch.cat([model(chunk)[0] for chunk in X.split(eval_chunk_rows(model))])
    return torch.softmax(scores, dim=1).cpu().numpy()
