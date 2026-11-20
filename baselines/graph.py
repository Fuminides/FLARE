"""The compared methods on graph node classification.

Every function takes `data`, an `experiments.data.graphs.GraphData` (its X, y, adjacency, name and metric), and the
(training, validation, test) node positions of `split_indices`; each one returns the class probabilities of the test
nodes. The methods are

* graph neural networks, written in PyTorch without extra packages. `sgc`: logistic regression on the features propagated
  with S = D^-1/2 (A + I) D^-1/2 (Wu et al., 2019); `gcn`: a two-layer GCN (Kipf and Welling, 2017) on features standardised
  with the training nodes; `gcn_standard` and `gat_standard`: GCN and GAT as published (row-normalised features and the
  training settings of their authors), on the citation graphs;
* logistic regression on the node's own variables, and on those plus the mean of the neighbours and the logarithm of the
  degree (`logistic`), the fair counterpart of the models that read the graph;
* methods that do not read the graph, which are given the same information as extra columns, `propagated_table`: the own
  variables, the mean of each variable over the neighbours at 1, ..., `hops` hops, and log10(1 + degree). Random forests
  and EBM (`table_model`) and the rule and tree learners of baselines/tabular.py (`rule_learner`) read it with two hops.

Training nodes only fit the standardisation and the choice of hyperparameters uses the validation nodes only. As for
FLARE, a method that fits on pooled training and validation nodes (EMOGI, `GraphData.pool_validation`) does so on
exactly those.
"""
import multiprocessing
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from baselines import tabular, workers
from flare.graph import hop_matrices, row_normalise

RULE_LEARNERS = tabular.DISCRETE_RULES + tabular.DIFF_RULES + tabular.DIFF_TREES + tabular.INTERPRETABLE
ISOLATED = ('CORELS',)           # fitted in a child process: its C code crashes on some wide tables


def graph_scores(metric, y_true, probabilities):
    """Accuracy, macro-F1 and balanced accuracy of the most probable class, and for the dataset whose metric is the area
    under the precision-recall curve (EMOGI), that area and the ROC AUC of class 1."""
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
    prediction = probabilities.argmax(axis=1)
    row = {'test_acc': accuracy_score(y_true, prediction), 'test_f1': f1_score(y_true, prediction, average='macro'),
           'test_bal_acc': balanced_accuracy_score(y_true, prediction)}
    if metric == 'auprc':
        row.update(auprc=average_precision_score(y_true, probabilities[:, 1]), auroc=roc_auc_score(y_true, probabilities[:, 1]))
    return row


def fit_nodes(data, train, val):
    """The nodes a model without a validation set fits on: training plus validation when the graph pools them."""
    return np.concatenate([train, val]) if data.pool_validation else train


# --- logistic regression and the propagated tables -----------------------------------------------------------------

def neighbour_mean_frame(X, adjacency):
    """The variables, the mean of each over the neighbours and log10(1 + degree): the graph baseline of logistic regression."""
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    means = pd.DataFrame(row_normalise(adjacency) @ X.values, index=X.index, columns=['NB_' + c for c in X.columns])
    return pd.concat([X, means, pd.Series(np.log10(degree + 1.0), index=X.index, name='DEG')], axis=1)


def propagated_table(data, hops):
    """Own variables, the mean over the neighbours at each hop up to `hops`, and log10(1 + degree)."""
    frames = [data.X]
    for hop, matrix in enumerate(hop_matrices(data.adjacency, hops)):
        frames.append(pd.DataFrame(matrix @ data.X.values, index=data.X.index, columns=[f'NB{hop + 1}_{c}' for c in data.X.columns]))
    degree = np.asarray(data.adjacency.sum(axis=1)).ravel()
    frames.append(pd.Series(np.log10(degree + 1.0), index=data.X.index, name='DEG'))
    return pd.concat(frames, axis=1)


def logistic(data, train, val, test, neighbours, balanced):
    """Logistic regression on the standardised variables of the node, plus its neighbour means and log degree when
    `neighbours`. Returns the test probabilities and the number of columns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    fit = fit_nodes(data, train, val)
    X = (neighbour_mean_frame(data.X, data.adjacency) if neighbours else data.X).values.astype(float)
    X = StandardScaler().fit(X[fit]).transform(X)
    model = LogisticRegression(max_iter=5000, class_weight='balanced' if balanced else None).fit(X[fit], data.y[fit])
    return model.predict_proba(X[test]), X.shape[1]


def table_model(name, seed, balanced, n_classes):
    """Random forest (500 trees, balanced class weights when `balanced`) or EBM (14 outer bags, 10 interaction pairs) on
    a propagated table."""
    if name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=500, n_jobs=4, random_state=seed, class_weight='balanced' if balanced else None)
    return tabular.build('EBM', {}, seed, n_classes)


def table_probabilities(name, data, train, val, test, seed, hops=2):
    """Test probabilities of RF or EBM on the table of `hops` hops (the variables as they are, without scaling)."""
    table = propagated_table(data, hops).values.astype(float)
    fit = fit_nodes(data, train, val)
    model = table_model(name, seed, data.metric == 'auprc', data.n_classes).fit(table[fit], data.y[fit])
    return model.predict_proba(table[test]), table.shape[1]


# --- the rule and tree learners ------------------------------------------------------------------------------------

def _fit_proba(name, seed, X, y, X_test, n_classes):
    model = tabular.build(name, dict(tabular.DEFAULTS[name]), seed, n_classes).fit(X, y)
    P, own = tabular.probabilities(model, X_test, n_classes)
    return P, own, tabular.n_rules(name, model)


def _in_child(name, seed, X, y, X_test, n_classes):
    """_fit_proba in a forked process, so that a crash of the method's C code fails this fit only, as a RuntimeError."""
    context = multiprocessing.get_context('fork')
    receive, send = context.Pipe(duplex=False)

    def child():
        try:
            send.send(('ok', _fit_proba(name, seed, X, y, X_test, n_classes)))
        except Exception as error:                                      # noqa: BLE001 - reported to the parent
            send.send(('error', f'{type(error).__name__}: {error}'[:300]))
        send.close()
    process = context.Process(target=child)
    process.start()
    send.close()
    try:
        status, value = receive.recv()                                  # EOFError when the child dies before sending
    except EOFError:
        status, value = 'error', None
    process.join()
    if status == 'ok':
        return value
    raise RuntimeError(value or f'{name} crashed (exit code {process.exitcode})')


def rule_learner(name, data, train, val, test, seed):
    """Test probabilities of a rule or tree learner of baselines/tabular.py at its defaults on the two-hop table, whose columns
    are standardised with the fit nodes and held in float32 (the precision the models are given). Returns the probabilities,
    whether they are the method's own (else one-hot of its predictions) and its number of rules (NaN if not defined)."""
    from sklearn.preprocessing import StandardScaler
    table = propagated_table(data, 2).values.astype(np.float32).astype(np.float64)
    fit = fit_nodes(data, train, val)
    X = StandardScaler().fit(table[fit]).transform(table)
    n_classes = data.n_classes
    if name == 'GRANDE':                    # trains on the training nodes and stops on the validation nodes
        with tempfile.TemporaryDirectory() as tmp:
            r = workers.run('GRANDE', tmp, dict(X_train=X[train], y_train=data.y[train], X_val=X[val], y_val=data.y[val], X_full=X[val],
                                                X_test=X[test]), mode='default', seed=seed)
            return np.asarray(r['proba_test']), True, float('nan')
    if name == 'CLASSY':
        with tempfile.TemporaryDirectory() as tmp:
            r = workers.run('CLASSY', tmp, dict(X=X[fit], y=data.y[fit], X_test=X[test]), mode='default', seed=seed, tune_seed=seed)
            return np.eye(n_classes)[np.asarray(r['pred_test']).astype(int)], False, float(r['complexity'])
    run = _in_child if name in ISOLATED else _fit_proba
    return run(name, seed, X[fit], data.y[fit], X[test], n_classes)


# --- graph neural networks -----------------------------------------------------------------------------------------

def sym_norm(adjacency):
    """D^-1/2 (A + I) D^-1/2."""
    A = adjacency + sp.eye(adjacency.shape[0], dtype=np.float32)
    d = np.asarray(A.sum(1)).ravel()
    return (sp.diags(1.0 / np.sqrt(d)) @ A @ sp.diags(1.0 / np.sqrt(d))).tocsr()


def validation_score(data, y, probabilities):
    from sklearn.metrics import accuracy_score, average_precision_score
    return average_precision_score(y, probabilities[:, 1]) if data.metric == 'auprc' else accuracy_score(y, probabilities.argmax(1))


def sgc(data, train, val, test, hops, balanced):
    """Logistic regression on [X, S X, ..., S^hops X] of the standardised variables, with C chosen on the validation nodes.
    Returns the test probabilities and C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = data.X.values.astype(float)
    X = StandardScaler().fit(X[train]).transform(X)
    S = sym_norm(data.adjacency)
    blocks, current = [X], X
    for _ in range(hops):
        current = S @ current
        blocks.append(current)
    Z = np.hstack(blocks)
    best = None
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=C, max_iter=5000, class_weight='balanced' if balanced else None).fit(Z[train], data.y[train])
        score = validation_score(data, data.y[val], model.predict_proba(Z[val]))
        if best is None or score > best[0]:
            best = (score, C, model)
    return best[2].predict_proba(Z[test]), best[1]


def gcn(data, train, val, test, seed, hidden, epochs, balanced):
    """Two-layer GCN, dropout 0.5, Adam with step 0.01 and weight decay 5e-4, on the standardised variables; the epoch with the
    best validation score is kept."""
    from sklearn.preprocessing import StandardScaler
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = data.X.values.astype(np.float32)
    X = StandardScaler().fit(X[train]).transform(X).astype(np.float32)
    S = sym_norm(data.adjacency).tocoo()
    St = torch.sparse_coo_tensor(np.vstack([S.row, S.col]), S.data.astype(np.float32), S.shape).coalesce()
    Xt, y = torch.tensor(X), torch.tensor(data.y, dtype=torch.long)
    n_classes = data.n_classes
    W1, W2 = torch.nn.Linear(X.shape[1], hidden), torch.nn.Linear(hidden, n_classes)
    optimizer = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=0.01, weight_decay=5e-4)
    weight = None
    if balanced:
        weight = torch.tensor(len(train) / (n_classes * np.bincount(data.y[train], minlength=n_classes)), dtype=torch.float32)
    tr = torch.tensor(train)

    def forward(training):
        h = torch.relu(torch.sparse.mm(St, W1(torch.nn.functional.dropout(Xt, 0.5, training))))
        return torch.sparse.mm(St, W2(torch.nn.functional.dropout(h, 0.5, training)))

    best = (-1, None)
    for _ in range(epochs):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(forward(True)[tr], y[tr], weight=weight).backward()
        optimizer.step()
        with torch.no_grad():
            probabilities = torch.softmax(forward(False), 1).numpy()
        score = validation_score(data, data.y[val], probabilities[val])
        if score > best[0]:
            best = (score, probabilities)
    return best[1][test]


def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _glorot(*shape):
    w = torch.empty(*shape, device=_device())
    torch.nn.init.xavier_uniform_(w)
    return torch.nn.Parameter(w)


def row_normalised_features(data):
    """Each node's variables divided by their sum, the preprocessing of the code of GCN and GAT."""
    X = data.X.values.astype(np.float32)
    s = X.sum(1, keepdims=True)
    return torch.tensor(np.divide(X, s, out=np.zeros_like(X), where=s != 0), device=_device())


def gcn_standard(data, train, val, test, seed):
    """GCN as published (Kipf and Welling, 2017): 16 hidden units, no bias, Glorot initialisation, dropout 0.5, Adam with step
    0.01, L2 5e-4 on the weights of the first layer (in the loss, so also in the validation loss), at most 200 epochs, stop
    when the validation loss exceeds the mean of the previous 10; the last model is scored. Returns the test probabilities
    and the epochs trained."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    S = sym_norm(data.adjacency).tocoo()
    St = torch.sparse_coo_tensor(np.vstack([S.row, S.col]), S.data.astype(np.float32), S.shape, device=device).coalesce()
    Xt, y = row_normalised_features(data), torch.tensor(data.y, dtype=torch.long, device=device)
    W1, W2 = _glorot(Xt.shape[1], 16), _glorot(16, data.n_classes)
    optimizer = torch.optim.Adam([W1, W2], lr=0.01)
    tr, va = torch.tensor(train, device=device), torch.tensor(val, device=device)
    F = torch.nn.functional

    def cost(logits, nodes):
        return F.cross_entropy(logits[nodes], y[nodes]) + 5e-4 * (W1 ** 2).sum() / 2

    def forward(training):
        h = torch.relu(torch.sparse.mm(St, F.dropout(Xt, 0.5, training) @ W1))
        return torch.sparse.mm(St, F.dropout(h, 0.5, training) @ W2)

    validation_costs = []
    for epoch in range(200):
        optimizer.zero_grad()
        cost(forward(True), tr).backward()
        optimizer.step()
        with torch.no_grad():
            validation_costs.append(cost(forward(False), va).item())
        if epoch > 10 and validation_costs[-1] > np.mean(validation_costs[-11:-1]):
            break
    with torch.no_grad():
        probabilities = torch.softmax(forward(False), 1).cpu().numpy()
    return probabilities[test], epoch + 1


class GATLayer(torch.nn.Module):
    """Graph attention (Velickovic et al., 2018) over an edge list with self loops: alpha_ij = softmax_j(LeakyReLU_0.2(a_dst . W h_i
    + a_src . W h_j)), dropout on alpha, out_i = sum_j alpha_ij W h_j for each head, the heads concatenated or averaged."""

    def __init__(self, fan_in, fan_out, heads, concat):
        super().__init__()
        self.heads, self.fan_out, self.concat = heads, fan_out, concat
        self.W = _glorot(fan_in, heads * fan_out)
        self.a_src, self.a_dst = _glorot(1, heads, fan_out), _glorot(1, heads, fan_out)
        self.bias = torch.nn.Parameter(torch.zeros(heads * fan_out if concat else fan_out, device=_device()))

    def forward(self, x, src, dst, dropout):
        n = x.shape[0]
        h = (x @ self.W).view(n, self.heads, self.fan_out)
        e = torch.nn.functional.leaky_relu((h * self.a_dst).sum(-1)[dst] + (h * self.a_src).sum(-1)[src], 0.2)
        top = torch.full((n, self.heads), -torch.inf, device=x.device).index_reduce_(0, dst, e, 'amax')
        w = torch.exp(e - top[dst])
        alpha = w / torch.zeros(n, self.heads, device=x.device).index_add_(0, dst, w)[dst]
        alpha = torch.nn.functional.dropout(alpha, dropout, self.training)
        out = torch.zeros(n, self.heads, self.fan_out, device=x.device).index_add_(0, dst, alpha[..., None] * h[src])
        return (out.reshape(n, -1) if self.concat else out.mean(1)) + self.bias


def gat_standard(data, train, val, test, seed):
    """GAT as published (Velickovic et al., 2018): 8 heads of 8 units with ELU, then one output head (PubMed: 8 averaged),
    dropout 0.6 on the inputs of both layers and on the attention, Adam with step 0.005 and L2 5e-4 on the weights (PubMed:
    0.01 and 1e-3). Stopping as in their code: patience of 100 epochs on the validation loss and accuracy, and the last model
    at which both were at their best is scored. Returns the test probabilities and the epochs trained."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device, pubmed = _device(), data.name == 'pubmed'
    A = (data.adjacency + sp.eye(data.adjacency.shape[0], dtype=np.float32)).tocoo()     # edge j -> i for every i ~ j and i = j
    src, dst = torch.tensor(A.col, device=device), torch.tensor(A.row, device=device)
    Xt, y = row_normalised_features(data), torch.tensor(data.y, dtype=torch.long, device=device)
    model = torch.nn.ModuleList([GATLayer(Xt.shape[1], 8, 8, True), GATLayer(64, data.n_classes, 8 if pubmed else 1, False)])
    weights = [p for name, p in model.named_parameters() if not name.endswith('bias')]
    biases = [p for name, p in model.named_parameters() if name.endswith('bias')]
    optimizer = torch.optim.Adam([dict(params=weights, weight_decay=1e-3 if pubmed else 5e-4), dict(params=biases, weight_decay=0)],
                                 lr=0.01 if pubmed else 0.005)
    tr, va = torch.tensor(train, device=device), torch.tensor(val, device=device)
    F = torch.nn.functional

    def forward():
        h = F.elu(model[0](F.dropout(Xt, 0.6, model.training), src, dst, 0.6))
        return model[1](F.dropout(h, 0.6, model.training), src, dst, 0.6)

    best_acc, best_loss, kept, wait = -1.0, np.inf, None, 0
    for epoch in range(10000):
        model.train()
        optimizer.zero_grad()
        F.cross_entropy(forward()[tr], y[tr]).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = forward()
            loss, acc = F.cross_entropy(logits[va], y[va]).item(), (logits[va].argmax(1) == y[va]).float().mean().item()
        if acc >= best_acc or loss <= best_loss:
            if acc >= best_acc and loss <= best_loss:
                kept = torch.softmax(logits, 1).cpu().numpy()
            best_acc, best_loss, wait = max(acc, best_acc), min(loss, best_loss), 0
        else:
            wait += 1
            if wait == 100:
                break
    return kept[test], epoch + 1
