"""Scores of a trained model: accuracy, macro-F1, balanced accuracy, ROC AUC, log-loss and the size of the rule set.

Everything is computed on the discrete model (`GradMode.PURE`): the rules that are printed are the rules that decide.
"""
import numpy as np
import torch

from .selectors import GradMode


def eval_chunk_rows(model, budget=2 ** 27):
    """Rows per evaluation forward: 8192, fewer when the largest intermediate of a row (slots x rules x variables x 2
    floats, the switched memberships of every slot) would exceed `budget` elements. For example 1000 rules over 112
    variables with 8 slots get 74 rows (about 0.8 GiB at the peak instead of an out-of-memory error). Rows are computed
    independently, so the chunking does not change the results."""
    try:
        per_row = model.rules.variable_logits.shape[0] * model.nrules * model.total_vars * 2
    except AttributeError:                      # not a FLARE model
        return 8192
    return max(1, min(8192, budget // per_row))


def balanced_class_weights(y, n_classes, device):
    """sklearn-style balanced weights n / (n_present_classes * count); classes absent from y get 1."""
    counts = np.bincount(y, minlength=n_classes)
    weights = np.where(counts > 0, len(y) / ((counts > 0).sum() * np.maximum(counts, 1)), 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, loader, k=0.0, mode=GradMode.PURE):
    """(accuracy, macro-F1, balanced accuracy) of one pass over the loader.

    The F1 of a class is 2 P R / (P + R) with a 1e-6 added to each denominator, and classes that are absent from both
    the labels and the predictions are skipped (as scikit-learn does). The balanced accuracy is the mean recall over
    the classes present in the labels."""
    model.set_inference_mode(mode, k)
    model.eval()

    # Predictions are accumulated on the device and reduced to one confusion matrix, so an evaluation costs a single
    # host-device sync. The loader is still iterated batch by batch (a pass draws from the random generator), but the
    # rows are evaluated in large chunks: a forward costs about the same for 32 or 8192 rows, as it is bound by kernel
    # launches, and every row is computed independently.
    inputs, targets = [], []
    for X, y in loader:
        inputs.append(X)
        targets.append(y)
    predictions = [model(X)[0].argmax(dim=1) for X in torch.cat(inputs).split(eval_chunk_rows(model))]

    yhat, y = torch.cat(predictions), torch.cat(targets)
    n_class = model.n_class
    confusion = torch.bincount(y * n_class + yhat, minlength=n_class * n_class).view(n_class, n_class).cpu()   # [true, predicted]
    true_positives = [float(v) for v in confusion.diag().tolist()]
    false_positives = [float(v) for v in (confusion.sum(dim=0) - confusion.diag()).tolist()]
    false_negatives = [float(v) for v in (confusion.sum(dim=1) - confusion.diag()).tolist()]

    f1_scores = []
    for ix in range(n_class):
        if true_positives[ix] + false_positives[ix] + false_negatives[ix] == 0:
            continue
        precision = true_positives[ix] / (true_positives[ix] + false_positives[ix] + 1e-6)
        recall = true_positives[ix] / (true_positives[ix] + false_negatives[ix] + 1e-6)
        f1_scores.append(2 * (precision * recall) / (precision + recall + 1e-6))

    support = confusion.sum(dim=1)
    recalls = [true_positives[ix] / float(support[ix]) for ix in range(n_class) if support[ix] > 0]
    return sum(true_positives) / len(y), np.mean(f1_scores), np.mean(recalls)


@torch.no_grad()
def output_scores(model, loader):
    """[N, K] class scores (before the softmax) of the model's output on a loader's rows, and the labels."""
    model.eval()
    parts = []
    for chunk in loader.dataset.X.to(next(model.parameters()).device).split(eval_chunk_rows(model)):
        parts.append(model(chunk)[0].float().cpu())
    return torch.cat(parts).double(), loader.dataset.y.cpu().long()


def probability_metrics(model, loader, calibration=None):
    """ROC AUC (binary: of class 1; multiclass: one-vs-rest, macro) and log-loss of the softmax of the model's scores on
    a loader's rows, in the inference mode set by the last `evaluate`. TabArena scores binary tasks by AUC and
    multiclass ones by log-loss. Probabilities are clipped at 1e-15, as scikit-learn's log_loss does. calibration:
    (T, b) from `calibration.calibrate_probabilities`, applied as softmax(s / T + b)."""
    from sklearn.metrics import log_loss, roc_auc_score
    scores, y = output_scores(model, loader)
    if calibration is not None:
        scores = scores / calibration[0] + torch.as_tensor(calibration[1], dtype=scores.dtype)
    P, y = torch.softmax(scores, dim=1).numpy(), y.numpy()
    P = np.clip(P, 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    labels = list(range(P.shape[1]))
    present = np.unique(y)
    if P.shape[1] == 2:
        auc = roc_auc_score(y, P[:, 1]) if len(present) == 2 else np.nan
    else:
        auc = roc_auc_score(y, P[:, present] / P[:, present].sum(axis=1, keepdims=True), multi_class='ovr', average='macro',
                            labels=present) if len(present) > 2 else np.nan
    return {'test_roc_auc': float(auc), 'test_logloss': float(log_loss(y, P, labels=labels))}


def rule_set_size(model):
    """(number of distinct rules, mean number of conditions per distinct rule, number of distinct conditions summed
    over the columns) of the whole rule base, as printed. Duplicates are merged (np.unique over the rows)."""
    rules, _, _ = model.extract_rules()
    rules = np.unique(rules, axis=0)
    conditions = np.mean([np.sum(np.not_equal(rule, -1)) for rule in rules])
    distinct = sum(np.sum(np.not_equal(np.unique(rules[:, i]), -1)) for i in range(rules.shape[1]))
    return rules.shape[0], conditions, distinct


@torch.no_grad()
def rule_set_complexity(model, X):
    """Rules that decide at least one row of X under PURE (distinct conditions plus consequent), their mean number of
    distinct conditions, and the mean Jaccard overlap of the rows that they cover (truth at least 0.5). Unlike counting
    the rules of the model, this ignores duplicates and rules that never win a row."""
    model.set_inference_mode(GradMode.PURE)
    winners, covered = [], []
    for chunk in X.split(eval_chunk_rows(model)):
        _, scores = model(chunk)
        winners.append(scores.max(dim=2)[0].argmax(dim=1))
        covered.append(model.cache_truth_rule_values.exp() >= 0.5)
    winners = torch.unique(torch.cat(winners)).cpu().numpy()
    coverage = torch.cat(covered)[:, torch.as_tensor(winners, device=X.device)].float()                       # [N, deciding rules]
    inter = coverage.T @ coverage
    size = inter.diag()
    jaccard = inter / (size.unsqueeze(0) + size.unsqueeze(1) - inter).clamp_min(1.0)
    off_diagonal = ~torch.eye(len(winners), dtype=torch.bool, device=X.device)
    overlap = float(jaccard[off_diagonal].mean()) if len(winners) > 1 else 0.0
    antecedents, _, consequents = model.extract_rules()
    deciding = {tuple(int(v) for v in antecedents[r]) + (int(consequents[r]),) for r in winners}
    conditions = float(np.mean([sum(v != -1 for v in rule[:-1]) for rule in deciding])) if deciding else 0.0
    return len(deciding), conditions, overlap
