import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from flare.calibration import calibrate_probabilities
from flare.metrics import evaluate, probability_metrics
from types import SimpleNamespace


class FixedScores(torch.nn.Module):
    """Returns preset class scores for sample indices, mimicking FLAREModel's (output, rules) tuple."""

    def __init__(self, scores, n_class):
        super().__init__()
        self.scores = scores
        self.n_class = n_class
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def set_inference_mode(self, *args):
        pass

    def forward(self, X):
        return self.scores[X.long().squeeze(1)], None


def loader_of(y, batch_size=3):
    return DataLoader(TensorDataset(torch.arange(len(y)).unsqueeze(1), y), batch_size=batch_size)


def test_macro_f1_ignores_label_ids_absent_from_labels_and_predictions():
    # Four label ids, but id 3 never occurs (a class that is missing from a fold).
    y = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    predictions = torch.tensor([0, 1, 1, 1, 2, 0, 0, 2])
    accuracy, macro_f1, _ = evaluate(FixedScores(torch.nn.functional.one_hot(predictions, 4).float(), 4), loader_of(y))
    assert accuracy == pytest.approx(5 / 8)
    assert macro_f1 == pytest.approx(f1_score(y, predictions, average='macro'), abs=1e-5)


def test_evaluate_matches_sklearn_on_an_imbalanced_problem():
    # Class 2 has one sample and is never predicted, so balanced accuracy and accuracy must differ.
    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2])
    predictions = torch.tensor([0, 0, 0, 1, 0, 1, 0, 0])
    accuracy, macro_f1, balanced = evaluate(FixedScores(torch.nn.functional.one_hot(predictions, 3).float(), 3), loader_of(y))
    assert accuracy == pytest.approx(accuracy_score(y, predictions))
    assert macro_f1 == pytest.approx(f1_score(y, predictions, average='macro'), abs=1e-5)
    assert balanced == pytest.approx(balanced_accuracy_score(y, predictions))
    assert balanced != pytest.approx(accuracy)


def test_probability_metrics_match_sklearn_on_the_softmax_of_the_output():
    """TabArena's metrics: ROC AUC of class 1 (binary), one-vs-rest macro AUC and log-loss (multiclass)."""
    torch.manual_seed(0)
    for k in (2, 3):
        X = torch.randn(200, 4)
        y = torch.randint(0, k, (200,))
        linear = torch.nn.Linear(4, k)
        model = TupleOutput(linear)
        got = probability_metrics(model, SimpleNamespace(dataset=SimpleNamespace(X=X, y=y)))
        P = torch.softmax(linear(X), dim=1).detach().numpy().astype(np.float64)
        P = P / P.sum(axis=1, keepdims=True)
        auc = roc_auc_score(y.numpy(), P[:, 1]) if k == 2 else roc_auc_score(y.numpy(), P, multi_class='ovr', average='macro')
        assert got['test_roc_auc'] == pytest.approx(auc, abs=1e-6)
        assert got['test_logloss'] == pytest.approx(log_loss(y.numpy(), P), abs=1e-6)


class TupleOutput(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x), None


class Scaled(torch.nn.Module):
    def __init__(self, scale, shift=0.0):
        super().__init__()
        self.scale, self.shift = scale, shift
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.stack([torch.zeros(len(x)), self.scale * x[:, 0] + self.shift], dim=1), None


def test_calibration_undoes_an_overconfident_temperature_and_keeps_the_binary_auc():
    torch.manual_seed(0)
    X = torch.randn(4000, 3)
    y = (torch.rand(4000) < torch.sigmoid(X[:, 0])).long()      # true log-odds x0
    loader, model = SimpleNamespace(dataset=SimpleNamespace(X=X, y=y)), Scaled(4.0)           # scores 4x the true log-odds
    T, b = calibrate_probabilities(model, loader)
    raw, cal = probability_metrics(model, loader), probability_metrics(model, loader, (T, np.zeros(2)))
    assert abs(T - 4.0) < 0.4 and cal['test_logloss'] < raw['test_logloss']
    assert cal['test_roc_auc'] == pytest.approx(raw['test_roc_auc'], abs=1e-12)


def test_calibration_can_be_kept_from_sharpening_and_its_biases_penalised():
    torch.manual_seed(1)
    X = torch.randn(3000, 3)
    y = (torch.rand(3000) < torch.sigmoid(2.0 * X[:, 0])).long()
    loader, model = SimpleNamespace(dataset=SimpleNamespace(X=X, y=y)), Scaled(1.0, 0.5)      # half the true log-odds, shifted
    free, _ = calibrate_probabilities(model, loader)
    kept, b0 = calibrate_probabilities(model, loader, min_temperature=1.0)
    _, b1 = calibrate_probabilities(model, loader, min_temperature=1.0, bias_penalty=1.0)
    assert free < 0.8 and kept >= 1.0 and np.abs(b1).sum() < np.abs(b0).sum()
