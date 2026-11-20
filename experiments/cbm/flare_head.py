"""FLARE as the head of a concept bottleneck model: it reads the concepts and predicts the class.

Each concept c in [0, 1] is one variable with two fixed conditions: TRUE, with truth c, and FALSE, with truth 1 - c.
There is no partition to learn, so FLARE is trained as eight averaged models of 100 one-condition rules plus 20
interaction rules of up to two conditions (slots), with the settings of the tabular experiments otherwise.

`ConceptInputs` is the inputs module that FLARE's model reads instead of the learned partitions (flare/inputs.py states
the contract). The rest is the recipe of `flare.train.fit_fold`, given a `Fold` made from the concepts as they are (no
standardisation: they are already truth degrees) and the validation split of the concept export.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from flare.config import flare
from flare.data import batched_loader, to_dataset
from flare.metrics import evaluate, rule_set_complexity, rule_set_size
from flare.preprocessing import Fold
from flare.printing import rules_text
from flare.selectors import GradMode
from flare.train import fit_fold

CONDITIONS = ('FALSE', 'TRUE')


class ConceptInputs(nn.Module):
    """[N, V] truth degrees of V concepts -> [N, V, 2] memberships (FALSE: 1 - c, TRUE: c). Nothing is learned."""

    n_conditions = 2

    def __init__(self, n_concepts):
        super().__init__()
        self.n_variables = n_concepts

    def forward(self, c):
        return torch.stack([1.0 - c, c], dim=2)

    def condition_text(self, variable, condition, name, category_names=None):
        return f'{name} IS {CONDITIONS[condition]}'

    def crossing_parameters(self):
        return []

    def space_parameters(self):
        return []

    def inherit(self, first):
        """Nothing to copy: the memberships of every model are the same fixed ones."""


def concept_config(**changes):
    """FLARE with two slots per interaction rule and no label runs (a concept has two conditions only). `labels` (7, the
    tabular default) is not used by the memberships; it only enters the estimate of the size of a step with which
    `fit_fold` decides whether the eight models train at once."""
    return flare(slots=2, label_runs='none', **changes)


def concept_fold(splits, train_source, test_source):
    """The Fold of a concept export: training and validation rows of `train_source`, test rows of `test_source`."""
    def part(split, source):
        return np.asarray(splits[split][source], dtype=float), np.asarray(splits[split]['y']).astype(int)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = part('train', train_source), part('val', train_source), part('test', test_source)
    n_classes = int(max(y_tr.max(), y_val.max(), y_te.max())) + 1
    return Fold(X_tr, y_tr, X_val, y_val, X_te, y_te, np.zeros(X_tr.shape[1], dtype=bool), n_classes=n_classes)


def fit_flare(names, splits, train_source, test_source, seed, config=None, device=None, verbose=False):
    """Train FLARE on the concepts of `train_source` (the training split; the validation split picks the epoch) and
    return the FitResult. The batch has ceil(training rows / 50) rows, or 32 if more: about 50 updates per epoch."""
    config = config or concept_config()
    fold = concept_fold(splits, train_source, test_source)
    batch = max(config.batch_size, math.ceil(len(fold.y_train) / config.steps_per_epoch)) if config.steps_per_epoch else config.batch_size
    config = config.replace(batch_size=batch, steps_per_epoch=0)
    # Evaluation in a fixed order (see flare.data.fold_loaders); changing it changes the random draws.
    return fit_fold(fold, config, [], seed, device=device, verbose=verbose, fixed_order_evaluation=True,
                    inputs_factory=lambda device: ConceptInputs(len(names)))


def score_flare(result, C, y):
    """Accuracy and macro-F1 of the deployed (discrete) model on the concepts C with classes y."""
    model = result.model
    device = next(model.parameters()).device
    acc, f1, _ = evaluate(model, batched_loader(to_dataset(C, y, device), 512, False), 0.0, GradMode.PURE)
    return float(acc), float(f1)


def rule_counts(result):
    """The rules of the deployed model: rules, distinct conditions, and rules that decide some training row."""
    model = result.model
    n_rules, _, unique_conditions = rule_set_size(model)
    device = next(model.parameters()).device
    deciding, _, _ = rule_set_complexity(model, result.train_loader.dataset.X.to(device))
    return {'n_rules': n_rules, 'unique_conditions': unique_conditions, 'n_deciding_rules': deciding}


def printed_rules(result, names):
    return rules_text(result.model, feat_names=list(names))
