"""The additive evidence head, the rule layer and the model built from them."""
import numpy as np
import pytest
import torch

from flare.categorical import CategoricalConditions
from flare.head import EvidenceHead
from flare.inputs import TabularInputs
from flare.model import FLAREModel
from flare.partitions import quantile_partitions
from flare.printing import rules_text
from flare.rules import RuleLayer
from flare.selectors import EPSILON, GradGraft, GradMode, process_weights, select

FRAME = np.array([[0.0, 0, 1.0], [0.2, 1, 0.7], [0.6, 2, 0.4], [1.0, 0, 0.1], [0.4, 1, 0.5], [0.8, 2, 0.2],
                  [0.3, 0, 0.8], [0.9, 1, 0.6]], dtype=np.float32)
MASK = np.array([False, True, False])


def mixed_model(n_rules=8, n_slots=3, joint_bags=1, seed=11, mode=GradMode.GRAFT):
    torch.manual_seed(seed)
    partitions = quantile_partitions(FRAME[:, ~MASK], 3)
    inputs = TabularInputs(partitions, 'none', None, None, 'cpu')
    model = FLAREModel(inputs, 2, MASK, [3], n_rules * joint_bags, n_slots, 1.0, mode, joint_bags)
    return model, torch.tensor(FRAME), torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])


def test_head_adds_truth_weighted_points_to_a_bias():
    torch.manual_seed(0)
    head = EvidenceHead(3, 5, 1, 'cpu')
    with torch.no_grad():
        head.bias.copy_(torch.tensor([0.5, -1.0, 2.0]))
    log_truth = torch.log(torch.rand(4, 5) * 0.9 + 0.1)
    out, scores = head(log_truth)
    expected = head.bias + torch.exp(log_truth) @ head.weights
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(scores.sum(dim=1) + head.bias, expected)


def test_single_literal_rules_express_a_linear_boundary():
    """Two rules "x1 is High" and "x2 is High" give +1 to class 1; the bias favours class 0. With both rules at truth 0.6
    the sum (1.2) exceeds the bias; with one of them it does not: evidence accumulates across rules."""
    head = EvidenceHead(2, 2, 1, 'cpu')
    with torch.no_grad():
        head.weights.copy_(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        head.bias.copy_(torch.tensor([1.0, 0.0]))
    assert head(torch.log(torch.tensor([[0.6, 0.6]])))[0].argmax() == 1
    assert head(torch.log(torch.tensor([[0.6, 0.0001]])))[0].argmax() == 0


def test_joint_head_is_the_mean_of_independent_blocks():
    torch.manual_seed(0)
    head = EvidenceHead(2, 6, 3, 'cpu')
    log_truth = torch.log(torch.rand(5, 6) * 0.9 + 0.1)
    out, _ = head(log_truth)
    per_block = [head.bias[b] + torch.exp(log_truth[:, 2 * b:2 * b + 2]) @ head.weights[2 * b:2 * b + 2] for b in range(3)]
    torch.testing.assert_close(out, sum(per_block) / 3)
    torch.testing.assert_close(head.bag_outputs, torch.stack(per_block, dim=1))


def test_training_forward_equals_pure_and_every_parameter_gets_gradient():
    model, X, y = mixed_model()
    model.set_inference_mode(GradMode.PURE)
    with torch.no_grad():
        pure, _ = model(X)
        pure_truth = model.cache_truth_rule_values.clone()
    model.set_train_mode(GradMode.GRAFT, 0.35)
    out, _ = model(X)
    torch.testing.assert_close(out.detach(), pure, rtol=0, atol=0)               # grafting: the forward is the discrete model
    torch.testing.assert_close(model.cache_truth_rule_values.detach(), pure_truth, rtol=0, atol=0)
    torch.nn.functional.cross_entropy(out, y).backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []
    for parameter in (model.rules.label_logits, model.rules.variable_logits, model.categorical.cat_weights[0], model.head.weights):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert model.head.bias.grad is not None


def test_residual_changes_gradients_but_not_forward():
    grads, outputs = [], []
    for residual in (False, True):
        model, X, y = mixed_model()
        model.residual = residual
        model.set_train_mode(GradMode.GRAFT, 0.35)
        out, _ = model(X)
        torch.nn.functional.cross_entropy(out, y).backward()
        grads.append(model.rules.label_logits.grad.clone())
        outputs.append(out.detach())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert torch.isfinite(grads[1]).all() and not torch.allclose(grads[0], grads[1])


def per_variable_categorical(layer, x, mode, k, temp=1.0):
    """One pass per categorical variable (what the grouped forward replaces)."""
    res = torch.zeros((x.shape[0], layer.n_rules, len(layer.cat_variables)))
    for i in range(x.shape[1]):
        one_hot = torch.nn.functional.one_hot(x[:, i].to(torch.int64), layer.cat_variables[i]).to(torch.float32)
        one_hot = torch.clamp(one_hot + layer.floor, min=0.0, max=1.0)
        shares = process_weights(layer.cat_weights[i], dim=0, temperature=temp)
        res[:, :, i] = select(one_hot.unsqueeze(2), shares.unsqueeze(0), k, 1, mode)
    return res


@pytest.mark.parametrize('mode', [GradMode.GRAFT, GradMode.PURE])
def test_grouped_categorical_forward_equals_the_per_variable_loop(mode):
    torch.manual_seed(1)
    cardinalities = [4, 3, 4, 2, 3, 4]
    layer = CategoricalConditions(cardinalities, 7, 0.1, 'cpu')
    x = torch.stack([torch.randint(c, (9,)) for c in cardinalities], dim=1).float()
    grouped = layer(x, mode, 0.3, 1.0)
    reference = per_variable_categorical(layer, x, mode, 0.3)
    assert torch.equal(grouped, reference)
    if mode == GradMode.GRAFT:
        upstream = torch.randn_like(grouped)
        grads = torch.autograd.grad((grouped * upstream).sum(), list(layer.cat_weights))
        reference_grads = torch.autograd.grad((reference * upstream).sum(), list(layer.cat_weights))
        for a, b in zip(grads, reference_grads):     # the backward sums in another order: a few float32 ulps apart
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def per_slot_forward(layer, x, mode, k, temp, categorical_truth, residual):
    """One (switch, select, log) pass per slot (what the batched slots replace)."""
    label_shares = process_weights(layer.label_logits, dim=3, temperature=temp)
    variable_shares = process_weights(layer.variable_logits, dim=3, temperature=temp)
    chosen = select(x.unsqueeze(1), label_shares, k, 3, mode)
    chosen = torch.cat([chosen, categorical_truth], dim=2)
    current, log_truth = chosen, None
    for s in range(variable_shares.shape[0]):
        if s > 0:
            shares = process_weights(layer.switch_logits[s], dim=3, temperature=temp)
            current = select(torch.stack([chosen, torch.ones_like(chosen)], dim=3), shares, k, 3, mode)
        candidate = select(current, variable_shares[s], k, 2, mode)
        log_truth = torch.log(candidate + EPSILON) if log_truth is None else log_truth + torch.log(candidate + EPSILON)
    if residual and mode != GradMode.PURE:
        log_truth = GradGraft.apply(log_truth.detach(), log_truth + torch.sum(chosen, dim=2) * min(0.1, k))
    return log_truth


@pytest.mark.parametrize('mode', [GradMode.GRAFT, GradMode.PURE])
def test_batched_slots_equal_the_per_slot_loop(mode):
    torch.manual_seed(3)
    layer = RuleLayer(5, 2, 3, 11, 4, True, 'cpu')
    x, categorical = torch.rand(9, 5, 3), torch.rand(9, 11, 2)
    batched = layer(x, mode, 0.3, 1.0, categorical, True)
    reference = per_slot_forward(layer, x, mode, 0.3, 1.0, categorical, True)
    assert torch.equal(batched, reference)
    if mode == GradMode.GRAFT:
        upstream = torch.randn_like(batched)
        parameters = [layer.label_logits, layer.variable_logits, layer.switch_logits]
        for a, b in zip(torch.autograd.grad((batched * upstream).sum(), parameters), torch.autograd.grad((reference * upstream).sum(), parameters)):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


def test_a_repeated_condition_counts_as_off_only_when_repeats_are_not_allowed():
    torch.manual_seed(0)
    layer = RuleLayer(3, 0, 3, 4, 2, False, 'cpu')
    with torch.no_grad():
        layer.variable_logits.zero_()
        layer.variable_logits[..., 0] = 5.0                                   # both slots of every rule take variable 0
        layer.switch_logits[..., 0] = 5.0                                     # and neither slot is off
        layer.label_logits.zero_()
        layer.label_logits[..., 1] = 5.0
    x = torch.full((1, 3, 3), 0.5)
    single = layer(x, GradMode.PURE, 0.0, 1.0)
    layer.allow_repeated_conditions = True
    squared = layer(x, GradMode.PURE, 0.0, 1.0)
    torch.testing.assert_close(single.exp(), torch.full((1, 4), 0.5), atol=1e-5, rtol=0)
    torch.testing.assert_close(squared.exp(), torch.full((1, 4), 0.25), atol=1e-5, rtol=0)


def test_restrict_to_single_condition_caps_the_first_rules():
    model, X, _ = mixed_model(n_rules=8, n_slots=3)
    assert model.restrict_to_single_condition(5) == 5
    assert model.interaction_rows().tolist() == [False] * 5 + [True] * 3
    rules, weights, consequents = model.extract_rules()
    assert (rules[:5] != -1).sum(axis=1).max() == 1                          # a capped rule reads one column
    model.set_train_mode(GradMode.GRAFT, 0.5)
    out, _ = model(X)
    out.sum().backward()
    assert model.rules.switch_logits.grad[1:, 0, :5].abs().sum() == 0        # and its later slots receive no gradient
    assert model.rules.variable_logits.grad[1:, 0, :5].abs().sum() == 0


def test_joint_model_caps_the_first_rules_of_every_block():
    model, _, _ = mixed_model(n_rules=6, joint_bags=3)
    assert model.restrict_to_single_condition(4, bag_size=6) == 12
    assert model.interaction_rows().tolist() == ([False] * 4 + [True] * 2) * 3


def test_consequent_is_the_class_with_most_points():
    model, _, _ = mixed_model()
    _, weights, consequents = model.extract_rules()
    np.testing.assert_array_equal(consequents, model.head.weights.detach().numpy().argmax(axis=1))
    np.testing.assert_allclose(np.abs(weights).sum(axis=1), 1.0, atol=1e-6)


def test_extract_rules_reads_the_selectors_in_the_data_column_order():
    model, _, _ = mixed_model(n_rules=2, n_slots=2)
    with torch.no_grad():
        model.rules.variable_logits.zero_()
        model.rules.variable_logits[0, 0, 0, 2] = 9.0        # rule 0, slot 0: the categorical variable (third of the layer)
        model.rules.variable_logits[0, 0, 1, 1] = 9.0        # rule 1, slot 0: the second real variable (data column 2)
        model.rules.switch_logits[1:, ..., 1] = 9.0          # every later slot is off
        model.rules.label_logits.zero_()
        model.rules.label_logits[0, 1, 1, 2] = 9.0           # rule 1 takes label 2 of that variable
        model.categorical.cat_weights[0].zero_()
        model.categorical.cat_weights[0][2, 0] = 9.0         # rule 0 tests category 2
    rules, _, _ = model.extract_rules()
    assert rules[0].tolist() == [-1, 2, -1] and rules[1].tolist() == [-1, -1, 2]
    text = rules_text(model, feat_names=['a', 'b', 'c'])
    assert text.splitlines()[0].startswith('IF b IS 2 THEN') and 'c IS High [' in text.splitlines()[1]


def test_mixed_model_without_real_variables_runs():
    torch.manual_seed(0)
    model = FLAREModel(None, 2, np.array([True, True]), [3, 2], 5, 2, 1.0, GradMode.GRAFT)
    out, scores = model(torch.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]]))
    assert out.shape == (3, 2) and scores.shape == (3, 5, 2)
