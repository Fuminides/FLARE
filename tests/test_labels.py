"""Learned Ruspini partitions and conditions on runs of labels."""
import numpy as np
import pytest
import torch

from flare.labels import (LabelRunFuzzifier, RuspiniFuzzifier, breakpoints, initial_gaps, label_run_list, run_label_name,
                          ruspini_chain)
from flare.printing import rules_text
from flare.selectors import GradMode
from helpers import toy_model, toy_partitions


@pytest.mark.parametrize('n_labels', [2, 3, 5, 7])
def test_tied_gaps_rebuild_the_quantile_partition(n_labels):
    """The quantile partitions are Ruspini (c_k = a_{k+1}, d_k = b_{k+1}), so the 2n - 1 tied gaps rebuild them."""
    _, partitions = toy_partitions(n_labels)
    log_gaps, starts, ends = initial_gaps(partitions)
    assert log_gaps.shape == (3, 2 * n_labels - 1) and len(ruspini_chain(n_labels)) == 2 * n_labels - 1
    rebuilt = breakpoints(torch.tensor(starts, dtype=torch.float32), torch.tensor(ends, dtype=torch.float32),
                          torch.tensor(log_gaps, dtype=torch.float32))
    torch.testing.assert_close(rebuilt, torch.tensor(partitions, dtype=torch.float32), rtol=0, atol=1e-5)


def test_label_runs_list_single_labels_first_and_never_the_whole_domain():
    names = ['VL', 'L', 'ML', 'M', 'MH', 'H', 'VH']
    assert label_run_list(7, 'none') == [(k, k) for k in range(7)]
    shoulders, every = label_run_list(7, 'shoulders'), label_run_list(7, 'all')
    assert len(shoulders) == 17 and len(every) == 27 and len(set(every)) == 27
    assert shoulders[:7] == every[:7] == [(k, k) for k in range(7)] and set(shoulders) <= set(every)
    assert (0, 6) not in every
    assert [run_label_name(names, run) for run in [(3, 3), (3, 6), (0, 2), (2, 4)]] == ['M', 'at least M', 'at most ML', 'ML to MH']
    with pytest.raises(ValueError):
        label_run_list(7, 'both')


def _fuzzifier(n_labels=7, runs='all'):
    _, partitions = toy_partitions(n_labels)
    base = RuspiniFuzzifier(partitions, 'cpu')
    run_list = label_run_list(n_labels, runs)
    return base, LabelRunFuzzifier(base, run_list, n_labels, 'cpu'), run_list, partitions


def test_a_run_of_labels_is_one_wider_set():
    """Neighbouring labels sum to one where they overlap, so "at least label k" rises over label k's left ramp and stays at
    one, and it is the complement of "at most label k - 1"."""
    base, fuzzifier, runs, partitions = _fuzzifier()
    x = torch.linspace(float(partitions[0, 0, 0]), float(partitions[0, -1, -1]), 2001).unsqueeze(1).repeat(1, 3)
    with torch.no_grad():
        labels, memberships = base(x), fuzzifier(x)
    torch.testing.assert_close(labels.sum(dim=2), torch.ones_like(labels.sum(dim=2)), rtol=0, atol=1e-5)
    torch.testing.assert_close(memberships[..., :7], labels)
    at_least, at_most = memberships[..., runs.index((3, 6))], memberships[..., runs.index((0, 2))]
    torch.testing.assert_close(at_least + at_most, torch.ones_like(at_least), rtol=0, atol=1e-5)
    assert (at_least[1:, 0] >= at_least[:-1, 0] - 1e-6).all()                  # one ramp up, no dip
    a, b = partitions[0, 3, :2]
    assert (at_least[x[:, 0] <= a, 0] < 1e-6).all() and (at_least[x[:, 0] >= b, 0] > 1 - 1e-6).all()


def test_learned_crossings_keep_a_ruspini_partition():
    """However the tied log gaps move (one pushed far down: label 1 loses its plateau), neighbouring labels still sum to
    one, "at least label 3" is one ramp and the complement of "at most label 2", and it is half true at its crossing;
    the crossings stay in order inside the domain."""
    base, fuzzifier, runs, _ = _fuzzifier(runs='shoulders')
    torch.manual_seed(0)
    with torch.no_grad():
        base.deltas.add_(torch.randn_like(base.deltas))
        base.deltas[:, 2] = -30.0                                                # gap 2 is label 1's plateau
    x = torch.linspace(base.begin_points.min().item(), base.end_points.max().item(), 4001).unsqueeze(1).repeat(1, 3)
    with torch.no_grad():
        labels, memberships = base(x), fuzzifier(x)
    torch.testing.assert_close(labels.sum(dim=2), torch.ones_like(labels.sum(dim=2)), rtol=0, atol=1e-5)
    at_least, at_most = memberships[..., runs.index((3, 6))], memberships[..., runs.index((0, 2))]
    torch.testing.assert_close(at_least + at_most, torch.ones_like(at_least), rtol=0, atol=1e-5)
    assert (at_least[1:] >= at_least[:-1] - 1e-6).all()
    cuts = base.crossings()
    assert cuts.shape == (3, 6) and (cuts[:, 1:] >= cuts[:, :-1]).all()
    assert (cuts >= base.begin_points[:, None]).all() and (cuts <= base.end_points[:, None]).all()
    with torch.no_grad():
        half = fuzzifier(cuts[:, 2].unsqueeze(0))[0, :, runs.index((3, 6))]
    torch.testing.assert_close(half, torch.full_like(half, 0.5), rtol=0, atol=1e-4)


def test_label_runs_model_selects_among_the_runs_names_them_and_prints_their_cuts():
    model, X = toy_model(n_labels=7, n_slots=2, label_runs='shoulders')
    assert model.rules.label_logits.shape[3] == 17
    partition = model.inputs.partition
    assert partition.deltas.requires_grad
    out = model(X)[0]
    out.sum().backward()
    assert partition.deltas.grad is not None and partition.deltas.grad.abs().sum() > 0     # the crossings are learned
    with torch.no_grad():
        model.rules.label_logits.fill_(0.0)
        model.rules.label_logits[..., 9] = 5.0                                   # every condition: "at least" label 3
    model.set_inference_mode(GradMode.PURE)
    text = rules_text(model, feat_names=['a', 'b', 'c'])
    assert 'IS at least 3' in text and 'IS 9' not in text
    cuts = {f'{c:.3g}' for c in partition.crossings()[:, 2].tolist()}
    printed = [part.split(']')[0] for part in text.split('IS at least 3 [>= ')[1:]]
    assert printed and set(printed) <= cuts


def test_frozen_partition_is_computed_once_and_follows_changes():
    base, _, _, partitions = _fuzzifier(runs='none')
    x = torch.tensor(np.random.default_rng(1).normal(size=(10, 3)), dtype=torch.float32)
    base.deltas.requires_grad_(False)
    first = base(x).clone()
    assert torch.equal(base(x), first)
    with torch.no_grad():
        base.deltas[:, 0] += 2.0                                  # not a shift of every logit: the softmax ignores it
    assert not torch.equal(base(x), first)
