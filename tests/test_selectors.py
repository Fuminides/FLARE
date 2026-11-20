"""Discrete selectors: the forward value of gradient grafting is the discrete model, the backward pass is the
continuous surrogate; ties are broken by index; Gumbel weights stay finite."""
import pytest
import torch

from flare.selectors import (GradGraft, GradMode, first_max_mask, generate_concrete_weights, grafted_propagation, init_weights,
                             pure_propagation, select)


def surrogate(x, w, k, dim):
    """The continuous selector: (1 - k) one-hot + k uniform, times the weights."""
    mask = ((1.0 - k) * first_max_mask(w, dim) + k * torch.ones_like(w) / w.shape[dim]).detach()
    return torch.sum(x * (w * mask), dim=dim)


def test_graft_selector_has_pure_forward_and_continuous_backward():
    x = torch.tensor([[0.2, 0.7, 0.4], [0.9, 0.1, 0.3]], requires_grad=True)
    weights = torch.tensor([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]], requires_grad=True)
    k = 0.35
    graft_output = select(x, weights, k, 1, GradMode.GRAFT)
    pure_output = pure_propagation(x.detach(), weights.detach(), dim=1)
    weighted_output = torch.sum(x.detach() * weights.detach(), dim=1)
    torch.testing.assert_close(graft_output.detach(), pure_output, rtol=0, atol=0)
    assert not torch.allclose(graft_output.detach(), weighted_output)
    graft_gradients = torch.autograd.grad(graft_output.sum(), (x, weights))
    sx, sw = x.detach().clone().requires_grad_(True), weights.detach().clone().requires_grad_(True)
    surrogate_gradients = torch.autograd.grad(surrogate(sx, sw, k, 1).sum(), (sx, sw))
    for graft_gradient, surrogate_gradient in zip(graft_gradients, surrogate_gradients):
        torch.testing.assert_close(graft_gradient, surrogate_gradient)


@pytest.mark.parametrize('k', [0.0, 0.35, 1.0])
def test_grafted_propagation_equals_the_two_separate_propagations(k):
    torch.manual_seed(0)
    x = torch.rand(5, 1, 4, 3)
    w = torch.softmax(torch.randn(1, 6, 4, 3), dim=3)
    xa, wa = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    xb, wb = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    fused = grafted_propagation(xa, wa, k, dim=3)
    with torch.no_grad():
        hard = pure_propagation(xb, w=wb, dim=3)
    continuous = torch.sum(xb * wb, dim=3) if k == 1.0 else surrogate(xb, wb, k, 3)
    separate = GradGraft.apply(hard, continuous)
    assert torch.equal(fused, separate)
    upstream = torch.randn_like(fused)
    for a, b in zip(torch.autograd.grad((fused * upstream).sum(), (xa, wa)), torch.autograd.grad((separate * upstream).sum(), (xb, wb))):
        assert torch.equal(a, b)


def test_tied_selectors_choose_the_first_candidate_only():
    w = torch.tensor([[0.2, 0.5, 0.5, 0.1]])
    assert torch.equal(first_max_mask(w, 1), torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    x = torch.tensor([[0.0, 1.0, 1.0, 0.0]])                     # both tied candidates hold: only the first counts
    for mode in (GradMode.PURE, GradMode.GRAFT):
        assert torch.allclose(select(x, w, 0.3, 1, mode), torch.tensor([1.0])), mode


ROWS = {'one zero': [0.0, 0.3, 0.7], 'one negative': [-0.2, 0.5, 0.7], 'all zero': [0.0, 0.0, 0.0],
        'all negative': [-0.1, -0.3, -0.2], 'positive': [0.2, 0.3, 0.5]}


@pytest.mark.parametrize('k', [0.0, 1.0])
@pytest.mark.parametrize('row', list(ROWS))
def test_concrete_weights_finite_with_gradient(row, k):
    """The Gumbel weights stay finite when a candidate weight reaches zero or below: the logarithm is floored."""
    torch.manual_seed(0)
    w = torch.tensor([ROWS[row]] * 4, requires_grad=True)
    out = generate_concrete_weights(w, dim=1, k=k, temperature=5.0)
    assert torch.isfinite(out).all(), out
    assert torch.allclose(out.sum(dim=1), torch.ones(4))
    (out * torch.arange(3.0)).sum().backward()
    assert torch.isfinite(w.grad).all(), w.grad


def test_concrete_weights_of_positive_selectors_are_the_plain_gumbel_softmax():
    w = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    torch.manual_seed(1)
    out = generate_concrete_weights(w, dim=1, k=1.0, temperature=5.0)
    torch.manual_seed(1)
    assert torch.equal(out, torch.nn.functional.gumbel_softmax(torch.log(w), 5.0, dim=1, hard=False))


def test_concrete_selector_is_hard_at_k_zero():
    torch.manual_seed(0)
    out = select(torch.rand(4, 5), torch.softmax(torch.randn(4, 5), dim=1), 0.0, 1, GradMode.CONCRETE)
    assert out.shape == (4,)
    w = generate_concrete_weights(torch.softmax(torch.randn(4, 5), dim=1), 1, 0.0)
    assert torch.equal(w.max(dim=1)[0], torch.ones(4))


def test_init_weights_draw_uniform_noise_before_the_normal_values():
    """`rand` is drawn and then overwritten by `normal_`; the draws that follow depend on both, so both are kept."""
    torch.manual_seed(0)
    init_weights((3, 4), 1, 'cpu')
    state = torch.get_rng_state()
    torch.manual_seed(0)
    torch.nn.init.normal_(torch.rand(3, 4))
    assert torch.equal(torch.get_rng_state(), state)
    torch.manual_seed(0)
    torch.nn.init.normal_(torch.empty(3, 4))
    assert not torch.equal(torch.get_rng_state(), state)
