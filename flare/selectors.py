"""Discrete selectors and how they are differentiated.

FLARE makes discrete choices: every condition slot of a rule takes ONE variable, and ONE label of that variable. The
final model is exactly this discrete model. To train it with gradients, each selector is a set of logits over Q
candidates, and there are three ways to run it, selected by `GradMode`:

* `PURE`     the discrete forward only: the candidate with the largest weight is chosen, the others are ignored.
             This is what is evaluated and printed.
* `GRAFT`    gradient grafting (Wang et al., 2021). The forward value is the PURE one. The backward pass takes the
             gradient of a continuous surrogate, in which each selector is a weighted mixture of its candidates blended
             with a uniform share `k` (annealed from 1 towards 0 while training, see `train.py`):
                 h~ = (1 - k) * rho_{q*} * h_{q*} + k / Q * sum_q rho_q * h_q,     rho = softmax(phi / temperature)
* `CONCRETE` the Gumbel alternative: a Gumbel-softmax relaxation of the choice (hard, i.e. straight-through, when
             k == 0).

Everything here is a small autograd helper; the layers that call it are in `rules.py`, `categorical.py` and `head.py`.
"""
from enum import Enum

import torch
import torch.nn.functional as F

EPSILON = 1e-6


class GradMode(Enum):
    GRAFT = 2
    PURE = 4
    CONCRETE = 6

    @classmethod
    def from_string(cls, name):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown gradient mode {name!r}; use 'graft' or 'concrete'") from None


class GradGraft(torch.autograd.Function):
    """Gradient grafting: the forward value is `X`, the backward gradient goes to `Y` (which has the same shape).

    Calling `GradGraft.apply(discrete, continuous)` therefore outputs the discrete value and trains as if the output
    were the continuous one."""

    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


# While a CUDA graph is being recorded (`cuda_graph.GraphedStep`) this holds 0-dim device tensors for k, 1 - k and
# min(0.1, k). A Python number would be frozen into the recording; the tensors are updated in place every epoch.
RECORDING_K = None


def k_terms(k):
    """(1 - k, k, min(0.1, k)): Python numbers, or device tensors while a CUDA graph is recorded."""
    if RECORDING_K is None:
        return 1.0 - k, k, min(0.1, k)
    return RECORDING_K['one_minus_k'], RECORDING_K['k'], RECORDING_K['residual']


def first_max_mask(x, dim):
    """One-hot of the FIRST maximal entry along `dim`. Comparing with the maximum instead would select every tied
    entry (a near-uniform selector can round to a tie on one device and not on another), so ties are broken by index."""
    return torch.zeros_like(x).scatter_(dim, x.argmax(dim=dim, keepdim=True), 1.0)


def process_weights(logits, dim, temperature):
    """Selector shares rho = softmax(logits / temperature) along `dim`."""
    return torch.nn.functional.softmax(logits / temperature, dim=dim)


def pure_propagation(x, w, dim):
    """The discrete selection: sum over the candidates of x times the one-hot of the best-weighted candidate."""
    return torch.sum(x * first_max_mask(w, dim), dim=dim)


def grafted_propagation(x, w, k, dim):
    """PURE forward with the continuous surrogate (see the module docstring) grafted on the backward pass.

    The hard mask is computed once and shared by both. When k == 1 (the first epoch) the surrogate has no hard part
    and is simply sum(x * w), so the mask is skipped."""
    with torch.no_grad():
        discrete = first_max_mask(w, dim)
        hard = torch.sum(x * discrete, dim=dim)
        if k != 1.0:
            one_minus_k, k_value, _ = k_terms(k)
            mask = one_minus_k * discrete + k_value * (torch.ones_like(w) / w.shape[dim])
    continuous = torch.sum(x * (w * mask if k != 1.0 else w), dim=dim)
    return GradGraft.apply(hard, continuous)


# The Gumbel-softmax's own temperature. The model's annealed `temp` (set by the training loop in CONCRETE mode) only
# scales the selector logits before the relaxation; this one stays at 1.
CONCRETE_TEMPERATURE = 1.0


def generate_concrete_weights(w, dim, k, temperature=CONCRETE_TEMPERATURE):
    """Gumbel-softmax sample over the candidates, one-hot (straight-through) when k == 0. The weights are floored at
    EPSILON before the log: a weight at or below 0 would give log = -inf and NaN gradients."""
    w = torch.clamp(w, min=EPSILON)
    return F.gumbel_softmax(torch.log(w), temperature, dim=dim, hard=k == 0.0)


def select(x, w, k, dim, mode):
    """Choose among the candidates of `x` (summed over `dim`) with the shares `w`, forward and backward as in `mode`."""
    if mode == GradMode.GRAFT:
        return grafted_propagation(x, w, k, dim)
    if mode == GradMode.PURE:
        return pure_propagation(x, w, dim)
    if mode == GradMode.CONCRETE:
        return torch.sum(x * generate_concrete_weights(w, dim, k), dim=dim)
    raise ValueError(mode)


def init_weights(shape, dim, device):
    """Selector logits: uniform noise overwritten by a normal with the LeCun standard deviation 1 / sqrt(fan_in).

    The uniform draw is overwritten by `normal_`, but it advances the random generator, so removing it changes every
    later draw (the other layers' initial values, the batch order)."""
    init_tensor = torch.rand(shape, device=device, dtype=torch.float32)
    std = 1 / (init_tensor.size(dim) ** 0.5 + EPSILON)
    return torch.nn.init.normal_(init_tensor, mean=0.0, std=std).to(torch.float32)
