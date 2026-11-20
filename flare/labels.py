"""Learned Ruspini partitions and conditions on runs of labels.

A real variable x_j has L trapezoidal labels that form a Ruspini partition: label l falls on the same ramp on which
label l + 1 rises, so the memberships of a value always add up to one. The partition is described by the 2L - 1 lengths
(gaps) between consecutive breakpoints, starting at the smallest training value and ending at the largest. The gaps are
learned through a softmax of free logits theta_j (initialised at the log of the quantile gaps),

    delta_j = (M_j - m_j) softmax(theta_j),

so the partition always covers the whole range and no segment collapses to zero (a clamped gap that reached zero would
get no gradient again, and Adam's fixed-size steps collapse labels that way). The point where two neighbouring labels
are both 0.5 true is a "crossing": training the gaps moves the crossings, and with them the cut points of every rule
that reads them.

Because the labels add up to one, "at least label k" (the sum of labels k.. L) and "at most label k" are one trapezoid
that goes between 1 and 0 along a single ramp. `LabelRunFuzzifier` adds these runs as extra conditions that a rule
can choose (`label_runs="shoulders"` is the default).
"""
import numpy as np
import torch
import torch.nn as nn

from .selectors import EPSILON


def label_run_list(n_labels, mode):
    """The (first, last) labels of each condition a rule can choose on a real variable. 'none': the single labels;
    'shoulders': also "at least label k" and "at most label k"; 'all': every run of neighbouring labels. The single
    labels come first, so condition indices below n_labels keep their meaning; the run of all the labels (always true)
    is left out."""
    runs = [(k, k) for k in range(n_labels)]
    if mode == 'shoulders':
        runs += [(k, n_labels - 1) for k in range(1, n_labels - 1)] + [(0, k) for k in range(1, n_labels - 1)]
    elif mode == 'all':
        runs += [(a, b) for a in range(n_labels) for b in range(a + 1, n_labels) if (a, b) != (0, n_labels - 1)]
    elif mode != 'none':
        raise ValueError(f"label_runs must be 'none', 'shoulders' or 'all', got {mode!r}")
    return runs


def run_label_name(names, run):
    """Readable name of a run of labels: "Low", "at least Medium", "at most Low" or "Low to Medium"."""
    first, last = run
    if first == last:
        return names[first]
    if last == len(names) - 1:
        return f'at least {names[first]}'
    if first == 0:
        return f'at most {names[last]}'
    return f'{names[first]} to {names[last]}'


def ruspini_chain(n_labels):
    """Breakpoints of a Ruspini partition of n trapezoids in increasing order, as groups of (label, parameter) pairs
    that share one point, after a0 = b0 = the domain start: label k falls on the same ramp on which label k + 1 rises
    (c_k = a_{k+1}, d_k = b_{k+1}), and c_{n-1} = d_{n-1} = the domain end. 2n - 1 gaps."""
    chain = []
    for k in range(n_labels - 1):
        chain += [((k, 2), (k + 1, 0)), ((k, 3), (k + 1, 1))]
    return chain + [((n_labels - 1, 2), (n_labels - 1, 3))]


def initial_gaps(partitions):
    """Log gaps of the Ruspini chain, and the domain start and end of each variable, from quantile partitions
    [variables, labels, 4] (partitions.py). A tied quantile (an integer-valued variable) gets a gap that is tiny
    instead of zero, since its logarithm must exist."""
    n_variables, n_labels = partitions.shape[:2]
    chain = [group[0] for group in ruspini_chain(n_labels)]
    gaps = np.zeros((n_variables, len(chain)))
    starts = partitions[:, 0, 0].astype('float64')
    ends = partitions[:, -1, -1].astype('float64')
    for i in range(n_variables):
        previous = starts[i]
        for j, (label, index) in enumerate(chain):
            gaps[i, j] = partitions[i, label, index] - previous
            previous = partitions[i, label, index]
    return np.log(np.maximum(gaps, 1e-6 * (ends - starts)[:, None])), starts, ends


def breakpoints(starts, ends, log_gaps):
    """Trapezoid parameters [variables, labels, 4] from the log gaps: the softmax of the logits scaled to the range
    gives the gaps, which are accumulated one after another from the domain start."""
    n_labels = (log_gaps.shape[1] + 1) // 2
    chain = ruspini_chain(n_labels)
    gaps = torch.softmax(log_gaps, dim=1) * (ends - starts).unsqueeze(1)
    parameters = [[None] * 4 for _ in range(n_labels)]
    parameters[0][0] = parameters[0][1] = starts
    point = starts
    for j, group in enumerate(chain):
        point = point + gaps[:, j]
        for label, index in group:
            parameters[label][index] = point
    parameters[n_labels - 1][3] = parameters[n_labels - 1][2]
    return torch.stack([torch.stack(label_parameters, dim=1) for label_parameters in parameters], dim=1)


def trapezoid_constants(trapezoid_parameters):
    """Everything in `trapezoid_memberships` that does not depend on the input: domain ends, sorted breakpoints and
    slope denominators. With frozen partitions these are constants, so they are computed once."""
    a, b, c, d = (trapezoid_parameters[0, :, :, i] for i in range(4))
    domain_start = a[:, 0].unsqueeze(1).unsqueeze(0)
    domain_end = d[:, -1].unsqueeze(1).unsqueeze(0)
    # The breakpoints are sorted ascending; torch.sort is differentiable (it is a permutation).
    p = torch.sort(torch.stack([a, b, c, d], dim=-1), dim=-1).values.permute(2, 0, 1)
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    return domain_start, domain_end, p0, p1, p2, p3, torch.clamp(p1 - p0, min=EPSILON), torch.clamp(p3 - p2, min=EPSILON)


def trapezoid_memberships(x, constants):
    """[N, V] values -> [N, V, L] memberships of the trapezoids. Values outside the training range are clamped to it, so
    the first (last) label is fully true below (above) it."""
    domain_start, domain_end, p0, p1, p2, p3, rise, fall = constants
    x = x.unsqueeze(2)
    x = torch.where(x <= domain_start, domain_start, x)
    x = torch.where(x >= domain_end, domain_end, x)
    rising = (x - p0) / rise
    falling = (p3 - x) / fall
    plateau = torch.logical_and(x >= p1, x <= p2).float()
    return torch.max(torch.clamp(torch.min(rising, falling), 0.0, 1.0), plateau)


class RuspiniFuzzifier(nn.Module):
    """The learned Ruspini partition of all the real variables: [N, V] -> [N, V, L] memberships.

    `deltas` holds the logits theta of the gaps and is the only parameter. It is frozen (`requires_grad_(False)`) in the
    averaged models after the first, which read the partition the first one learned."""

    def __init__(self, partitions, device):
        super().__init__()
        log_gaps, starts, ends = initial_gaps(partitions)
        # Rounded to float32 once, on the host.
        self.deltas = nn.Parameter(torch.tensor(log_gaps.astype('float32'), device=device))
        # Buffers, so that moving the model to another device moves them too (not in the state: they follow from the data).
        self.register_buffer('begin_points', torch.tensor(starts.astype('float32'), device=device), persistent=False)
        self.register_buffer('end_points', torch.tensor(ends.astype('float32'), device=device), persistent=False)

    def forward(self, x):
        # A frozen partition has constant breakpoints: compute them once per version of the logits instead of
        # rebuilding them from ~40 small operations on every forward.
        key = (self.deltas._version, self.deltas.device)
        if self.deltas.requires_grad or getattr(self, '_frozen_key', None) != key:
            parameters = breakpoints(self.begin_points, self.end_points, self.deltas)
            constants = trapezoid_constants(parameters.unsqueeze(0))
            self._frozen_key, self._frozen_constants = (None, None) if self.deltas.requires_grad else (key, constants)
        else:
            constants = self._frozen_constants
        return trapezoid_memberships(x, constants)

    @torch.no_grad()
    def crossings(self):
        """[vars, labels - 1] points where neighbouring labels k and k + 1 are both 0.5 true (the middle of their
        shared ramp): where a run of labels cuts."""
        p = breakpoints(self.begin_points, self.end_points, self.deltas)
        return (p[:, :-1, 2] + p[:, :-1, 3]) / 2


class LabelRunFuzzifier(nn.Module):
    """A fuzzifier whose conditions are runs of neighbouring labels of `base` (label_run_list). Neighbouring trapezoids
    of the partition sum to one where they overlap and the others do not meet, so the membership of a run is the sum
    of its labels' memberships: one wider trapezoid, or a shoulder when the run reaches an end of the domain."""

    def __init__(self, base, runs, n_labels, device):
        super().__init__()
        self.base = base
        cover = torch.zeros((n_labels, len(runs)), device=device)
        for j, (first, last) in enumerate(runs):
            cover[first:last + 1, j] = 1.0
        self.register_buffer('cover', cover)

    def forward(self, x):
        return torch.clamp(self.base(x) @ self.cover, max=1.0)               # [N, V, L] -> [N, V, runs]
