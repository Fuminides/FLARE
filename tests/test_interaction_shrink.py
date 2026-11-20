"""The slower interaction rate: the recorded rewind must give the same parameters as the plain operations."""
import numpy as np
import torch

from flare.cuda_graph import InteractionShrink
from flare.inputs import TabularInputs
from flare.model import FLAREModel
from flare.partitions import quantile_partitions
from flare.selectors import GradMode


def model_and_data():
    rng = np.random.RandomState(0)
    X = rng.rand(64, 4).astype(np.float32)
    torch.manual_seed(0)
    model = FLAREModel(TabularInputs(quantile_partitions(X, 3), 'none', None, None, 'cpu'), 2, np.zeros(4, dtype=bool), [], 10, 2, 2.0, GradMode.GRAFT)
    model.set_train_mode(GradMode.GRAFT, 0.5)
    model.restrict_to_single_condition(6)                                  # rules 6-9 are the interaction rows
    return model, torch.tensor(X), torch.tensor(rng.randint(0, 2, 64))


def train(shrink_kind, steps=6, factor=0.3):
    model, x, y = model_and_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    rows = [(p, dim, model.interaction_rows().nonzero().flatten()) for p, dim in model.rule_parameters()]
    shrink = InteractionShrink(rows, factor) if shrink_kind == 'class' else None
    for step in range(steps):
        optimizer.zero_grad()
        batch = slice(step * 8, step * 8 + 16)
        torch.nn.functional.cross_entropy(model(x[batch])[0], y[batch]).backward()
        if shrink is not None:
            shrink.before_step()
            optimizer.step()
            shrink.after_step()
        else:                                                              # the plain operations it replaces, written out
            before = [p.detach().index_select(dim, idx).clone() for p, dim, idx in rows]
            optimizer.step()
            with torch.no_grad():
                for (p, dim, idx), old in zip(rows, before):
                    p.index_copy_(dim, idx, old + factor * (p.index_select(dim, idx) - old))
    return model


def test_the_shrink_is_bit_identical_to_the_plain_operations():
    eager, shrunk = train('eager'), train('class')
    for (name, a), (_, b) in zip(eager.named_parameters(), shrunk.named_parameters()):
        assert torch.equal(a, b), name


def test_the_shrink_only_touches_interaction_rows():
    """With a factor of 0 the interaction rows never move; the other rows train as usual."""
    start = {n: p.detach().clone() for n, p in model_and_data()[0].named_parameters()}
    trained = train('class', factor=0.0)
    rows = trained.interaction_rows().nonzero().flatten()
    w0, w1 = start['head.weights'], trained.head.weights.detach()
    assert torch.equal(w1[rows], w0[rows])
    assert not torch.equal(w1[:6], w0[:6])
