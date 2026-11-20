"""CUDA graphs (flare/cuda_graph.py). Skipped without a GPU."""
import copy

import pytest
import torch

from flare.cuda_graph import GraphedStep
from flare.selectors import GradMode
from helpers import toy_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need a GPU')


def test_graphed_step_is_bit_identical_to_the_eager_step_across_k_changes_and_batch_shapes():
    eager, X = toy_model(n_labels=3, n_rules=5, n_slots=2, label_runs='none', device='cuda')
    y = (torch.arange(len(X), device='cuda') % 2)
    graphed = copy.deepcopy(eager)
    X, y = X[:17], y[:17]                                                              # batches of 6, 6 and 5 rows
    optimizers = [torch.optim.Adam(m.parameters(), lr=0.1) for m in (eager, graphed)]
    step = GraphedStep(lambda a, b: torch.nn.functional.cross_entropy(graphed(a)[0], b), graphed.parameters())
    for k in (0.9, 0.35, 0.0):
        for m in (eager, graphed):
            m.set_train_mode(GradMode.GRAFT, k)
        step.set_k(k)
        for start in range(0, 17, 6):
            xb, yb = X[start:start + 6], y[start:start + 6]
            optimizers[0].zero_grad()
            eager_loss = torch.nn.functional.cross_entropy(eager(xb)[0], yb)
            eager_loss.backward()
            assert torch.equal(step(xb, yb), eager_loss)
            for m, optimizer in zip((eager, graphed), optimizers):
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                optimizer.step()
            for (name, a), b in zip(eager.named_parameters(), graphed.parameters()):
                assert torch.equal(a, b), (name, k, start)
    assert len(step.recordings) == 2 and not step.failed


def test_graphed_step_falls_back_when_recording_fails_and_skips_epochs_with_k_equal_one():
    model, X = toy_model(n_labels=3, n_rules=5, n_slots=2, label_runs='none', device='cuda')

    def broken_loss(a, b):
        raise ValueError('not recordable')

    step = GraphedStep(broken_loss, model.parameters())
    step.set_k(1.0)
    assert not step.usable()                                   # k == 1 uses the unmasked selector path: run eagerly
    step.set_k(0.5)
    assert step.usable()
    with pytest.warns(UserWarning, match='CUDA graph recording failed'):
        assert step(X, torch.zeros(len(X), dtype=torch.long, device='cuda')) is None
    assert step.failed and not step.usable()
