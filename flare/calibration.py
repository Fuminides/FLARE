"""Post-hoc recalibration of the class probabilities.

The scores of the model are recalibrated on the validation rows with a temperature T and one bias per class,
softmax(s / T + b), fitted by maximum likelihood. The model, its rules and its decisions do not change; only the
probabilities do. T is kept at least `min_temperature` (1 means it may only soften the scores), and the class biases
are shrunk with an L2 penalty: the validation rows also chose the epoch, which makes the model look more certain than
it is, and an unconstrained fit sharpens the probabilities on small tasks.
"""
import torch

from .metrics import output_scores


def calibrate_probabilities(model, loader, min_temperature=0.0, bias_penalty=0.0):
    """(T, b) fitted on the loader's rows with L-BFGS; K + 1 parameters."""
    s, y = output_scores(model, loader)
    u = torch.zeros((), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(s.shape[1], dtype=torch.float64, requires_grad=True)
    temperature = (lambda: min_temperature + torch.exp(u)) if min_temperature > 0 else (lambda: torch.exp(u))
    optimizer = torch.optim.LBFGS([u, b], lr=0.5, max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(s / temperature() + b, y) + bias_penalty * b.pow(2).sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(temperature().detach()), b.detach().numpy()
