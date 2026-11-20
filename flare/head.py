"""The additive evidence head.

The score of class c is  s_c(x) = b_c + sum_r w_rc T_r(x):  every rule adds its signed points, weighted by its truth
degree, to the class scores, and the predicted class is the one with the highest score. The rules therefore add up
instead of competing, and the head is linear in the points. That linearity is what lets the average of several
models be again one FLARE model: the union of their rules with the points divided by the number of models.

With `joint_bags = B > 1` the rules form B blocks of the same size, each with its own bias, and the decision is the mean
of the blocks. The blocks are independent models trained at once (see bagging.py); the per-block scores are kept in
`bag_outputs` so that each block can have its own loss.
"""
import torch
import torch.nn as nn

from .selectors import init_weights


class EvidenceHead(nn.Module):
    def __init__(self, n_classes, n_rules, joint_bags, device):
        super().__init__()
        self.n_classes = n_classes
        # The "logits" initialisation is drawn first and then overwritten by N(0, 0.1); the first draw advances the
        # random generator, so removing it changes every later draw.
        self.weights = nn.Parameter(init_weights((n_rules, n_classes), dim=1, device=device))
        with torch.no_grad():
            self.weights.normal_(0.0, 0.1)
        self.joint_bags = int(joint_bags)
        self.bias = nn.Parameter(torch.zeros(self.joint_bags, n_classes, device=device) if self.joint_bags > 1
                                 else torch.zeros(n_classes, device=device))

    def forward(self, log_truth):
        """log_truth: [N, R] -> (class scores [N, C], contributions of every rule [N, R, C])."""
        scores = torch.exp(log_truth).unsqueeze(2) * self.weights.unsqueeze(0)
        if self.joint_bags > 1:
            per_bag = scores.view(scores.shape[0], self.joint_bags, -1, scores.shape[2]).sum(dim=2) + self.bias.unsqueeze(0)   # [N, K, C]
            self.bag_outputs = per_bag                  # the training loss scores every block on its own rows
            return per_bag.mean(dim=1), scores
        return self.bias + scores.sum(dim=1), scores
