"""Conditions on categorical variables.

A categorical variable with a few categories gets one crisp label per category: the condition "x_j IS a" is true (1)
for a row of category a. For every categorical variable and every rule there is a selector over the categories (the
logits `cat_weights[j][:, r]`), so rule r picks the category it tests. Every categorical variable is one candidate of
the slot selectors of `rules.py`, next to the real variables.

A category that does not match keeps a small truth `floor` (0.1) instead of 0, so a condition that is false for a row
still passes some gradient. The floor is part of the value of the condition in every mode, the discrete model
included: the truth of the condition is clamp(one-hot + floor, max=1) before the selection.
"""
import torch
import torch.nn as nn

from .selectors import init_weights, process_weights, select


class CategoricalConditions(nn.Module):
    def __init__(self, cardinalities, n_rules, floor, device):
        """cardinalities: number of categories of each categorical variable (the categories are coded 0..K-1)."""
        super().__init__()
        self.cat_variables = list(cardinalities)
        self.n_rules = n_rules
        self.cat_weights = nn.ParameterList([nn.Parameter(init_weights((self.cat_variables[i], n_rules), dim=0, device=device))
                                             for i in range(len(self.cat_variables))])
        self.floor = floor

    def forward(self, x, mode, k, temp):
        """x: [N, n_categorical] codes -> [N, n_rules, n_categorical] truth of the condition each rule has on each variable."""
        res = torch.zeros((x.shape[0], self.n_rules, len(self.cat_variables)), device=x.device)
        # Variables with the same number of categories are processed together (stacked selectors, one softmax, one
        # selection), so the Python loop runs once per cardinality instead of once per variable.
        for cardinality, variables, index in self._cardinality_groups(x.device):
            columns = x[:, index].to(torch.int64)                                                          # [N, G]
            one_hot_x = torch.nn.functional.one_hot(columns, cardinality).to(torch.float32)                # [N, G, card]
            weights = torch.stack([self.cat_weights[i] for i in variables], dim=0)                         # [G, card, R]
            shares = process_weights(weights, dim=1, temperature=temp)
            one_hot_x = torch.clamp(one_hot_x + self.floor, min=0.0, max=1.0)
            chosen = select(one_hot_x.unsqueeze(3), shares.unsqueeze(0), k, 2, mode)                       # [N, G, R]
            res[:, :, index] = chosen.permute(0, 2, 1)
        return res

    def _cardinality_groups(self, device):
        """[(cardinality, variable positions, the same positions as a device tensor)], cached per device (indexing with
        a Python list would copy it to the device on every forward, a host-device synchronisation)."""
        cache = self.__dict__.setdefault('_group_cache', {})
        if device not in cache:
            groups = {}
            for i, cardinality in enumerate(self.cat_variables):
                groups.setdefault(int(cardinality), []).append(i)
            cache[device] = [(cardinality, variables, torch.tensor(variables, device=device)) for cardinality, variables in groups.items()]
        return cache[device]

    def chosen_categories(self, temp):
        """[variables, rules] category each rule tests on each variable (the argmax of its selector)."""
        return torch.stack([process_weights(weights, dim=0, temperature=temp).max(dim=0)[1] for weights in self.cat_weights])
