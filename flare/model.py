"""The FLARE model: conditions, rules and the additive head put together.

    x  ->  conditions (truth of each label / category the rules may use)
       ->  rules (every rule: up to S conditions picked by selectors, truth = product of the conditions)    rules.py
       ->  class scores  s_c = b_c + sum_r w_rc T_r(x)                                                       head.py

The real variables enter through an `inputs` module (inputs.py: the learned fuzzy partitions of `TabularInputs`, or
fixed memberships supplied by an experiment); the categorical variables are read directly (categorical.py).

Besides the forward pass, this module has what is specific to FLARE's rule mix: `restrict_to_single_condition` makes
most of the rules one-condition rules (the main effects of an additive model) and leaves the others with up to S
conditions (the interactions), `rule_parameters` lists the tensors indexed by rule (needed to train several models
as the blocks of one, bagging.py) and `extract_rules` reads the discrete rules back from the selectors.
"""
import numpy as np
import torch
import torch.nn as nn

from .categorical import CategoricalConditions
from .head import EvidenceHead
from .rules import RuleLayer


class FLAREModel(nn.Module):
    def __init__(self, inputs, n_classes, categorical_mask, cardinalities, n_rules, n_slots, temp, mode, joint_bags=1,
                 allow_repeated_conditions=True, categorical_floor=0.1, residual=True, device='cpu'):
        """
        inputs:           an inputs module (see inputs.py) for the real variables, or None if there are none.
        categorical_mask: [columns] bool, True for the categorical columns of the data (in the data's column order).
        cardinalities:    number of categories of each categorical column (coded 0..K-1), in column order.
        n_rules:          rules in the model (the blocks of a joint model included).
        joint_bags:       number of independent models trained at once as blocks of this one (bagging.py).
        """
        super().__init__()
        self.categorical_mask = np.asarray(categorical_mask, dtype=bool)
        self.n_class = n_classes
        self.nrules = n_rules
        self.gradMode = mode
        self.temp = temp
        self.k = 0.1
        self.residual = residual
        self.num_vars = inputs.n_variables if inputs is not None else 0
        self.ncategorical_variables = len(cardinalities)
        self.total_vars = self.num_vars + self.ncategorical_variables
        # Position of each variable of the rule layer (real ones first, then categorical) among the data's columns.
        self.real_columns = [i for i, c in enumerate(self.categorical_mask) if not c]
        self.categorical_columns = [i for i, c in enumerate(self.categorical_mask) if c]

        # Build order: inputs, rule layer, categorical, head. Each one draws its initial values from the global generator,
        # so changing the order changes the initial values.
        self.inputs = inputs
        n_conditions = inputs.n_conditions if inputs is not None else 0
        self.rules = RuleLayer(self.num_vars, self.ncategorical_variables, n_conditions, n_rules, n_slots,
                               allow_repeated_conditions, device)
        self.categorical = (CategoricalConditions(cardinalities, n_rules, categorical_floor, device)
                            if self.ncategorical_variables > 0 else None)
        self.head = EvidenceHead(n_classes, n_rules, joint_bags, device)
        if self.ncategorical_variables > 0:
            self.register_buffer('_cat_columns', torch.as_tensor(self.categorical_columns, dtype=torch.long, device=device), persistent=False)
            self.register_buffer('_real_columns', torch.as_tensor(self.real_columns, dtype=torch.long, device=device), persistent=False)
        self.n_single_condition = 0
        self.single_condition_mask = None

    # --- modes ---------------------------------------------------------------------------------------------------

    def set_train_mode(self, mode, k):
        """Run in `mode` (GRAFT or CONCRETE) with the uniform share k while training."""
        self.gradMode = mode
        self.k = k

    def set_inference_mode(self, mode, k=0.0):
        """Run in `mode` without training: PURE is the discrete model that is deployed."""
        self.eval()
        self.gradMode = mode
        self.k = k

    def set_temperature(self, temp):
        self.temp = temp

    # --- forward ---------------------------------------------------------------------------------------------------

    def forward(self, x):
        """[N, columns] -> (class scores [N, C], contribution of every rule to every class [N, R, C])."""
        x_real, categorical_truth = x, None
        if self.ncategorical_variables > 0:
            categorical_truth = self.categorical(x[:, self._cat_columns], self.gradMode, self.k, self.temp)
            x_real = x[:, self._real_columns]
        memberships = self.inputs(x_real) if self.num_vars > 0 else None
        log_truth = self.rules(memberships, self.gradMode, self.k, self.temp, categorical_truth, self.residual)
        self.cache_truth_rule_values = log_truth
        return self.head(log_truth)

    # --- the rule mix -----------------------------------------------------------------------------------------------

    @torch.no_grad()
    def restrict_to_single_condition(self, n_rules, logit=5.0, bag_size=0):
        """Cap the first `n_rules` rules at one condition: their later slots are switched off (the "off" logit is large)
        and receive no gradient, so the search only chooses which single condition each of them tests. The remaining
        rules keep every slot. With joint bags (`bag_size` rules per block) the cap applies to the first n_rules of every
        block. Returns how many rules were capped."""
        layer = self.rules
        index = torch.arange(self.nrules, device=layer.label_logits.device)
        capped = (index % bag_size < n_rules) if bag_size > 0 else (index < n_rules)
        self.single_condition_mask = capped.clone()
        if layer.rule_size < 2 or not bool(capped.any()):
            self.n_single_condition = int(capped.sum())
            return self.n_single_condition
        layer.switch_logits[1:, 0, capped, :, 0] = -logit
        layer.switch_logits[1:, 0, capped, :, 1] = logit

        def frozen_slots(parameter):
            mask = torch.ones_like(parameter)
            mask[1:, 0, capped] = 0.0
            parameter.register_hook(lambda grad: grad * mask)

        frozen_slots(layer.switch_logits)
        frozen_slots(layer.variable_logits)
        self.n_single_condition = int(capped.sum())
        return self.n_single_condition

    def interaction_rows(self):
        """[R] bool: the rules that may hold more than one condition (the interaction rules)."""
        if self.rules.rule_size < 2:
            return torch.zeros(self.nrules, dtype=torch.bool, device=self.head.weights.device)
        rows = torch.ones(self.nrules, dtype=torch.bool, device=self.head.weights.device)
        if self.single_condition_mask is not None:
            rows &= ~self.single_condition_mask.to(rows.device)
        return rows

    def rule_parameters(self):
        """[(parameter, rule dimension)] for every parameter indexed by rule."""
        layer = self.rules
        pairs = [(layer.label_logits, 1), (layer.variable_logits, 2), (layer.switch_logits, 2), (self.head.weights, 0)]
        if self.ncategorical_variables > 0:
            pairs += [(weights, 1) for weights in self.categorical.cat_weights]
        return pairs

    # --- reading the rules ------------------------------------------------------------------------------------------

    def active_rules(self):
        """[R] bool, all True: every rule takes part in inference."""
        return torch.ones(self.nrules, dtype=torch.bool, device=self.rules.label_logits.device)

    def extract_rules(self):
        """The discrete rules. Returns (rules [R, columns], rule_weights [R, C], consequents [R]) as numpy arrays:
        rules[r, j] is the condition (label index, or category) that rule r tests on column j, -1 if it does not read
        j; rule_weights are the points of the rule normalised to sum |.| = 1 and the consequent is the class with
        the largest normalised points (the rule's "main" class; the model itself adds up all the classes)."""
        n_columns = len(self.categorical_mask)
        rule_matrix = torch.zeros((self.nrules, self.total_vars)) - 1
        if self.num_vars > 0:
            rule_matrix[:, :self.num_vars] = self.rules.chosen_conditions(self.temp)[:, :self.num_vars].cpu().to(rule_matrix.dtype)
        if self.ncategorical_variables > 0:
            categories = self.categorical.chosen_categories(self.temp)[:, :self.nrules]
            rule_matrix[:, self.num_vars:self.num_vars + self.ncategorical_variables] = categories.T.cpu().to(rule_matrix.dtype)

        # The variable chosen by each slot, and whether the slot is off.
        selected = self.rules.chosen_variables(self.temp)                                              # [slots, rules]
        off = self.rules.switched_off(selected).cpu().numpy()
        selected = selected.cpu().numpy()
        chosen = np.zeros((self.nrules, self.total_vars), dtype=int) - 1
        for i in range(self.nrules):
            for j in range(self.rules.rule_size):
                variable = int(selected[j, i])
                if j == 0 or not off[j, i]:
                    chosen[i, variable] = rule_matrix[i, variable]

        # Back to the data's column order: real variables first inside the layer, categorical ones after.
        columns = np.zeros((self.nrules, n_columns), dtype=int) - 1
        for position, column in enumerate(self.real_columns):
            columns[:, column] = chosen[:, position]
        for position, column in enumerate(self.categorical_columns):
            columns[:, column] = chosen[:, self.num_vars + position]

        points = self.head.weights
        rule_weights = points / points.abs().sum(dim=1, keepdim=True).clamp(min=1e-12)
        consequents = rule_weights.argmax(dim=1)
        return columns, rule_weights.detach().cpu().numpy(), consequents.detach().cpu().numpy()


class BagEnsemble(nn.Module):
    """The union rule base of several models trained one after another (the sequential route of bagging.py): the head
    is linear in the points, so the mean of the members' outputs equals one rule base holding every member's rules with
    the points divided by the number of members. Attributes that are not defined here come from the first member."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.nrules = sum(m.nrules for m in members)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.members[0], name)

    def forward(self, x, **kwargs):
        outputs, scores = zip(*[member(x, **kwargs) for member in self.members])
        self.cache_truth_rule_values = torch.cat([member.cache_truth_rule_values for member in self.members], dim=1)
        return sum(outputs) / len(self.members), torch.cat(scores, dim=1) / len(self.members)

    def set_inference_mode(self, *args, **kwargs):
        for member in self.members:
            member.set_inference_mode(*args, **kwargs)

    def set_train_mode(self, *args, **kwargs):
        for member in self.members:
            member.set_train_mode(*args, **kwargs)

    def extract_rules(self):
        parts = [member.extract_rules() for member in self.members]
        return tuple(np.concatenate([part[i] for part in parts]) for i in range(3))
