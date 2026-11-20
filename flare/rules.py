"""Learning conditions and rules.

A rule has S condition slots. For every rule r:

* `variable_logits[s, r, j]`   how much slot s of rule r wants variable j. The slot takes the variable with the
                               largest weight. Real variables come first, then the categorical ones.
* `label_logits[r, j, l]`      how much rule r wants label l of the real variable j (the same label for all its slots).
                               With label runs, "labels" are the conditions of `labels.py` (single labels, "at least",
                               "at most").
* `switch_logits[s, r, j, :]`  (entry 0 is "on", entry 1 is "off") a slot after the first can be switched off, so rules
                               can be as short as needed. The slot is off when its off weight is larger than its on
                               weight for the variable it selected.

The truth of a rule is the product of the truths of its active slots (product t-norm); the layer works with the sum of
their logarithms, so what it returns is log T_r(x), shape [N, R]. Everything that chooses (variable, label, on/off)
goes through `selectors.select`, i.e. discrete in the forward pass and grafted (or Gumbel) in the backward pass.
"""
import torch
import torch.nn as nn

from .selectors import EPSILON, GradGraft, GradMode, init_weights, k_terms, process_weights, select


class RuleLayer(nn.Module):
    def __init__(self, n_real, n_categorical, n_conditions, n_rules, n_slots, allow_repeated_conditions, device):
        super().__init__()
        self.num_vars = n_real
        self.num_cat_vars = n_categorical
        # Three tensors of selector logits (initial values: see selectors.init_weights). Shapes keep a singleton
        # dimension that broadcasts against the batch.
        self.label_logits = nn.Parameter(init_weights((1, n_rules, n_real, n_conditions), dim=3, device=device))
        self.variable_logits = nn.Parameter(init_weights((n_slots, 1, n_rules, n_real + n_categorical), dim=2, device=device))
        # The switch of slot 0 never exists (the first slot is always on): its entries are kept so that the tensor has
        # one row per slot, but they are never read and receive no gradient.
        self.switch_logits = nn.Parameter(init_weights((n_slots, 1, n_rules, n_real + n_categorical, 2), dim=4, device=device))
        self.rule_size = n_slots
        # True (the default): a slot may repeat the variable of an earlier active slot of its rule, and the membership is
        # then multiplied in again. False: such a slot counts as off.
        self.allow_repeated_conditions = allow_repeated_conditions

    def forward(self, memberships, mode, k, temp, categorical_truth=None, residual=True):
        """memberships: [N, V, L] of the real variables (None if there are none); categorical_truth: [N, R, Vc] from
        `CategoricalConditions`. Returns log T_r(x), [N, R]."""
        label_shares = process_weights(self.label_logits, dim=3, temperature=temp)
        variable_shares = process_weights(self.variable_logits, dim=3, temperature=temp)

        # Step 1: every rule picks one label (condition) of every real variable. [N, R, V]
        if memberships is not None:
            chosen = select(memberships.unsqueeze(1), label_shares, k, 3, mode)
            if categorical_truth is not None:
                chosen = torch.cat([chosen, categorical_truth], dim=2)
        else:
            chosen = categorical_truth

        # Step 2: all slots at once. Slot 0 sees the chosen truths; the slots after it see them after their own
        # on/off switch (a candidate "the condition" against a candidate "1", i.e. no condition).
        n_slots = variable_shares.shape[0]
        per_slot = chosen.unsqueeze(0)                                                                  # [1, N, R, V]
        if n_slots > 1:
            switch_shares = process_weights(self.switch_logits[1:], dim=4, temperature=temp)           # [S-1, 1, R, V, 2]
            with_ones = torch.stack([chosen, torch.ones_like(chosen)], dim=3).unsqueeze(0)             # [1, N, R, V, 2]
            switched = select(with_ones, switch_shares, k, 4, mode)                                    # [S-1, N, R, V]
            per_slot = torch.cat([per_slot, switched], dim=0)                                          # [S, N, R, V]

        # Step 3: every slot picks one variable. [S, N, R]
        slot_truth = select(per_slot, variable_shares, k, 3, mode)
        if not self.allow_repeated_conditions and n_slots > 1:
            # A repeated condition would square its membership. Its slot contributes 1 instead (as if off) in the
            # forward pass, so PURE and the printed rules never repeat; the backward pass keeps the slot's gradient.
            repeated = self.repeated_condition_mask(variable_shares).unsqueeze(1)                      # [S, 1, R]
            hard = torch.where(repeated, torch.ones_like(slot_truth), slot_truth)
            slot_truth = hard if mode == GradMode.PURE else GradGraft.apply(hard.detach(), slot_truth)

        # Step 4: product of the slots = sum of their logs (a sequential sum, in the order of the slots).
        slot_logs = torch.log(slot_truth + EPSILON)
        log_truth = slot_logs[0]
        for ix in range(1, n_slots):
            log_truth = log_truth + slot_logs[ix]

        # Residual term: every chosen membership gets a direct gradient path to the truth of its rule. It is grafted
        # onto the backward pass only, so the forward values are unchanged.
        if residual and mode != GradMode.PURE:
            log_truth = GradGraft.apply(log_truth.detach(), log_truth + torch.sum(chosen, dim=2) * k_terms(k)[2])
        return log_truth

    def repeated_condition_mask(self, variable_shares):
        """[S, R] True where an active slot selects the variable of an earlier active slot of the same rule (the
        forward's hard choices, without host synchronisation)."""
        selected = variable_shares[:, 0].argmax(dim=2)                                                  # [S, R]
        active = torch.ones_like(selected, dtype=torch.bool)
        silence = self.switch_logits[1:, 0]                                                             # [S-1, R, V, 2]
        picked = torch.gather(silence, 2, selected[1:, :, None, None].expand(-1, -1, 1, 2)).squeeze(2)  # [S-1, R, 2]
        active[1:] = ~(picked[..., 1] > picked[..., 0])
        repeated = torch.zeros_like(active)
        for s in range(1, selected.shape[0]):
            repeated[s] = ((selected[:s] == selected[s]) & active[:s]).any(dim=0)
        return repeated & active

    # The discrete choices of every rule, for printing: which label, which variable and whether the slot is off.

    def chosen_conditions(self, temp):
        """[rules, real variables] condition (label) each rule takes of each real variable."""
        return torch.max(process_weights(self.label_logits, dim=3, temperature=temp)[0], dim=2)[1]

    def chosen_variables(self, temp):
        """[slots, rules] variable taken by every slot of every rule."""
        return torch.max(process_weights(self.variable_logits, dim=3, temperature=temp)[:, 0], dim=2)[1]

    def switched_off(self, selected):
        """[slots, rules] bool: the slot is off for the variable it selected (its off weight beats its on weight)."""
        switch = self.switch_logits[:, 0]                                                               # [S, R, V, 2]
        off = switch[..., 1] > switch[..., 0]
        return torch.gather(off, 2, selected.long().unsqueeze(2)).squeeze(2)
