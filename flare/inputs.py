"""The inputs of the rule layer: how the real variables become condition truths.

`FLAREModel` (model.py) does not read raw variables. It reads, for every real variable j, the truth of each of the
conditions that a rule may choose on it: a tensor of memberships [N, V, L]. Whatever produces that tensor is an
"inputs" module. `TabularInputs` is the one used for tabular data: learned Ruspini partitions (labels.py), optional
runs of labels, and optional learned category spaces for high-cardinality categoricals.

Extension point. Experiments in which the memberships are given (concept bottleneck models: the truth of each concept
comes from a network; graphs: truths computed from neighbourhoods) replace `TabularInputs` by their own module. The
contract that `FLAREModel` relies on is:

    n_variables       number of real variables V
    n_conditions      number of conditions L that a rule may choose on each of them (same for every variable)
    forward(x)        [N, V] values -> [N, V, L] truths in [0, 1]
    condition_text(variable, condition, name, category_names=None) -> str
                      the text of a condition for printed rules, e.g. "age IS at least Medium [>= 0.4]"
    crossing_parameters() / space_parameters()
                      the parameters that are trained with their own step size (empty lists for fixed memberships)
    inherit(first)    copy from `first` (an inputs module of the first averaged model) what it learned and freeze it

Categorical variables are not part of the inputs: `FLAREModel` reads them directly (categorical.py).
"""
import torch
import torch.nn as nn

from .labels import LabelRunFuzzifier, RuspiniFuzzifier, label_run_list, run_label_name
from .partitions import label_names


class TabularInputs(nn.Module):
    def __init__(self, partitions, label_runs, category_spaces, space_codes, device):
        """partitions: [V, L, 4] quantile trapezoids (partitions.py). category_spaces: {variable index: warmed-up
        CategorySpace} (the variable's conditions are then the space's regions); space_codes: {variable index: original
        category codes by frequency rank}, used to print them."""
        super().__init__()
        self.n_variables, n_labels = partitions.shape[:2]
        self.label_names = label_names(n_labels)
        self.runs = label_run_list(n_labels, label_runs) if label_runs != 'none' else None
        self.n_conditions = len(self.runs) if self.runs else n_labels
        partition = RuspiniFuzzifier(partitions, device)
        self.fuzzifier = LabelRunFuzzifier(partition, self.runs, n_labels, device) if self.runs else partition
        self.category_spaces = nn.ModuleDict({str(k): v for k, v in category_spaces.items()}) if category_spaces else None
        self.space_codes = space_codes or {}

    @property
    def partition(self):
        return getattr(self.fuzzifier, 'base', self.fuzzifier)

    def forward(self, x):
        m = self.fuzzifier(x)
        if self.category_spaces is None:
            return m
        # The conditions of a category-space variable are the regions of its space (under label runs, the bounded
        # union of the regions that each run covers) instead of the fuzzy partition's labels.
        cover = self.fuzzifier.cover if self.runs else None
        m = m.clone()
        for variable, space in self.category_spaces.items():
            regions = space(x[:, int(variable)])
            m[:, int(variable), :] = torch.clamp(regions @ cover.to(regions.dtype), max=1.0) if cover is not None else regions
        return m

    # Parameters with their own step size (train.py).

    def crossing_parameters(self):
        return [p for p in (self.partition.deltas,) if p.requires_grad]

    def space_parameters(self):
        return [p for p in self.category_spaces.parameters() if p.requires_grad] if self.category_spaces is not None else []

    def space_loss(self, device):
        """The spaces' reconstruction and radius terms (spaces frozen for later bags contribute nothing)."""
        terms = [space.loss() for space in self.category_spaces.values() if space.points.requires_grad]
        return sum(terms) if terms else torch.zeros((), device=device)

    def inherit(self, first):
        """One partition and one vocabulary of groups for the union rule base: the later models of a bag read the
        crossings and the category spaces of the first one, frozen."""
        gaps = self.partition.deltas
        with torch.no_grad():
            gaps.copy_(first.partition.deltas)
        gaps.requires_grad_(False)
        if self.category_spaces is not None:
            self.category_spaces.load_state_dict(first.category_spaces.state_dict())
            self.category_spaces.requires_grad_(False)

    # Printed rules.

    def condition_text(self, variable, condition, name, category_names=None, shown=5):
        space = self.category_spaces[str(variable)] if self.category_spaces is not None and str(variable) in self.category_spaces else None
        if space is not None:
            return self._space_text(variable, condition, space, name, category_names, shown)
        run = self.runs[condition] if self.runs else (condition, condition)
        try:
            label = run_label_name(self.label_names, run)
        except Exception:                                        # noqa: BLE001 - a missing name must not stop printing
            label = str(condition)
        cuts = self._cuts()
        first, last = run
        lower = f'{cuts[variable, first - 1]:.3g}' if first > 0 else None
        upper = f'{cuts[variable, last]:.3g}' if last < cuts.shape[1] else None
        suffix = f' [>= {lower}]' if upper is None else f' [<= {upper}]' if lower is None else f' [{lower} to {upper}]'
        return f'{name} IS {label}{suffix}'

    def _cuts(self):
        # Where each real condition cuts, in the model's input units (the half-true points of the learned ramps).
        if getattr(self, '_cuts_cache', None) is None:
            self._cuts_cache = self.partition.crossings().cpu().numpy()
        return self._cuts_cache

    def _space_text(self, variable, condition, space, name, category_names, shown):
        """"X IN {A, F, G, ...} (n categories)": the categories that each region (or union of regions) holds."""
        if getattr(self, '_space_cache', None) is None:
            self._space_cache = {}
        if variable not in self._space_cache:
            cover = self.fuzzifier.cover if self.runs else None
            self._space_cache[variable] = space.groups(cover)
        held = self._space_cache[variable][condition]
        codes = self.space_codes[variable]
        names = category_names

        def label(code):
            return names[int(code)] if names is not None and 0 <= int(code) < len(names) else str(int(code))
        listed = ', '.join(label(codes[i]) for i in held[:shown])
        more = ', ...' if len(held) > shown else ''
        return f'{name} IN {{{listed}{more}}} ({len(held)} categories)'
