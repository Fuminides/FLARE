"""Configuration of FLARE: every hyperparameter in one place, the presets and the tuning space.

Two presets are built by `flare()` and `flare_explainability()`:

* FLARE (performance): 8 averaged models of 120 rules each (100 with one condition and 20 with up to 3), 7 labels per
  real variable.
* FLARE-explainability: one model of 40 rules (30 + 10), the smallest configuration that stays strong.

Both use runs of labels ("at least / at most") on the real variables and learned partition crossings. For TabArena
(`tabarena=True`) FLARE also learns category spaces for categoricals of more than 32 training categories and
recalibrates its probabilities. With `gumbel=True` the selectors are trained with a Gumbel-softmax relaxation instead of
gradient grafting.
"""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class FLAREConfig:
    # --- structure of one model --------------------------------------------------------------------------------
    rules: int = 120                        # rules per model
    single_condition_rules: int = 100       # of them, capped at one condition (the main effects); the rest are interactions
    slots: int = 3                          # condition slots of an interaction rule
    labels: int = 7                         # labels per real variable
    label_runs: str = 'shoulders'           # 'none' | 'shoulders' ("at least / at most") | 'all' (every run of labels)
    bags: int = 8                           # averaged models (1 = a single model)
    parallel_bags: bool = True              # train the averaged models at once when that is faster (bagging.py)
    allow_repeated_conditions: bool = True  # True: a slot may repeat a condition of its rule; False: it counts as off
    categorical_floor: float = 0.1          # truth of a category that does not match
    # --- training ---------------------------------------------------------------------------------------------
    grad_mode: str = 'graft'                # 'graft' (gradient grafting) | 'concrete' (Gumbel-softmax)
    epochs: int = 100
    patience: int = 0                       # stop after this many epochs without improvement (0: never)
    batch_size: int = 32
    steps_per_epoch: int = 50               # the batch grows with the data, to about this many updates per epoch
    lr: float = 0.1                         # step of the points and the biases
    lr_structure: float = 0.3               # step of the selectors, relative to lr
    lr_interactions: float = 0.3            # factor on the update of the interaction rules
    crossings_lr: float = 0.01              # step of the partitions' crossings, relative to the selectors'
    category_space_lr: float = 0.1          # step of the category spaces, relative to the selectors'
    weight_decay: float = 0.0
    temp: float = 5.0                       # temperature of the selectors
    class_weight: str = 'balanced'          # 'balanced' (inverse class frequency) | 'none'
    residual: bool = True                   # residual gradient path
    k_upper_train: float = 1.0              # the uniform share k is clipped to [k_lower_train, k_upper_train]
    k_lower_train: float = 0.0
    cyclical_k: bool = True                 # k = xi + 0.2 sin(4 pi xi) instead of the linear schedule xi
    checkpoint_metric: str = 'acc'          # validation metric that picks the epoch: 'acc' | 'f1'
    cuda_graph: bool = True                 # replay the training step as a CUDA graph (GPU only; same numbers as eager mode)
    # --- category spaces (TabArena) -----------------------------------------------------------------------------
    category_space: int = 0                 # dimension K of a category space (0: none, categoricals keep one label per category)
    category_space_min: int = 32            # a categorical of more training categories than this gets a space
    category_space_init: str = 'effect'     # 'effect': the first axis starts at the effect of each category on the class
    category_space_rec: float = 0.1         # weight of the reconstruction loss
    category_space_radius: float = 1.0      # weight of the radius loss
    # --- probabilities (TabArena) --------------------------------------------------------------------------------
    calibrate: bool = False                 # also report the probability metrics recalibrated on the validation rows
    calibration_min_temperature: float = 1.0
    calibration_bias_penalty: float = 0.05

    def replace(self, **changes):
        return replace(self, **changes)

    def as_dict(self):
        return asdict(self)


def flare(tabarena=False, gumbel=False, **overrides):
    """FLARE (performance)."""
    config = FLAREConfig()
    if tabarena:
        config = config.replace(category_space=10, calibrate=True)
    if gumbel:
        config = config.replace(grad_mode='concrete')
    return config.replace(**overrides)


def flare_explainability(tabarena=False, gumbel=False, **overrides):
    """FLARE-explainability: one model of 30 + 10 rules."""
    return flare(tabarena, gumbel, **{'rules': 40, 'single_condition_rules': 30, 'bags': 1, 'parallel_bags': False, **overrides})


PRESETS = {'flare': flare, 'flare-explainability': flare_explainability}


# --- hyperparameter search (tuned FLARE) -----------------------------------------------------------------------------

def tuning_space(trial):
    """The options that tuned FLARE searches; the rest of the configuration stays. `trial` is an Optuna trial (or a
    FixedTrial). The order of the suggestions matters: the TPE sampler's draws depend on it."""
    f, i, c = trial.suggest_float, trial.suggest_int, trial.suggest_categorical
    rules = c('rules', [60, 120, 240, 480])
    return dict(lr=f('lr', 0.02, 0.3, log=True), temp=f('temp', 2.0, 10.0), labels=i('vl', 5, 11, step=2), rules=rules,
                single_condition_rules=int(round(rules * f('single_share', 0.6, 0.95))),
                lr_structure=f('lr_structure', 0.1, 1.0, log=True), lr_interactions=f('lr_interactions', 0.1, 1.0, log=True),
                crossings_lr=f('crossings_lr', 0.003, 0.03, log=True), class_weight=c('class_weight', ['balanced', 'none']),
                bags=c('point_bags', [5, 8]))


# Names under which two of the tuned options appear in the result files.
TUNED_NAMES = {'labels': 'vl', 'bags': 'point_bags'}
