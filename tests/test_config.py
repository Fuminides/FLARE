import pytest

from flare.config import FLAREConfig, flare, flare_explainability, tuning_space


def test_flare_is_eight_models_of_120_rules_with_label_runs():
    c = flare()
    assert (c.bags, c.rules, c.single_condition_rules, c.slots, c.labels, c.label_runs) == (8, 120, 100, 3, 7, 'shoulders')
    assert (c.epochs, c.steps_per_epoch, c.lr, c.lr_structure, c.lr_interactions, c.crossings_lr, c.temp) == (100, 50, 0.1, 0.3, 0.3, 0.01, 5.0)
    assert (c.class_weight, c.grad_mode, c.checkpoint_metric, c.patience, c.category_space) == ('balanced', 'graft', 'acc', 0, 0)
    assert not c.calibrate


def test_flare_explainability_is_one_model_of_40_rules():
    c = flare_explainability()
    assert (c.bags, c.parallel_bags, c.rules, c.single_condition_rules, c.slots, c.labels) == (1, False, 40, 30, 3, 7)


def test_tabarena_adds_category_spaces_and_recalibration_and_gumbel_changes_the_estimator():
    c = flare(tabarena=True)
    assert (c.category_space, c.category_space_min, c.category_space_init, c.category_space_rec) == (10, 32, 'effect', 0.1)
    assert c.calibrate and c.calibration_min_temperature == 1.0 and c.calibration_bias_penalty == 0.05
    assert flare_explainability(tabarena=True, gumbel=True).grad_mode == 'concrete'
    assert flare(tabarena=True, category_space_min=5).category_space_min == 5          # any option can be overridden


def test_configurations_are_immutable_values():
    c = flare()
    assert c.replace(bags=5).bags == 5 and c.bags == 8
    assert FLAREConfig(**c.as_dict()) == c
    with pytest.raises(Exception):
        c.bags = 2


class Recorder:
    """An Optuna-like trial that records the order of the suggestions and returns the lowest value of each range."""

    def __init__(self):
        self.names = []

    def suggest_float(self, name, low, high, log=False):
        self.names.append(name)
        return low

    def suggest_int(self, name, low, high, step=1):
        self.names.append(name)
        return low

    def suggest_categorical(self, name, choices):
        self.names.append(name)
        return choices[0]


def test_tuning_space_keeps_the_order_of_the_suggestions_and_the_ranges_of_the_paper():
    trial = Recorder()
    params = tuning_space(trial)
    # the order matters: the TPE sampler's draws depend on it
    assert trial.names == ['rules', 'lr', 'temp', 'vl', 'single_share', 'lr_structure', 'lr_interactions', 'crossings_lr',
                           'class_weight', 'point_bags']
    assert params == dict(lr=0.02, temp=2.0, labels=5, rules=60, single_condition_rules=36, lr_structure=0.1, lr_interactions=0.1,
                          crossings_lr=0.003, class_weight='balanced', bags=5)
    assert FLAREConfig().replace(**params).rules == 60
