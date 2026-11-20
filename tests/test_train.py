"""Training: the schedule of k, the checkpoint, and the routes of the averaged models."""
import numpy as np
import pytest
import torch

from flare.metrics import evaluate
from flare.model import BagEnsemble, FLAREModel
from flare.printing import rules_text
from flare.selectors import GradMode
from flare.train import fit_fold, k_schedule, score_fold
from helpers import small_config, toy_fold


def test_the_uniform_share_starts_at_one_and_ends_near_the_value_of_the_paper():
    assert k_schedule(0, 100) == pytest.approx(1.0)
    assert k_schedule(99, 100) == pytest.approx(0.035, abs=1e-3)                # about 0.035 after 100 epochs
    ks = [k_schedule(e, 100) for e in range(100)]
    assert any(b > a for a, b in zip(ks, ks[1:]))                                # two oscillations raise it again
    assert k_schedule(25, 100, cyclical=False) == pytest.approx(0.75)


def test_the_kept_model_is_the_best_validation_epoch():
    fold, cardinalities = toy_fold(categorical_sizes=(3,))
    result = fit_fold(fold, small_config(epochs=8), cardinalities, random_seed=1, device='cpu')
    kept = evaluate(result.model, result.val_loader, 0.0, GradMode.PURE)[0]
    assert kept == pytest.approx(max(result.val_accs), abs=0)
    assert len(result.losses) == 8


@pytest.mark.parametrize('patience, expected_epochs', [(3, 4), (0, 6)])
def test_patience_stops_after_epochs_without_improvement(patience, expected_epochs):
    fold, cardinalities = toy_fold()
    # A step size of zero freezes the model, so only epoch 0 sets a new best.
    result = fit_fold(fold, small_config(epochs=6, lr=0.0, patience=patience), cardinalities, random_seed=1, device='cpu')
    assert len(result.losses) == expected_epochs


def test_a_run_is_reproducible_and_depends_on_the_seed():
    fold, cardinalities = toy_fold(categorical_sizes=(3,))
    a, b, c = (fit_fold(fold, small_config(), cardinalities, random_seed=seed, device='cpu') for seed in (1, 1, 2))
    assert a.losses == b.losses and a.losses != c.losses
    for key, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[key])


ROUTES = [  # bags, parallel_bags, categorical sizes, forced limit, expected route
    (1, False, (3,), None, 'single'),
    (3, False, (3,), None, 'sequential'),            # the user asked for one after another
    (2, True, (3,), None, 'sequential'),             # two models: nothing to train in parallel after the first
    (3, True, (3,), 0, 'sequential'),                # the step is too large for the parallel route
    (3, True, (3,), None, 'staged'),                 # learned partitions: the first model alone, then the others as blocks
    (3, True, (), None, 'staged'),
]


@pytest.mark.parametrize('bags, parallel, sizes, limit, route', ROUTES)
def test_routes_of_the_averaged_models(bags, parallel, sizes, limit, route):
    fold, cardinalities = toy_fold(categorical_sizes=sizes)
    options = {} if limit is None else {'max_parallel_elements': limit}
    result = fit_fold(fold, small_config(bags=bags, parallel_bags=parallel), cardinalities, random_seed=1, device='cpu', **options)
    assert result.route == route
    model = result.model
    if route == 'single':
        assert isinstance(model, FLAREModel) and model.nrules == 8
    elif route == 'sequential':
        assert isinstance(model, BagEnsemble) and len(model.members) == bags and model.nrules == 8 * bags
        first, *later = model.members
        for member in later:                     # one partition for the union rule base: the first model's, frozen
            assert not member.inputs.partition.deltas.requires_grad
            assert torch.equal(member.inputs.partition.deltas, first.inputs.partition.deltas)
    else:
        assert isinstance(model, FLAREModel) and model.nrules == 8 * bags and model.head.joint_bags == bags
        assert not model.inputs.partition.deltas.requires_grad
    row = score_fold(result)
    assert 0 <= row['f1'] <= 1 and row['n_rules'] <= model.nrules
    scores = model(result.test_loader.dataset.X)[0]
    assert scores.shape == (len(result.test_loader.dataset), 2) and torch.isfinite(scores).all()


def test_categorical_only_data_trains_the_averaged_models_at_once():
    fold, cardinalities = toy_fold(n_real=0, categorical_sizes=(3, 4, 3), rows=160)
    result = fit_fold(fold, small_config(bags=3, parallel_bags=True), cardinalities, random_seed=1, device='cpu')
    assert result.route == 'parallel' and result.model.head.joint_bags == 3 and result.model.inputs is None
    assert score_fold(result)['n_rules'] > 0


def test_an_averaged_model_is_the_mean_of_its_blocks_with_the_points_shared_out():
    """The head is linear: the average of the models is the model of their union with points divided by the models."""
    fold, cardinalities = toy_fold(categorical_sizes=(3,))
    result = fit_fold(fold, small_config(bags=3, parallel_bags=False), cardinalities, random_seed=1, device='cpu')
    X = result.test_loader.dataset.X
    result.model.set_inference_mode(GradMode.PURE)
    union = result.model(X)[0]
    parts = []
    for member in result.model.members:
        member.set_inference_mode(GradMode.PURE)
        parts.append(member(X)[0])
    torch.testing.assert_close(union, sum(parts) / 3)


def test_category_spaces_are_learned_by_the_first_model_and_frozen_in_the_others():
    fold, cardinalities = toy_fold(categorical_sizes=(40, 3), category_space=10, rows=260)
    assert list(fold.category_space) == [3]
    config = small_config(bags=3, parallel_bags=True, category_space=4, category_space_min=10)
    result = fit_fold(fold, config, cardinalities, random_seed=1, device='cpu')
    assert result.route == 'staged'
    spaces = result.model.inputs.category_spaces
    assert len(spaces) == 1 and not any(p.requires_grad for p in spaces.parameters())
    assert rules_text(result.model, feat_names=list('abcde'))


def test_the_gumbel_estimator_trains_with_finite_losses():
    fold, cardinalities = toy_fold(categorical_sizes=(3,))
    result = fit_fold(fold, small_config(grad_mode='concrete', epochs=3), cardinalities, random_seed=1, device='cpu')
    assert np.isfinite(result.losses).all() and len(result.val_accs) == 3
    assert result.model.temp == pytest.approx(0.01 ** (2 / 3))       # annealed every epoch and not restored afterwards


def test_scores_include_recalibrated_probabilities_on_request():
    fold, cardinalities = toy_fold(categorical_sizes=(3,))
    row = score_fold(fit_fold(fold, small_config(calibrate=True), cardinalities, random_seed=1, device='cpu'))
    assert {'test_roc_auc', 'test_logloss', 'test_roc_auc_cal', 'test_logloss_cal', 'cal_temperature'} <= set(row)
    assert row['cal_temperature'] >= 1.0


def test_a_wide_step_trains_the_averaged_models_one_after_another_unless_the_limit_is_raised():
    """The step of the parallel routes holds batch x rules x models x (conditions + slots x 2 per column) elements; above
    40 million (bagging.PARALLEL_BAGS_MAX_ELEMENTS) the models train one at a time."""
    fold, cardinalities = toy_fold(rows=140, n_real=500)
    config = small_config(bags=3, parallel_bags=True, rules=60, single_condition_rules=45, labels=5, epochs=1, label_runs='shoulders')
    # 32 rows x 1.3 (pooled with the validation rows) x 60 rules x 3 models x (500 x 11 + 500 x 4) = 56M, and the first model is
    # trained alone, so the staged route needs 2/3 of it (37M) under the limit; with every run of labels (14) it is 45M.
    assert fit_fold(fold, config.replace(label_runs='all'), cardinalities, random_seed=1, device='cpu').route == 'sequential'
    assert fit_fold(fold, config, cardinalities, random_seed=1, device='cpu').route == 'staged'
