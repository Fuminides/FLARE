"""Learned category space (flare/category_space.py)."""
import numpy as np
import pytest
import torch

from flare.category_space import CategorySpace
from flare.inputs import TabularInputs
from flare.model import FLAREModel
from flare.partitions import quantile_partitions
from flare.selectors import GradMode
from flare.train import fit_fold
from helpers import small_config, toy_fold


def test_loss_terms_can_be_removed_independently_without_changing_initialisation():
    def make(rec, radius):
        torch.manual_seed(33)
        return CategorySpace([8, 4, 1], 2, dim=3, seed=33, reconstruction_weight=rec, radius_weight=radius)
    full, reconstruction, radius = make(.1, 1), make(.1, 0), make(0, 1)
    for name, value in full.state_dict().items():
        assert torch.equal(value, reconstruction.state_dict()[name])
        assert torch.equal(value, radius.state_dict()[name])
    losses, grads = [], []
    for model in (full, reconstruction, radius):
        torch.manual_seed(42)
        loss = model.loss()
        loss.backward()
        losses.append(loss.detach())
        grads.append(model.points.grad)
    torch.testing.assert_close(losses[0], losses[1] + losses[2])
    torch.testing.assert_close(grads[0], grads[1] + grads[2])


def space(counts=(500, 300, 100, 20, 5, 2, 1, 1), regions=3, dim=4):
    torch.manual_seed(0)
    return CategorySpace(np.array(counts), n_regions=regions, dim=dim, seed=0).warm_up(steps=400)


def test_regions_have_one_membership_per_region_and_unseen_categories_use_the_extra_point():
    s = space()
    m = s(torch.tensor([0.0, 3.0, 7.0, 99.0, -1.0]))
    assert m.shape == (5, 3) and torch.all((m >= 0) & (m <= 1))
    assert torch.equal(s.index(torch.tensor([99.0, -1.0, 2.0])), torch.tensor([8, 8, 2]))


def test_warm_up_puts_rare_categories_further_from_the_centre():
    s = space()
    radius = s.points[:-1].norm(dim=1).detach().numpy()
    assert radius[0] < radius[3] < radius[-1]
    assert s.target_radius[0] < s.target_radius[-2]


def test_printed_groups_are_exactly_the_categories_at_least_half_true():
    s = space()
    m = s.regions(torch.arange(s.n_categories)).detach().numpy()
    for k, held in enumerate(s.groups()):
        assert set(held.tolist()) == set(np.flatnonzero(m[:, k] >= 0.5).tolist())
        assert list(held) == sorted(held, key=lambda c: -s.counts[c])          # most frequent first


def test_the_first_axis_starts_at_the_effects_and_keeps_it_through_the_warm_up():
    counts = np.array([400, 300, 100, 50, 10, 5])
    effects = np.array([0.9, 0.1, 0.5, 0.7, 0.3, 0.5])
    s = CategorySpace(counts, n_regions=3, dim=4, seed=0, effects=effects, reconstruction_weight=0.1).warm_up(steps=200)
    axis = s.points[:-1, 0].detach().numpy()
    w = counts / counts.sum()
    z = (effects - (w * effects).sum()) / np.sqrt((w * (effects - (w * effects).sum()) ** 2).sum())
    assert np.allclose(axis, z, atol=1e-6) and s.points[-1, 0].item() == 0.0
    assert np.argsort(axis, kind='stable').tolist() == np.argsort(effects, kind='stable').tolist()


def test_a_model_with_a_category_space_runs_prints_the_groups_and_its_loss_has_a_gradient():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.integers(0, 40, 200), rng.normal(size=200)]).astype(np.float32)
    torch.manual_seed(0)
    codes = np.argsort(-np.bincount(X[:, 0].astype(int), minlength=40), kind='stable')
    space_ = CategorySpace(np.bincount(X[:, 0].astype(int), minlength=40)[codes] + 1, n_regions=3, dim=4).warm_up(steps=50)
    inputs = TabularInputs(quantile_partitions(X, 3), 'shoulders', {0: space_}, {0: codes}, 'cpu')
    model = FLAREModel(inputs, 2, np.array([False, False]), [], 6, 2, 1.0, GradMode.GRAFT)
    out = model(torch.as_tensor(X))[0]
    (out.sum() + inputs.space_loss('cpu')).backward()
    assert out.shape == (200, 2) and space_.points.grad is not None and space_.centers.grad is not None
    assert set(p for p in inputs.space_parameters()) == set(space_.parameters())
    text = inputs.condition_text(0, 0, 'city', [f'town{i}' for i in range(40)], shown=2)
    assert text.startswith('city IN {town') and 'categories)' in text
    inputs.category_spaces.requires_grad_(False)                 # frozen spaces (a later bag) contribute no loss and no step
    assert inputs.space_parameters() == [] and float(inputs.space_loss('cpu')) == 0.0


def test_the_warm_up_leaves_no_gradient_behind():
    s = CategorySpace(np.array([400, 300, 100, 50, 10, 5]), n_regions=3, dim=4, seed=0).warm_up(steps=20)
    assert all(p.grad is None for p in s.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need a GPU')
def test_the_later_models_of_a_bag_keep_the_space_of_the_first_one_through_their_training():
    fold, cardinalities = toy_fold(categorical_sizes=(40,), category_space=32)
    config = small_config(bags=3, parallel_bags=False, cuda_graph=True, category_space=4, category_space_min=32)
    members = fit_fold(fold, config, cardinalities, random_seed=1, device='cuda').model.members
    first = members[0].inputs.category_spaces.state_dict()
    assert len(members) == 3 and first
    for member in members[1:]:
        for key, value in first.items():
            assert torch.equal(value, member.inputs.category_spaces.state_dict()[key]), key
