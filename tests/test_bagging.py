"""Averaged models: the blocks of one joint model behave as separately trained models, each with its own split,
gradient, clipping and checkpoint."""
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from flare import bagging
from flare.data import BatchedDataset, batched_loader
from flare.selectors import GradMode
from test_model import mixed_model

RULES, BLOCKS = 5, 3


def joint_model(blocks, seed=0):
    model, _, _ = mixed_model(n_rules=RULES, n_slots=2, joint_bags=blocks, seed=seed)
    model.restrict_to_single_condition(3, bag_size=RULES if blocks > 1 else 0)
    model.inputs.partition.deltas.requires_grad_(False)      # the partition is read, frozen, by the blocks
    return model


def block_model(joint, b):
    """A separate model holding block b of the joint model."""
    single = joint_model(1, seed=1)
    with torch.no_grad():
        for (p, dim), (q, _) in zip(joint.rule_parameters(), single.rule_parameters()):
            q.copy_(p.narrow(dim, b * RULES, RULES))
        single.head.bias.copy_(joint.head.bias[b])
    return single


def data():
    x = mixed_model()[1].repeat(5, 1)
    y = torch.tensor(np.arange(40) % 2)
    member = (torch.rand(40, BLOCKS, generator=torch.Generator().manual_seed(3)) < 0.7).float()
    return x, y, member


def test_every_block_gets_the_gradient_it_would_get_alone():
    joint = joint_model(BLOCKS)
    x, y, member = data()
    weights = torch.tensor([0.7, 1.3])
    joint(x)
    bagging.parallel_bag_loss(joint.head.bag_outputs, y, member, weights).backward()
    for b in range(BLOCKS):
        single = block_model(joint, b)
        rows = member[:, b].bool()
        torch.nn.functional.cross_entropy(single(x[rows])[0], y[rows], weight=weights).backward()
        for (p, dim), (q, _) in zip(joint.rule_parameters(), single.rule_parameters()):
            assert torch.allclose(p.grad.narrow(dim, b * RULES, RULES), q.grad, atol=1e-6)
        assert torch.allclose(joint.head.bias.grad[b], single.head.bias.grad, atol=1e-6)


def test_blocks_are_clipped_separately():
    joint = joint_model(BLOCKS)
    for p, _ in bagging.block_parameters(joint, BLOCKS):
        p.grad = torch.randn_like(p) * 3
    singles = [block_model(joint, b) for b in range(BLOCKS)]
    for b, single in enumerate(singles):
        for (p, dim), (q, _) in zip(joint.rule_parameters(), single.rule_parameters()):
            q.grad = p.grad.narrow(dim, b * RULES, RULES).clone()
        single.head.bias.grad = joint.head.bias.grad[b].clone()
        torch.nn.utils.clip_grad_norm_([q for q, _ in bagging.block_parameters(single, 1)], max_norm=1.0)
    bagging.clip_grad_norm_blocks(joint, BLOCKS, max_norm=1.0)
    for b, single in enumerate(singles):
        for (p, dim), (q, _) in zip(joint.rule_parameters(), single.rule_parameters()):
            assert torch.allclose(p.grad.narrow(dim, b * RULES, RULES), q.grad, atol=1e-6)


def test_copy_blocks_touches_only_the_chosen_blocks():
    joint = joint_model(BLOCKS)
    state = {n: v.detach().clone() for n, v in joint.state_dict().items()}
    with torch.no_grad():
        for p, _ in bagging.block_parameters(joint, BLOCKS):
            p.add_(1.0)
    bagging.copy_blocks(state, joint, BLOCKS, [1])
    points = state['head.weights']
    assert torch.equal(points[RULES:2 * RULES], joint.head.weights[RULES:2 * RULES].detach())
    original = joint_model(BLOCKS).state_dict()                    # same seed: the values before the change
    assert torch.equal(points[:RULES], original['head.weights'][:RULES])
    assert torch.equal(points[2 * RULES:], original['head.weights'][2 * RULES:])
    assert torch.equal(state['head.bias'][1], joint.head.bias[1].detach())


def pooled_loaders():
    X = torch.rand(50, 3)
    y = torch.tensor(np.arange(50) % 2)
    return (batched_loader(BatchedDataset(X[:40], y[:40]), 8, shuffle=True), batched_loader(BatchedDataset(X[40:], y[40:]), 8, shuffle=False), y)


def test_loaders_pack_the_sequential_bags_splits():
    train, val, y = pooled_loaders()
    train_loader, _ = bagging.parallel_bag_loaders(train, val, 3, seed=33, batch_size=8)
    packed = train_loader.dataset.y
    assert torch.equal(packed[:, 0], y)
    assert packed[:40, 1].eq(1).all() and packed[40:, 4].eq(1).all()          # block 0: the fold's own split
    for b in range(1, 3):
        train_ix, val_ix = train_test_split(np.arange(50), test_size=10, random_state=33 + 1000 * b, stratify=y.numpy())
        assert packed[train_ix, 1 + b].eq(1).all() and packed[val_ix, 1 + b].eq(0).all()
        assert packed[val_ix, 4 + b].eq(1).all() and int(packed[:, 4 + b].sum()) == 10
    assert train_loader.batch_size == 10                                       # 8 rows of each block's 40 per batch
    loaders = bagging.sequential_bag_loaders(train, val, 1, 33, 8)             # the same split for the sequential route
    train_ix, val_ix = train_test_split(np.arange(50), test_size=10, random_state=33 + 1000, stratify=y.numpy())
    assert len(loaders[0].dataset) == 40 and len(loaders[1].dataset) == 10


def test_loaders_can_start_at_a_later_bag():
    """The staged route trains bag 0 alone and the others in parallel: the blocks are then the bags 1, 2, ..."""
    train, val, _ = pooled_loaders()
    all_bags = bagging.parallel_bag_loaders(train, val, 3, seed=33, batch_size=8)[0].dataset.y
    later = bagging.parallel_bag_loaders(train, val, 2, seed=33, batch_size=8, first_bag=1)[0].dataset.y
    assert torch.equal(later[:, 0], all_bags[:, 0])
    assert torch.equal(later[:, 1:3], all_bags[:, 2:4]) and torch.equal(later[:, 3:5], all_bags[:, 5:7])


def test_join_blocks_gives_the_joint_model_of_the_parts():
    """The first bag (one block) and the other bags (one joint model) joined are the joint model of all the bags."""
    joint = joint_model(BLOCKS)
    with torch.no_grad():
        joint.head.bias.copy_(torch.randn_like(joint.head.bias))
    first, rest = block_model(joint, 0), joint_model(BLOCKS - 1, seed=2)
    with torch.no_grad():
        for (p, dim), (q, _) in zip(joint.rule_parameters(), rest.rule_parameters()):
            q.copy_(p.narrow(dim, RULES, RULES * (BLOCKS - 1)))
        rest.head.bias.copy_(joint.head.bias[1:])
    joined = joint_model(BLOCKS, seed=5)
    bagging.join_blocks(joined, [first, rest])
    x = data()[0]
    for model in (joint, joined, first, rest):
        model.set_inference_mode(GradMode.PURE, 0.0)
    got = joined(x)[0]
    assert torch.allclose(got, joint(x)[0], atol=1e-6)
    assert torch.allclose(got, (first(x)[0] + (BLOCKS - 1) * rest(x)[0]) / BLOCKS, atol=1e-5)


def test_block_validation_scores_each_block_on_its_own_validation_rows():
    joint = joint_model(BLOCKS)
    x, y, _ = data()
    val = (torch.rand(40, BLOCKS, generator=torch.Generator().manual_seed(4)) < 0.4).long()
    packed = torch.cat([y.unsqueeze(1), torch.zeros(40, BLOCKS, dtype=torch.long), val], dim=1)
    loader = batched_loader(BatchedDataset(x, packed), 40, shuffle=False)
    scores = bagging.block_validation(joint, loader, BLOCKS, 'acc')
    joint.set_inference_mode(GradMode.PURE, 0.0)
    for b in range(BLOCKS):
        rows = val[:, b].bool()
        single = block_model(joint, b)
        single.set_inference_mode(GradMode.PURE, 0.0)
        assert scores[b] == float((single(x[rows])[0].argmax(dim=1) == y[rows]).float().mean())
