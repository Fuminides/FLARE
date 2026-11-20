"""Averaging several models to reduce variance.

FLARE trains B models, each on its own split of the training rows into training and validation rows, and averages them.
Because the head is linear in the points, the average is again a FLARE model: the union of the rules of the models,
with the points divided by B. The first model learns the partitions and the category spaces and the other B - 1 read
them frozen. There are three ways of training them, chosen by `train.fit_fold`:

* sequential  one model after the other; the average is a `model.BagEnsemble` (used when a step of the parallel
              routes would not fit, see `PARALLEL_BAGS_MAX_ELEMENTS`, or when there are only two models);
* parallel    when nothing is learned in common (no real variable, no category space) all B models train at once as the
              blocks of ONE model (`joint_bags = B`): B independent rule bases whose outputs are averaged by the head;
* staged      the first model alone, then the other B - 1 as parallel blocks reading its frozen partitions, and finally
              the blocks are joined (`join_blocks`) in one model of B blocks.

A block is an independent model: it has its own training rows and validation rows (packed next to the label by
`parallel_bag_loaders`), its own loss (`parallel_bag_loss`), its own gradient clipping (`clip_grad_norm_blocks`) and its
own best epoch (`block_validation`, `copy_blocks`). So the union of blocks is exactly the union of separately trained
models, only faster, since a small step is bound by kernel launches and B blocks cost about one.
"""
import numpy as np
import torch

from .data import BatchedDataset, batched_loader
from .metrics import eval_chunk_rows
from .selectors import GradMode

# Above this many elements in one step (see fit_fold), the models train one at a time: the step is then bound by
# arithmetic and memory, and the parallel blocks stop being faster.
PARALLEL_BAGS_MAX_ELEMENTS = 40e6


def parallel_bag_loaders(train_loader, val_loader, bags, seed, batch_size, first_bag=0):
    """Parallel union bagging: the pooled training and validation rows, with each block's split packed next to the label
    (column 0 the class, then one training and one validation membership bit per block). Bag 0 keeps the fold's split;
    bag b resplits the pooled rows (stratified, same validation size, random_state seed + 1000 b), as the sequential
    route does. The blocks are the bags first_bag, first_bag + 1, ... (the staged route trains bag 0 alone
    and the others here). The pooled batch is scaled so that every block gets about the batch size it would get alone,
    and the same number of steps per epoch."""
    from sklearn.model_selection import train_test_split
    X = torch.cat([train_loader.dataset.X, val_loader.dataset.X])
    y = torch.cat([train_loader.dataset.y, val_loader.dataset.y])
    n_train, n = len(train_loader.dataset.y), len(y)
    train_bits = torch.zeros(n, bags, dtype=torch.long, device=y.device)
    val_bits = torch.zeros(n, bags, dtype=torch.long, device=y.device)
    for block, bag in enumerate(range(first_bag, first_bag + bags)):
        if bag == 0:
            train_bits[:n_train, block], val_bits[n_train:, block] = 1, 1
            continue
        train_ix, val_ix = train_test_split(np.arange(n), test_size=n - n_train, random_state=seed + 1000 * bag, stratify=y.cpu().numpy())
        train_bits[torch.as_tensor(train_ix, device=y.device), block] = 1
        val_bits[torch.as_tensor(val_ix, device=y.device), block] = 1
    dataset = BatchedDataset(X, torch.cat([y.long().unsqueeze(1), train_bits, val_bits], dim=1))
    pooled_batch = max(1, int(round(batch_size * n / n_train)))
    return batched_loader(dataset, pooled_batch, shuffle=True), batched_loader(dataset, pooled_batch, shuffle=False)


def sequential_bag_loaders(train_loader, val_loader, bag, seed, batch_size):
    """The training and validation loaders of bag `bag` >= 1 of the sequential route: the pooled rows resplit with
    random_state seed + 1000 bag, with the size of the original validation split."""
    from sklearn.model_selection import train_test_split
    pooled_X = torch.cat([train_loader.dataset.X, val_loader.dataset.X])
    pooled_y = torch.cat([train_loader.dataset.y, val_loader.dataset.y])
    train_ix, val_ix = train_test_split(np.arange(len(pooled_y)), test_size=len(val_loader.dataset.y),
                                        random_state=seed + 1000 * bag, stratify=pooled_y.cpu().numpy())
    return [batched_loader(BatchedDataset(pooled_X[ix], pooled_y[ix]), batch_size, shuffle=True)
            for ix in (torch.as_tensor(train_ix, device=pooled_X.device), torch.as_tensor(val_ix, device=pooled_X.device))]


def parallel_bag_loss(per_bag, y, member, class_weights=None):
    """Sum over blocks of each block's weighted mean cross-entropy on its own training rows (member [B, K] in {0, 1}), so
    every block gets exactly the gradient it would get alone on those rows. per_bag: [B, K, C] block outputs."""
    blocks = per_bag.shape[1]
    element = torch.nn.functional.cross_entropy(per_bag.reshape(-1, per_bag.shape[2]), y.repeat_interleave(blocks),
                                                weight=class_weights, reduction='none').view(-1, blocks)
    norm = member * (class_weights[y].unsqueeze(1) if class_weights is not None else 1.0)
    return ((element * member).sum(dim=0) / norm.sum(dim=0).clamp(min=1e-12)).sum()


def block_parameters(model, blocks):
    """[(parameter, block dimension)] of a model trained as `blocks` independent rule blocks. Every trainable parameter
    must belong to exactly one block, otherwise the blocks would not be independent models."""
    pairs = list(model.rule_parameters()) + [(model.head.bias, 0)]
    covered = {id(p) for p, _ in pairs}
    shared = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in covered]
    if shared:
        raise ValueError(f'parallel bags need every trainable parameter split by block; shared: {shared}')
    return pairs


@torch.no_grad()
def clip_grad_norm_blocks(model, blocks, max_norm=1.0):
    """clip_grad_norm_ applied to every block on its own (the bagged models are clipped separately, as if alone)."""
    pairs = [(p.grad, dim) for p, dim in block_parameters(model, blocks) if p.grad is not None]
    squared = sum(g.movedim(dim, 0).reshape(blocks, -1).pow(2).sum(dim=1) for g, dim in pairs)
    coefficient = (max_norm / (squared.sqrt() + 1e-6)).clamp(max=1.0)
    for g, dim in pairs:
        shape = [1] * g.dim()
        shape[dim] = g.shape[dim]
        g.mul_(coefficient.repeat_interleave(g.shape[dim] // blocks).view(shape))


@torch.no_grad()
def block_validation(model, loader, blocks, metric='acc'):
    """[blocks] validation score of every block on its own validation rows (PURE forward, the block's output alone)."""
    model.set_inference_mode(GradMode.PURE, 0.0)
    model.eval()
    X = torch.cat([batch[0] for batch in loader])
    packed = torch.cat([batch[1] for batch in loader])
    predictions = []
    for chunk in X.split(eval_chunk_rows(model)):
        model(chunk)
        predictions.append(model.head.bag_outputs.argmax(dim=2))
    pred, y, val = torch.cat(predictions), packed[:, 0], packed[:, 1 + blocks:].bool()
    if metric != 'f1':
        return (((pred == y.unsqueeze(1)) & val).sum(dim=0).float() / val.sum(dim=0).clamp(min=1)).cpu().numpy()
    scores = []
    for b in range(blocks):                               # macro-F1 over the classes present, as metrics.evaluate computes it
        yb, pb = y[val[:, b]], pred[val[:, b], b]
        f1s = []
        for c in range(model.n_class):
            tp, fp, fn = float(((pb == c) & (yb == c)).sum()), float(((pb == c) & (yb != c)).sum()), float(((pb != c) & (yb == c)).sum())
            if tp + fp + fn > 0:
                precision, recall = tp / (tp + fp + 1e-6), tp / (tp + fn + 1e-6)
                f1s.append(2 * precision * recall / (precision + recall + 1e-6))
        scores.append(np.mean(f1s))
    return np.array(scores)


@torch.no_grad()
def copy_blocks(state, model, blocks, which):
    """Copy the listed blocks' parameters from the model into `state` (a state_dict), leaving the other blocks."""
    names = {id(p): name for name, p in model.named_parameters()}
    for p, dim in block_parameters(model, blocks):
        size = p.shape[dim] // blocks
        for b in which:
            state[names[id(p)]].narrow(dim, b * size, size).copy_(p.detach().narrow(dim, b * size, size))


@torch.no_grad()
def join_blocks(model, parts):
    """Fill `model`, a joint-bag model, with the rule blocks of `parts` in order: models of the same architecture that
    hold fewer blocks each (the staged route: the first model alone, then the others trained at once)."""
    for (target, dim), *sources in zip(model.rule_parameters(), *[part.rule_parameters() for part in parts]):
        target.copy_(torch.cat([p for p, _ in sources], dim=dim))
    bias = model.head.bias
    bias.copy_(torch.cat([part.head.bias.reshape(-1, bias.shape[-1]) for part in parts]))
