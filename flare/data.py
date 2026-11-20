"""PyTorch batches of a preprocessed fold (see preprocessing.py for how a `Fold` is built).

The three loaders of a fold (training, validation, test) are plain `torch.utils.data.DataLoader`s over tensors that
already live on the device, with one indexing call per batch. All three shuffle, validation and test included, even
though their order does not change any score. The reason is the random generator: every pass over a shuffling loader
draws two numbers from the global torch generator, so the batches of all later training epochs depend on how many
passes came before (`train.py` makes one dummy pass over the validation loader per epoch, and one pass per
evaluation). Changing this changes the random draws and therefore the results.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BatchedDataset(Dataset):
    """Rows `X` with labels `y`, served a whole batch at once (`__getitems__`) instead of one row at a time, which on a
    GPU would cost two tiny kernel calls per row."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __getitems__(self, indices):
        index = torch.as_tensor(indices, device=self.X.device)
        return self.X[index], self.y[index]


def _whole_batch(batch):
    return batch


def batched_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False, collate_fn=_whole_batch)


def to_dataset(X, y, device):
    """float32 features and int64 labels on the device."""
    return BatchedDataset(torch.tensor(np.asarray(X), dtype=torch.float32).to(device=device),
                          torch.tensor(np.asarray(y), dtype=torch.long).to(device=device))


def fold_loaders(fold, batch_size, device, fixed_order_evaluation=False):
    """(train, validation, test) loaders of a preprocessing.Fold, all shuffling (see the module docstring).

    fixed_order_evaluation=True: the validation and test loaders keep their order, in batches of 512 rows (the training
    rows still shuffle). Scores do not change; the later training batches do, because these loaders then draw nothing
    from the generator."""
    datasets = [to_dataset(X, y, device) for X, y in ((fold.X_train, fold.y_train), (fold.X_val, fold.y_val), (fold.X_test, fold.y_test))]
    if fixed_order_evaluation:
        return (batched_loader(datasets[0], batch_size, True), batched_loader(datasets[1], 512, False),
                batched_loader(datasets[2], 512, False))
    return tuple(batched_loader(d, batch_size, shuffle=True) for d in datasets)


def fold_batch_size(n_train_fold, batch_size, steps_per_epoch):
    """The batch grows with the data so that an epoch has about `steps_per_epoch` updates: 32 rows, or a fiftieth of the
    80% of the training fold that is trained on, if larger. `n_train_fold` is the training fold before the validation
    split."""
    if steps_per_epoch > 0:
        return max(batch_size, int(np.ceil(0.8 * n_train_fold / steps_per_epoch)))
    return batch_size
