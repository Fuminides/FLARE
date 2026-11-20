import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from flare.data import BatchedDataset, batched_loader, fold_batch_size, fold_loaders
from helpers import toy_fold


class RowDataset(Dataset):
    """The default way of serving rows: one by one."""

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def test_batched_loader_yields_the_same_batches_and_rng_stream_as_the_default_loader():
    X = torch.arange(70, dtype=torch.float32).view(35, 2)
    y = torch.arange(35) % 3
    torch.manual_seed(7)
    reference = [(a.clone(), b.clone()) for _ in range(2) for a, b in DataLoader(RowDataset(X, y), batch_size=8, shuffle=True)]
    reference_rng = torch.get_rng_state()
    torch.manual_seed(7)
    batched = [(a, b) for _ in range(2) for a, b in batched_loader(BatchedDataset(X, y), 8, shuffle=True)]
    assert len(batched) == len(reference) == 10
    for (a, b), (ra, rb) in zip(batched, reference):
        assert torch.equal(a, ra) and torch.equal(b, rb)
    assert torch.equal(torch.get_rng_state(), reference_rng)


def test_a_pass_over_a_shuffling_loader_advances_the_global_generator_even_if_unused():
    """This is why training iterates the validation loader once per epoch (train.py): the draws shape later epochs."""
    loader = batched_loader(BatchedDataset(torch.zeros(10, 2), torch.zeros(10, dtype=torch.long)), 4, shuffle=True)
    torch.manual_seed(0)
    before = torch.get_rng_state()
    for _ in loader:
        pass
    assert not torch.equal(before, torch.get_rng_state())


def test_fold_loaders_hold_float32_features_and_shuffle_every_split():
    fold, _ = toy_fold()
    train, val, test = fold_loaders(fold, 16, 'cpu')
    assert train.dataset.X.dtype == torch.float32 and train.dataset.y.dtype == torch.long
    assert [len(d.dataset) for d in (train, val, test)] == [100, 30, 30]
    assert all(type(loader.sampler).__name__ == 'RandomSampler' for loader in (train, val, test))


def test_batch_grows_with_the_data_to_about_fifty_steps_per_epoch():
    assert fold_batch_size(768 * 4 // 5, 32, 50) == 32                     # a fiftieth of 80% of the rows is below 32
    assert fold_batch_size(40000, 32, 50) == int(np.ceil(0.8 * 40000 / 50))
    assert fold_batch_size(40000, 32, 0) == 32
