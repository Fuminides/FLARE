import os

# The reference runs of tests/test_equivalence.py use two threads: the order of floating-point sums inside a threaded
# reduction depends on their number.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import pytest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: reruns whole training runs (minutes); deselect with -m "not slow"')


@pytest.fixture(autouse=True)
def _cpu_threads():
    import torch
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
