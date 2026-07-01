"""Modular addition task: (a, b) -> (a + b) mod p, one-hot inputs, p-way classification.

Cheap, exact ground truth, no real dataset needed -- the diagnostic task named in
docs/methodology.md for fast depth/width probes. Used here as the sanity-check task for
thread 6 (muP-style hyperparameter transfer): we only care whether the LR-sweep minimum
is stable across width, not whether the task itself is hard.
"""

import torch


def make_dataset(p: int, seed: int, n_train: int, n_test: int):
    """Returns (x_train, y_train, x_test, y_test) as one-hot float tensors / long labels."""
    g = torch.Generator().manual_seed(seed)
    n_total = n_train + n_test
    all_pairs = torch.randint(0, p, (n_total, 2), generator=g)
    a, b = all_pairs[:, 0], all_pairs[:, 1]
    labels = (a + b) % p

    x = torch.zeros(n_total, 2 * p)
    x[torch.arange(n_total), a] = 1.0
    x[torch.arange(n_total), p + b] = 1.0

    perm = torch.randperm(n_total, generator=g)
    x, labels = x[perm], labels[perm]
    return x[:n_train], labels[:n_train], x[n_train:], labels[n_train:]


def input_dim(p: int) -> int:
    return 2 * p


def num_classes(p: int) -> int:
    return p
