"""Modular addition task: (a, b) -> (a + b) mod p, one-hot inputs, p-way classification.

Cheap, exact ground truth, no real dataset needed -- the diagnostic task named in
docs/methodology.md for fast depth/width probes. Used here as the sanity-check task for
thread 6 (muP-style hyperparameter transfer): we only care whether the LR-sweep minimum
is stable across width, not whether the task itself is hard.
"""

import torch


def make_dataset(p: int, seed: int, n_train: int, n_test: int):
    """Returns (x_train, y_train, x_test, y_test) as one-hot float tensors / long labels.

    Splits by enumerating the p*p unique (a, b) pairs and slicing a random permutation of
    them disjointly, so train and test never share a pair. (A previous version drew
    n_train+n_test pairs *with replacement* from p*p possibilities and then split, which
    for small p meant most "test" pairs also appeared in train -- caught in review, see
    docs/threads/06-mup-hparam-transfer.md's post-hoc note.)
    """
    assert n_train + n_test <= p * p, (
        f"requested {n_train + n_test} examples but p={p} only has {p * p} unique pairs"
    )
    g = torch.Generator().manual_seed(seed)
    a_all, b_all = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a_all, b_all = a_all.reshape(-1), b_all.reshape(-1)
    perm = torch.randperm(p * p, generator=g)
    a_all, b_all = a_all[perm], b_all[perm]

    n_total = n_train + n_test
    a, b = a_all[:n_total], b_all[:n_total]
    labels = (a + b) % p

    x = torch.zeros(n_total, 2 * p)
    x[torch.arange(n_total), a] = 1.0
    x[torch.arange(n_total), p + b] = 1.0

    return x[:n_train], labels[:n_train], x[n_train:], labels[n_train:]


def input_dim(p: int) -> int:
    return 2 * p


def num_classes(p: int) -> int:
    return p
