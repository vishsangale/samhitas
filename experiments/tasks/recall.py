"""Associative recall: n (key, value) pairs, then a repeated key as a query; the model
must output the value that was paired with that key earlier in the sequence. Requires
content-based routing over the full sequence -- the diagnostic task named in
docs/methodology.md, used here to test thread 1's structured-recurrence prediction.
"""

import torch


def make_batch(n_pairs: int, vocab_size: int, batch_size: int, seed: int):
    """Returns (tokens, targets). Sequence layout: key_1 val_1 key_2 val_2 ... key_n val_n
    query, where query repeats one of the n keys and the target is its paired value.
    seq_len is always 2*n_pairs + 1 -- see seq_len_for()."""
    g = torch.Generator().manual_seed(seed)
    keys = torch.randint(0, vocab_size, (batch_size, n_pairs), generator=g)
    values = torch.randint(0, vocab_size, (batch_size, n_pairs), generator=g)
    query_idx = torch.randint(0, n_pairs, (batch_size,), generator=g)

    seq_len = seq_len_for(n_pairs)
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tokens[:, 0:2 * n_pairs:2] = keys
    tokens[:, 1:2 * n_pairs:2] = values
    tokens[:, -1] = keys[torch.arange(batch_size), query_idx]
    targets = values[torch.arange(batch_size), query_idx]
    return tokens, targets


def seq_len_for(n_pairs: int) -> int:
    return 2 * n_pairs + 1
