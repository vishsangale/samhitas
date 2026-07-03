"""Character-level language modeling on tiny-shakespeare -- the "tiny LM for loss-vs-compute
trend measurement" task named in docs/methodology.md, and the setting docs/threads/
06-mup-hparam-transfer.md's real run requires (a non-algorithmic task, to kill the
modular-arithmetic grokking-dynamics confound the three CPU smoke tests were suspected to
be measuring instead of a genuine width effect).

Corpus: `data/tinyshakespeare.txt` -- the canonical Karpathy char-rnn / nanoGPT corpus
(1,115,394 chars, 65-char ASCII vocab; the concatenated works are public-domain Shakespeare).
It is committed into the repo verbatim rather than downloaded at run time so the task is
fully reproducible on the user's GPU machine with no network dependency and no risk of the
source text changing under the experiment.

Layout mirrors modular_arith.py / recall.py: plain functions, no framework magic. The task
is next-char prediction from a fixed context window, identical for both of thread 6's arms
(baseline MLP and thread-1 recurrence): given `context_len` consecutive characters, predict
the single character that immediately follows. Cross-entropy at init is log(65) ~= 4.17 nats.
"""

import functools
from pathlib import Path

import torch

_DATA_PATH = Path(__file__).resolve().parent / "data" / "tinyshakespeare.txt"


@functools.lru_cache(maxsize=1)
def _corpus():
    """Loads and encodes the corpus once. Returns (data, chars, stoi):
    data is a 1-D long tensor of character indices; chars is the sorted vocab; stoi maps
    char -> index. Cached so repeated batching doesn't re-read/re-encode the 1.1MB file."""
    text = _DATA_PATH.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, chars, stoi


def vocab_size() -> int:
    _, chars, _ = _corpus()
    return len(chars)


def make_split(train_frac: float = 0.9):
    """Contiguous split: the first `train_frac` of the corpus is train, the remainder test.

    Contiguous (not random-window) specifically so train and test share no character
    positions -- a random-window split would leak windows that straddle or overlap the
    boundary, the same train/test-leakage failure mode that bit the modular-arith task's
    first version (see docs/threads/06-mup-hparam-transfer.md's post-hoc note). Returns
    (train_data, test_data) as 1-D long tensors.
    """
    data, _, _ = _corpus()
    n_train = int(len(data) * train_frac)
    return data[:n_train], data[n_train:]


def make_batch(split_data: torch.Tensor, context_len: int, batch_size: int,
               generator: torch.Generator):
    """Samples `batch_size` fixed-context windows from a single split's data.

    For each sampled start index i: input is split_data[i : i+context_len], target is
    split_data[i+context_len] (the next character). Windows are drawn only from within the
    passed split, so a train batch never reaches into the test region and vice versa.

    Returns (contexts, targets): contexts is (batch_size, context_len) long, targets is
    (batch_size,) long.
    """
    max_start = split_data.shape[0] - context_len - 1
    assert max_start >= 0, "split too short for this context_len"
    start = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    offsets = torch.arange(context_len)
    contexts = split_data[start[:, None] + offsets[None, :]]
    targets = split_data[start + context_len]
    return contexts, targets
