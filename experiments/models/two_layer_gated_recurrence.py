"""Arm (a) of thread 17 (docs/threads/17-recall-mechanism-ladder.md): two
`GatedLinearRecurrentBlock` instances (thread 9's exact, unmodified construction) stacked
sequentially -- the first block's full output-state sequence feeds as the input sequence to
the second. Tests whether composition alone (no new primitive, just depth-2 stacking of the
existing mechanism) is enough for one layer to detect key-matches and a second to act on
them, per the literature's 1-vs-2-layer separation for attention (open for this family).

Reuses `experiments.models.gated_linear_recurrence.GatedLinearRecurrentBlock` unmodified in
both layers.
"""

import torch.nn as nn

from experiments.models.gated_linear_recurrence import GatedLinearRecurrentBlock


class TwoLayerGatedBlock(nn.Module):
    """Two independent GatedLinearRecurrentBlock instances (separate parameters, separate
    orthogonal cores, separate gates), composed: block1(x) -> block2(block1(x))."""

    def __init__(self, hidden: int, input_dim: int, eps: float):
        super().__init__()
        self.block1 = GatedLinearRecurrentBlock(hidden, input_dim, eps)
        self.block2 = GatedLinearRecurrentBlock(hidden, hidden, eps)

    def forward(self, x):
        """x: (batch, seq_len, input_dim) -> (batch, seq_len, hidden)."""
        h1 = self.block1(x)
        h2 = self.block2(h1)
        return h2


class TwoLayerGatedRecallModel(nn.Module):
    """Embedding -> TwoLayerGatedBlock -> readout at the final (query) position. Exposes
    .embed / .recur / .readout so experiments/harness/gradient_flow.py's
    gradient_norm_ratio() works unchanged, matching every other recall model in this repo."""

    def __init__(self, vocab_size: int, hidden: int, eps: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.recur = TwoLayerGatedBlock(hidden, hidden, eps)
        self.readout = nn.Linear(hidden, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)
        states = self.recur(x)
        return self.readout(states[:, -1, :])
