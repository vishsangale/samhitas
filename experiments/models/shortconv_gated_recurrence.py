"""Arm (b) of thread 17 (docs/threads/17-recall-mechanism-ladder.md): a depthwise causal
1D convolution inserted before thread 9's existing, unmodified `GatedLinearRecurrentBlock`,
giving each timestep's gate/write decision direct access to a short window of
immediately-preceding tokens -- the "shift" primitive Zoology's literature identifies as
one of three missing primitives for multi-pair recall in single-gated-recurrence models.

Reuses `experiments.models.gated_linear_recurrence.GatedLinearRecurrentBlock` unmodified --
this arm is only about the effect of adding a short conv in front of it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.models.gated_linear_recurrence import GatedLinearRecurrentBlock


class ShortConvGatedBlock(nn.Module):
    """Depthwise causal conv (kernel_size k) over the input sequence, then thread 9's
    gated recurrence on the convolved representation.

    Causal via symmetric padding (k-1 on both sides) followed by truncation to the
    original sequence length -- output position j depends only on input positions
    [max(0, j-(k-1)), j], never on future positions (standard causal-conv construction,
    equivalent to left-padding only, used here because it's a one-line change to a
    stock nn.Conv1d call rather than a custom padding function)."""

    def __init__(self, hidden: int, input_dim: int, eps: float, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size,
                               padding=kernel_size - 1, groups=input_dim, bias=True)
        self.gated = GatedLinearRecurrentBlock(hidden, input_dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, seq_len, hidden)."""
        seq_len = x.shape[1]
        xt = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        conv_out = self.conv(xt)[:, :, :seq_len]  # causal truncation
        conv_out = F.gelu(conv_out).transpose(1, 2)  # (batch, seq_len, input_dim)
        return self.gated(conv_out)


class ShortConvGatedRecallModel(nn.Module):
    """Embedding -> ShortConvGatedBlock -> readout at the final (query) position.
    Exposes .embed / .recur / .readout so experiments/harness/gradient_flow.py's
    gradient_norm_ratio() works unchanged, matching every other recall model in this repo."""

    def __init__(self, vocab_size: int, hidden: int, eps: float, kernel_size: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.recur = ShortConvGatedBlock(hidden, hidden, eps, kernel_size)
        self.readout = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        states = self.recur(x)
        return self.readout(states[:, -1, :])
