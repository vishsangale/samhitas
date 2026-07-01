"""Independent read/write gates on top of thread 1's spectrally-constrained recurrence
core, built to test thread 11's falsifiable prediction
(docs/threads/11-dual-gate-spectral-recurrence.md): does splitting thread 9's single shared
gate into independent retain/write gates (LSTM-style, instead of GRU-style interpolation)
resolve the depth-8 recall failure both thread 9 (direct training) and thread 10
(curriculum) found?

Reuses `experiments.models.linear_recurrence._A()`'s exact `orthogonal` construction
unmodified, same as thread 9's `GatedLinearRecurrentBlock`.
"""

import torch
import torch.nn as nn

from experiments.models.linear_recurrence import LinearRecurrentBlock


class DualGateLinearRecurrentBlock(nn.Module):
    """h_t = f_t * (A h_{t-1}) + w_t * (B x_t), elementwise on the hidden dimension, with
    f_t = sigmoid(W_f x_t + b_f) and w_t = sigmoid(W_w x_t + b_w) independent (no
    f_t + w_t = 1 constraint, unlike thread 9's single shared gate). Both depend only on
    x_t, never on h_{t-1}, so the LTV structure (linear in h for a fixed input sequence)
    still holds.

    Bias init: b_f=+4 (retain gate starts ~0.982, almost fully open/retaining) and
    b_w=-4 (write gate starts ~0.018, almost fully closed) -- together the model starts at
    init behaving almost exactly like ungated orthogonal (mostly retain, rarely write),
    the same "start close to non-selective baseline" rationale as thread 9's single-gate
    bias init, applied to two independent gates with complementary target values instead
    of one shared value.
    """

    F_BIAS_INIT = 4.0
    W_BIAS_INIT = -4.0

    def __init__(self, hidden: int, input_dim: int, eps: float):
        super().__init__()
        self.core = LinearRecurrentBlock(hidden, input_dim, "orthogonal", eps)
        self.forget_gate = nn.Linear(input_dim, hidden)
        self.write_gate = nn.Linear(input_dim, hidden)
        nn.init.constant_(self.forget_gate.bias, self.F_BIAS_INIT)
        nn.init.constant_(self.write_gate.bias, self.W_BIAS_INIT)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, seq_len, hidden) full state sequence."""
        A = self.core._A()
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden, device=x.device)
        bx = self.core.B(x)
        f = torch.sigmoid(self.forget_gate(x))  # (batch, seq_len, hidden)
        w = torch.sigmoid(self.write_gate(x))
        states = []
        for t in range(seq_len):
            h = f[:, t, :] * (h @ A.transpose(-1, -2)) + w[:, t, :] * bx[:, t, :]
            states.append(h)
        return torch.stack(states, dim=1)


class DualGateRecallModel(nn.Module):
    """Embedding -> DualGateLinearRecurrentBlock -> readout at the final (query) position.
    Mirrors thread 9's `GatedRecallModel` shape exactly (only the recurrence block
    differs) for a fair, matched-architecture comparison."""

    def __init__(self, vocab_size: int, hidden: int, eps: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.recur = DualGateLinearRecurrentBlock(hidden, hidden, eps)
        self.readout = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        states = self.recur(x)
        return self.readout(states[:, -1, :])
