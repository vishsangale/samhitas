"""Minimal input-dependent gate on top of thread 1's spectrally-constrained recurrence
core, built to test thread 9's two falsifiable predictions
(docs/threads/09-gated-spectral-recurrence.md): does a Mamba-style retention gate make
associative recall solvable, and does it preserve thread 1's predictable-training-range
property (within a stated 2x tolerance)?

Reuses `experiments.models.linear_recurrence._A()`'s exact `orthogonal` construction
unmodified -- this thread is only about the effect of adding the gate, not about
re-litigating thread 1's own cross-parameterization result.
"""

import torch
import torch.nn as nn

from experiments.models.linear_recurrence import LinearRecurrentBlock


class GatedLinearRecurrentBlock(nn.Module):
    """h_t = (1 - g_t) * (A h_{t-1}) + g_t * (B x_t), elementwise on the hidden dimension,
    g_t = sigmoid(W_g x_t). g_t depends only on the current input token, never on h_{t-1},
    so for a fixed input sequence this is still a linear (time-varying) map from h_{t-1} to
    h_t -- see the thread doc's "architectural hypothesis" section for why that matters for
    the gradient-flow analysis carrying over from thread 1.

    Only "orthogonal" mode is supported here (see thread doc's "explicitly out of scope"
    section -- diag_lowrank is thread 1's own cross-parameterization question, not this
    thread's).

    Gate bias init: caught during the pre-implementation smoke test (before any numbers
    were reported) that `nn.Linear`'s default init makes `sigmoid(W_g x_t)` average ~0.5
    with essentially no dependence on `eps` -- at init this would make the gate's own
    random init dominate the decay rate rather than the nominal spectral bound, which
    would make thread 09's prediction B fail near-trivially regardless of the actual
    mechanism. Fixed with a negative bias (-4, sigmoid(-4)~=0.018) so the gate starts
    mostly *closed* (h_t ~= A h_{t-1} at init, matching the ungated control almost
    exactly) and must learn to open where the task needs content-based writing -- the
    same "start close to a non-selective baseline, let training turn on selection"
    convention Mamba's own selection-parameter init uses.
    """

    GATE_BIAS_INIT = -4.0

    def __init__(self, hidden: int, input_dim: int, eps: float):
        super().__init__()
        self.core = LinearRecurrentBlock(hidden, input_dim, "orthogonal", eps)
        self.gate = nn.Linear(input_dim, hidden)
        nn.init.constant_(self.gate.bias, self.GATE_BIAS_INIT)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, seq_len, hidden) full state sequence."""
        A = self.core._A()
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden, device=x.device)
        bx = self.core.B(x)
        g = torch.sigmoid(self.gate(x))  # (batch, seq_len, hidden)
        states = []
        for t in range(seq_len):
            g_t = g[:, t, :]
            h = (1 - g_t) * (h @ A.transpose(-1, -2)) + g_t * bx[:, t, :]
            states.append(h)
        return torch.stack(states, dim=1)


class GatedRecallModel(nn.Module):
    """Embedding -> GatedLinearRecurrentBlock -> readout at the final (query) position.
    Mirrors `linear_recurrence.RecallModel`'s shape exactly so the two are a fair,
    matched-architecture comparison (same embed/readout, only the recurrence differs)."""

    def __init__(self, vocab_size: int, hidden: int, eps: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.recur = GatedLinearRecurrentBlock(hidden, hidden, eps)
        self.readout = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        states = self.recur(x)
        return self.readout(states[:, -1, :])
