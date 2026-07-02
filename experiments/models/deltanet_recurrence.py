"""Arm (c) of thread 17 (docs/threads/17-recall-mechanism-ladder.md): a minimal, single-
head DeltaNet-style outer-product matrix state, replacing every other arm's vector state
`h_t in R^hidden` with a matrix state `S_t in R^{hidden x hidden}` that can store multiple,
separately-addressable key-value associations -- the literature's highest-confidence
mechanism for recall at this scale (DeltaNet, arXiv:2406.06484; fast weight programmers).

Delta-rule update (Schlag et al. 2021 / Yang et al. 2024's DeltaNet): at each step, the
state is decayed, then corrected toward reducing the error between what it currently
retrieves for the current key and the current value:

    S_t = S_{t-1} @ decay^T                          (state-retention decay)
    pred_v_t = S_t @ k_t                              (what the state currently associates with k_t)
    S_t = S_t - beta_t * (pred_v_t - v_t) (x) k_t      (delta-rule correction, outer product)

`decay` is thread 1's exact `orthogonal` construction (`(1-eps) * expm(skew(theta))`,
reusing `_skew_symmetric` unmodified), applied on the state's key-addressing axis, so the
state-retention half of the update inherits the same provable spectral-radius property
every other construction in this repo uses -- this is what "spectrally constrained decay"
means for a matrix state, the arm's one deliberate point of continuity with threads 1/9/11.
`k_t` is L2-normalized (standard DeltaNet practice, keeps the outer-product update's scale
bounded). `beta_t` (per-timestep write strength, in [0,1] via sigmoid) is bias-initialized
to -4 (`sigmoid(-4)~=0.018`), matching this repo's established "start close to a
non-selective baseline" convention (thread 9/11's gate init) -- at init, `beta_t~=0` means
the delta-rule correction is nearly inert and the state just decays, the matrix-state
analogue of "start close to doing nothing, let training turn on writing where needed."

Not a claim to reproduce full DeltaNet (which pairs the delta rule with multi-head
structure and specific normalization) -- deliberately the minimal single-head version, per
the thread doc's explicitly-out-of-scope section.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.models.linear_recurrence import _skew_symmetric


class DeltaNetBlock(nn.Module):
    BETA_BIAS_INIT = -4.0

    def __init__(self, hidden: int, input_dim: int, eps: float):
        super().__init__()
        self.hidden = hidden
        self.eps = eps
        self.theta = nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        self.W_k = nn.Linear(input_dim, hidden, bias=False)
        self.W_v = nn.Linear(input_dim, hidden, bias=False)
        self.W_beta = nn.Linear(input_dim, 1)
        nn.init.constant_(self.W_beta.bias, self.BETA_BIAS_INIT)

    def _decay(self) -> torch.Tensor:
        Q = torch.matrix_exp(_skew_symmetric(self.theta))
        return (1 - self.eps) * Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, seq_len, hidden), the per-timestep
        self-retrieval S_t @ k_t (state read back with that same step's own key)."""
        batch, seq_len, _ = x.shape
        decay = self._decay()  # (hidden, hidden)
        k = F.normalize(self.W_k(x), dim=-1)  # (batch, seq_len, hidden), unit-norm keys
        v = self.W_v(x)                        # (batch, seq_len, hidden)
        beta = torch.sigmoid(self.W_beta(x))    # (batch, seq_len, 1)

        S = torch.zeros(batch, self.hidden, self.hidden, device=x.device)
        states = []
        for t in range(seq_len):
            S = S @ decay.transpose(-1, -2)
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            beta_t = beta[:, t, :]
            pred_v = torch.einsum("bij,bj->bi", S, k_t)
            error = pred_v - v_t
            S = S - beta_t.unsqueeze(-1) * torch.einsum("bi,bj->bij", error, k_t)
            o_t = torch.einsum("bij,bj->bi", S, k_t)
            states.append(o_t)
        return torch.stack(states, dim=1)


class DeltaNetRecallModel(nn.Module):
    """Embedding -> DeltaNetBlock -> readout at the final (query) position. Exposes
    .embed / .recur / .readout so experiments/harness/gradient_flow.py's
    gradient_norm_ratio() works unchanged, matching every other recall model in this repo."""

    def __init__(self, vocab_size: int, hidden: int, eps: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.recur = DeltaNetBlock(hidden, hidden, eps)
        self.readout = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        states = self.recur(x)
        return self.readout(states[:, -1, :])
