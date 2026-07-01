"""Linear recurrent block with three state-matrix parameterizations, built to test
thread 1's falsifiable prediction: does "trainable range scales with spectral bound"
hold consistently across different ways of enforcing that bound, not just one.

Recurrence is intentionally linear (h_t = A h_{t-1} + B x_t) -- see
docs/threads/01-stability-constrained-recurrence.md: the thread's pre-registered claim is
explicitly scoped to the linear regime, where "spectral radius of A" is a single
well-defined number. A nonlinearity would make the Jacobian state-dependent and put the
result outside this thread's claim.
"""

import torch
import torch.nn as nn


def _skew_symmetric(raw: torch.Tensor) -> torch.Tensor:
    return raw - raw.transpose(-1, -2)


def _power_iteration_spectral_norm(A: torch.Tensor, n_iters: int = 8) -> torch.Tensor:
    """Estimates the largest singular value of A via power iteration on A^T A. Cheap at
    the small hidden sizes this repo uses; no need for a full SVD every forward pass."""
    h = A.shape[-1]
    v = torch.randn(h, device=A.device)
    v = v / v.norm()
    for _ in range(n_iters):
        v = A.transpose(-1, -2) @ (A @ v)
        v = v / (v.norm() + 1e-12)
    return (A @ v).norm() / (v.norm() + 1e-12)


class LinearRecurrentBlock(nn.Module):
    """h_t = A h_{t-1} + B x_t, with A parameterized per `mode`:

    - "free": an unconstrained learnable matrix, initialized with a plain small-Gaussian
      ("vanilla RNN") init -- no special spectral properties, free to drift arbitrarily
      under gradient updates. The baseline arm the constrained variants are compared to.
    - "orthogonal": A = (1 - eps) * expm(skew(theta)). expm of a skew-symmetric matrix is
      exactly orthogonal, so its spectral radius (== spectral norm; orthogonal matrices are
      normal) is exactly 1 - eps by construction, for any theta, throughout training.
    - "diag_lowrank": A = diag(d) + E, where E is a rank-`low_rank` term. Naively capping
      ||A||_2 (spectral norm) via power iteration does *not* bound the spectral *radius*
      for a non-normal matrix like this one -- caught empirically: an early version capped
      the norm but the actual eigenvalues came out far smaller than the target, decaying
      much faster than the intended 1-eps would predict. Fixed via the Bauer-Fike theorem:
      for a normal matrix D (diag(d) is normal) perturbed by E, every eigenvalue of D+E
      lies within ||E||_2 of some eigenvalue of D. So capping |d_i| <= (1-eps-delta) and
      ||E||_2 <= delta guarantees spectral radius <= (1-eps-delta) + delta = 1-eps,
      exactly, not just an upper-bound-in-spirit like the norm-capping version was.

      That bound is a ceiling on the *largest* eigenvalue, not a claim about the rest of
      them -- a second bug, caught by an Opus 4.8 review, not by running it: the diagonal
      was initialized as diag_scale*tanh(N(0,4)), which spreads magnitudes across the
      whole [0, diag_scale] range (mean |d_i| ~= 0.72*diag_scale, only the max entry near
      the cap). Gradient decay over many steps is governed by the eigenvalue *bulk*, not
      the single largest one, so this parameterization's *effective* decay rate came out
      2-5x its nominal (1-eps), silently invalidating any matched-eps comparison against
      orthogonal. Fixed by initializing the diagonal to saturate tanh for (almost) every
      entry, so the whole diagonal -- not just its max -- sits near the target magnitude.
    """

    def __init__(self, hidden: int, input_dim: int, mode: str, eps: float = None,
                 low_rank: int = 4):
        super().__init__()
        assert mode in ("free", "orthogonal", "diag_lowrank")
        self.mode = mode
        self.eps = eps
        self.hidden = hidden
        self.B = nn.Linear(input_dim, hidden, bias=False)

        if mode == "free":
            assert eps is None, "free mode has no spectral constraint to set eps for"
            self.A_raw = nn.Parameter(torch.randn(hidden, hidden) / (hidden ** 0.5))
        elif mode == "orthogonal":
            assert eps is not None
            self.theta = nn.Parameter(torch.randn(hidden, hidden) * 0.01)
        else:  # diag_lowrank
            assert eps is not None
            # Random sign, magnitude saturating tanh (tanh(3) = 0.995) with only a little
            # per-entry jitter -- so the diagonal's *bulk*, not just its max, sits near the
            # cap. (v1 used torch.randn(hidden)*2.0, which spreads magnitudes across the
            # whole range and understates the effective spectral radius -- see class
            # docstring.)
            sign = torch.where(torch.rand(hidden) < 0.5, -1.0, 1.0)
            self.d_raw = nn.Parameter(sign * 3.0 + torch.randn(hidden) * 0.3)
            self.U = nn.Parameter(torch.randn(hidden, low_rank) / (hidden ** 0.5))
            self.V = nn.Parameter(torch.randn(hidden, low_rank) / (hidden ** 0.5))
            # delta reserved for E in the Bauer-Fike bound, scaled with eps rather than a
            # fixed constant -- a fixed delta (tried 0.05) ate a large, eps-dependent
            # fraction of the target radius at small eps (e.g. 0.05 out of a 0.02 budget),
            # shifting the *effective* bound to 1-eps-delta well below the intended 1-eps
            # and confounding the cross-parameterization comparison this thread is about.
            self.perturbation_budget = 0.1 * eps

    def _A(self) -> torch.Tensor:
        if self.mode == "free":
            return self.A_raw
        if self.mode == "orthogonal":
            Q = torch.matrix_exp(_skew_symmetric(self.theta))
            return (1 - self.eps) * Q
        delta = self.perturbation_budget
        diag_scale = max(1 - self.eps - delta, 0.0)
        d = diag_scale * torch.tanh(self.d_raw)
        E = self.U @ self.V.transpose(-1, -2)
        sigma_E = _power_iteration_spectral_norm(E)
        E = E * torch.clamp(delta / (sigma_E + 1e-8), max=1.0)
        return torch.diag(d) + E

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) -> (batch, seq_len, hidden), the full state
        sequence, so callers can read out any timestep (e.g. the query position)."""
        A = self._A()
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden, device=x.device)
        bx = self.B(x)
        states = []
        for t in range(seq_len):
            h = h @ A.transpose(-1, -2) + bx[:, t, :]
            states.append(h)
        return torch.stack(states, dim=1)


class RecallModel(nn.Module):
    """Embedding -> LinearRecurrentBlock -> readout at the final (query) position."""

    def __init__(self, vocab_size: int, hidden: int, mode: str, eps: float = None,
                 low_rank: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.recur = LinearRecurrentBlock(hidden, hidden, mode, eps, low_rank)
        self.readout = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, seq_len) int64 -> logits at the final position: (batch, vocab)."""
        x = self.embed(tokens)
        states = self.recur(x)
        return self.readout(states[:, -1, :])
