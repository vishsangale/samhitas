"""muP-scaled char-LM recurrence: embedding -> thread 1's orthogonal LinearRecurrentBlock
-> readout at the final context position. This is thread 6's declared "first novel case"
(docs/threads/06-mup-hparam-transfer.md's architectural hypothesis explicitly names thread
1's structured-recurrence layer as the first non-baseline arm).

Reuses `LinearRecurrentBlock("orthogonal")` UNMODIFIED for the A = (1-eps)*expm(skew(theta))
construction (exact spectral radius 1-eps by construction); this class only applies muP
init/LR on top of it. Layer-type mapping (muP; base_width n0, width W):

  embed      Embedding(vocab, W)   input-like : per-coord init 1/sqrt(vocab) (width-indep),
                                                 LR mult 1
  recur.B    Linear(W, W)          hidden     : init 1/sqrt(W), LR mult n0/W
  recur.theta  (W, W) generator    hidden*    : init THETA_BASE_STD*sqrt(n0/W)  (1/sqrt(W)-
                                                 scaled), LR mult n0/W  (*see THETA note)
  readout    Linear(W, vocab)      output     : init 1/sqrt(W), forward mult n0/W, LR mult n0/W

THETA note -- the one genuine open subtlety in muP-izing this layer (see thread 6 doc
addendum for the full reasoning and the numerical check behind it):

  theta's original thread-1 init is a FIXED std=0.01 regardless of width. A does not depend
  linearly on theta -- it goes through expm(skew(theta)) -- and A's SPECTRAL RADIUS is
  pinned at exactly 1-eps by construction for ANY theta (eps alone controls it, by design).
  So the norm/decay quantity muP normally protects for a hidden matrix is ALREADY protected
  here by eps: for THAT quantity theta is effectively exempt from the usual hidden-layer
  treatment. gradient_flow.py's effective-decay-rate diagnostic confirms this -- it reads
  ~(1-eps) at every width for either theta scaling, because ||A^t|| = (1-eps)^t*||Q^t|| =
  (1-eps)^t with Q orthogonal, i.e. the decay diagnostic is BLIND to theta by construction.

  What eps does NOT pin is the ROTATION / mixing structure of Q = expm(skew(theta)), which
  is governed by theta's scale. Measured directly: with a FIXED std=0.01, skew(theta)'s
  spectral radius (the largest rotation angle) grows as ~sqrt(width) (0.21 rad @ W=64 ->
  1.80 rad @ W=4096) -- a genuine width-dependent change in the layer's forward dynamics,
  exactly the kind of drift muP's hidden-layer init exists to prevent. Scaling theta's std
  as 1/sqrt(W) holds it essentially flat (~0.21-0.23 across the same range). So we scale
  theta's INIT as a hidden layer (THETA_BASE_STD at base_width, 1/sqrt(W) beyond it),
  preserving thread 1's near-identity-at-base-width regime while making the mixing structure
  width-invariant.

  theta's LR multiplier is set hidden-like (n0/W) too, as the principled default, but the
  exact LR exponent for a matrix-exp-parameterized generator is NOT derived in closed form
  here (the expm nonlinearity is precisely the kind of thing thread 6 asks whether vanilla
  muP covers). It is validated empirically by the coordinate check at the pinned widths
  (thread06_charlm_coord_check.py); a future adverse real-run reading would make theta's LR
  exponent the first suspect. This residual uncertainty is flagged, not silently resolved.

SP (the naive/unscaled control): every constructor default -- N(0,1) embedding,
Kaiming-uniform B and readout, and LinearRecurrentBlock's own fixed std=0.01 theta -- with
no forward multiplier and flat LR. (Those defaults ARE the naive/unscaled choice, so SP
does no re-init at all.)
"""

import math

import torch
import torch.nn as nn

from experiments.models.linear_recurrence import LinearRecurrentBlock


class CharLMRecurrence(nn.Module):
    THETA_BASE_STD = 0.01  # theta std at base_width; scaled as sqrt(base_width/width) beyond

    def __init__(self, vocab: int, width: int, eps: float = 0.05,
                 base_width: int = 64, parametrization: str = "mup"):
        super().__init__()
        assert parametrization in ("sp", "mup")
        self.parametrization = parametrization
        self.base_width = base_width
        self.width = width
        self.vocab = vocab
        self.eps = eps

        self.embed = nn.Embedding(vocab, width)
        self.recur = LinearRecurrentBlock(width, width, "orthogonal", eps)
        self.readout = nn.Linear(width, vocab)

        self._output_mult = base_width / width if parametrization == "mup" else 1.0
        if parametrization == "mup":
            self._mup_init()
        # SP: leave every constructor default in place -- those defaults are exactly the
        # naive/unscaled control (N(0,1) embedding, Kaiming B/readout, fixed-0.01 theta).

    def _mup_init(self):
        # Embedding: input weight -> per-coordinate init variance width-independent.
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(self.vocab))
        fan_in = self.width
        # B: hidden layer.
        nn.init.normal_(self.recur.B.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
        # theta: hidden-like init, scaled 1/sqrt(width) (see THETA note in module docstring).
        theta_std = self.THETA_BASE_STD * math.sqrt(self.base_width / self.width)
        nn.init.normal_(self.recur.theta, mean=0.0, std=theta_std)
        # readout: output layer.
        nn.init.normal_(self.readout.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
        nn.init.zeros_(self.readout.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, context_len) int64 -> next-char logits: (batch, vocab)."""
        x = self.embed(tokens)
        states = self.recur(x)                     # (batch, context_len, width)
        return self.readout(states[:, -1, :]) * self._output_mult

    def param_groups(self, base_lr: float):
        """Per-layer-type Adam LR groups with muP's multipliers baked in.

        embed is input-like (LR mult 1); B, theta, and readout get base_width/width. Under
        SP every group gets base_lr (mult 1).
        """
        hidden_mult = self.base_width / self.width if self.parametrization == "mup" else 1.0
        return [
            {"params": self.embed.parameters(), "lr": base_lr},
            {"params": self.recur.B.parameters(), "lr": base_lr * hidden_mult},
            {"params": [self.recur.theta], "lr": base_lr * hidden_mult},
            {"params": self.readout.parameters(), "lr": base_lr * hidden_mult},
        ]
