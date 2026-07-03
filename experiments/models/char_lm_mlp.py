"""Fixed-context char-LM MLP (Bengio et al. style neural LM) with SP and muP
parametrizations -- thread 6's baseline arm on the char-LM task
(docs/threads/06-mup-hparam-transfer.md). Reuses `MuPMLP`'s exact per-layer-type muP table
(Yang & Hu et al., "Tensor Programs V"), the table thread 14's coordinate check already
validated for the modular-arith MuPMLP.

Layer-type mapping (muP; base_width n0, width W):

  token_embed  Embedding(vocab, d_embed)    input-like : per-coord init 1/sqrt(vocab)
                                                          (width-independent), LR mult 1
  proj         Linear(context*d_embed, W)   input-like : fan_in = context*d_embed is FIXED
                                                          (d_embed does NOT scale with W),
                                                          init 1/sqrt(fan_in), LR mult 1
  hidden[i]    Linear(W, W)                 hidden     : init 1/sqrt(W), LR mult n0/W
  output       Linear(W, vocab)             output     : init 1/sqrt(W), forward mult n0/W,
                                                          LR mult n0/W

`d_embed` is a FIXED small constant, NOT the width -- exactly analogous to `MuPMLP`'s fixed
`input_dim` on modular arithmetic, where only the hidden width scales. This is deliberate:
it keeps the concatenated input dimension (context*d_embed) width-independent, so `proj` is
a genuine muP *input* layer (fixed fan_in, LR mult 1) rather than a hidden layer with a
context*W fan-in, and it keeps the parameter count O(W^2) instead of O(context*W^2). The
token embedding itself involves no width (vocab->d_embed, both fixed), so it is a fixed
learned featurizer; `proj` is the first width-scaling layer.

SP (the naive/unscaled control thread 6's prediction says should NOT transfer a tuned LR
across width): the standard defaults -- N(0,1) embedding, Kaiming-uniform linears, no
forward multiplier, flat LR on every group. Matches `MuPMLP`'s SP convention.
"""

import math

import torch
import torch.nn as nn


class CharLMMLP(nn.Module):
    def __init__(self, vocab: int, width: int, context_len: int, d_embed: int = 64,
                 depth: int = 4, base_width: int = 64, parametrization: str = "mup"):
        super().__init__()
        assert parametrization in ("sp", "mup")
        assert depth >= 2, "depth counts proj(input) + output; need at least those two"
        self.parametrization = parametrization
        self.base_width = base_width
        self.width = width
        self.context_len = context_len
        self.d_embed = d_embed
        self.vocab = vocab

        self.token_embed = nn.Embedding(vocab, d_embed)
        self.proj = nn.Linear(context_len * d_embed, width)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(width, width) for _ in range(depth - 2)
        )
        self.output_layer = nn.Linear(width, vocab)
        self.activation = nn.ReLU()

        self._output_mult = base_width / width if parametrization == "mup" else 1.0
        self._init_weights()

    def _init_weights(self):
        def mup_linear(layer: nn.Linear):
            fan_in = layer.weight.shape[1]
            nn.init.normal_(layer.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            nn.init.zeros_(layer.bias)

        if self.parametrization == "sp":
            # Naive defaults on every layer -- the unscaled control arm.
            nn.init.normal_(self.token_embed.weight, mean=0.0, std=1.0)
            for layer in [self.proj, *self.hidden_layers, self.output_layer]:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.zeros_(layer.bias)
        else:
            # muP: embedding is an input weight -> per-coordinate init variance is
            # width-independent (fan_in = vocab, fixed). proj/hidden/output follow the table.
            nn.init.normal_(self.token_embed.weight, mean=0.0, std=1.0 / math.sqrt(self.vocab))
            for layer in [self.proj, *self.hidden_layers, self.output_layer]:
                mup_linear(layer)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, context_len) int64 -> next-char logits: (batch, vocab)."""
        e = self.token_embed(tokens)                       # (batch, context_len, d_embed)
        h = self.activation(self.proj(e.reshape(e.shape[0], -1)))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h) * self._output_mult

    def param_groups(self, base_lr: float):
        """Per-layer-type Adam LR groups with muP's multipliers baked in.

        token_embed and proj are input-like (LR mult 1); hidden and output get the
        base_width/width multiplier. Under SP every group gets base_lr (mult 1).
        """
        hidden_mult = self.base_width / self.width if self.parametrization == "mup" else 1.0
        return [
            {"params": self.token_embed.parameters(), "lr": base_lr},
            {"params": self.proj.parameters(), "lr": base_lr},
            {"params": self.hidden_layers.parameters(), "lr": base_lr * hidden_mult},
            {"params": self.output_layer.parameters(), "lr": base_lr * hidden_mult},
        ]
