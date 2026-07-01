"""A small MLP with two parametrizations: standard (SP) and muP.

muP scaling follows the width-scaling ("abc-parametrization") table for Adam from
Yang & Hu et al., "Tensor Programs V" / the muTransfer paper: relative to a chosen
base_width n0,

  layer     init var      forward multiplier      Adam LR multiplier (relative to base_lr)
  input     1/fan_in      1                        1                      (fan_in fixed)
  hidden    1/fan_in      1                        n0/width               (fan_in = width)
  output    1/fan_in      n0/width                 n0/width               (fan_in = width)

Standard parametrization (SP) is the common default: Kaiming-uniform init on every layer,
no forward multiplier, no per-layer LR scaling -- this is the control arm thread 6's
prediction says should NOT transfer a tuned LR across width.

This is a from-scratch implementation for the falsification harness, not a wrapper around
an existing muP library, so the exact exponents above are the thing thread 6 is testing --
see docs/threads/06-mup-hparam-transfer.md.
"""

import math

import torch
import torch.nn as nn


class MuPMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, num_classes: int, depth: int = 3,
                 base_width: int = 64, parametrization: str = "mup"):
        super().__init__()
        assert parametrization in ("sp", "mup")
        assert depth >= 2, "depth counts input + output layers; need at least one hidden"
        self.parametrization = parametrization
        self.base_width = base_width
        self.width = width

        self.input_layer = nn.Linear(input_dim, width)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(width, width) for _ in range(depth - 2)
        )
        self.output_layer = nn.Linear(width, num_classes)
        self.activation = nn.ReLU()

        self._output_mult = base_width / width if parametrization == "mup" else 1.0
        self._init_weights()

    def _init_weights(self):
        def mup_init(layer: nn.Linear):
            fan_in = layer.weight.shape[1]
            nn.init.normal_(layer.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            nn.init.zeros_(layer.bias)

        if self.parametrization == "sp":
            for layer in [self.input_layer, *self.hidden_layers, self.output_layer]:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.zeros_(layer.bias)
        else:
            for layer in [self.input_layer, *self.hidden_layers, self.output_layer]:
                mup_init(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h) * self._output_mult

    def param_groups(self, base_lr: float):
        """Per-layer-type parameter groups with muP's Adam LR multipliers baked in.

        Under SP, every group gets the same base_lr (multiplier 1) -- this is the
        naive/unscaled control the falsifiable prediction compares against.
        """
        if self.parametrization == "sp":
            hidden_mult = output_mult = 1.0
        else:
            hidden_mult = output_mult = self.base_width / self.width

        return [
            {"params": self.input_layer.parameters(), "lr": base_lr},
            {"params": self.hidden_layers.parameters(), "lr": base_lr * hidden_mult},
            {"params": self.output_layer.parameters(), "lr": base_lr * output_mult},
        ]
