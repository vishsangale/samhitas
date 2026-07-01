"""Plain, unnormalized, pointwise-activation deep MLP for thread 2's criticality-guided-
init falsification test (docs/threads/02-criticality-guided-init.md): stacked
Linear -> tanh blocks, no normalization, no residual connections -- exactly the regime
the mean-field / edge-of-chaos theory in experiments/harness/meanfield.py applies to.

Per-layer weight/bias init variance (sigma_w^2, sigma_b^2) is a constructor argument
rather than a fixed default, since the whole point of this thread is sweeping it against
the theory's predicted critical value.
"""

import math

import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, depth: int, num_classes: int,
                 sigma_w2: float, sigma_b2: float):
        super().__init__()
        assert depth >= 2, "depth counts the input-projection layer + output readout"
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden)])
        self.layers.extend(nn.Linear(hidden, hidden) for _ in range(depth - 2))
        self.readout = nn.Linear(hidden, num_classes)
        self._init_weights(sigma_w2, sigma_b2)

    def _init_weights(self, sigma_w2: float, sigma_b2: float):
        for layer in [*self.layers, self.readout]:
            fan_in = layer.weight.shape[1]
            nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(sigma_w2 / fan_in))
            nn.init.normal_(layer.bias, mean=0.0, std=math.sqrt(sigma_b2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return self.readout(h)
