"""Analytic FLOP counting, stated explicitly per docs/methodology.md's compute-accounting
rule: report FLOPs alongside wall-clock, and state the counting method.

Counting method: for a Linear(fan_in, fan_out), forward is 2*fan_in*fan_out FLOPs/sample
(multiply-add counted as 2), backward (grad wrt input + grad wrt weight) is ~2x forward,
so ~6*fan_in*fan_out FLOPs/sample total for a forward+backward pass through that layer.
This is the standard hand-derived approximation used in scaling-law literature (e.g.
Kaplan et al. 2020); it ignores activation/bias FLOPs, which are asymptotically negligible
next to the matmuls for anything but pathologically narrow layers.
"""

from experiments.models.mlp import MuPMLP


def mlp_flops_per_sample(model: MuPMLP) -> int:
    linears = [model.input_layer, *model.hidden_layers, model.output_layer]
    return sum(6 * layer.weight.shape[0] * layer.weight.shape[1] for layer in linears)


def train_flops(model: MuPMLP, batch_size: int, n_steps: int) -> int:
    return mlp_flops_per_sample(model) * batch_size * n_steps
