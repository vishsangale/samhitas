"""Measures backprop-through-time gradient decay: how much does the loss (computed from
the model's final-position output) depend on the input embedding at the first sequence
position vs. the last -- the direct empirical signature of vanishing/exploding gradients
over a long sequence, and the mechanism-level quantity thread 1's prediction is about
(docs/threads/01-stability-constrained-recurrence.md). One forward+backward pass, no
training loop required.
"""

import torch
import torch.nn.functional as F

# Thread 1's pre-registered "healthy" band for the first/last gradient-norm ratio.
HEALTHY_BAND = (0.1, 10.0)


def gradient_norm_ratio(model, tokens: torch.Tensor, targets: torch.Tensor) -> dict:
    x = model.embed(tokens)
    x.retain_grad()
    states = model.recur(x)
    logits = model.readout(states[:, -1, :])
    loss = F.cross_entropy(logits, targets)
    model.zero_grad(set_to_none=True)
    loss.backward()

    first_grad_norm = x.grad[:, 0, :].norm().item()
    last_grad_norm = x.grad[:, -1, :].norm().item()
    ratio = first_grad_norm / (last_grad_norm + 1e-12)
    lo, hi = HEALTHY_BAND
    return {
        "loss": loss.item(),
        "first_grad_norm": first_grad_norm,
        "last_grad_norm": last_grad_norm,
        "ratio_first_over_last": ratio,
        "healthy": lo <= ratio <= hi,
    }
