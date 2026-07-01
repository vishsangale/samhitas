"""Measures backprop-through-time gradient decay: how much does the loss (computed from
the model's final-position output) depend on the input embedding at the first sequence
position vs. the last -- the direct empirical signature of vanishing/exploding gradients
over a long sequence, and the mechanism-level quantity thread 1's prediction is about
(docs/threads/01-stability-constrained-recurrence.md). One forward+backward pass, no
training loop required.

v2: the ratio alone conflates two different failure modes that the first smoke test
showed matter a lot in practice -- a config whose *ratio* is in-band but whose *raw*
gradient magnitude has already blown up or underflowed is not actually healthy, and a
config that fails on ratio because it decayed gracefully is a very different, much less
concerning situation than one that fails because it's numerically exploding. Both used to
just show up as `healthy=False`; now `failure_mode` says which.
"""

import torch
import torch.nn.functional as F

# Thread 1's pre-registered "healthy" band for the first/last gradient-norm ratio.
HEALTHY_BAND = (0.1, 10.0)

# Absolute magnitude guardrails, checked before the ratio: cross-entropy gradients at
# init on a small model are naturally O(1) or smaller (bounded by softmax probabilities
# times a modestly-scaled output layer), so a raw gradient norm many orders of magnitude
# above or below that is a real numerical problem regardless of what the ratio says.
# Heuristic thresholds, not derived from theory -- flagged as such, revisit if they turn
# out to misclassify real cases once used at larger scale.
EXPLOSION_ABS_THRESHOLD = 1e4
VANISHING_ABS_THRESHOLD = 1e-8


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

    if first_grad_norm > EXPLOSION_ABS_THRESHOLD or last_grad_norm > EXPLOSION_ABS_THRESHOLD:
        failure_mode = "exploding_absolute"
    elif first_grad_norm < VANISHING_ABS_THRESHOLD and last_grad_norm < VANISHING_ABS_THRESHOLD:
        failure_mode = "numerically_dead"  # both endpoints below float noise floor
    elif ratio > hi:
        failure_mode = "exploding_ratio"  # first >> last: gradient grows going backward in time
    elif ratio < lo:
        failure_mode = "vanishing_ratio"  # first << last: gradient shrinks going backward in time
    else:
        failure_mode = "healthy"

    return {
        "loss": loss.item(),
        "first_grad_norm": first_grad_norm,
        "last_grad_norm": last_grad_norm,
        "ratio_first_over_last": ratio,
        "failure_mode": failure_mode,
        "healthy": failure_mode == "healthy",
    }
