# Thread 1: Stability-constrained recurrence

**Math source:** control theory (linear systems, spectral radius / eigenvalue placement),
Lyapunov stability, Koopman operator theory (linearizing nonlinear dynamics via a lifted
linear operator).

## Motivation

A recurrent state update `h_{t} = f(h_{t-1}, x_t)` is a discrete-time dynamical system.
Classical control theory gives exact conditions for when such a system is stable (bounded
state, non-exploding/non-vanishing sensitivity to past inputs): for a linear update
`h_t = A h_{t-1} + B x_t`, stability and long-range memory are governed entirely by the
eigenvalue spectrum of `A`. This is precisely the mechanism S4/Mamba-style state-space
models exploit (HiPPO-derived `A` matrices with a controlled spectrum), but it's usually
presented as "this specific matrix works well" rather than as a general design rule:
*parameterize `A` so its spectrum is constrained by construction, and predict trainability
from the constraint, rather than discovering it by trial and error per architecture.*

## Architectural hypothesis

If we parameterize any recurrent/residual state update so the linearization of its
Jacobian has a spectral radius provably in a target band (e.g. via a structured
parameterization — orthogonal/unitary, or diagonal-plus-low-rank with bounded diagonal),
the network should be trainable at proportionally greater depth/sequence length than an
unconstrained update of the same parameter count, and the relationship should be
predictable from the spectral bound itself, not just qualitatively better.

## Falsifiable prediction

For a family of recurrent blocks with spectral radius constrained to `1 - eps`, the
maximum depth (or sequence length) at which the model trains without gradient
explosion/vanishing (measured via gradient norm ratio between first and last layer staying
within a fixed band, e.g. [0.1x, 10x]) scales like `O(1/eps)`. An unconstrained baseline
(same parameter count, free spectrum) should plateau well below this and show no such
scaling relationship with any single tunable knob.

This is a *mechanism-level* prediction (gradient-flow behavior, trainable depth) — it does
not claim anything about downstream capability, which keeps it testable at small scale.

## Minimal experiment

- Tiny recurrent/SSM blocks (<= a few M params), depths swept from 4 to 128 (or sequence
  lengths from 64 to 8192 for a fixed-depth model).
- Task: associative recall / selective copy (see `docs/methodology.md`).
- Compare: (a) unconstrained linear recurrence, (b) spectral-constrained recurrence at 2-3
  different `eps` values, matched FLOPs and matched tuning sweep.
- Measure: max trainable depth/length, gradient norm ratio curves, task accuracy.

## Compute budget

Well under a GPU-day: all models are small, sequence lengths modest, no large dataset
needed (synthetic task).

## Bitter-lesson check

- Encodes a general dynamical-systems property (stability), not task-specific knowledge. Low risk.
- Structured spectral constraints (orthogonal/diagonal-plus-low-rank) are parallel-scan
  friendly — this is exactly the class of update S4/Mamba already run efficiently on
  current accelerators, so hardware plausibility is well precedented, not speculative.

## Known prior work / risk of reinventing

This is close to S4/HiPPO/Mamba's actual mechanism, and to unitary/orthogonal RNNs
(uRNN, expRNN). The contribution here isn't a new matrix family — it's testing whether the
*general rule* ("trainable depth is predictable from the spectral bound") holds across
different structured parameterizations, not just the one HiPPO happened to pick. If it
doesn't generalize beyond HiPPO's specific construction, that's a real (and useful)
falsification.

## Status

Not yet run.
