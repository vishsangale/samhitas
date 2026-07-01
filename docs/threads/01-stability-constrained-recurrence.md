# Thread 1 (priority 2): Stability-constrained recurrence

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

If we parameterize a *linear* recurrent state update so its transition matrix has a
spectral radius provably in a target band (e.g. via a structured parameterization —
orthogonal/unitary, or diagonal-plus-low-rank with bounded diagonal), the network should
be trainable at proportionally greater depth/sequence length than an unconstrained update
of the same parameter count, and the relationship should be predictable from the spectral
bound itself — and, more importantly, that predictability should hold *across different
structured parameterizations that enforce the same bound*, not just for one specific
construction (this is the part that would actually be new; the linear-regime scaling law
itself is close to a known identity — see prediction below).

## Falsifiable prediction (pre-registered, revised after review)

Two predictions, scoped separately because they carry different risk:

1. **Linear-regime prediction (low risk, near-identity).** For a strictly *linear*
   recurrence `h_t = A h_{t-1} + B x_t` with spectral radius constrained to `1 - eps`, the
   maximum depth (or sequence length) at which the model trains without gradient
   explosion/vanishing (gradient norm ratio between first and last layer staying within a
   fixed band, e.g. [0.1x, 10x]) scales like `O(1/eps)`. This follows almost directly from
   contraction/mixing-time arguments and is expected to hold — it's included mainly as a
   harness sanity check, not as the interesting claim.
2. **Cross-parameterization prediction (the actual falsifiable core).** The `O(1/eps)`
   relationship holds *regardless of which structured parameterization enforces the
   spectral bound* — i.e., it is not an artifact specific to HiPPO's construction.
   Concretely: orthogonal/unitary parameterizations and diagonal-plus-low-rank
   parameterizations, both constrained to the same `eps`, should show max-trainable-depth
   curves that agree with each other and with the `O(1/eps)` prediction within a factor of
   2, at matched FLOPs. If the two parameterization families diverge from each other by
   more than that factor, the claim that "trainable depth is predictable from the spectral
   bound alone, independent of construction" is falsified, even though the linear-regime
   math above would still be technically correct for each construction individually.

**Nonlinear recurrence is explicitly out of scope for this thread's falsifiable claim.**
For a nonlinear update, the Jacobian spectrum is state- and time-dependent, so "the
spectral radius" is not a single well-defined number, and the clean mixing-time argument
does not transfer without a much more careful (uniform-in-state) stability analysis. If a
nonlinear variant is tried, it is exploratory and must not be reported under this thread's
pre-registered prediction — it would need its own falsifiable statement (e.g., a stability
margin computed along the actual training trajectory, not the linearization at init).

An unconstrained baseline (same parameter count, free spectrum) is expected to plateau
well below both constrained variants and show no such scaling relationship with any single
tunable knob — this is the comparison arm, not the main claim.

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

Not yet run. Priority 2 — build second, after thread 6's harness exists (this thread is
also thread 6's first novel-layer test case).
