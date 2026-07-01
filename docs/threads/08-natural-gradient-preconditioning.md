# Thread 8 (priority 5, added after review): Fisher-preconditioned / natural-gradient optimization

**Math source:** information geometry (Fisher information metric, natural gradient),
K-FAC-style Kronecker-factored curvature approximations.

## Motivation

Added after review as a stronger substitute for thread 3's slot: it's "general
computational/statistical structure" in the same sense thread 6 (muP) is, but targets
optimization-step efficiency via curvature rather than width-transfer via scale. Plain
gradient descent treats parameter space as Euclidean; the natural gradient rescales the
update by the inverse Fisher information metric, which is the geometrically correct notion
of "steepest descent" over the space of distributions the model represents, not over raw
parameter coordinates. The Fisher metric's conditioning (its eigenvalue spread) is a
concrete, computable property of a given architecture, and it makes a sharp prediction:
architectures whose Fisher metric is better-conditioned should need fewer optimization
steps to reach a target loss, by a factor tied to the conditioning number, not just
"train better" vaguely.

## Architectural hypothesis

For a given architecture, preconditioning gradient updates by a Kronecker-factored (K-FAC-
style) approximation to the Fisher information matrix reduces the number of optimization
steps needed to reach a target loss, relative to plain SGD/Adam at matched per-step
compute, by a factor predictable from the *condition number* of the (approximate) Fisher
metric — and, more interestingly for this repo, the Fisher conditioning number itself can
be used as a cheap, architecture-comparison design signal even before training to
convergence (i.e., a poorly-conditioned architecture is predictably going to be harder to
optimize, and this can be checked near initialization).

## Falsifiable prediction (pre-registered)

For 2-3 small architectures (a plain baseline plus at least one novel layer type from
another thread), estimate the Fisher/K-FAC condition number near initialization (via a
Kronecker-factored curvature estimate, which is tractable at small scale). Prediction:
the ranking of architectures by (estimated condition number near init) should match the
ranking of (steps-to-target-loss under plain Adam, matched per-step compute) — well-
conditioned architectures reach target loss faster in step count. Separately: applying
K-FAC-style preconditioning should reduce steps-to-target-loss for the *worse*-conditioned
architecture by a larger factor than for the *better*-conditioned one (since there's more
curvature pathology for preconditioning to correct) — a specific, checkable
cross-architecture pattern, not just "K-FAC generally helps." Pre-registered pass/fail: a
Spearman correlation >= 0.7 between init-time condition number and steps-to-target across
>= 6 (architecture x seed) combinations; if the correlation is weak or the preconditioning
benefit doesn't scale with how poorly-conditioned the architecture is, the thread is
falsified.

## Minimal experiment

- Small models (plain baseline + 1-2 novel layer types from other threads once
  implemented), small classification/LM task.
- Estimate K-FAC-style Fisher condition number near init (cheap: a handful of
  forward/backward passes, block-diagonal Kronecker approximation, no need for the full
  Fisher matrix).
- Train each architecture to a fixed target loss under (a) plain Adam and (b) K-FAC-
  preconditioned updates, matched per-step compute accounting for preconditioning
  overhead (K-FAC has real extra cost per step — must be charged honestly, same FLOP+
  wall-clock discipline as every other thread).
- Correlate init-time condition number against steps-to-target and against
  preconditioning's step-count benefit.

## Compute budget

Small models/datasets; the curvature estimation itself is cheap relative to training.
Main cost is the extra per-step overhead of K-FAC updates during the comparison runs —
still well within a GPU-day given small model/data scale.

## Bitter-lesson check

- General optimization-geometry statement, no task-specific knowledge. Low risk.
- Explicit FLOP+wall-clock charge for preconditioning overhead is required (per
  `docs/methodology.md`'s compute-accounting rule) — K-FAC's per-step cost is real and
  architecture-dependent (bigger for layers with large weight matrices), so a naive
  step-count-only comparison would be misleading; this thread's own prediction is designed
  around matched *compute*, not matched steps.

## Known prior work / risk of reinventing

K-FAC (Martens & Grosse 2015) and natural gradient methods (Amari) are well established;
this is not a new optimizer. The contribution is using the Fisher condition number as a
*cross-architecture design diagnostic* computed cheaply near initialization — i.e., as a
falsifiable predictor of which of this repo's proposed layer types will be easy or hard to
optimize, before spending a full training run finding out — rather than as a claim that
K-FAC itself is novel.

## Status

Not yet run. Priority 5 — run after thread 4, once at least one novel layer type exists to
compare against the baseline.
