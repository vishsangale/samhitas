# Thread 2 (priority 3, demoted after review): Criticality-guided initialization

**Math source:** statistical mechanics / mean-field theory of deep networks (order-to-chaos
phase transition; Poole et al. 2016, Schoenholz et al. 2017 "deep information propagation").

## Motivation

Mean-field theory treats a deep network's forward pass as a dynamical system over signal
variance/correlation, propagated layer to layer. For a given (activation function,
normalization scheme, residual/skip topology), there is an analytically derivable "edge of
chaos" — an initialization variance at which the correlation between two nearby inputs
neither collapses (order phase) nor decorrelates exponentially (chaos phase) as depth
grows, allowing gradient signal to propagate to much greater depth. This is usually derived
once per architecture family as a post-hoc explanation. The hypothesis here is to treat it
as a *design procedure*: for any new activation/layer/skip combination we invent (in other
threads or otherwise), derive its critical initialization first, and use the theory to
predict trainable depth before running anything.

## Architectural hypothesis

For a novel **pointwise activation, unnormalized** layer/topology combination, the
mean-field recursion for the input-input correlation map has a fixed point `chi_1(sigma)`
(the derivative of the correlation map at the fixed point) that predicts trainable depth:
at the critical point (`chi_1 = 1`), trainable depth should scale as the theory's own
depth-scale `xi ~ 1 / |log chi_1(sigma)|`, and depth should saturate early, at a scale set
by `xi`, once `sigma` moves off criticality.

**Scope, tightened after review:** this thread's falsifiable claim applies only to
pointwise nonlinearities without normalization layers. Standard mean-field theory assumes
no normalization; LayerNorm/BatchNorm break the clean correlation-map recursion and need a
separate (harder, and not attempted in this thread) mean-field treatment. A layer design
that includes normalization is out of scope for this thread's prediction.

## Falsifiable prediction (pre-registered, revised after review)

Given a derived critical variance `sigma*^2` and its associated depth-scale
`xi(sigma) = 1 / |log chi_1(sigma)|` for a specific pointwise-activation layer design
(not a hand-picked offset — the pass/fail band comes directly from the theory's own
`xi`), models should reach a fixed depth threshold `D_ref` (e.g. the depth reachable at
`sigma*` with default compute) at variance `sigma` if and only if `xi(sigma) >~ D_ref` up
to a small constant factor. Concretely: compare training success (loss reaches a fixed
threshold, gradient norms non-degenerate) across a `sigma` sweep and a depth sweep, and
check whether the *boundary* of the trainable region in (sigma, depth) space matches the
theory's `xi(sigma)` curve — not a hand-picked 0.8x/1.2x offset. If the theory's derived
`sigma*^2` or the shape of the `xi(sigma)` curve don't match where the empirical trainable
boundary actually falls, the specific derivation is falsified without needing to falsify
mean-field theory in general.

**Cost caveat (added after review):** deriving `chi_1(sigma)` and `sigma*` in closed form
for a genuinely novel activation is not a cheap preprocessing step — it requires Gaussian
integrals of the nonlinearity that often have no closed form and may need numerical
quadrature or Monte Carlo estimation of the recursion itself. Budget for this as a real
(if still small-compute) analysis task, not something to derive in an afternoon before the
"real" experiment. If a closed form isn't tractable within a bounded effort, numerically
estimating `chi_1(sigma)` via Monte Carlo forward passes is an acceptable substitute and
should be reported as such.

## Minimal experiment

- Pick 1-2 pointwise-activation, unnormalized layer/topology combinations not already
  characterized in the literature (or deliberately re-derive a known one, e.g. tanh-MLP,
  as a cross-check that the harness reproduces Poole/Schoenholz's own numbers first).
- Derive `chi_1(sigma)` and `sigma*^2` analytically, or via numerical quadrature/Monte
  Carlo if closed form is intractable (see cost caveat above) — do this and validate it
  against the tanh-MLP cross-check *before* committing to a novel activation.
- Depth sweep (e.g. 10 to 500 layers) crossed with a `sigma` sweep bracketing `sigma*^2`,
  on a small vision task (CIFAR-10 subset) or synthetic regression task, matched compute
  and LR sweep per depth/variance combination.
- Measure: the empirical (sigma, depth) trainability boundary; compare its shape to the
  theory's `xi(sigma)` curve, not just success/failure at a couple of hand-picked points.

## Compute budget

Depth x variance sweeps at small width and small dataset are cheap in training compute;
the numerically expensive part is the fixed-point derivation, which is analysis effort,
not GPU time, but should be budgeted honestly as nontrivial (see cost caveat above) rather
than assumed free. Training itself fits well under a GPU-day.

## Bitter-lesson check

- This is a *derivation procedure*, not a fixed prior — it's meant to be re-run for
  whatever new layer a different thread invents, so it composes with the rest of the
  portfolio rather than competing with it.
- No task-specific knowledge encoded; purely a statement about signal propagation through
  a fixed computation graph, independent of data domain.

## Known prior work / risk of reinventing

Directly builds on Poole et al., Schoenholz et al., and later work explaining why
ResNets/Fixup/ReZero-style init work (residual connections push the critical exponent
toward polynomial rather than exponential decay, extending trainable depth). Novelty here
is operationalizing the derivation as a standard step applied to *new* layers proposed in
this repo, with the falsification protocol from `docs/methodology.md`, rather than as a
one-off explanation of an existing architecture.

## Status

Not yet run. Priority 3 (demoted from the original draft's priority-1 slot after review —
the derivation cost was previously undersold; run after threads 6 and 1).
