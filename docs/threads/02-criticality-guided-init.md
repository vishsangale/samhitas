# Thread 2: Criticality-guided initialization

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

For a novel layer/activation/topology combination, the mean-field recursion for the
input-input correlation map has a fixed point whose stability (derivative of the map at
the fixed point) predicts trainable depth: at the critical point, trainable depth should
scale much more steeply with a tunable "distance from criticality" parameter than off the
critical point, where depth saturates early regardless of other hyperparameters.

## Falsifiable prediction

Given a derived critical variance `sigma*^2` for a specific layer design, models
initialized at `sigma*^2` should train successfully (loss decreasing, non-degenerate
gradient norms) at depths at least an order of magnitude greater than models initialized
even moderately off `sigma*^2` (e.g. `sigma*^2 * 0.8` or `sigma*^2 * 1.2`), with the
transition sharp enough to be visible in a depth sweep at fixed compute budget per depth.
If the theory's derived `sigma*^2` is wrong, the empirical optimum will not sit near it —
that alone falsifies the specific derivation without needing to falsify the general
mean-field approach.

## Minimal experiment

- Pick 2-3 layer/activation/topology combinations not already characterized in the
  literature (or deliberately re-derive a known one, e.g. a physics-inspired activation
  from another thread, as a cross-check).
- Derive `sigma*^2` analytically (or numerically via the recursion if closed form is
  intractable).
- Depth sweep (e.g. 10 to 500 layers) at several init variances bracketing `sigma*^2`, on a
  small vision task (CIFAR-10 subset) or synthetic regression task, matched compute and LR
  sweep per depth/variance combination.
- Measure: max depth at which training loss reaches a fixed threshold; gradient norm decay
  rate per layer.

## Compute budget

Depth sweeps at small width and small dataset are cheap; the numerically expensive part
(deriving the fixed point) is analysis, not training. Fits well under a GPU-day.

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

Not yet run.
