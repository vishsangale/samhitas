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

**Post-hoc note, 2026-07-07 (`harness/meanfield.py`, `models/deep_mlp.py`,
`scripts/thread02_criticality_sanity.py`): prediction A falsified exactly as
pre-registered, but for reasons that decouple the *loss-based* operationalization from the
theory's actual claim — not chased further under this label, needs its own fresh thread.**

Built the mean-field numerics (`chi_1(sigma_w2, sigma_b2)`, `xi = 1/|log(chi_1)|` via
Gauss-Hermite quadrature, root-finding `q*` by bisection rather than plain fixed-point
iteration — iteration's convergence rate is governed by `chi_1` itself, so it's slowest
exactly at criticality, "critical slowing down," caught during smoke-testing before any
numbers were reported). Cross-checked three ways before trusting it: exact match to the
analytically-derivable `sigma_b2=0` special case (critical `sigma_w2=1.0` to 8 decimal
places), an independent Monte Carlo estimate of the actual correlation-map derivative
(agreed within MC noise), and the expected qualitative Poole et al. trend (critical
`sigma_w2` rises monotonically with `sigma_b2`).

Ran the pre-registered protocol: `sigma_b2=0.1` fixed, 13-point `sigma_w2` grid bracketing
the theory's own derived `sigma_w2*=1.9861` on both sides, 13-point geometric depth grid
(4 to 256), modular arithmetic (p=17), 5-point LR grid x 3 seeds x 150 Adam steps per
cell, "trainable" = training loss <= 1.0 within budget. 2535 configs, ~77 min CPU.

**Result: the empirical (sigma_w2, depth) trainability boundary is nearly flat.** It sits
at depth 8-16 for almost the entire grid, including right at the exact critical point
(`xi` ~ 2.9e10, i.e. theoretically near-unbounded trainable depth) and at the grid's
extreme edges (`xi` as low as ~4-6) — `xi` spans about 9 orders of magnitude across the
grid while the empirical boundary moves by at most ~3x. **Falsified as literally
specified** ("matches up to a small constant factor" cannot survive a 9-orders-of-magnitude
vs. 3x mismatch).

Sent the full result to an independent Opus 4.8 review before drawing any conclusion, per
this repo's process. The review re-derived the harness numerics itself (confirmed correct
via its own independent checks) and live-re-ran 7 cells, reproducing the stored results to
every digit — not a harness bug. It then named three concrete, verified confounds that
decouple the *loss-reaching* metric from the theory's actual signal-propagation claim:

1. **The task doesn't need depth.** Reach rate is ~100% at depth 6-8 for essentially every
   `sigma_w2`, and 0% at depth >=32 for essentially every `sigma_w2` — modular addition is
   shallow-solvable, so added depth is a pure handicap regardless of criticality. There's
   no regime in this task where extra depth helps, which is the regime the theory is
   actually about.
2. **The LR grid saturates at its own ceiling.** Among cells that reached target, the
   winning LR was consistently the grid maximum (0.01) — deeper nets are being cut off by
   the LR grid's own edge, not by criticality.
3. **The binary threshold inverts the ranking right at the boundary.** At depth 16, the
   exact critical point's best final loss (1.184) just misses the 1.0 cutoff while the
   `sigma_w2=3.0` chaotic-edge point passes — a threshold artifact, not a real trainability
   difference; 150 steps is also short enough that deep-net optimization has little room to
   express any criticality benefit before the shallow-solvable-task ceiling dominates.

**The theory's actual claim (a signal-propagation *depth scale*, not "does a fixed-budget
Adam run hit a loss target") looks more supported when tested with the right diagnostic —
same lesson thread 1 already learned once (task accuracy was the wrong metric there too,
replaced by a gradient-flow diagnostic).** The review measured the init-time gradient-flow
*decay/growth length* (fit `log||d loss/dW_1||` vs. depth, not raw magnitude) and reported
a length constant within roughly a constant ~2x factor of theory `xi` on both the ordered
and chaotic sides, peaking at criticality with the correct decay-to-growth sign flip.

**I independently spot-checked this specific claim myself** (5 `sigma_w2` points, 12
seeds, depths 4-128, before writing any of this into the permanent record) rather than
taking the review's positive framing at face value, symmetric to the review not taking my
original write-up at face value. The qualitative pattern reproduced: empirical length
peaks near criticality (44.3 vs. 6.75-18.5 on the ordered side of the same range) and the
sign flips correctly. But the precise "~2x constant factor" is optimistic as a general
characterization — at `sigma_w2=2.15` (`xi=29.8`) my fit gave a length of 280 (ratio ~9x,
not ~2x), traced directly to per-seed noise: individual init-time gradient norms at that
point span roughly 50x across 12 seeds at a single depth, comparable to or larger than the
actual depth-trend signal over the tested range, so a naive log-linear fit is unstable
there with this seed count. The qualitative mechanism (peak at criticality, correct
decay/growth sign flip, order-of-magnitude-plus improvement in propagation length near
criticality vs. far from it) is real and consistent across both my check and the review's.
The specific quantitative "small constant factor" claim is not yet cleanly established —
it needs more seeds and/or a depth range individually matched to each `sigma_w2`'s own
`xi` (points with large `xi` need proportionally deeper grids to resolve; points with
`chi_1` very close to 1 have a weak trend that per-seed init noise can dominate) before
being reported as a verified quantitative match.

**Verdict for this thread's specific pre-registered claim: falsified as specified** (the
loss-based operationalization). **Not closing the underlying idea as a negative result** —
per this repo's own discipline (don't retrofit a falsified prediction; a genuinely
different operationalization needs its own fresh pre-registration, not a retroactive
metric swap under this thread's existing claim), a follow-up thread testing the
gradient-flow depth-scale claim specifically — with a properly designed per-`sigma_w2`
depth grid and enough seeds to resolve the trend against per-seed init noise, pre-
registered with its own numeric pass/fail band *before* that run — is the natural next
step, not yet started.
