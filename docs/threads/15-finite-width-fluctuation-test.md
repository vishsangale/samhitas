# Thread 15 (idea I2 + I5's width-only piece): finite-width fluctuation test for the
criticality anomaly

**Math source:** finite-width perturbation theory around the mean-field infinite-width
limit -- Hanin (arXiv:1801.03744), Hanin & Nica (arXiv:1812.05994), Roberts-Yaida-Hanin
(arXiv:2106.10165). Proposed by the 2026-07-07 portfolio review (`docs/reviews/
2026-07-07-portfolio-review.md`, idea I2) as the structurally different measurement threads
12 and 13 both called for when closing the criticality-guided-init measurement-refinement
sub-line.

## Why this thread exists

Threads 12 and 13 (`docs/threads/12-gradient-flow-depth-scale.md`,
`docs/threads/13-robust-gradient-flow-depth-scale.md`) both found a systematic,
chaotic-phase-only bias: the empirical gradient-flow-length estimate undershoots
mean-field theory's `xi` by 2.7-4x at several `sigma_w2` points, and even sign-flips at
`sigma_w2=2.05`. An Opus review of thread 13 found this is consistent with a specific,
quantitative *other* theory, not just "noisy measurement": finite-width theory predicts
`log||grad||` is asymptotically Gaussian (grad norm log-normal) with `Var[log||grad||]`
growing proportional to `depth/width`, and mean-field's own controlled regime requires
`r = depth/width << 1` -- threads 12/13's grid (fixed `hidden=32`) reaches `r ~ 11` at
depth 362, deep outside that regime. Under a log-normal with depth-growing variance, a
median-based estimator (what threads 12/13 both used, for robustness to seed noise) is
*mathematically required* to undershoot a mean-based (`E[grad^2]`) prediction, and by a
computable amount: `log(E[X]) - log(median(X)) = Var[log X] / 2` for log-normal `X`. This
thread tests that specific mechanism directly and quantitatively, rather than treating the
residual bias as an unexplained nuisance.

A second, competing explanation was also identified (Opus meta-review of thread 13): the
per-depth *median* itself turns over non-monotonically (rises then sags), which a median
being tail-robust cannot be caused by heavy tails alone -- it would require an actual
change in central tendency, e.g. tanh saturation compounding along individual chaotic
trajectories at large finite depth (a finite-*depth* story, not finite-*width*). This
thread's design (varying width while holding the task/model/depth-grid otherwise fixed)
directly distinguishes the two: the finite-width story predicts the effect *shrinks* as
width grows; the finite-depth-saturation story predicts no such width dependence.

## Falsifiable predictions (pre-registered)

Reuse `experiments/models/deep_mlp.py` and `experiments/harness/meanfield.py` unchanged
(no new model code). Same task/model constants as threads 12/13: `p=17`, `SIGMA_B2=0.1`,
`BATCH=64`, single forward+backward pass per cell (no training loop), first-layer weight
gradient norm as the measured quantity (`model.layers[0].weight.grad.norm()`).

**Grid:** `sigma_w2` in `{2.05, 2.2, 2.4, 2.8}` -- the four chaotic-phase points thread 13
flagged with undershoot (2.2: 3.98x, 2.4: 2.69x, 2.8: 2.34x) or sign-flip (2.05) --
deliberately restricted to the anomalous branch rather than repeating the full 9-point grid,
since the ordered-phase points already matched theory well in thread 13 and aren't the
question here. `width` (= `hidden`) in `{32, 64, 128}` (`32` matches threads 12/13 exactly,
for direct comparability; `64`/`128` test the width-scaling prediction). `depth` -- same
16-point grid as threads 12/13: `{2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256,
362}`. `seeds` = 50 per cell (matches thread 13). Total: 4 x 3 x 16 x 50 = 9,600
forward+backward passes.

### Prediction A: variance-growth slope scales as 1/width

For each `(sigma_w2, width)`, compute `Var[log(grad_norm)]` across the 50 seeds at each of
the 16 depths, then fit `Var[log(grad_norm)]` vs. `depth` by ordinary least squares to get a
`variance_growth_slope(sigma_w2, width)`.

- **Pass** iff, for every `sigma_w2`, the three widths' `slope * width` values are pairwise
  within a `[0.5, 2.0]` ratio band of each other (i.e., `slope` is compatible with `c/width`
  for some roughly-constant `c`, as Hanin-Nica predicts), **and** `variance_growth_slope` is
  positive (variance actually grows with depth, not flat/negative) at `width=32` for at
  least 3 of the 4 `sigma_w2` points (a positive-control check reproducing what threads
  12/13 already observed qualitatively -- if this fails, the base phenomenon itself isn't
  replicating and the width-scaling test is moot).
- **Fail** otherwise.

### Prediction B: mean/median slope gap matches the log-normal identity

For the same 12 `(sigma_w2, width)` cells, compute two depth-fit slopes over the same
16-point depth grid: `slope_median` (Theil-Sen on per-depth median `log(grad_norm)` across
seeds -- thread 13's exact method) and `slope_mean` (ordinary least squares on
`log(mean(grad_norm))` per depth, arithmetic mean of the raw, non-logged values -- the
quantity theory's `E[grad^2]`-style prediction is actually stated in terms of). Define the
empirical gap `gap_emp = slope_mean - slope_median` and the theory-predicted gap
`gap_theory = variance_growth_slope / 2` (from Prediction A's fit, the log-normal identity
`log(E[X]) - log(median(X)) = Var[log X]/2` differentiated w.r.t. depth).

- **Pass** iff, pooling all 12 cells:
  1. **Sign match:** `gap_emp > 0` in at least 10 of 12 cells (theory predicts the mean-based
     slope exceeds the median-based one whenever variance is growing with depth).
  2. **Magnitude:** among cells where `gap_theory > 0.001` (a floor excluding
     near-zero-variance-growth cells from a division), the *median* of `gap_emp / gap_theory`
     across those cells falls within `[0.33, 3.0]` (a looser 3x band than the sub-line's
     usual magnitude criterion, since this is a second-order quantity derived from a
     difference of two independently-fit slopes and is expected to be noisier).
- **Fail** otherwise.

**Both A and B pass** -> strong, quantitative confirmation that the threads 12/13 residual
chaotic-phase anomaly is the finite-width log-normal-gradient phenomenon Hanin-Nica predict
-- converts "unexplained systematic bias" into a positive confirmation of next-order theory.
**A passes, B fails (or vice versa)** -> partial support; informative but not a full
confirmation -- Prediction C (below) becomes the tiebreaker between the finite-width and
finite-depth-saturation stories. **Both fail** -> the finite-width explanation is itself
falsified for this residual anomaly; the finite-depth-saturation story (Opus meta-review's
alternative) becomes the leading candidate and would need its own fresh thread with
per-layer tracking to test directly.

### Prediction C (informational, non-gating): per-layer forward-statistic diagnostic

At the single most anomalous cell (`sigma_w2=2.2`, `width=32`, depth=362, the thread-13
worst-magnitude point) and a handful of seeds (10), track `phi'(pre-activation)^2` averaged
per layer across the full depth. If per-layer `E[phi'^2]` stays close to the fixed-point-
predicted value throughout (no systematic drift toward zero/saturation at later layers) even
while per-seed *gradient* variance explodes, that favors the finite-width story (the
nonlinearity isn't structurally changing; only its finite-sample gradient statistics are).
If `E[phi'^2]` visibly drifts/saturates at later layers, that favors the finite-depth
-saturation story instead. Reported alongside A/B but does not gate the pass/fail verdict --
this is a single-cell qualitative disambiguator, not a swept quantitative claim.

## Explicitly out of scope

- Not re-running the ordered phase (`sigma_w2 <= 1.9`) -- thread 13 already found good
  agreement there; this thread targets only the anomalous chaotic branch.
- Not the orthogonal-init arm from idea I5 (dynamical-isometry init, Xiao et al.
  arXiv:1806.05393) -- deferred to a possible follow-up if this thread's result is
  ambiguous between the finite-width and finite-depth stories; not needed if A+B already
  distinguish them.
- Not fixed-point-matched inputs (`q_0=q*`) -- also an I5 sub-idea, same deferral logic.
- No training loop, no LR sweep -- this is a forward/backward structural diagnostic exactly
  like threads 12/13.

## Minimal experiment / compute budget

Timing pilot (not a separate script -- see commit message for the numbers): single-cell
cost at the largest depth (362) ranges from ~2ms (width=32) to ~5ms (width=128); a full
16-depth grid at one `sigma_w2`/width takes 0.14s (width=32) to 0.36s (width=128) for one
seed. Extrapolated full-grid estimate (4 `sigma_w2` x 50 seeds x each width): ~29s
(width=32) + ~40s (width=64) + ~73s (width=128) =~ 142s total, comfortably under the
review's "< 1 CPU-hr" budget for this idea and this repo's usual per-experiment ceiling.

## Bitter-lesson check

Not a novel-architecture claim -- a measurement of an established next-order correction
(finite-width perturbation theory) to a theory this repo already tested. Value is purely
diagnostic: distinguishing which of two named, literature-grounded mechanisms explains an
already-observed residual anomaly.

## Known prior work / risk of reinventing

Hanin (2018), Hanin & Nica (2018), Roberts-Yaida-Hanin (2021) -- established finite-width
perturbation theory for deep nets at initialization, not novel here. The log-normal
mean/median identity used in Prediction B is elementary log-normal-distribution algebra, not
a new result. Novelty here is applying this specific, already-published correction to
explain this repo's own specific residual anomaly, with a pre-registered, falsifiable
quantitative test rather than treating it as qualitative color commentary.

## Status

Not yet run. Pre-registered 2026-07-07 (session-label date, per this repo's now-standard
doc-date-vs-commit-date convention -- see `docs/reviews/2026-07-07-portfolio-review.md`'s
bookkeeping note).
