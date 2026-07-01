# Thread 12 (follow-up to thread 2): gradient-flow depth scale vs. mean-field `xi`

**Math source:** same as thread 2 (mean-field theory / statistical mechanics, order-to-
chaos phase transition; Poole et al. 2016, Schoenholz et al. 2017). This thread changes
the *operationalization* of "trainable depth" — from "does a fixed-budget Adam run hit a
loss target" to "what is the actual signal-propagation length scale" — not the underlying
theory or model. Per this repo's own rule, a differently-operationalized claim gets its own
pre-registration rather than a retrofit of thread 2's falsified prediction.

## Why this thread exists (disclosure of prior exploratory numbers)

Thread 2's pre-registered prediction A (empirical loss-based trainability boundary matches
`xi(sigma_w2)`) was falsified as specified: see
`docs/threads/02-criticality-guided-init.md`'s 2026-07-07 addendum. An independent Opus
review, and a follow-up spot-check I ran myself, both found that when the *right* quantity
is measured instead — the init-time gradient-flow decay/growth length (fit
`log||d loss/dW_1||` vs. depth, not raw magnitude and not loss-reaching within a step
budget) — the length constant peaks at criticality with the correct decay-to-growth sign
flip, and lands roughly within a factor of ~1.3-2.4x of theory `xi` at several points
tested (with one noisy outlier at ~9x, traced to per-seed init-gradient variance dominating
the depth trend at that specific `sigma_w2` with only 12 seeds).

**Full disclosure, since this creates an obvious risk of the band below being reverse-
engineered from already-seen numbers:** those numbers exist, from an informal 5-point
spot-check (12 seeds each) done during thread 2's closeout, not from a systematic run under
this thread's own protocol. To keep this a real test rather than a confirmation of numbers
already in hand, this thread (a) uses a different, larger sigma_w2 grid than the 5 points
already spot-checked (some overlap is unavoidable near the bracketing extremes, but the
grid below was chosen for even xi coverage, not to reuse exact prior points), (b) uses 30
seeds per cell instead of 12 specifically to resolve the noise problem the spot-check
diagnosed, and (c) sets the pass/fail band from precedent (thread 1's established
factor-of-2 bound for a structurally similar "ratio within a constant factor" claim) rather
than fit to the 1.3-2.4x range already observed — loosened to a factor of 3, justified
below, not because 3x happens to comfortably cover the already-seen numbers, but because
this measurement (a fitted slope over noisy per-seed gradient norms) is intrinsically
noisier than thread 1's closed-form `ratio_first_over_last`, which had no fitting step at
all.

## Architectural hypothesis

Unchanged from thread 2: plain, unnormalized, tanh MLP (`models/deep_mlp.py`), per-layer
init variance `(sigma_w2, sigma_b2)`. No new model code.

## Falsifiable prediction (pre-registered)

Fixed `sigma_b2=0.1` (same as thread 2, for direct comparability). Sigma_w2 grid: `{1.3,
1.4, 1.6, 1.8, 1.9, 2.05, 2.2, 2.4, 2.8}` (9 points, chosen for roughly even `log(xi)`
coverage on both sides of the theory's critical point `sigma_w2*=1.9861`, and restricted to
`xi` in the ~5-75 range so the depth grid below can actually resolve the decay/growth
trend — thread 2's review flagged that points with `xi` in the hundreds-to-billions range
exceed what's resolvable with a CPU-feasible depth grid, so this thread doesn't attempt
them). Depth grid: `{2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256, 362}` (16
points, geometric ratio ~sqrt(2), spanning well past `5x` the largest `xi` in the grid).
30 seeds per `(sigma_w2, depth)` cell. Metric: at init (no training — this is a pure
forward+backward-pass diagnostic), first-hidden-layer weight gradient norm from a fixed
64-example batch of the modular-arithmetic task (p=17, same task as thread 2). For each
`sigma_w2`, fit `log(grad_norm)` vs. `depth` by linear regression across the per-seed
values (not just the per-depth mean — the fit uses all `16 depths x 30 seeds` points
together, both for the point estimate and to get a bootstrap-style spread), giving an
empirical decay/growth length `L = 1/|slope|`.

**Pass (two joint criteria, both must hold):**
1. **Shape:** across the 9-point `sigma_w2` grid, `log(L)` vs. `log(xi_theory)` has Pearson
   correlation >= 0.8, AND the `sigma_w2` with the largest `L` is adjacent (one grid point
   away or closer) to the theoretical critical point `sigma_w2*=1.9861` (i.e. `1.9` or
   `2.05` in this grid).
2. **Magnitude:** for the subset of `sigma_w2` points whose theoretical `xi` is well inside
   the resolvable range of the depth grid (`xi` between 5 and 60, a conservative interior
   band avoiding both ends of the tested depth range) the ratio `L / xi_theory` falls within
   a factor of 3 of 1.0 for every point in that subset (not just on average) — i.e.
   `1/3 <= L/xi_theory <= 3`.

**Fail:** either criterion fails to hold — e.g., no meaningful shape correlation, the peak
lands somewhere other than near the theoretical critical point, or the ratio is wildly
inconsistent (as in thread 2's exploratory ~9x outlier) even after tripling the seed count.

**Explicitly not retrying if this fails:** unlike threads 9→10→11's escalating attempts
within one gate family, if this specific operationalization also fails to show a clean
shape+magnitude match, that's reasonably strong evidence that whatever qualitative
resemblance the exploratory numbers showed doesn't survive a properly-powered systematic
test, and this thread's finding should be logged as falsified without a further immediate
follow-up in this sub-line.

## Explicitly out of scope

- Not attempting the near-critical `xi` > ~100 region — acknowledged as unresolvable within
  a CPU-feasible depth grid, per thread 2's review finding, not chased here.
- Not re-testing the loss-based/training-based operationalization — thread 2's addendum
  already covers that verdict.
- Not sweeping `sigma_b2` — fixed at 0.1 throughout, matching thread 2.

## Minimal experiment

- Model: `models/deep_mlp.py`, unchanged.
- Task: `tasks/modular_arith.py`, p=17, unchanged from thread 2.
- No training loop — single forward+backward pass per `(sigma_w2, depth, seed)` cell, so
  the full `9 x 16 x 30 = 4320`-cell grid is CPU-cheap (~1 minute measured in a timing
  check on one full `sigma_w2` column before committing to the grid).

## Compute budget

Well under a minute of CPU time for the full grid (measured, not estimated) — far under
this repo's per-experiment budget ceiling.

## Bitter-lesson check

Unchanged from thread 2: this is a derivation/diagnostic procedure applied to a general
init-variance knob, not a task-specific mechanism.

## Known prior work / risk of reinventing

Gradient-norm-vs-depth as the empirical signature of vanishing/exploding gradients is
standard (Schoenholz et al. 2017 measure exactly this quantity, calling it "depth scale of
gradient propagation"). Novelty, if any, is only in applying it as a direct pre-registered
falsification test on top of thread 2's already-built harness, not in the diagnostic
itself.

## Status

Not yet run. This doc exists to satisfy `docs/methodology.md`'s pre-registration rule
before any of this thread's analysis code is written or run.

**Post-hoc note, 2026-07-07 (`scripts/thread12_gradient_flow_depth_scale.py`): falsified
exactly as pre-registered — both criteria fail, concentrated entirely in the chaotic
phase.**

Ran the pre-registered protocol exactly: 9 `sigma_w2` points x 16 depths x 30 seeds, single
init-time forward+backward pass per cell, one global `log(grad_norm)` vs. `depth` fit per
`sigma_w2`.

```
sigma_w2   xi_theory   L (empirical)   ratio
1.30       4.82        7.94            1.650
1.40       6.00        9.29            1.549
1.60       10.15       13.74           1.354
1.80       23.07       21.76           0.943
1.90       51.92       29.45           0.567
2.05       73.85       63.93           0.866
2.20       23.19       850.46          36.681   <- anomaly
2.40       12.71       45.18           3.555
2.80       7.13        18.47           2.588
```

**Shape criterion: FAIL.** `log(L)` vs. `log(xi_theory)` correlation = 0.524 (need >= 0.8);
peak `L` at `sigma_w2=2.2`, not adjacent to the theoretical critical point (need `{1.9,
2.05}`). **Magnitude criterion: FAIL.** The `sigma_w2=2.2` ratio (36.7x) and `sigma_w2=2.4`
ratio (3.56x, just outside the band) both violate the pre-registered `[1/3, 3]` requirement
on every interior point. **Overall: FAIL, as pre-registered.**

Notably — worth stating precisely, since it sharpens what this negative result does and
doesn't rule out — **the ordered phase alone (`sigma_w2` <= 1.9) already matches theory
well under this exact fit**: ratios `1.65 -> 1.55 -> 1.35 -> 0.94 -> 0.57`, monotonically
decreasing toward 1 as `sigma_w2` approaches the critical point from below. The failure is
concentrated entirely in the chaotic phase (`sigma_w2` > 1.9861), not a whole-curve
mismatch.

Sent the result to an independent Opus 4.8 review before finalizing, per this repo's
process. The review re-ran a slice itself (15 seeds at `sigma_w2` in `{1.3, 1.9, 2.2,
2.8}`, plus an exact 30-seed reproduction of the 2.2 column) and reproduced every number to
the digit — confirmed not a harness bug, RNG issue, or numerical overflow (0/30 non-finite
values even at the deepest chaotic cells). It also confirmed the correct theory curve
(`chi_1` crosses 1.0 at `sigma_w2*=1.9861`, `xi(sigma_w2)` is V-shaped around it) is what's
being compared against.

**The review corrected my own working explanation for the `sigma_w2=2.2` anomaly.** While
investigating before writing this up, I attributed the large-depth blowup to forward
tanh-derivative saturation ("finite-width trajectories individually saturating tanh's
derivative once compounding chaotic growth pushes pre-activations into the saturating
tail"). The review measured this directly and **found it's false**: the forward
pre-activation saturation fraction at the last hidden layer stays low (~0.01) and flat with
depth at `sigma_w2=2.2` — expected, since the variance map's fixed point `q*` is bounded
(~0.94), so forward activations don't run away. The forward pass isn't the problem. The
actual mechanism, verified directly by the review: **heavy-tailed, seed-to-seed
multiplicative variance on the *backward* pass at large depth in the chaotic phase** — the
log-grad-norm standard deviation across seeds rises from ~0.3 at depth 32 to ~3.0 at depth
256, and the log-mean itself goes non-monotonic (rises to depth ~90, drops at depth 256,
rises again at depth 362). A single global log-linear slope fit over a trend that
rises-then-falls-then-rises collapses toward zero, inflating `L` to the hundreds.

The review also confirmed the transient-vs-asymptotic confound I'd flagged as a risk in
this thread's "explicitly out of scope" reasoning is real and large: theory's `xi` is
strictly an asymptotic near-fixed-point rate, but this task's near-orthogonal one-hot
inputs start far from the fixed point, so early depth is a transient, not the asymptotic
regime — and the pre-registered single global fit (depth 2 to 362) conflates that
transient, a genuine near-asymptotic window, and the large-depth backward-variance-
corrupted regime into one number, dominated by whichever regime has the most leverage (for
chaotic-phase points, that's the corrupted deep tail). As a check on this (exploratory
only — **not used to change the verdict above**), the review refit `sigma_w2=2.2` with the
depth range restricted:

```
window            shape corr   peak    2.2 ratio
2-362 (pre-reg)   0.538        2.2     31-37x
depth <= 90       0.865        2.05    3.15
depth <= 45       0.906        2.05    2.71
```

Restricting to a near-asymptotic-only window would pass the shape criterion and bring the
2.2 outlier close to (though still slightly outside) the magnitude band — suggesting the
underlying theory-resemblance may be real and the failure here is specifically in the
*estimator* (one global slope averaging three different regimes), not necessarily in the
theory's applicability.

**Verdict: falsified as specified, per the pre-registered plan — no do-over under this
thread's label.** The window-restricted numbers above are a diagnosed confound, not a
result; per this repo's own discipline they don't get to retroactively pass this thread.
If this is pursued further, it needs its own fresh pre-registration with a differently
defined estimator (e.g. a near-asymptotic-only fit window, or a piecewise/robust fit)
specified *before* running — chosen for principled reasons stated in advance, not because
it happens to recover the pattern already seen here.
