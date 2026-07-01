# Thread 13 (second follow-up to thread 2 / thread 12): robust-regression gradient-flow
depth scale

**Math source:** same as threads 2 and 12 (mean-field theory, order-to-chaos phase
transition). This is the third pre-registered attempt in this specific measurement-
refinement sub-line (thread 2's loss-based metric -> thread 12's OLS gradient-flow-length
metric -> this thread's robust-regression gradient-flow-length metric). Per this repo's own
precedent (the gate-family sub-line, threads 9->10->11, took three attempts before
closing), **this is the last planned attempt on this specific sub-line** — if it also
fails, the honest conclusion is to close the sub-line as a negative result rather than try
a fourth estimator variant.

## Why this thread exists, and what was ruled out first

Thread 12 (`docs/threads/12-gradient-flow-depth-scale.md`) pre-registered and ran a global
ordinary-least-squares fit of `log(grad_norm)` vs. `depth` (2 to 362) per `sigma_w2`,
comparing the fitted length to theory's `xi`. It failed both criteria, but an independent
Opus review found the failure concentrated entirely in the chaotic phase and traced it to
**heavy-tailed, seed-to-seed multiplicative variance on the backward pass at large depth**
(log-grad-norm std across seeds rising from ~0.3 at depth 32 to ~3.0 at depth 256, with the
log-mean itself going non-monotonic) — a single global OLS slope is not robust to that kind
of outlier-heavy, non-monotonic corruption at a handful of large-depth points.

Before designing this thread, I checked (and am disclosing) two things that did NOT pan
out, to be transparent about what actually motivates the design below rather than
retrofitting a story to a window that already worked:

1. **A "transient before reaching the asymptotic regime" explanation, via the
   *correlation* map.** `harness/meanfield.py` now has `correlation_map()` and
   `transient_depth_to_fixed_point()` (cross-checked: `C(1)=1` exactly, and its
   finite-difference slope at `c=1` matches the closed-form `chi_1` to 5 decimal places).
   But iterating the actual correlation map from this task's near-orthogonal starting
   correlation (`c0~0`) revealed that **`c=1` is a repelling fixed point in the chaotic
   phase** (`chi_1>1` means the map's derivative at `c=1` exceeds 1) — trajectories move
   *away* from `c=1` there, converging instead toward some other, smaller stable
   correlation. So "depth to reach `c~1`" is well-defined only in the ordered phase and
   doesn't even apply in the chaotic phase; it's also the wrong recursion for this
   question in the first place — the correlation map governs two-input
   representational-distinguishability decay, a different question from single-input
   backward gradient magnitude scaling.
2. **The actually-relevant transient — the *variance* map's own convergence to `q*`,
   which governs a single input's forward pre-activation distribution and is what the
   per-layer backward multiplicative factor `sigma_w2 * E[phi'(sqrt(q_l) z)^2]` actually
   depends on — converges in about 6 layers across the entire `sigma_w2` grid tested,
   regardless of phase.** Far too short to explain a failure that persists out to
   depth 100+. Ruled out as the mechanism.

Given neither transient story holds up, the design below targets the mechanism the review
actually found (heavy-tailed large-depth noise) directly, with a standard tool for it
(robust regression), rather than picking a depth window by trial.

## Falsifiable prediction (pre-registered)

**Same `sigma_w2` grid, same depth grid, same task, same model as thread 12** — `{1.3, 1.4,
1.6, 1.8, 1.9, 2.05, 2.2, 2.4, 2.8}` x `{2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128,
181, 256, 362}` — specifically to keep this a fair test of a *different estimator* on the
*same data-generating design*, not a search over depth windows. Two changes from thread 12:

1. **50 seeds per cell instead of 30** (more seeds directly reduces the standard error of a
   median-based statistic; decided in advance as a generic power increase, not tuned to any
   observed ratio).
2. **Theil-Sen robust regression instead of ordinary least squares.** For each `sigma_w2`:
   compute the per-depth *median* `log(grad_norm)` across the 50 seeds (first level of
   robustness — a median at each depth is insensitive to a single extreme seed), then take
   the *median of all pairwise slopes* between the 16 depth-median points (Theil-Sen — a
   second level of robustness, standard practice for regression with a minority of
   corrupted/outlier points, breaks down only if more than ~29% of the pairwise slopes are
   themselves corrupted). Empirical length `L = 1 / |Theil-Sen slope|`.

**Pass (both criteria, unchanged structure from thread 12):**
1. **Shape:** `log(L)` vs. `log(xi_theory)` Pearson correlation >= 0.8 across the 9-point
   grid, AND the `sigma_w2` with the largest `L` is adjacent to the theoretical critical
   point (`sigma_w2*=1.9861`), i.e. in `{1.9, 2.05}`.
2. **Magnitude:** for `sigma_w2` points with `xi_theory` in `[5, 60]` (the same
   "resolvable interior" band as thread 12), `L / xi_theory` falls within a factor of 3 of
   1.0 for every point in that subset.

**Fail:** either criterion fails.

**This is the last planned attempt on this measurement-refinement sub-line.** If this also
fails, log it as a negative result for the sub-line (the loss-based, OLS-gradient-flow, and
robust-gradient-flow operationalizations all failed) without a fourth estimator variant —
any further attempt on this specific idea (mean-field-predicted depth scale for this task)
would need a structurally different measurement (e.g. a genuinely different task whose
inputs start closer to the theory's fixed point, or per-layer rather than per-model
gradient tracking) and its own fresh thread, matching how the gate-family sub-line closed
after three attempts (`docs/threads/11-dual-gate-spectral-recurrence.md`).

## Explicitly out of scope

- Not re-deriving a transient-depth-based window (ruled out above).
- Not sweeping `sigma_b2`, hidden width, or the task — held fixed for direct comparability
  to threads 2 and 12.
- Not attempting the near-critical `xi` > ~100 region — same reason as before (unresolvable
  within a CPU-feasible depth grid).

## Minimal experiment

- Model/task: unchanged (`models/deep_mlp.py`, `tasks/modular_arith.py`, p=17).
- No new model code. `harness/meanfield.py`'s existing `chi_1`/`depth_scale` are reused
  unchanged; the new `correlation_map`/`transient_depth_to_fixed_point` additions are kept
  in the harness as validated infrastructure but are not central to this thread's protocol
  (see "why this thread exists" above for why that avenue didn't pan out).
- Compute: 9 x 16 x 50 = 7200 single forward+backward passes (no training loop) — thread
  12's 9x16x30 grid took ~1 minute; this is ~1.7x more cells, still well under a couple of
  minutes, to be timed before the full run regardless.

## Compute budget

CPU-trivial, well under this repo's per-experiment ceiling.

## Bitter-lesson check

Unchanged from threads 2/12 — a general diagnostic procedure, not task-specific machinery.
Theil-Sen itself is a standard, decades-old robust-regression method, not something
invented to rescue this result.

## Known prior work / risk of reinventing

Theil-Sen estimator (Theil 1950, Sen 1968) is standard robust regression, chosen here
specifically because it is provably robust to up to ~29% corrupted points without any
tuning — a direct, principled response to the review's diagnosed "heavy-tailed noise at a
subset of large-depth points" mechanism, not a novel technique.

## Status

Not yet run. This doc exists to satisfy `docs/methodology.md`'s pre-registration rule
before any of this thread's analysis code is written or run.
