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

**Post-hoc note, 2026-07-07 (`scripts/thread13_robust_gradient_flow.py`): falsified on the
pre-registered joint criterion — but with the strongest partial support of the three
attempts in this sub-line. Sub-line closed as pre-registered; no fourth estimator
variant.**

Ran the exact pre-registered protocol: same `sigma_w2`/depth grid as thread 12, 50 seeds,
Theil-Sen robust regression (median of pairwise slopes on per-depth median `log(grad_norm)`
across seeds) instead of ordinary least squares.

```
sigma_w2   xi_theory   L (Theil-Sen)   ratio
1.30       4.82        8.14            1.690
1.40       6.00        9.90            1.651
1.60       10.15       14.64           1.442
1.80       23.07       25.89           1.122
1.90       51.92       39.43           0.759
2.05       73.85       138.23          1.872
2.20       23.19       92.30           3.981   <- only band violation
2.40       12.71       34.19           2.691
2.80       7.13        16.73           2.344
```

**Shape criterion: PASSES.** `log(L)` vs. `log(xi_theory)` correlation = 0.872 (need
>=0.8), peak `L` at `sigma_w2=2.05` (in the required `{1.9, 2.05}` set) — a real
improvement from thread 12's 0.524 correlation and wrong-location peak. **Magnitude
criterion: FAILS**, by one interior point: `sigma_w2=2.2` gives ratio 3.981 against the
`[1/3, 3]` band (all 6 other interior points are comfortably inside it) — down from thread
12's 36.7x outlier at the same point, but still outside the pre-registered band.
**Overall: FAIL**, since both criteria are jointly required and one interior point misses.

Sent the result to an independent Opus 4.8 review before finalizing, per this repo's
process. The review reproduced every number exactly (full grid re-run, deterministic given
the fixed seeding) and independently re-implemented `theil_sen_slope` to check it against a
vectorized reference — matched to 6 decimals, confirming the estimator itself is correctly
implemented, not a bug inflating or deflating the result.

**The review found the `sigma_w2=2.2` failure is not an isolated one-point fluke — it's
the visible edge of a systematic chaotic-phase bias.** Comparing the theory's own per-layer
log-slope (`log(chi_1)`) to the empirical Theil-Sen slope across the whole chaotic branch:
the empirical slope is systematically shallower than theory at every chaotic-phase point
tested — undershooting by ~2.7x at `sigma_w2=2.4`, ~4x at `sigma_w2=2.2`, and actually
**flipping sign** at `sigma_w2=2.05` (theory predicts positive/growing, the empirical
Theil-Sen slope came out slightly negative). The review traced this directly to the
mechanism thread 12's review already found: at large depth in the chaotic phase, the
per-seed `log(grad_norm)` distribution develops a heavy left tail (at `sigma_w2=2.2`,
depth 362: median -0.75 but min -9.9, max +6.6; std rises from 0.06 at depth 2 to 3.4 at
depth 362) and the *median itself* turns over non-monotonically (rises to depth ~90, then
sags). Theil-Sen substantially tames the effect of the heavy tail (that's why the worst
ratio dropped from thread 12's 36.7x to 4x here, and why the shape criterion now passes
cleanly) — but it cannot remove a genuine turnover in the underlying median curve, which
flattens the net fitted slope toward zero regardless of estimator robustness.

**Important nuance the review flagged, worth stating precisely:** the magnitude
criterion's interior-xi window (`[5, 60]`) happens to exclude `sigma_w2=2.05` — which is
where theory and the empirical estimate disagree *most* (the sign flip). So "`sigma_w2=2.2`
is the one failure" is partly an artifact of which points the window happens to test; the
underlying chaotic-phase slope undershoot is systematic across the branch (2.7x-4x
undershoot at 2.2/2.4, sign flip at 2.05), not a single localized miss. The review also
checked an alternative aggregation (`log(mean(grad))` instead of `median(log(grad))` —
arguably more theory-faithful, since the theory is stated in terms of `E[grad^2]`) and
found it pulls `sigma_w2=2.2` into the band (ratio 2.53) but blows up `sigma_w2=2.05` to
13x instead — confirming no single untuned point estimator cleanly passes every point here;
this is a genuine systematic effect, not an estimator away from working.

**Verdict, adopting the review's framing: falsified on the pre-registered joint criterion —
with the strongest partial support of the three attempts in this sub-line.** Shape went
from failing (thread 12: 0.524, wrong peak) to a clean pass (0.872, correct peak).
Worst-point magnitude went from 36.7x (thread 12) to 3.98x (this thread) — a real,
substantial improvement from switching to robust regression, just not enough to clear the
pre-registered band. This is not "the theory is wrong here" so much as "the depth-scale
*ordering and peak location* now clearly track theory, but a real systematic bias in the
chaotic-phase backward pass (heavy-tailed large-depth variance plus a non-monotonic
turnover) prevents a clean magnitude match with any untuned point estimator tried so far."

**Per the pre-registered plan, this was the last planned attempt in this measurement-
refinement sub-line (thread 2's loss metric -> thread 12's OLS fit -> this thread's
Theil-Sen fit) — closing it here rather than trying a fourth estimator variant.** The
residual failure is not the kind of thing a different regression trick would likely fix
(the review found no single untuned estimator passes every point); a genuine next attempt
on this idea needs a structurally different measurement — e.g. a task whose inputs start
closer to the theory's fixed point (avoiding the transient/turnover regime this task's
near-orthogonal one-hot inputs sit in), or per-layer rather than per-model gradient
tracking — and its own fresh thread doc, matching how the gate-family sub-line closed after
three attempts (`docs/threads/11-dual-gate-spectral-recurrence.md`).
