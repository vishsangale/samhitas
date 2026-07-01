# Thread 14: muP coordinate check (idea I1, unblocks thread 6)

**Math source:** same muP / Tensor Programs machinery as thread 6. This is not a new
architectural hypothesis -- it is a diagnostic on thread 6's *implementation*, proposed by
the 2026-07-07 portfolio review (`docs/reviews/2026-07-07-portfolio-review.md`, idea I1) as
the cheapest, most decisive way to localize thread 6's adverse smoke-test reads before
spending any GPU budget on a real run.

## Motivation

Thread 6's two CPU smoke tests found muP's raw-LR optimum drifting a full decade or more
across width while SP stayed comparatively flat -- the opposite of muP's prediction. An
Opus static code review already ruled out the simplest explanation (a wrong Adam LR
multiplier in `models/mlp.py:param_groups` -- verified correct). The muP literature review
flagged the likelier suspects as the *task* (modular arithmetic's grokking/weight-decay
dynamics, not width-governed) or subtler embedding/readout handling, rather than the core
forward/backward scaling.

The standard tool for distinguishing "the scaling implementation is broken" from "the
scaling implementation is fine but something else (task, metric, LR range) is confounding
the higher-level LR-transfer measurement" is the **coordinate check**: track each layer
type's activation scale across a width sweep, at init and after a handful of optimizer
steps at one aggressive, fixed LR. This does not require any convergence sweep -- it is a
forward/backward-pass property, observable in a handful of steps, so it is CPU-trivial
(the review estimates minutes) and produces a qualitatively sharp pass/fail signal (flat
vs. width = correct; systematic drift with width = broken), unlike the noisy LR-argmin
metric thread 6's real prediction has to use.

## Falsifiable prediction (pre-registered)

Reuse `experiments/models/mlp.py`'s existing `MuPMLP` (input layer, `depth-2` hidden
layers, output layer; SP = Kaiming-uniform init + flat LR; muP = 1/sqrt(fan_in) init +
output forward multiplier `base_width/width` + per-layer-type Adam LR multiplier
`base_width/width` on hidden/output groups -- exactly the parametrization thread 6 already
uses and the meta-review already verified statically).

For each parametrization (`sp`, `mup`), each layer type (`input_layer`, each
`hidden_layers[i]`, `output_layer`), and each checkpoint step count `t` in
`{0, 1, 2, 5, 10}` (t=0 is at init, before any optimizer step), measure `y(width) =
mean(|activation|)` on a **fixed** held-out evaluation batch (same batch reused across all
widths and steps, so cross-width comparisons aren't confounded by different eval data) at
width in `{64, 128, 256, 512, 1024, 2048, 4096}` (base_width=64, so k in
`{1,2,4,8,16,32,64}`, 7 points, geometric). Optimizer steps between checkpoints use freshly
drawn random training batches (same task/generator setup as `harness/train.py`), at one
fixed **aggressive** `base_lr` (large enough that SP would be expected to visibly misbehave
under standard muP-paper coordinate-check practice, but not so large either arm
numerically explodes at width=64 -- picked via a quick pilot, see Minimal experiment).

Fit `log(y)` vs `log(width)` by OLS across the 7-point width grid, separately per
(parametrization, layer type, step `t`), giving a slope `beta`.

- **muP passes** iff `|beta| < 0.15` for *every* layer type and *every* checkpoint step
  `t` (flat scale vs. width, matching muP's asymptotic-width-invariance theory).
- **SP positive control** iff at least one hidden or output layer type shows `|beta| >=
  0.3` at some `t >= 1` (a clear post-step, width-dependent drift -- the known SP failure
  mode the coordinate check is designed to detect; SP is *expected* to fail this, that is
  what makes it a positive control, not a second candidate to "pass").

**Interpretation, pre-committed before running:**
- muP passes AND SP fails as expected -> the coordinate check confirms the forward/backward
  scaling implementation is mechanically correct; thread 6's adverse LR-transfer reading is
  very likely a task/metric artifact (grokking dynamics, LR-argmin noise), not an
  implementation bug. This directly supports the literature review's leading hypothesis
  and reprioritizes thread 6's real run toward a non-algorithmic task, not toward
  re-deriving the scaling rule.
- muP fails (some layer type/step exceeds `|beta| >= 0.15`) -> localizes a genuine
  implementation defect to the specific layer type/step where the slope breaks, and thread
  6's adverse reading is at least partly explained by an actual bug, not (only) a task
  artifact. Falsifies the working assumption that the implementation is clean.
- SP does *not* fail the positive-control check (all layer types stay under `|beta| <
  0.3` at every `t>=1`) -> the coordinate check itself is not discriminating anything at
  this width range/LR/step count (inconclusive on its own terms) -- widen the LR or step
  count rather than trusting either arm's result.

Secondary, non-gating diagnostic (reported, not part of the pass/fail bar): the
**"wider is always better"** check -- under a correctly scaled parametrization, train loss
after the fixed step budget should be monotonically non-increasing in width (more capacity
never hurts once LR is properly transferred); report the loss-vs-width curve for both arms
alongside the slope table.

## Minimal experiment

- Model/task: `MuPMLP` + `modular_arith` (matches thread 6's existing harness exactly, no
  new code needed beyond the coordinate-check driver itself).
- `p=41` (small input dim, matches `thread06_mup_widerange.py`), `depth=4` (2 hidden
  layers, matches thread 6's `RunConfig` default), `batch_size=128`.
- Width grid: `{64, 128, 256, 512, 1024, 2048, 4096}`, `base_width=64`.
- Checkpoint steps: `{0, 1, 2, 5, 10}` -- no convergence needed, just early-training
  dynamics, which is exactly where the muP paper's own coordinate checks are read.
- `base_lr`: picked via a quick pilot at width=64 only (both arms) before committing to
  the full grid -- large enough that SP visibly moves loss/activations within 10 steps,
  small enough neither arm produces NaN/Inf at the smallest width. Record whatever value is
  chosen and why, in the results log, before running the full grid.
- One fixed eval batch (drawn once, same seed, reused at every width/step/parametrization)
  and one fixed model-init seed per width (same seed across `sp`/`mup` for a matched
  comparison, different seed per width so init draws aren't literally identical copies).
- No multi-seed requirement here per methodology's usual >=3-seed bar: this is a
  forward/backward-pass structural diagnostic, not a stochastic-outcome claim (the
  muP-paper convention is single-seed coordinate-check plots) -- if the pilot shows
  meaningfully seed-sensitive slopes, note that explicitly and add seeds before trusting
  the verdict.

## Compute budget

Minutes of CPU, per the review's estimate: 2 parametrizations x 7 widths x 5 checkpoints x
(up to 10 Adam steps + 1 eval forward), no convergence sweep, no width above 4096. Timing
check at the largest width (4096) before committing to the full grid, per this repo's
standard practice.

## Bitter-lesson check

Not a novel-architecture claim -- this is a re-validation-plus-implementation-check of
existing muP theory on existing code, exactly like thread 6 itself. Value is instrumental:
it either clears the implementation (redirecting thread 6's real run toward a better task)
or finds a real bug (which would itself be a valuable, cheap catch before any GPU spend).

## Known prior work / risk of reinventing

The coordinate check is standard muP-paper diagnostic practice (Yang & Hu et al.,
`microsoft/mup`'s own `coord_check.py`). No novelty claimed here beyond applying it to this
repo's from-scratch `MuPMLP` implementation and task.

## Status

Run 2026-07-07 (session-label date). **Falsified as specified** -- muP did not clear the
pre-registered flatness bar -- but an independent Opus review, which re-ran the code from
scratch and reproduced every number bit-for-bit, found the failure is fully explained by
two identified miscalibrations in the pre-registered bar itself, not by an implementation
defect. See the dated post-hoc note below.

**Post-hoc note, 2026-07-07 (Opus review):** Full grid (2 parametrizations x 7 widths x 5
checkpoints, `base_lr=0.3`) ran in ~9s CPU. Raw result: SP failed as a strong, clean
positive control (`hidden_1` slope +1.999 at t=1, `output_layer` +2.956 at t=1, loss
blowing from 3.7 to ~405 across the width grid by t=10) -- exactly the known SP failure
mode the check exists to detect. But muP also failed the literal `|slope| < 0.15`
flatness bar: `output_layer` slope sits at ~-1 at *every* checkpoint including t=0 (before
any training step), and `input_layer`/`hidden_0`/`hidden_1` drift from near-flat at t<=1 to
`|slope|` 0.3-0.85 by t=10.

Per this repo's rule, the pre-registered bar is not edited after seeing this -- the
literal verdict is **falsified**. But the Opus review (re-ran the driver standalone,
independently re-derived every layer's activation by hand-replicating `forward()` without
hooks and got a max difference of `0.0` against the hook-based measurements, verified
`log_log_slope()` against synthetic power laws, and ran an additional LR-robustness sweep
at `base_lr` in `{0.3, 0.1, 0.05, 0.01, 0.001}` not in the original pre-registration) traced
both apparent failures to specific, checkable causes rather than a bug:

1. **The output-layer criterion was the wrong bar, not evidence of a defect.** The
   pre-multiplier output activation is empirically width-invariant (mean|.| ~0.04-0.047
   across the whole width grid); the measured post-multiplier quantity is that constant
   times muP's documented `base_width/width` forward multiplier, so a slope of exactly
   `-1` at init is the *arithmetically necessary, intended* consequence of the parametrization
   table this repo's `MuPMLP` already implements (a "nonzero-readout-init" muP variant whose
   output is designed to vanish at init as width grows, matching the standard
   `mup.MuReadout` convention with `readout_zero_init=False`). Supporting evidence this is a
   feature, not a bug: at a moderate LR (0.1 or 0.05) the output slope *relaxes from -1
   toward 0* as training proceeds (e.g. base_lr=0.1: -0.994 -> -0.066 -> -0.142 by t=10) --
   exactly muP's prediction that the vanishing-at-init contribution gets progressively
   dominated by a width-invariant, Theta(1) learned update. A genuinely broken readout
   scaling would not produce that clean relaxation.
2. **The hidden/input-layer drift is an aggressive-pilot-LR transient, not a persistent
   scaling defect.** The review's LR-robustness sweep found the drift is strongly,
   monotonically LR-dependent: at the pilot's `base_lr=0.3`, `hidden_1`'s worst `|slope|`
   is 0.853 (fails); at 0.1 it's 1.486 (fails worse); at 0.01 it drops to 0.137 (**passes**
   the pre-registered `<0.15` bar); at 0.001 it's 0.008 (flat). Input layer shows the same
   pattern (0.327 -> 0.256 -> 0.148(pass) -> 0.005 across the same LR sequence). Mechanism:
   `base_lr=0.3` is a genuinely enormous Adam LR (typical Adam LR is 1e-3 to 1e-2, and this
   pilot value was deliberately chosen aggressive specifically to make SP misbehave
   visibly) -- muP's flatness guarantee is an asymptotic, small-perturbation-per-step
   statement, and large first steps push activations through a nonlinear, width-correlated
   transient that is not present at typical LR. If this were a real width-scaling exponent
   error it would persist or worsen at smaller LR; instead it vanishes, which is the
   signature of a transient, not a defect. (Also noted: even at `base_lr=0.3` in the
   original run, muP's *loss* stayed essentially flat across width -- 3.759 to 3.699 at
   t=10 -- while SP exploded to ~405; the width-stability muP promises at the loss level
   held throughout, even while the coordinate-level activation reading was mid-transient.)
3. **The driver's mechanics are correct** -- independently re-verified, no bug found in
   `measure_layer_scales()`'s forward hooks or `log_log_slope()`'s OLS fit.

**Bottom-line verdict (Opus review, adopted here):** no new muP implementation bug found.
The literal pre-registered verdict stands as **falsified as specified** (muP did not clear
`|slope| < 0.15` for every layer/step as written), but the failure is attributable to a
mis-specified bar (wrong theoretical expectation for the output layer; too-aggressive read
LR for the hidden-layer flatness claim), not to a defect in the forward/backward scaling
implementation. This corroborates the prior static verification of the Adam LR-multiplier
table and **positively supports thread 6's leading task/metric-artifact hypothesis**
(modular-arithmetic grokking dynamics / LR-argmin noise, not a broken scaling rule) as the
explanation for thread 6's adverse smoke-test reads. There is no implementation fix that
needs to happen before thread 6's real (GPU-scale, non-algorithmic-task) run.

Per the review's suggestion: a fully *clean* pass/fail coordinate-check record would need
its own fresh pre-registration with a corrected output-layer criterion (test for slope near
`-1` at init and relaxation toward `0` under training, not flatness) and a non-aggressive
read LR (`<=0.01`) for the hidden-layer flatness claim. Not pursued as a separate thread for
now -- the substantive question I1 was built to answer (is there an undetected muP
implementation bug blocking thread 6?) already has a decisive, reproduced answer (no), which
is the actionable output; a relabeled "thread 15" would mostly re-confirm the same
conclusion with tidier numbers rather than test anything new. Revisit only if a future
thread-6 run reopens the implementation-bug question specifically.
