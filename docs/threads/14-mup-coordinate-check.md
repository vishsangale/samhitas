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

Not yet run. Pre-registered 2026-07-07 (session-label date; see `docs/reviews/
2026-07-07-portfolio-review.md`'s bookkeeping note on doc-date-vs-commit-date drift).
