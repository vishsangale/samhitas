# Methodology: the falsification loop

## The loop

1. **Theory.** State the mathematical structure being imported and the derivation from it
   to an architectural consequence (layer, init, parameterization, scaling rule). This
   should be short enough to write on a page; if it needs ten pages, it's probably not a
   single testable hypothesis yet.
2. **Prediction.** Extract a quantitative, falsifiable prediction. Bad: "should help
   long-range tasks." Good: "trainable depth scales linearly with 1/(spectral radius - 1)
   for the constrained variant, and plateaus by depth 8 for the unconstrained baseline."
   A prediction must specify a measurable quantity, a direction, and ideally a functional
   form or threshold, so it can fail.
3. **Minimal falsification experiment.** The smallest model/dataset/compute budget that
   could break the prediction. Rules:
   - **Matched compute.** Report both FLOPs and wall-clock, not just params — see
     "Compute accounting" below, this is load-bearing enough to warrant its own section.
   - **Matched tuning budget.** Sweep learning rate (log-spaced, >=5 points) and any
     theory-implied hyperparameter for *both* the novel variant and the baseline, with the
     *number* of tuning trials matched between arms — see "Tuning-budget symmetry" below.
   - **Multi-seed.** >=3 seeds minimum before a result counts as "supported." Report
     mean +/- spread, not a single curve. 3 seeds is a floor for estimating a mean, not
     sufficient on its own to claim a *difference* between arms is real — see the
     effect-size criterion below.
   - **Ablate one assumption at a time.** If the theory says "X because of stability
     property Y," the experiment must isolate Y (e.g., compare matched-parameter-count
     variants that differ only in the stability-relevant parameterization) rather than
     bundling in unrelated changes.
4. **Verdict.** One of:
   - *Falsified* — prediction did not hold; write down what broke and why, keep the doc
     (negative results are the actual product of this repo).
   - *Supported so far* — prediction held under the minimal experiment **and** cleared the
     thread's pre-registered effect-size criterion (see below); eligible for a scaling
     check.
   - *Inconclusive* — the experiment couldn't distinguish the prediction from noise/
     baseline, or cleared seed-to-seed noise but missed the pre-registered effect size;
     this usually means the prediction wasn't sharp enough — go back to step 2.
5. **Scaling check** (only for "supported so far" threads). A 4-6 point sweep across model
   size (and/or data size), fit a simple trend (power law or the theory's own predicted
   functional form) for both the novel variant and baseline. We are checking whether the
   *sign* of the effect and its rough trend survive growth, not chasing a new best number.
   This is the cheap proxy for "would this still matter at frontier scale" — the same
   logic scaling-law and muP papers use to make claims about regimes they don't directly
   train in.

## Pre-registration

*(Added after the Opus 4.8 advisor review of draft v0 — see `RESEARCH.md` section 5 for
the full review summary.)*

A thread doc's "Falsifiable prediction" section — the numeric thresholds, functional form,
and pass/fail band — must be written and committed to git *before* the corresponding
experiment is run. If a result suggests the original threshold was mis-calibrated, don't
edit it in place after seeing the data: add a dated addendum paragraph ("Post-hoc note,
<date>: ...") explaining what changed and why, and re-run under the revised prediction if
you want the new version to count as a fresh verdict. This is what stops "the ratio
loosely tracks the predicted order" from becoming an adjustable-after-the-fact judgment
call — every thread doc in this repo as of the v0 review now states its numeric prediction
and pass/fail band up front, before any experiment code exists for it.

## Compute accounting: FLOPs and wall-clock

FLOP-matching is doing a lot of unexamined work whenever a comparison spans structurally
different operations (FFT vs. dense attention vs. parallel scan vs. multi-stage
integrators vs. K-FAC-preconditioned updates) — and FLOPs do not track real hardware
utilization (FFT and gather/scatter-heavy ops have very different efficiency than dense
matmul on the same accelerator). Every comparison must report **both**:

- an analytic FLOP count, with the counting method stated (which ops are counted, whether
  the count is hand-derived or came from a profiler), and
- measured wall-clock on the actual hardware used for the run.

A result that only wins on analytic FLOPs but loses on wall-clock is not a hardware-
plausible win — it fails the bitter-lesson hardware-friendliness check regardless of what
the FLOP count says.

## Tuning-budget symmetry and effect-size criterion

The novel arm in a comparison usually has extra theory-implied hyperparameters (a
spectral-radius `eps`, an integrator order, an init-variance offset, a preconditioning
damping term, ...) beyond whatever the baseline has, so sweeping LR "for both arms" can
still leave the novel arm with more total tuning draws than the baseline — a "best of N
configs" inflation that makes the novel arm look better for a reason that has nothing to
do with the theory. Two mitigations, apply both:

1. **Match total tuning trial count between arms.** If the novel arm gets an LR x `eps`
   grid, give the baseline an equivalently sized grid over its own free knobs (even where
   that means arguably over-tuning the baseline).
2. **Report the full sweep surface**, not just the argmax, in the run output, so it's
   possible to see how peaked a win is rather than trusting a single best point.

For the verdict itself: non-overlapping means across the 3-seed floor is not sufficient
evidence that a difference is real. Every thread's pre-registered prediction (see above)
must state an explicit effect-size criterion — e.g., a minimum gap relative to the pooled
seed spread, or a minimum correlation coefficient for ranking-based predictions — and
"supported so far" verdicts require clearing that stated bar, not just non-overlapping
error bars.

## Compute budget discipline

Falsification-phase experiments should fit in roughly a single GPU-day on one consumer or
cloud GPU. If a theory needs more than that to show a first signal, treat that as a signal
the prediction (step 2) isn't sharp/mechanistic enough yet — sharpen it rather than
scaling the experiment up. Scaling checks (step 5) get a slightly larger but still bounded
budget (a handful of small runs, not one big run) — the point is a trend line, not a
record.

## What counts as a "diagnostic task"

Prefer tasks known to mechanistically separate architectures at small scale rather than
generic benchmark accuracy:

- **Associative recall / selective copy** — needs content-based routing over long
  context; famously separates plain RNNs/SSMs from attention and from later SSM variants
  (S4 -> H3 -> Mamba's selection mechanism was validated exactly this way).
- **Modular arithmetic / parity / sparse-symbolic tasks** — cheap, exact ground truth,
  good for testing whether a layer can represent a specific algebraic structure at all
  before asking whether it does so efficiently.
- **Tiny language modeling** (char-level Shakespeare, TinyStories-scale) — for
  loss-vs-compute trend measurement.
- **Small vision classification** (CIFAR-10/100 subsets) — for depth-scaling and
  criticality-related predictions, where convolutional/MLP depth behavior is well studied
  and cheap to probe.

## Reporting format

Every experiment that produces a verdict should leave behind, in `experiments/`: the
config used, the raw metrics (loss/accuracy vs. step, per seed), the fitted trend if a
scaling check was run, and a short verdict paragraph appended to the relevant thread doc.
