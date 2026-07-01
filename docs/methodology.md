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
   - **Matched compute.** Report accuracy/loss vs. FLOPs, not just vs. params. If the
     theory's mechanism is expected to change FLOPs/param (e.g., structured matrices), say
     so up front and report both axes.
   - **Matched tuning effort.** Sweep learning rate (log-spaced, >=5 points) and any
     theory-implied hyperparameter for *both* the novel variant and the baseline. No
     hand-picked LR for one side only.
   - **Multi-seed.** >=3 seeds minimum before a result counts as "supported." Report
     mean +/- spread, not a single curve.
   - **Ablate one assumption at a time.** If the theory says "X because of stability
     property Y," the experiment must isolate Y (e.g., compare matched-parameter-count
     variants that differ only in the stability-relevant parameterization) rather than
     bundling in unrelated changes.
4. **Verdict.** One of:
   - *Falsified* — prediction did not hold; write down what broke and why, keep the doc
     (negative results are the actual product of this repo).
   - *Supported so far* — prediction held under the minimal experiment; eligible for a
     scaling check.
   - *Inconclusive* — the experiment couldn't distinguish the prediction from noise/
     baseline; this usually means the prediction wasn't sharp enough — go back to step 2.
5. **Scaling check** (only for "supported so far" threads). A 4-6 point sweep across model
   size (and/or data size), fit a simple trend (power law or the theory's own predicted
   functional form) for both the novel variant and baseline. We are checking whether the
   *sign* of the effect and its rough trend survive growth, not chasing a new best number.
   This is the cheap proxy for "would this still matter at frontier scale" — the same
   logic scaling-law and muP papers use to make claims about regimes they don't directly
   train in.

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
