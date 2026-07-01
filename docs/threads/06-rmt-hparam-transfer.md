# Thread 6: RMT-guided hyperparameter transfer

**Math source:** random matrix theory (spectral statistics of weight matrices during
training), infinite-width mean-field limits (Tensor Programs / muP).

## Motivation

muP (maximal update parameterization) already shows that a specific per-layer scaling of
init and learning rate lets hyperparameters tuned at small width transfer, in principle
exactly, to much larger width — this is the closest existing precedent to the entire
methodology this repo is built around (falsify/tune cheaply at small scale, trust the
theory to carry the result to larger scale). The open question worth treating as its own
thread: does the same style of derivation extend cleanly to the *other* architectural
mechanisms proposed in this portfolio (structured recurrence from thread 1, integrator
blocks from thread 4, learned-generator layers from thread 5), or does each new mechanism
need its own scaling derivation because it changes the relevant weight-matrix spectral
statistics in a way vanilla muP doesn't cover?

## Architectural hypothesis

For any of this repo's proposed layer types, a per-layer learning-rate/init scaling rule
can be derived from the layer's forward/backward Jacobian statistics (following the same
program as muP) such that hyperparameters swept and selected at a small width transfer,
within a small pre-registered tolerance on final loss, to a width several multiples larger
— without re-tuning.

## Falsifiable prediction

For a chosen layer type (start with the plain baseline to validate the protocol, then apply
to at least one novel layer from another thread), the learning rate that is optimal at
width `W` under the derived scaling rule should also be within a factor of ~2 of optimal at
width `k*W` (k in {4, 8, 16}), measured by directly sweeping LR at each width and comparing
where the minimum lands, whereas a naive (unscaled) parameterization should show the
optimal LR drifting by an order of magnitude or more across the same width range. If the
derived scaling rule's transfer tolerance is no better than the naive parameterization's,
the derivation for that specific layer type is falsified (even if muP itself remains valid
for the layers it was originally derived for).

## Minimal experiment

- Pick one baseline layer type (sanity check against known muP results) and one novel
  layer type from threads 1/4/5.
- Widths swept across at least 3 multiples (e.g. base, 4x, 8x), LR swept log-spaced at
  each width, small model/dataset (tiny LM or small classification task).
- Measure: location of the LR sweep minimum at each width, under both the derived scaling
  rule and an unscaled control parameterization.

## Compute budget

This is the one thread that most directly needs a genuine, if small, scaling sweep (by
construction — the prediction is about transfer across width). Still bounded: small
widths, small dataset, LR sweep at each of ~3 widths. Should stay within a small multiple
of a GPU-day, not require anything resembling a large training run.

## Bitter-lesson check

- Lowest risk in the portfolio: this thread's entire content *is* the "test cheap, confirm
  the trend survives scale" methodology, applied to hyperparameters instead of to a novel
  mechanism. Its main value is instrumental — if it works for the novel layers, it justifies
  trusting small-scale results from the other threads more; if it doesn't transfer for a
  given layer type, that's an important warning that the layer's small-scale falsification
  results in this repo may not be scale-stable, and should be flagged as such wherever they
  appear.

## Known prior work / risk of reinventing

Directly extends Tensor Programs / muP (Yang & Hu et al.). Novelty is applying the
derivation as a required checklist item for every other thread's novel layer, and treating
transfer failure as a first-class, reportable outcome rather than skipping the check.

## Status

Not yet run.
