# Thread 6 (priority 1): muP-style hyperparameter transfer for novel layers

**Math source:** infinite-width mean-field limits / Tensor Programs (muP). Originally
labeled "random matrix theory" — that was inaccurate. RMT (Marchenko-Pastur spectra etc.)
does no load-bearing work in this derivation; the actual machinery is muP's
forward/backward Jacobian scaling program. Naming corrected after review.

## Motivation

muP (maximal update parameterization) shows that a specific per-layer scaling of init and
learning rate lets hyperparameters tuned at small width transfer, in principle exactly, to
much larger width — this is the closest existing precedent to the entire methodology this
repo is built around (falsify/tune cheaply at small scale, trust the theory to carry the
result to larger scale). The open question worth treating as its own thread: does the same
style of derivation extend cleanly to the *other* architectural mechanisms proposed in this
portfolio (structured recurrence from thread 1, integrator blocks from thread 4), or does
each new mechanism need its own scaling derivation because it changes the relevant
forward/backward statistics in a way vanilla muP doesn't cover?

This thread is promoted to priority 1 (was priority 3 in the original draft) because it is
a **logical dependency for the rest of the repo**, not just another candidate idea: the
entire premise of "falsify at small scale, trust the trend at larger scale" rests on
hyperparameter (and, implicitly, effect) transfer actually holding. If it doesn't hold for
the novel layers this repo proposes, every other thread's small-scale verdict needs an
explicit caveat. Build and run this first, on the plain baseline and on Thread 1's layer,
before trusting any other thread's small-scale result at face value.

## Architectural hypothesis

For any of this repo's proposed layer types, a per-layer learning-rate/init scaling rule
can be derived from the layer's forward/backward Jacobian statistics (following the same
program as muP) such that hyperparameters swept and selected at a small width transfer,
within a pre-registered tolerance on final loss, to a width several multiples larger —
without re-tuning.

## Falsifiable prediction (pre-registered)

For a chosen layer type (start with the plain baseline to validate the protocol against
known muP results, then apply to Thread 1's structured-recurrence layer once that thread
has a stable implementation), the learning rate that is optimal at width `W` under the
derived scaling rule should also land within a factor of 2x of optimal at width `k*W`
(k in {4, 8, 16}), measured by directly sweeping LR at each width and comparing where the
sweep minimum lands. A naive (unscaled) parameterization, swept the same way, should show
the optimal LR drifting by an order of magnitude or more across the same width range.

Effect-size criterion (per `docs/methodology.md`): "transfer holds" requires the derived
rule's LR-drift-in-log-space across widths to be at least 3x smaller than the naive
parameterization's drift, not merely non-overlapping across 3 seeds. If the derived rule's
transfer tolerance is no better than the naive parameterization's by that margin, the
derivation for that specific layer type is falsified (even if muP itself remains valid for
the layers it was originally derived for).

## Minimal experiment

- Pick one baseline layer type (sanity check against known muP results) and Thread 1's
  structured-recurrence layer as the first novel case.
- Widths swept across at least 3 multiples (e.g. base, 4x, 8x), LR swept log-spaced at
  each width, small model/dataset (tiny LM or small classification task).
- Measure: location of the LR sweep minimum at each width, under both the derived scaling
  rule and an unscaled control parameterization. Report the full sweep surface, not just
  the argmax (methodology requirement).

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
transfer failure as a first-class, reportable outcome rather than skipping the check. This
thread is honestly a re-validation-plus-extension of known theory, not a new hypothesis —
its value is instrumental (validating the repo's core methodology), not novelty.

## Status

Not yet run for real. Priority 1 — build first.

**Post-hoc note, 2026-07-01 (harness smoke test, not the pre-registered run):** a CPU-scale
smoke test of the harness (`experiments/scripts/thread06_mup_sanity.py`, widths 32/128/512,
~4K train examples, 150 steps) confirmed the code path works end to end but produced a
non-informative LR-transfer signal — the task saturates across most of the LR grid at this
size, and Adam's own per-parameter normalization appears to mask width-scaling effects at
such small width, so the "best LR" argmin was noisy rather than tracking a real
trainability boundary (SP showed *less* drift than muP in this run, opposite the
prediction, which reads as a metric/scale problem, not evidence against the theory). Per
this repo's pre-registration rule, this does not change the prediction above or count as a
verdict. It does inform how the real run should be designed: use width multiples large
enough to be unambiguous (the pre-registered k in {4, 8, 16} should be applied to a
not-tiny base width, not 32), consider a harder task than modular arithmetic that doesn't
saturate across the whole LR grid, and consider tracking training stability / steps-to-
target-loss rather than final-loss argmin, which is a weak signal on any task easy enough
to reach near-zero loss under many configurations.
