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

**Post-hoc note, 2026-07-02 (harness smoke test v2, still not the pre-registered run):**
fixed three issues an Opus 4.8 code review found in v1 (train/test leakage in the dataset
split, a final-loss metric too weak to discriminate, and no effect-size check in
`summarize_sweep`). With those fixed and switched to a steps-to-target-loss metric on a
disjoint split, the harness now produces a real, non-degenerate signal for the first time
(126-run sweep, widths 64/256/512 = k in {1, 4, 8}, 3 seeds, ~82s on CPU) — and the signal
runs *against* the prediction as measured: SP's optimal raw `base_lr` was exactly flat
across all three widths (0.01, log10 drift = 0.0), while muP's optimal raw `base_lr`
shifted a full decade (0.01 -> 0.1, log10 drift = 1.0). Verdict per the pre-registered bar:
**fails** (drift ratio 0.0x, needs >=3x).

Two things temper how much this should be trusted as a real result, both structural, not
bugs: (1) the LR grid's spacing (~3.3x between points) is coarser than the 2x tolerance
the prediction itself is stated in, so the sweep can't actually resolve a 2x-scale claim
either way; (2) muP's known advantage is largely asymptotic in width ratio and is usually
demonstrated at ratios far larger than 8x (the original muTransfer paper transfers across
~100-1000x) — an 8x range may simply be too small for muP's compensation to show up over
Adam's own incidental scale-robustness at this size, independent of which parametrization
is "right." Converting to *effective* LR (base_lr times muP's internal width multiplier)
shows muP's effective LR was in fact close to flat (~0.01 -> ~0.0125, well under 2x) —
so muP's underlying claim about effective learning dynamics looks fine here; what fails is
the practical claim that the *same raw base_lr number* transfers without adjustment, which
is the actually-relevant claim for "no re-tuning needed" and is legitimately what the
thread's prediction is about.

Per the pre-registration rule, this still does not count as the thread's verdict (wrong
scale, wrong grid resolution, single toy task). It does sharpen what the real run needs
beyond the note above: an LR grid finer than 2x per step (not ~3.3x), and a width range
that reaches at least the pre-registered 16x and ideally further, since an 8x ratio may be
structurally too small to separate the two parametrizations regardless of which is
correct.
