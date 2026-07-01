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

**Correction, 2026-07-02, after a second Opus 4.8 review of this addendum itself:** the
paragraph originally here softened that result in a way the reviewer correctly called out
as motivated reasoning, on two counts, and it's worth recording both since getting this
wrong is itself informative about how easy it is to explain away an inconvenient result.

First, the grid-coarseness argument (spacing ~3.3x vs. a 2x tolerance) doesn't actually
apply: muP's optimum moved a full decade, roughly two grid steps, which the grid resolves
just fine. That objection argues against precision the result doesn't depend on.

Second, and more importantly, the original addendum claimed converting to *effective* LR
(`base_lr * base_width/width`) showed muP staying flat ("~0.01 -> ~0.0125, well under
2x") and used that to argue the underlying theory was fine even though the raw-`base_lr`
claim failed. This doesn't hold up two ways. (a) It's circular: the `base_width/width`
factor *is* muP's mechanism, so dividing it back out and then observing "flatness" is
recovering the input, not evidence about anything. The thread's actual pre-registered
prediction is stated in terms of "the learning rate that is optimal ... under the derived
scaling rule" — i.e. the raw `base_lr` a user sets and reuses, specifically because that's
what "transfer without re-tuning" has to mean in practice. (b) It was also arithmetically
wrong: it quoted only the width=64 and width=512 endpoints and dropped the width=256 point
(0.025), which the sanity gate had flagged as noise-tied, not as invalid — including it,
the effective-LR sequence is 0.010 -> 0.025 -> 0.0125, a 2.5x spread, which fails the very
&lt;2x bar the paragraph claimed it cleared.

The honest reading: **the smoke test ran cleanly against the prediction.** SP's raw
LR-optimum was flat where muP's moved a full decade — at this scale, muP made the exact
quantity the thread cares about worse, not better, than doing nothing. This is still not
the thread's real verdict (wrong scale, wrong task, and the width range is too narrow for
muP's usually-asymptotic advantage to plausibly appear at all — that caveat is legitimate
and is the one to keep), but it should not be read as "inconclusive, ignore it." Two
related bugs this exposed in `experiments/harness/report.py` are now fixed: a
both-arms-flat sweep no longer silently auto-passes the effect-size bar (it previously
divided by a zero drift and reported `ratio=inf`, which technically clears any threshold),
and the drift summary now reports which widths were actually used versus gated out instead
of silently dropping a noise-tied point the way this addendum's first draft implicitly
did.

What the real run needs, revised: width range genuinely matters and should reach well past
the pre-registered 16x (30-60x+, since 8x already looks insufficient) to give muP a fair
shot at its asymptotic regime — but extend it to find out whether muP starts winning, not
as a reason to discount the current adverse reading. The step-budget/threshold setup
(`target_train_loss` + `max_steps`) is also a real confounder the first draft of this note

**Post-hoc note, 2026-07-03 (`thread06_mup_widerange.py`, k up to 32x, dual thresholds,
still CPU/toy scale, last smoke-test iteration for now):** extending to widths
64/256/1024/2048 (k=1,4,16,32) with a finer LR grid (~2.3x spacing) and two loss
thresholds gives the same qualitative answer as the narrower run, not a different one:
muP's raw `base_lr` optimum keeps moving (0.015 -> 0.08 -> 0.4 -> 0.9 across widths at the
looser threshold), and its log10 drift (1.43 decades, using the two widths that cleared
the sanity gate) is still larger than SP's (0.33 decades) — ratio 0.23x, still failing the
>=3x bar, in the same direction as before. One new, honest wrinkle: at this wider range SP
is no longer perfectly flat either (its optimum drifts down: 0.015 -> 0.015 -> 0.007 ->
0.003), and *both* arms hit the edge of the LR grid at width=2048, which the sanity gate
correctly flags as inconclusive rather than reporting a misleading number — so the grid
itself is now too narrow in both directions, on top of everything else.

Stopping smoke-test iteration on this thread here rather than continuing to chase a
cleaner number: two independent runs at two different width ranges now agree that muP's
raw-LR transfer looks worse than SP's naive robustness at small/toy scale, and the
remaining open question (does this flip once the width range, task, and step budget match
what real muP validation actually needs — thousands of steps, a harder task, much larger
widths) can only be answered by the pre-registered GPU run, not by further CPU tuning.
Moving on to thread 1 (the first thread that's an actual candidate architecture, not a
methodology check) rather than continuing to refine this smoke test.
didn't flag: which configs count as "converged" is sensitive to one arbitrary cutoff and
step ceiling, so the real run should report steps-to-target across several thresholds (or
fit the loss curve) rather than hinge everything on one target value. LR grid resolution
was a secondary concern, not the binding one.
