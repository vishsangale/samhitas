# Thread 10 (follow-up to thread 9): Curriculum training for gated spectral recurrence

**Math source:** same as thread 9 (control theory + linear time-varying systems); this
thread adds no new mathematical claim of its own — it isolates a *training-protocol*
variable (curriculum vs. direct training) that thread 9's own falsification left open,
rather than the mechanism itself.

## Motivation

Thread 9 (`docs/threads/09-gated-spectral-recurrence.md`) pre-registered and ran a minimal
input-dependent retention gate on top of thread 1's spectrally-constrained linear core, and
found prediction A **falsified exactly as specified**: at the pre-registered depth
(n_pairs=8), direct training (2000 fresh-batch Adam steps) reached only 0.032 mean
held-out accuracy, far short of the 0.30 target. An independent Opus 4.8 review, asked to
check the diagnosis (not just the code) before any follow-up was designed, found: (a) no
harness bug, (b) the gate provably does inject query-dependent content-sensitivity that a
linear model cannot (verified directly — swap-response ratio 0.02 near-closed vs. 0.47
forced-open), and (c) accuracy collapses monotonically with depth regardless of LR/hidden
size/gate-bias-init — the *same* construction reached 0.32 accuracy at n_pairs=2 in the
review's own re-run, 0.11 at n_pairs=4, and ~0.01-0.03 at n_pairs=8 — with the learned gate
barely opening at n_pairs=8 (mean g ~= 0.01-0.16) in every configuration tried. That pattern
(works shallow, collapses deep, forcing the gate open doesn't help) looks like a credit-
assignment problem — the gate must learn to simultaneously protect several already-stored
associations *and* correctly write a new one, and gradient signal for "keep protecting an
association written many steps ago" is exactly the kind of thing that's easy to learn at
shallow depth and hard to learn when it's never rewarded until training already very deep.
Curriculum learning (train easy instances first, gradually increase difficulty) is the
standard, well-established fix for exactly this failure mode — not a novel idea being
introduced here, but a specific, falsifiable, cheap thing to test before concluding thread
9's mechanism doesn't work at the pre-registered depth.

## Architectural hypothesis

The `GatedRecallModel` architecture from thread 9 is unchanged — no new mechanism, no new
parameters. The change is entirely in the training *schedule*: instead of training only at
the target depth (n_pairs=8) for the full step budget, train in stages of increasing
depth (n_pairs=2, then 4, then 8), each stage the same architecture and continuing from the
previous stage's weights, with the **total step budget held identical** to thread 9's
already-run direct-training protocol (2000 steps) — this is a same-compute comparison, not
"train longer," which would confound curriculum's effect with more total training.

## Falsifiable prediction (pre-registered)

**Curriculum recovers depth-8 performance within the original compute budget.** Train
`GatedRecallModel` (vocab=512, hidden=64, eps=0.1 — identical to thread 9) with a 3-stage
curriculum over the *same total* 2000 Adam steps thread 9 used for direct training: 700
steps at n_pairs=2, 700 steps at n_pairs=4, 600 steps at n_pairs=8 (fresh random batch every
step within each stage, no repeated-epoch memorization risk, matching thread 9's "online"
protocol). Same LR grid (`{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`), same 5 seeds, evaluated the same
way (held-out n_pairs=8 batch, 512 examples). **Pass:** best-of-grid mean held-out accuracy
at n_pairs=8 >= 0.30 (thread 9's original bar). **Fail:** stays below 0.30, or clears 0.30
by a margin no larger than seed-to-seed noise (report full per-seed spread, not just the
mean, per `docs/methodology.md`'s effect-size discipline).

The comparison arm is **already collected, not re-run**: thread 9's direct-training result
at the identical total step budget, LR grid, and seeds (best-of-grid mean 0.032) is reused
as the matched-compute control — training on n_pairs=8 alone for the same 2000 steps. This
keeps the comparison honest (same architecture, same total compute, same tuning budget;
only the schedule differs) without spending extra compute re-deriving a number this repo
already has on record.

**What would NOT count as support for this prediction:** if curriculum training reaches
>=0.30 only by effectively using its early stages as "free" extra steps at a task so easy
the model would reach high accuracy on n_pairs=2 or n_pairs=4 alone almost immediately
(possible, since the review already found n_pairs=2 alone reaches 0.32) — the diagnostic
here is whether the **final n_pairs=8 stage's evaluation** genuinely reflects depth-8
capability, not leftover credit from an easy early stage that happens to correlate with
n_pairs=8's answer distribution. Since keys/values are independently random per batch with
no cross-stage structure to exploit, this risk is low, but it's flagged here rather than
assumed away.

## Explicitly out of scope for this thread's claim

- Does not test the "dual gate" (independent read/write gates) idea the review also
  raised — that is a separate, architecture-level hypothesis (changes the model, not just
  the schedule) and needs its own thread doc if pursued.
- No claim about whether curriculum training is a *general* fix for this class of gated
  recurrence beyond this specific task/depth/architecture combination.
- Does not re-run or revise thread 9's own falsified prediction A — that verdict stands as
  logged (falsified as pre-registered, n_pairs=8, direct training). This thread asks a
  narrower, different question (does a different *schedule* change the outcome), which is
  exactly why it gets its own pre-registration rather than amending thread 9's.

## Minimal experiment

- Model: `GatedRecallModel` (unchanged from thread 9,
  `experiments/models/gated_linear_recurrence.py`).
- New code needed: a curriculum training loop (stage list of `(n_pairs, n_steps)` tuples),
  otherwise reusing thread 9's driver script structure directly.
- Compute: same total step count as thread 9's already-timed protocol (2000 steps took
  ~15s per run there), so expect a comparable per-run cost; time the first run before
  committing to the full 5-seed x 5-LR grid, per this repo's own "measure, don't
  extrapolate" rule.

## Compute budget

Well under a GPU-day; CPU-smoke-testable here, matched to thread 9's already-demonstrated
budget.

## Bitter-lesson check

Curriculum learning is a training-schedule technique with a long, well-established
literature (Bengio et al. 2009 and much subsequent work) — using it here is not introducing
new task-specific machinery, just testing whether a standard optimization fix resolves an
optimization-shaped failure (the gate barely opens, forcing it open doesn't help,
performance degrades smoothly with depth) rather than a representational one.

## Known prior work / risk of reinventing

Curriculum learning for exactly this kind of length/depth-generalization failure in
sequence models is well precedented (e.g. curriculum schedules used in some S4/SSM and
Transformer length-generalization work). This thread isn't claiming novelty in the
curriculum idea itself — only in applying it as the specific, falsifiable next test for
thread 9's specific gated-spectral-recurrence construction.

## Status

Not yet implemented. This doc exists to satisfy `docs/methodology.md`'s pre-registration
rule before any curriculum training code is written.

**Post-hoc note, 2026-07-06 (`scripts/thread10_curriculum_sanity.py`): falsified as
pre-registered — curriculum barely moves the needle.**

Ran the exact pre-registered protocol: 3-stage curriculum (n_pairs=2 x 700 steps, n_pairs=4
x 700 steps, n_pairs=8 x 600 steps; 2000 total, matching thread 9's direct-training budget
exactly), same LR grid and 5 seeds, evaluated on held-out n_pairs=8 batches.

- **Best-of-grid mean accuracy: 0.039** (lr=3e-4), vs. the 0.30 target. **Falsified.**
- Comparison against thread 9's already-collected direct-training control (0.032, not
  re-run): the curriculum arm's 0.039 is only marginally higher, and the gap is within
  per-seed noise (curriculum per-seed spread at its best LR: 0.033-0.043). Curriculum
  training, at this specific stage allocation, did not meaningfully recover depth-8
  performance.

This is a more informative negative result than a plain "didn't work," though: the review
that motivated this thread found the *same* gated architecture reaches 0.32 accuracy when
trained and evaluated at n_pairs=2 alone. Here, the curriculum spends its first two stages
at exactly n_pairs=2 and n_pairs=4 (1400 of the 2000 total steps) before the final 600 steps
at n_pairs=8 — and still ends up indistinguishable from direct n_pairs=8-only training. That
pattern is consistent with the credit-assignment story from thread 9's addendum: whatever
the shared read/write gate learns at shallow depth does not survive further training at
n_pairs=8 well enough to help, rather than shallow training simply being wasted. It does not
by itself confirm that reading, though — two unpre-registered confounds are visible without
more experiments: (a) 700 steps at n_pairs=2 may not be enough on its own to reach the
review's 0.32 (that number's own training budget wasn't matched here), and (b) 600 final
steps at n_pairs=8 may not be enough to consolidate even a well-learned shallow-stage gate.
Not running further stage-allocation variants under this same "prediction" label — that
would be exactly the retrofitting `docs/methodology.md`'s pre-registration rule exists to
prevent. The honest bookkeeping: **this curriculum schedule, as pre-registered, is
falsified.** A different allocation, or the dual-gate (independent read/write) idea thread
9's review also raised, would need its own fresh pre-registration, not a rerun under this
one's numbers.

At this point, two independent attempts to recover thread 9's prediction A at the
pre-registered depth (direct training, then this curriculum) have both failed, while the
mechanism-level evidence (the gate provably injects content-dependence; shallow depths
train fine) still says the underlying idea isn't structurally dead. The next cheapest,
different-in-kind thing to try, if this line is pursued further, is architectural (the
dual-gate idea) rather than another training-schedule variant — but that is a decision for
a fresh thread doc, not an extension of this one.
