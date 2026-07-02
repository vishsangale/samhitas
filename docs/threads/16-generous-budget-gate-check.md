# Thread 16 (idea I4): generous-budget gate check

**Math source:** none new -- this is not an architecture thread. It directly tests a
specific, bolded, previously-recorded conclusion from the gate-family sub-line (threads
9/10/11), per the 2026-07-07 portfolio review (`docs/reviews/2026-07-07-portfolio-review.md`,
idea I4): the "learnability limit" half of thread 11's closing claim ("this is an
optimization/learnability limit... not a capacity or architecture one") was **never
decisively tested** -- the review found what's actually established is "no signal within
2000 online fresh-batch steps," not "no signal, period."

## Why this thread exists

Thread 11 closed the gate-family sub-line (single shared gate -> curriculum -> independent
read/write gates, all on top of thread 1's spectrally-constrained orthogonal core) as a
negative result: three architecturally distinct gate interventions all converged to ~0.03
mean accuracy at `n_pairs=8`, and the write-relevant gate barely moved from its near-closed
init value (~0.021 -> ~0.025-0.026) across all three, regardless of forced-open
initialization. The recorded closing framing was "optimization/learnability limit, no
discoverable gradient signal." But the specific experiment that would earn that framing --
a much more generous budget (10k+ steps, and/or repeated-batch overfitting of a small fixed
set) to distinguish "no gradient signal exists at all" from "2000 online fresh-batch steps
just aren't enough" -- was explicitly specified by the thread-11 review and never run. This
thread runs it.

Distinguishing these two readings matters for the whole sub-line's record: if a generous
budget still finds nothing, "no discoverable gradient signal, at this depth/mechanism" is
finally earned rather than assumed, and RESEARCH.md's bolded conclusion is confirmed with
actual evidence. If a generous budget *does* find a signal (accuracy climbs, gate opens),
the gate-family closure needs to be re-characterized as a budget artifact, and the sub-line
would need reopening under a fresh thread -- a materially different, correction-worthy
outcome that would matter for the still-open Zoology-literature question of whether this
gate family is architecturally insufficient regardless of optimization (portfolio review
section 2.2).

## Falsifiable predictions (pre-registered)

Reuse `experiments/models/dual_gate_recurrence.py`'s `DualGateRecallModel` unchanged (no
architecture change -- per the review, this deliberately tests the recorded claim about the
*existing* gate designs, not a fourth variant) and `experiments/tasks/recall.py` unchanged.
Fixed config matching threads 9/10/11 exactly: `vocab=512, hidden=64, eps=0.1, n_pairs=8`
(`seq_len=17`), `batch=32`, single best LR from thread 11's grid (`lr=3e-4`, thread 11's
best-of-grid choice -- reused directly rather than re-swept, since re-sweeping LR is a
different question from "does more budget help at the LR that already worked best"), 5
seeds (matching threads 9/10/11's seed count).

### Arm A: generous online budget (fresh-random-batch-per-step, 6x thread 11's budget)

Train `DualGateRecallModel` exactly as thread 11 did (fresh random batch drawn every step,
matching the recall task's generalization setting) but for **12,000 steps instead of
2,000** (6x). Track `mean(sigmoid(write_gate(x)))` on a fixed diagnostic batch at
checkpoints every 1,000 steps (informational, not gating), and final held-out accuracy on a
fresh eval batch (`eval_batch=512`, matching thread 9/10/11's protocol) at step 12,000.

- **Pass** (signal found; budget artifact) iff mean held-out accuracy across the 5 seeds
  `>= 0.30` (the same bar threads 9/10/11 used, kept for continuity) **or** the mean write
  gate value at step 12,000 is at least 2x its step-0 value (a directional movement signal,
  weaker than the accuracy bar, tracked because "gate never moves" was the specific
  symptom the sub-line documented -- if the gate is now clearly moving even without hitting
  the accuracy bar yet, that already contradicts "no discoverable gradient signal exists").
- **Fail** iff accuracy stays within noise of thread 11's 0.032 control **and** the gate
  stays within 2x of its init value.

### Arm B: repeated-batch memorization on a small fixed set

Generate a single fixed pool of 128 examples once (`recall.make_batch(n_pairs=8, vocab=512,
batch_size=128, seed=0)`, an 8x-oversized "training set" relative to the usual batch=32),
then train by repeatedly sampling minibatches (`batch=32`, with replacement) *from this
fixed pool only* -- no fresh generation -- for the same 12,000-step budget. This tests pure
memorization/overfitting capacity, isolated from generalization: if the write-gate pathway
can move productively under *any* amount of repetition, overfitting a small closed set
should be dramatically easier than generalizing to fresh sequences, since the model can in
principle memorize input-output pairs directly rather than learning the general recall
algorithm. Evaluate **training accuracy on the same fixed 128-example pool** (not a fresh
eval set -- this arm is deliberately about memorization, not generalization) at step
12,000, across the same 5 seeds.

- **Pass** (a genuine optimization dead end is contradicted) iff mean training accuracy on
  the fixed pool `>= 0.90` (near-memorization -- a much higher bar than Arm A's 0.30, since
  overfitting a small closed set should be far easier than generalizing if any gradient
  signal exists at all).
- **Fail** iff training accuracy on the fixed pool stays near chance (`<0.10`, well below
  even thread 9/10/11's already-low ~0.03 generalization accuracy would suggest as a
  ceiling) despite unlimited repetition on 128 examples.

**Interpretation, pre-committed:**
- **Both A and B fail** -> "no discoverable gradient signal, at this depth/mechanism, even
  under a 6x budget and even for pure memorization" is earned with actual evidence -- the
  gate-family sub-line's optimization-limit framing is confirmed, not merely asserted. The
  Zoology-literature architectural-insufficiency explanation (portfolio review section 2.2)
  becomes the more load-bearing account of *why* no signal exists, since a genuine
  optimization dead end even for memorization is consistent with (though doesn't prove) an
  information-theoretic/representational barrier, not merely "needed more steps."
- **B passes, A fails** -> the write-gate pathway *can* move productively under
  optimization (rules out a hard, structural optimization dead end), but the fresh-batch
  generalization problem specifically remains unsolved at this budget -- reframes the
  finding as "can memorize, cannot generalize to the general recall algorithm at this
  depth/budget," a materially different and more specific conclusion than either "no
  signal" or "budget artifact."
- **A passes** (regardless of B) -> the gate-family closure was a budget artifact; the
  sub-line needs reopening under a fresh thread with the corrected budget, and
  RESEARCH.md's bolded "optimization/learnability limit" conclusion needs a dated
  correction, not just a footnote.

## Explicitly out of scope

- Not re-sweeping LR -- reuses thread 11's already-best `lr=3e-4` directly.
- Not testing the single-shared-gate (thread 9) or curriculum (thread 10) architectures --
  dual-gate only, since it's the most capable of the three already-tested designs and the
  question is about budget, not re-litigating which architecture is best.
- Not a new gate variant -- per the review, this deliberately does not change the
  architecture, since the question is "does more budget help the existing designs," not
  "does a new design help."
- Not testing `diag_lowrank` -- same scope limit as threads 9/10/11.

## Minimal experiment / compute budget

Thread 11's original 2000-step runs took ~15-25s each on this sandbox. Timing check at the
full 12,000-step budget (both arms) before committing to the full 5-seed grid, per this
repo's standard practice; review's own cost estimate for this idea is ~1-2 CPU-hours.

## Bitter-lesson check

Not a novel-architecture claim -- a budget/optimization diagnostic on already-built,
already-tested models. Value is purely in firming up or correcting an existing bolded
conclusion with actual evidence rather than an unrun specification.

## Known prior work / risk of reinventing

"Train longer / overfit a small set to distinguish an optimization dead end from an
under-trained model" is standard ML debugging practice, not a novel technique. No claim of
novelty here beyond applying it to this repo's own specific recorded gap.

## Status

Not yet run. Pre-registered 2026-07-07 (session-label date, per this repo's now-standard
doc-date-vs-commit-date convention).
