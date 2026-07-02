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

Run 2026-07-07 (session-label date). **Both arms literally PASS their pre-registered
bars, but an independent Opus review found the honest headline is closer to "falsified in
spirit" than "budget artifact confirmed."** See the dated post-hoc note below.

**Post-hoc note, 2026-07-07 (Opus review):** Full grid (2 arms x 5 seeds x 12,000 steps)
ran in ~1003s CPU (~17 min). Raw result: **Arm A** mean held-out accuracy across 5 seeds =
**0.0316** (individual seeds 0.0195-0.0371, genuinely flat, no stray high-accuracy seed) --
statistically indistinguishable from thread 11's already-collected 2000-step control
(0.032) despite 6x more compute, and far below the 0.30 "solved recall" bar. But the write
gate's mean value moved from 0.0211 (init) to 0.0993 (step 12000), a **4.70x** increase,
clearing the pre-registered `>=2.0x` gate-growth clause -- so Arm A technically **PASSES**
via the OR criterion's weaker branch. **Arm B** reached **1.0000** mean training accuracy
on the fixed 128-example pool (near-perfect memorization), clearing its `>=0.90` bar --
**PASSES** outright.

Sent to an independent Opus review before drawing a conclusion, flagging explicitly that
Arm A's pass came entirely from the weak gate-growth clause and asking whether "budget
artifact, sub-line reopens" was the right headline. The review re-ran Arm A (seeds 0-1) and
Arm B (seed 0) from scratch at the full 12,000 steps and reproduced every number to the
digit (a concurrent reviewer's independent 3-seed re-run also matched), then added
diagnostics the original driver didn't collect:

1. **The pre-registered `>=2.0x` gate-growth clause was miscalibrated -- too permissive,
   for a specific, checkable reason.** The init gate value (0.0211) sits at logit -3.84, deep
   in the sigmoid's saturated tail; the 2x bar (0.0422, logit -3.12) needs only +0.72 nats of
   logit movement to clear, while an actually-open gate (0.5) would need +3.84 nats. The
   observed final value (0.0993, logit -2.20) is a real +1.63-nat move -- genuine directed
   movement (`write_gate.weight`'s norm grew 2.78x, the largest of any weight matrix, vs.
   `forget_gate.weight`'s 1.00-1.16x and `embed`'s 1.05x baseline -- not Adam diffusion) --
   but "the gate moved 4.7x in relative terms" dramatizes a move from "deeply closed" to
   "still quite closed," not to "open." The bar conflated "a usable gradient direction
   exists for the write gate" with "gradient signal exists toward solving recall" -- those
   turned out to diverge sharply here (see next point). A better-calibrated bar would have
   used an absolute threshold (e.g. mean gate `>0.5`) rather than a relative-to-a-near-zero-
   baseline one, or restricted "budget artifact" to the accuracy clause alone. Per this
   repo's rule, the literal grade (Arm A passes) is not edited after the fact -- this is
   logged as the interpretation alongside it.
2. **What the gate is opening toward: a recency-weighted in-context-copy shortcut, not
   partial content-addressed recall.** Per-query-position accuracy climbs near-monotonically
   from **0.000 for the oldest pairs (query index 0-3) to 0.08-0.13 for the most recent pair
   (index 7)** -- the fingerprint of a decaying recurrent memory where only the most
   recently-written pair survives to readout, not flat content-addressed routing. ~78-82% of
   predictions are tokens present in the input sequence (chance would be ~3.3%), and
   predictions are diverse (240-350 unique tokens, no single-token collapse) -- so this is
   learned in-context copying, not a degenerate frequency heuristic (which couldn't help
   anyway, since values are uniform over the vocab). A naive "always output the most recent
   value" heuristic alone scores 0.137, close to the observed idx-7 accuracy -- consistent
   with the ~16x-chance overall accuracy being almost entirely this shortcut, not partial
   recall of older pairs.
3. **Arm B's memorization is a real but low bar, and doesn't support "the gate mechanism
   specifically has usable signal for recall."** ~550 parameters per memorized example
   (~70k total params, 128 examples) makes near-perfect fixed-set memorization close to
   guaranteed for any model with functioning gradients -- passing it mainly rules out a
   catastrophic dead-gradient failure, not much more. Two findings sharpen this: the
   memorizing Arm-B model generalizes at **chance** on a fresh eval batch (0.0039, vs.
   chance 0.00195) -- memorization and the recall *algorithm* are fully decoupled -- and
   memorization is achieved with the write gate still mostly closed (~0.087, only 4.1x),
   suggesting the fixed-set fitting plausibly routes through the embedding/readout lookup
   rather than through the recurrent gating pathway doing content routing at all.

**Bottom-line verdict (Opus review, adopted here): not "budget artifact confirmed, sub-line
reopens" (reading the literal PASS this way would be exactly the kind of motivated-reasoning
error this repo's methodology exists to catch). Closer to "falsified in spirit despite a
literal Arm-A pass," with one genuine, specific correction to the prior record.** The
budget-artifact reading is not supported: 6x compute bought zero held-out-accuracy
improvement (0.0316 vs. the 0.032 control, flat across all 5 seeds, 10x below the 0.30 bar)
-- the PASS came entirely from a bar that, on inspection, didn't measure what "budget
artifact" needs it to measure. The core negative finding from threads 9/10/11 stands: more
budget alone does not solve generalized multi-pair recall for this gate family at this
depth. **One real correction is earned, though:** threads 9/10/11's phrasing that the
write-relevant gate "barely moved" / showed "no discoverable gradient signal" is **too
strong** and should be corrected -- given 12,000 steps there *is* a discoverable, directed
gradient signal (the gate moves 4.7x, monotonically, across all seeds; its weight matrix is
the most-updated in the model). The earlier "gate frozen" observation was a 2000-step
artifact, not evidence of a literally dead gradient. But this correction does **not**
reopen the sub-line -- if anything it strengthens the architectural-insufficiency reading
(portfolio review section 2.2, the Zoology citation) over a pure "just needed more steps"
reading: the pathway is optimizable, and given 6x more budget to use that optimizability,
it still converges on a shortcut rather than solving recall, consistent with this gate
family lacking one of the structural primitives (short-conv, two-layer composition, or
explicit key-value state) the literature says multi-pair recall actually needs.

**Action taken on the prior record:** `docs/threads/11-dual-gate-spectral-recurrence.md`'s
"write-relevant gate value moved from ~0.021 to only ~0.025-0.026" observation (thread 11's
2000-step measurement) is accurate as a 2000-step snapshot and is not edited -- but a
forward pointer to this thread's 12,000-step finding (real movement given more budget,
still no recall gain) is worth keeping in mind wherever that 2000-step number is cited as
if it characterized the gate's ceiling rather than its state at that specific, shorter
budget. Not pursuing a fourth gate variant or further budget escalation -- per idea I4's
own scope, this thread's purpose was to test the recorded claim about the *existing*
designs, which it has now done with a decisive, reproduced, and honestly-characterized
answer.
