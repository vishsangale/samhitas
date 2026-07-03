# Thread 18: Recall-protocol validation (idea C2 from the 2026-07-08 program regroup)

**Math source:** none new — this thread borrows no derivation of its own. It uses a
literature-established reference *architecture* (2-layer causal self-attention, the
textbook-sufficient mechanism for induction-head/associative-recall tasks) to test a
*harness* question, not a theory. See "Why this isn't a literature-screen kill" below for
why that distinction is the whole point of this thread.

**Proposed by:** `docs/reviews/2026-07-08-program-regroup.md`, section 4, item C2, itself
prompted by the user's direct question about why the repo produces "all negative results."
This is the first pre-registration written under `docs/methodology.md`'s amendments v2
(six new gates, adopted the same day) and the first to go through the new item-6 gate
(independent pre-run design review, before any full-grid code runs).

## Why this thread exists

Seven prior experiments (threads 9, 10, 11, 16, and all three arms of 17) tried to make
some variant of a gated/matrix-state linear recurrence solve associative recall at
`n_pairs=8` and failed, converging on a training loss near `ln(512)=6.238` (not even
fitting the training distribution) and a best-ever held-out accuracy of ~0.04 — below the
0.137 a naive "always output the most recent value" heuristic achieves. Every failure was
reviewed and (mostly) traced to real, specific mechanism-level causes: gates that open
toward a recency-copy shortcut and re-close under forcing (threads 11, 16), depth-2
optimization pathologies (17a), an untrained-conv confound (17b), a write gate with no
online gradient toward recall (17c). None of those reviews found a harness bug.

But not one of those seven experiments included an arm known to solve the task. The
2026-07-08 regroup's central diagnosis (`docs/reviews/2026-07-08-program-regroup.md`,
root cause RC4) is that this is a real, structural gap: "mechanism insufficient" and
"protocol unlearnable by anything at this budget" are indistinguishable without a positive
control, and the recall cluster never had one. This thread supplies it, retroactively, for
the whole cluster.

## Why this isn't a literature-screen kill (methodology v2, item 1)

**Literature screen, stated directly, before any bands are set:** the literature firmly
and unambiguously predicts that a 2-layer causal self-attention model can solve
induction-head-style associative recall at this scale. Elhage et al. 2021 ("A Mathematical
Framework for Transformer Circuits") identify induction heads mechanistically in exactly
this model class — 2-layer, attention-only toy transformers — and show *1 layer is
provably insufficient* (the "shift-then-match" circuit needs two layers minimum) while 2
layers reliably learns it. Olsson et al. 2022 ("In-context Learning and Induction Heads")
track the induction-head "phase change" across training in small models and tie it
directly to in-context copying/recall ability. Zoology (Arora et al. 2023,
arXiv:2312.04927) — already the load-bearing citation for thread 17's whole design — uses
attention as its explicit *known-sufficient upper-bound reference* when proving gated/SSM
architectures have a capacity lower bound below what multi-pair recall needs. If the
question this thread asked were "can attention solve associative recall," that question is
already closed in the literature and would fail methodology v2's kill rule (item 1) exactly
the way threads 7/8-as-originally-written did in the 2026-07-07 portfolio review.

**That is not this thread's question.** The claim under test is narrower and specific to
this repository: does *this repo's* exact harness — `experiments/tasks/recall.py`'s task
generator, the online fresh-batch-per-step training loop, the `n_pairs=8`/2000-step/`hidden=64`
protocol every gate-family thread inherited unexamined — actually admit a learnable
solution by *any* architecture, or does it have some repo-specific obstruction (a bug, a
miscalibrated budget, an accidentally-too-hard task variant) that no amount of literature
about attention's general capability would predict? That's an engineering/harness-validity
question idiosyncratic to this codebase, not answered anywhere in the published record, and
not low-information under item 1's kill rule — it's the precise gap RC4 names. Per item 1's
instruction to "reshape toward what is genuinely open" rather than drop a claim the
literature already answers: this doc reshapes "does attention solve recall" (closed, would
be killed) into "does this repo's harness let attention demonstrate what the literature
already knows it can do" (open, this repo's own unresolved question).

## Regime-validity statement (methodology v2, item 2)

Unlike threads 2/12/13/15, this isn't an asymptotic mean-field claim with a formal validity
condition (`depth/width << 1`, etc.) — the relevant "regime" question is whether this
thread's planned scale sits inside the range where induction-head formation is
*empirically* established, not derived from a limit.

- **Depth:** 2 layers — exactly the literature-identified minimum (Elhage et al. 2021 prove
  1 layer cannot implement the shift-then-match circuit; 2 is both necessary and
  sufficient in their toy models). This thread is not below the literature's floor.
- **Width:** `hidden=64` — on the smaller end of Elhage et al.'s and Olsson et al.'s studied
  toy-model range, but both papers explicitly report the phenomenon in small toy models
  built to demonstrate it cleanly, not only at production scale.
  Elhage et al.'s smallest models are comparable to or smaller than this.
- **Context/vocab:** `seq_len=17` (`n_pairs=8`) or `5` (`n_pairs=2`), `vocab=512` — both
  smaller than most published induction-head studies. If this differs from the literature's
  regime at all, it differs in the *easy* direction (shorter context to search, smaller
  output space), not an adverse one.
- **Training budget:** 2000-12000 steps x batch 32 (64,000-384,000 examples). Smaller than
  large-scale induction-head studies, but this thread does not rely on the literature's
  asymptotic guarantees alone — see the noise-floor pilot below, which directly measured
  non-trivial learning (loss escaping the `ln(512)` plateau, accuracy climbing from chance
  toward 0.07 and still rising) at exactly this scale and budget, at a single arbitrarily-
  chosen grid point. This is empirical evidence the regime is viable for *this* setup, not
  just a literature analogy.

**Conclusion:** this thread's planned configuration sits inside or favorably adjacent to
the literature's demonstrated regime on every axis. No out-of-regime quantitative band is
being imported here (contrast threads 13/15) — the pass bar (0.30 mean accuracy) is an
absolute, already-precedented number reused unchanged from threads 9-17, not derived from
an asymptotic formula.

## Architecture: `TinyAttentionModel` (`experiments/models/tiny_attention.py`)

Embedding (`vocab=512 -> hidden=64`) + learned absolute positional embedding (added, table
sized to the task's exact `seq_len`) -> 2x pre-LN causal transformer blocks (multi-head
self-attention, `n_heads=4`, causal mask; GELU MLP, `mlp_ratio=4`) -> final LayerNorm ->
linear readout at the final (query) position, matching every other model in this repo's
embed/.../readout convention (`experiments/models/linear_recurrence.py`,
`gated_linear_recurrence.py`).

Design choices, each made to maximize the reference architecture's fairness (any failure
should not be attributable to an unnecessarily crippled positive control):

- **Standard PyTorch default init throughout** (`nn.Linear`, `nn.Embedding`,
  `nn.LayerNorm` defaults) — no muP scaling, no repo-specific tuning. The most
  textbook-standard construction available, deliberately, so a failure can't be blamed on
  an idiosyncratic init the way thread 17b's untrained-conv-plus-GELU confound was.
- **Causal masking** (not bidirectional), matching the literature's decoder-only induction-
  head constructions. Since the model only ever reads out logits at the sequence's *final*
  position, causal vs. bidirectional attention makes no difference to what the *readout*
  position can see (there is nothing after it either way) — it only affects what
  *intermediate* positions can see, which matters for whether the standard 2-layer
  shift-then-match circuit (layer 1: "what token came before me," strictly a backward-
  looking relation) can form the way the literature describes it. Using causal masking is
  the conservative, literature-faithful choice.
- **Learned absolute positional embeddings, mandatory in every arm** (methodology v2 item
  2's regime statement plus C2's explicit call-out: "attention has no recurrence to encode
  order"). Self-attention is permutation-equivariant without positional information; the
  recall task's structure (value immediately follows its key) is inherently positional
  (fixed offset), so a model with no positional signal cannot distinguish "the token right
  after a matched key" from any other position. This is not an ablation arm — it is a
  fixed, non-negotiable part of the reference architecture in every arm below, because
  omitting it would make a failure uninformative (indicting the harness for a defect that's
  actually just a missing standard component).
- **4 heads, GELU MLP (`mlp_ratio=4`)** — a standard, non-minimal-to-a-fault transformer
  block (not the bare attention-only toy models Elhage et al. use to isolate the induction
  circuit for interpretability). More capacity than the theoretical minimum is a deliberate
  choice in the reference-architecture's favor: it reduces the chance that a failure is
  "this exact minimal circuit didn't form" rather than "the task truly isn't learnable
  here," which is the more decisive, informative failure mode this thread wants to isolate.
- **~165K parameters** at `n_pairs=8` (embed 32,768 + pos_embed 1,088 + 2x block
  ~49,152 + readout 32,768), smaller than every practical transformer, larger than any
  model in this repo's recall-family sub-line (`gated_linear_recurrence.py` etc. are all
  under 40K params) — consistent with "give the positive control a fair, non-stingy
  budget."

## Falsifiable predictions (pre-registered) — three arms

All three arms reuse `experiments/tasks/recall.py` unchanged (`vocab=512`, online
fresh-random-batch-per-step training, `batch=32`, Adam) and the pre-registered 0.30 mean
held-out accuracy bar every gate-family thread has used since thread 9 — reused unchanged
for direct comparability, not re-derived.

**Attention-centered LR grid (all arms):** `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}` — 6
points, log-spaced, shifted one decade lower and widened by one point relative to the
gate-family sub-line's `{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`. Stated reason, not just habit:
this repo has no prior experience with attention's optimum on this task (unlike the
recurrence family, which had thread 9's experience to anchor a grid), and named contributor
(c) from the regroup (`docs/reviews/2026-07-08-program-regroup.md` section 2) is exactly
"a fixed grid saturating at its own ceiling or being mis-centered for a new architecture" —
a materially wider net here is the direct fix, not a stylistic preference. (This is not a
same-thread comparison the tuning-budget-symmetry rule's "match total trials between arms"
clause applies to — thread 18 has no internal baseline arm being compared against the novel
arm within this thread; each arm is tested against an absolute, previously-fixed bar.)

### Arm 1 (primary): exact existing protocol
`n_pairs=8` (`seq_len=17`), `hidden=64`, 2000 online Adam steps, LR grid above, 5 seeds.

### Arm 2: `n_pairs=2` floor check
Identical to Arm 1 except `n_pairs=2` (`seq_len=5`). Same LR grid, same 2000-step budget,
5 seeds. Reference point: thread 9's independent review found the *gated recurrence*
mechanism — the one every subsequent gate-family thread found insufficient at `n_pairs=8` —
already reaches 0.32 mean accuracy at `n_pairs=2` in its own re-runs
(`docs/threads/09-gated-spectral-recurrence.md`). If a literature-sufficient attention model
cannot clear the same bar a previously-*failing* mechanism already cleared at this easier
depth, that is the single most diagnostic possible signal that something in the harness
itself (not task difficulty, not architecture) is broken.

### Arm 3: extended budget
Identical to Arm 1 (`n_pairs=8`) except 12,000 steps (6x, matching thread 16's precedent
exactly) at a **single LR: Arm 1's own best-of-grid LR** (determined empirically from Arm
1's sweep, not a fresh grid search — keeps this arm cheap and is directly analogous to
thread 16's Arm A, which also reused a previously-identified best LR rather than re-sweeping
at the larger budget). Same 5 seeds. Training is from step 0 (same per-seed data stream as
Arm 1 at that LR, just continued past step 2000, not a checkpoint-resume), so the first
2000 steps are identical to Arm 1's run at that LR and the extra 10,000 are the only new
signal.

### Interpretation matrix (pre-committed)

Let R1/R2/R3 be each arm's best-of-grid (Arm 1, 2) or single-LR (Arm 3) mean held-out
accuracy across 5 seeds.

| R1 (primary) | R2 (`n_pairs=2`) | R3 (extended) | Verdict |
|---|---|---|---|
| >= 0.30 | (any) | (any) | **Protocol validated at standard budget.** The recall-negative cluster (9-17) is confirmed as evidence about *mechanism* insufficiency, not harness or protocol failure. Thread 18 supplies the missing positive control retroactively for the whole cluster. |
| < 0.30 | >= 0.30 | >= 0.30 | **Budget was the confound, not the protocol.** Named contributor (b) (the 2000-step convention) extends from the gate family (thread 16) to the positive control too — the standard budget was too tight for *any* architecture at `n_pairs=8`, not evidence about mechanism sufficiency one way or the other at 2000 steps specifically. |
| < 0.30 | >= 0.30 | < 0.30 | **Depth-8 recall is genuinely hard at this scale for any tested architecture within this budget range**, but the harness/task-generator itself is not broken (proof: `n_pairs=2` is learnable). Recall-cluster verdicts stand but get a dated "difficulty/budget calibration, not purely mechanism" caveat; future capability claims at `n_pairs=8` need a budget bar recalibrated against what attention itself needed (if R3 also fails, that recalibration is an open question for a future thread, not answered here). |
| < 0.30 | < 0.30 | (any) | **Harness/protocol defect indicated.** A literature-known-sufficient architecture failing even the easier floor a previously-*failing* mechanism already cleared points at the task generator, eval harness, or training loop — not at task difficulty or architecture. Blocks trusting *any* prior recall-cluster verdict until found and fixed; would need its own follow-up debugging pass, not a new mechanism thread. |

Every cell resolves a real question; there is no "inconclusive" outcome by design, per
methodology v2 item 4's escape-clause logic (this thread's whole purpose is to be the
protocol validation gate 4 requires before any future recall capability claim).

## Noise-floor pilot (methodology v2, item 3) — run before bands above were frozen

`experiments/scripts/thread18_noise_floor_pilot.py`, exploratory only, firewalled from the
treatment comparison (its numbers are not used to hand-pick a winning LR for the arms
above — only to check resolvability at the planned seed count). Three checks, all run on
CPU, ~24s total plus a 24s follow-up:

1. **Chance-level check:** untrained `TinyAttentionModel`, held-out eval — `acc=0.0020`
   vs. nominal chance `1/512=0.00195`. Confirms the eval harness (batching, argmax,
   accuracy) is wired correctly before trusting anything downstream.
2. **Short-budget resolvability check** (400 steps, `{3e-4, 1e-3, 3e-3}` x 3 seeds): no LR
   showed learning distinguishable from chance yet at this short budget (max per-LR seed
   stdev `0.0052`, all final losses still near `ln(512)~=6.2-6.3`). Uninformative about
   whether 2000 steps is enough — see check 3 — but directly answers the resolvability
   question item 3 asks: seed-to-seed noise at this scale (`<=0.0052`) is over 50x smaller
   than the gap this thread's bands need to resolve (chance `~0.002` / control band
   `~0.02-0.03` vs. pass bar `0.30`, a gap of `~0.27-0.28`). Five seeds at the full budget
   is expected to resolve that gap comfortably; a narrower pre-registered band than this
   would have been inadmissible under item 3, but nothing here is that narrow.
3. **Single-point full-budget sanity check** (not part of the pilot script; run directly to
   rule out a dead-gradient bug before freezing bands around an untested architecture):
   `lr=1e-3`, seed=0, full 2000 steps. Loss fell from 6.375 to 5.681 — clearly escaping the
   `ln(512)=6.238` plateau every gate-family failure got stuck at — and held-out accuracy
   climbed from chance to 0.0703, still rising at step 2000 (0.0645 -> 0.0664 -> 0.0645 ->
   0.0703 over the last 800 steps). Not a treatment result (single seed, single
   arbitrarily-chosen grid point, not used to bias the real sweep), but decisive evidence
   against a plumbing bug: this exact construction does learn something on this exact task,
   unlike every previously-tested mechanism, which stayed pinned at the uniform-loss
   plateau throughout training.

**Measured throughput:** 84.1 steps/s (2000-step single-point run, 23.8s). Full-grid time
estimate from this measurement (not extrapolation-without-data, per `CLAUDE.md`'s
measure-don't-extrapolate practice): Arm 1 `~30 (LR x seed) x 2000/84 ~= 12 min`; Arm 2
similar or faster (shorter `seq_len`); Arm 3 `~5 x 12000/84 ~= 12 min`. Total `~30-35 min`
CPU across all three arms — affordable, in line with the rest of the recall-family
sub-line's 11-22 minute runtimes.

## Positive-control statement (methodology v2, item 4)

Item 4 requires every capability-claim experiment to include an arm known to achieve the
claimed capability under the same harness, *or* — if none exists — be blocked on validating
the protocol first, itself a separately pre-registered experiment. **This thread is that
escape-clause experiment.** Its own "positive control" is the literature citation chain
above (Elhage et al., Olsson et al., Zoology) plus the single-point sanity check's direct
empirical confirmation that this exact construction learns on this exact task — there is no
older, already-validated arm within this repo to point to, because no thread before this
one ever included one. Passing this thread is what supplies that missing arm for every
future recall-capability claim in this repo.

## Stated prior pass-probability (methodology v2, item 5)

**Headline prediction (Arm 1 clears the 0.30 bar at the standard 2000-step budget):
expected p(pass) ~ 0.70**, because: the literature gives high confidence the task is
*fundamentally* learnable by this architecture at this depth/scale (Elhage et al.'s 2-layer
minimum is met exactly, not approximated), and the single-point sanity check directly
confirmed real learning at this exact scale/budget/task (loss escaping the uniform-loss
plateau, accuracy still climbing at step 2000) — but that check used one arbitrary LR with
no tuning and reached only 0.07, well short of 0.30, and the literature describes
induction-head formation as a training-time "phase change" (Olsson et al. 2022) rather than
smooth improvement, so there's real uncertainty about whether the best of 6 tuned LRs
across 5 seeds completes that phase change within 2000 steps specifically, versus needing
more steps than the standard budget provides. Not higher than 0.70: the sanity check's own
trajectory (still visibly rising, not plateaued, at step 2000) is as consistent with
"nearly there" as with "would need meaningfully longer."

**Overall protocol-validated outcome (top two rows of the interpretation matrix — R1
passes, or R2+R3 both pass): expected p(pass) ~ 0.85.** Higher than the headline number
because the extended-budget arm gives the phase-change-needs-more-steps scenario a second,
independent chance to succeed, and both scenarios lead to the same "protocol is learnable"
conclusion.

**Harness-defect row (R1 and R2 both fail): expected p(pass-of-this-row, i.e. defect found)
~ 0.05.** The literature's confidence in attention at `n_pairs=2` (a strictly shorter,
easier version of a task a previously-*failing* gated-recurrence mechanism already reached
0.32 on) is very high; this row would require the harness itself to be broken in a way ten-
plus independent reviews across seven prior recall experiments never surfaced.

This entry is added to the calibration ledger in `RESEARCH.md` section 8 at registration
time (headline Arm-1 number), with the outcome appended once Arm 1 completes.

## Explicitly out of scope

- Not claiming any novel result about attention or induction heads — the literature already
  answers "can 2-layer attention do associative recall" (see the literature-screen section
  above). This thread's only claimed contribution is harness/protocol validation, specific
  to this repository.
- Not a gate-family reopening. No gated-recurrence variant is involved; the closure rules
  from threads 11 and 17 are untouched regardless of this thread's outcome.
- Not testing muP, scaling, or any width sweep — fixed `hidden=64` throughout, matching the
  existing recall-family protocol exactly (that is thread 6's separate, parallel-running
  concern, not this one's).
- Not running thread 9's deferred prediction B (does gradient-flow predictability survive
  learned selectivity) — that question is specific to the spectrally-constrained-recurrence
  family's own construction and doesn't transfer to an attention model, which has no
  analogous spectral-radius parameter.

## Compute budget

`~30-35 min` CPU total (measured throughput, see noise-floor pilot section), comfortably
inside this repo's per-thread budget discipline. No GPU needed or used.

## Bitter-lesson check

Causal self-attention is the single most hardware-optimized primitive in this repo's
portfolio by a wide margin (every production LLM as of this writing uses it; FlashAttention
and friends exist specifically because of how well it maps to accelerator hardware) — no
concern about this thread's positive control being a hardware-unfriendly curiosity.

## Known prior work / risk of reinventing

None claimed. Every architectural component (causal self-attention, learned positional
embeddings, pre-LN transformer blocks) is standard, decades-old-to-recent, off-the-shelf
machinery, deliberately — see "Architecture" above for why minimality was *not* the goal
here the way it was for thread 17's arms.

## Independent pre-run design review (methodology v2, item 6)

Pending — this section will be filled in with the review's outcome and any changes made
before Arm 1 is run, per the amendment's requirement that this happen *before* the
experiment executes, not after.

## Status

Pre-registered 2026-07-08 (session-label date). Noise-floor pilot run (see above); design
sent for independent pre-run review next; arms run only after that review completes.
