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
- **Width:** `hidden=64` — a genuine toy scale, deliberately matching this repo's own
  `hidden=64` convention (thread 9 onward) rather than a width chosen to match either
  paper's specific configurations. Both papers report the induction-head phenomenon in
  small toy models built to demonstrate it cleanly, not only at production scale, but this
  doc does not claim a precise width comparison to their smallest reported models (flagged
  by the independent design review, finding F3, as unverifiable from either reviewer's
  prior knowledge and possibly wrong — the transformer-circuits toy models are recalled as
  plausibly wider, `d_model` in the low hundreds). The regime argument this thread actually
  relies on is the *empirical* one below (the sanity check demonstrates real learning at
  this exact width/scale), not a claimed literature-width match.
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

**Implementation contract, pinned explicitly (added per the independent design review's D4
finding, since the arm-driver script doesn't exist yet and these details were previously
only implicit in the pilot script):** each arm constructs its own fresh `TinyAttentionModel`
instance with `seq_len` set to that arm's own `recall.seq_len_for(n_pairs)` (17 for Arms 1
and 3, 5 for Arm 2) — never reuse a model built for one `seq_len` on another arm's batches.
Per-step training batches use `recall.make_batch(n_pairs, VOCAB, BATCH, seed=seed*100_000 +
step)`; held-out eval batches use `recall.make_batch(n_pairs, VOCAB, EVAL_BATCH,
seed=seed+999_999)`, `EVAL_BATCH=512` — both formulas reused unchanged from the pilot
script and from every gate-family thread's own convention, so results stay comparable.

### Arm 1 (primary): exact existing protocol
`n_pairs=8` (`seq_len=17`), `hidden=64`, 2000 online Adam steps, LR grid above, 5 seeds.

### Arm 2: `n_pairs=2` floor check
Identical to Arm 1 except `n_pairs=2` (`seq_len=5`). Same LR grid, same 2000-step budget,
5 seeds. **Primary justification (reframed per the design review's F1 finding):** a
2-pair induction task is a far shallower search than `n_pairs=8` (one candidate match
instead of eight, over a sequence less than a third as long), and 2-layer attention is the
literature's own minimum-sufficient depth for this circuit — this arm should be close to
trivial for a correctly-wired reference model on its own merits, independent of any
cross-architecture comparison. **Secondary, approximate reference point:** thread 9's
independent review separately found the *gated recurrence* mechanism — the one every
subsequent gate-family thread found insufficient at `n_pairs=8` — reaching 0.32 accuracy
at `n_pairs=2` in exploratory re-runs (`docs/threads/09-gated-spectral-recurrence.md`);
that number was a single exploratory data point from the review's own sweep, not a stated
5-seed mean under this thread's identical protocol, so it's cited as directional context,
not as the primary bar. If a literature-sufficient attention model cannot clear 0.30 here
at all, that is strong evidence something in the harness itself (not task difficulty, not
architecture) is broken — though see Row 4's caveat below for the one alternative reading
this arm alone can't rule out.

### Arm 3: extended budget
Identical to Arm 1 (`n_pairs=8`) except 12,000 steps (6x, matching thread 16's precedent
exactly) at **two LRs: Arm 1's own best-of-grid LR, plus the next-lower grid point**
(determined empirically from Arm 1's sweep, not a fresh 6-point search — keeps this arm
cheap while hedging a specific risk the design review named, see D3 below). Same 5 seeds
per LR (10 runs total). Training is from step 0 (same per-seed data stream as Arm 1 at
each LR, just continued past step 2000, not a checkpoint-resume), so each LR's first 2000
steps are identical to Arm 1's run at that LR and the extra 10,000 are the only new signal.

**Why two LRs, not one (added per the independent design review's D3 finding):** thread
16's precedent (a single reused best LR) tested a *directional* question (does the gate
move at all under more budget); Arm 3 tests a *threshold crossing* (does accuracy cross
0.30), and the LR that moved fastest in the first 2000 steps is not guaranteed to be the
LR that best *completes* a phase change by step 12000 — a single-LR design risks a false
negative if the 2000-step winner is a fast-but-early-plateauing choice rather than the
eventual-best one. Two adjacent grid points is a cheap hedge (~+12 min CPU) against that
specific failure mode, not a full re-sweep.

### Interpretation matrix (pre-committed)

Let R1/R2/R3 be each arm's best-of-grid (Arm 1, 2: best of 6 LRs) or best-of-two-LR (Arm
3) mean held-out accuracy across 5 seeds (10, if the near-boundary escalation rule above
triggers for that arm).

| R1 (primary) | R2 (`n_pairs=2`) | R3 (extended) | Verdict |
|---|---|---|---|
| >= 0.30 | (any) | (any) | **Protocol validated at standard budget.** The recall-negative cluster (9-17) is confirmed as evidence about *mechanism* insufficiency, not harness or protocol failure. Thread 18 supplies the missing positive control retroactively for the whole cluster. |
| < 0.30 | >= 0.30 | >= 0.30 | **Budget was the confound, not the protocol.** Named contributor (b) (the 2000-step convention) extends from the gate family (thread 16) to the positive control too — the standard budget was too tight for *any* architecture at `n_pairs=8`, not evidence about mechanism sufficiency one way or the other at 2000 steps specifically. |
| < 0.30 | >= 0.30 | < 0.30 | **Depth-8 recall is genuinely hard at this scale for any tested architecture within this budget range**, but the harness/task-generator itself is not broken (proof: `n_pairs=2` is learnable). Recall-cluster verdicts stand but get a dated "difficulty/budget calibration, not purely mechanism" caveat; future capability claims at `n_pairs=8` need a budget bar recalibrated against what attention itself needed (if R3 also fails, that recalibration is an open question for a future thread, not answered here). |
| < 0.30 | < 0.30 | (any) | **Harness/protocol defect indicated (usual reading), with one named alternative.** A literature-known-sufficient architecture failing even the easier floor a previously-*failing* mechanism already cleared usually points at the task generator, eval harness, or training loop — not at task difficulty or architecture. Named alternative this row alone can't rule out (per the design review's F1 finding): `n_pairs=2` has no extended-budget backstop the way `n_pairs=8` does (Arm 3), so a slow-to-form phase change could in principle affect Arm 2 too, not only Arm 1 — considered unlikely (`n_pairs=2` is a strictly easier search than `n_pairs=8`, which already shows learning within 2000 steps in the sanity check) but not formally excluded by this design. Blocks trusting *any* prior recall-cluster verdict until resolved; would need its own follow-up debugging pass (or, if the alternative reading is live, a cheap Arm-2-at-extended-budget check) rather than a new mechanism thread. |

**Near-boundary escalation rule (added per the independent design review, see gate-6
section below):** the 0.30 bar is not treated as a hard knife-edge for any arm. If an
arm's best-of-grid (or, for Arm 3, best-of-two-LR) 5-seed mean falls within one pooled
standard-error-of-the-mean of 0.30 (`|mean - 0.30| < stdev/sqrt(5)`, computed from that
arm's own realized seed spread at its best config), that arm's result is **not** assigned
a matrix row yet — 5 additional seeds are run at the same (best) config, and only the
combined 10-seed mean determines the row. Rationale: induction-head formation is a
training-time phase change (Olsson et al. 2022), plausibly bimodal across seeds near the
budget where it first becomes reachable (my own sanity check's trajectory — see the
noise-floor section — looks like the phase change beginning, not completed, right at the
2000-step edge); a 5-seed mean straddling 0.30 could reflect which seeds were drawn rather
than the underlying protocol, and the noise-floor pilot below only bounds seed variance in
the null (nothing-learning) regime, not variance near the decision boundary itself. This
rule is the pre-registered fix for that specific gap, decided *before* Arms 1-3 run.

Every cell resolves a real question given the escalation rule above; there is no
open-ended "inconclusive, stop here" outcome by design, per methodology v2 item 4's
escape-clause logic (this thread's whole purpose is to be the protocol validation gate 4
requires before any future recall capability claim) — a near-boundary result escalates to
more seeds rather than terminating undecided.

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
   whether 2000 steps is enough — see check 3 — and this check's `0.0052` figure bounds
   seed noise only in the **null regime** (nothing learning, every seed near chance) —
   over 50x smaller than the `~0.27-0.28` gap between chance/control-band and the 0.30 pass
   bar, comfortably resolvable at 5 seeds *if* the real outcome lands far from the
   boundary. **What this check does not establish, flagged directly by the independent
   design review (finding D1) rather than smoothed over: seed variance *near* the 0.30
   decision boundary itself**, which is the variance that actually matters if training
   induces a phase change (see check 3) that different seeds complete at different rates.
   A pre-registered band chosen from null-regime noise alone would risk exactly the
   band-generation blind spot item 6 exists to catch — resolved not by re-running a bigger
   null-regime pilot (which still wouldn't measure boundary variance without touching the
   treatment comparison, violating item 3's own firewall) but by the near-boundary
   escalation rule added to the interpretation matrix above: any arm landing within one
   seed-SEM of 0.30 automatically gets 5 more seeds before a verdict is assigned, rather
   than trusting a single 5-seed mean at exactly the point where trust matters most.
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

**Measured throughput:** 84.1 steps/s (2000-step single-point run, 23.8s) — see the
"Compute budget" section below for the full-grid time estimate derived from this
measurement (updated there to account for Arm 3's two-LR hedge, D3 below).

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

**Reconciliation with the independent design review (finding D2), before any arm runs:**
the reviewer's own read, from the same sanity-check trajectory this doc cites, was Arm
1@2000 `~0.55-0.65` (lower than this doc's original 0.70 — the phase change looked to the
reviewer like it was just *beginning* at step 2000, making a 2000-step pass more of a
coin-flip-plus than a likely win) and overall protocol-validated `~0.85-0.92` (higher than
this doc's original 0.85 — the extended-budget arm, given how sharply loss was still
falling in the sanity check, looked more reliable to the reviewer than a 2000-step-only
read would suggest). Both reads use the same evidence and are within a plausible range of
each other, not a correctness dispute — recorded transparently rather than silently
overwritten, per this repo's pre-registration rule (even though no treatment data exists
yet, so nothing here is "editing after seeing results"). **Adopted final numbers, splitting
the difference deliberately rather than picking either source's exact figure:** Arm
1@2000 `p(pass) ~ 0.62`; overall protocol-validated `p(pass) ~ 0.87`. Harness-defect row
unchanged at `~0.05` (the review did not dispute this figure).

This entry (adopted numbers) is added to the calibration ledger in `RESEARCH.md` section 8
at registration time, with the outcome appended once Arm 1 completes.

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

`~45-50 min` CPU total at 84 steps/s uncontended (Arm 1 `~12 min`, Arm 2 `~10 min`, Arm 3
`~24 min` after the D3 two-LR hedge above; the near-boundary escalation rule adds more
only if triggered), comfortably inside this repo's per-thread budget discipline. No GPU
needed or used. **Contention caveat (design review finding F4):** the independent
review's own re-run of the sanity check hit a measured ~6x wall-clock slowdown from
transient CPU contention (149s vs. this doc's 24s for the identical 2000-step run) — this
sandbox has only 4 cores (`CLAUDE.md`), and another background stream (thread 6 prep) may
be running concurrently. Check `ps aux` before launching the full grid; the *throughput*
figure above is real and reproducible uncontended, but *wall-clock* could stretch
1-3x under contention.

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

**Verdict: approve-with-corrections.** An independent Opus review re-ran every load-bearing
claim itself rather than trusting the write-up (per this repo's standing convention,
applied here for the first time *before* any treatment data exists): the chance-level
check, the short-budget resolvability check, and the single-point full-budget sanity check
all reproduced essentially bit-for-bit (loss 6.375->5.681, accuracy climbing to 0.0703,
throughput within 1.2%). It independently verified `tiny_attention.py`'s correctness
directly rather than by inspection: confirmed the causal mask produces bit-identical
hidden states at position `t` when future positions are corrupted (max deviation
`0.0`), confirmed positional-embedding sizing is correct for both `seq_len=17` and `5`,
and confirmed `recall.py`'s duplicate-key ambiguity is negligible (eval ceiling `>=0.98`,
target never trivially equals the query above chance rate) so a pass cannot be hollow. It
found the item-1 literature-screen reframing sound (not a threads-7/8-style repeat) and
all six gates satisfied on their merits.

**Findings adopted (edits made to this doc before any arm runs, all above):**
- **D1 (substantive):** the noise-floor pilot bounds seed variance in the null regime only,
  not near the 0.30 decision boundary, where a phase-change dynamic could make a 5-seed
  mean noise-dominated. Fixed with a near-boundary escalation rule added to the
  interpretation matrix (auto-escalates to 10 seeds if within one seed-SEM of 0.30).
- **D2:** reviewer's independently-derived p(pass) estimates recorded alongside this doc's
  original numbers and reconciled to adopted final figures (Arm 1@2000: 0.70 -> 0.62
  adopted; overall: 0.85 -> 0.87 adopted).
- **D3:** Arm 3 hedged from one LR to two (best-of-grid plus the next-lower point) against
  a threshold-crossing false-negative risk thread 16's single-LR precedent didn't have to
  worry about.
- **D4:** implementation contract (per-arm `seq_len`, batch/eval seed formulas,
  `EVAL_BATCH=512`) pinned explicitly in this doc rather than left implicit in the pilot
  script.
- **D5:** calibration ledger entry added to `RESEARCH.md` section 8 (was still the empty
  placeholder) as part of this reconciliation.
- **F1:** Arm 2's primary justification reframed to rest on attention's own expected
  near-ceiling performance at this shallower task, with the gated-recurrence 0.32 number
  demoted to approximate/secondary context; Row 4 of the interpretation matrix given a
  one-line caveat about the untested alternative (n_pairs=2 phase-change lag).
- **F2:** noise-floor pilot section reworded to state plainly what it does and doesn't
  establish, rather than leading with the favorable "50x" framing alone.
- **F3:** an unverifiable claim about Elhage et al.'s toy-model widths softened/removed.
- **F4:** compute-budget section updated with the D3 cost increase and a contention
  caveat (the reviewer's own re-run hit a measured ~6x wall-clock slowdown from transient
  CPU contention).

No findings were rejected — every item the review raised was either a direct correctness
confirmation (nothing to fix) or judged worth adopting.

## Status

Pre-registered 2026-07-08 (session-label date). Noise-floor pilot run; independent pre-run
design review complete and reconciled (this section, above). Arms 1-3 not yet run.
