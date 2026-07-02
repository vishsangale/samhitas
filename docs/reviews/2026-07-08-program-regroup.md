# Program-level regroup: why is everything falsifying? — 2026-07-08

*(Session-label date, per the doc-date-vs-commit-date convention recorded in the
2026-07-07 portfolio review, section 2.6.4; wall clock at writing reads 2026-07-02.)*

**What this is.** The user asked directly: "we built this repo to find new NN
architectures through mathematical foundations and we are getting all negative results —
what is wrong with the setup?" This document is the answer, produced per the repo's own
regroup precedent (`docs/reviews/2026-07-07-portfolio-review.md`): a program-level audit
of *why* the verdict stream is nearly all falsifications, what that does and does not
imply, and a decided course correction. The 2026-07-07 review audited thread-level
correctness and interpretations; this one audits the program's design. Per this repo's
standard practice, an independent Opus review (reasoning-only, instructed to verify every
claim against the repo record and to attack the reasoning) was commissioned on this exact
draft; its reconciliation is recorded as a dated note at the bottom of this doc, and
nothing above that note is silently edited after the fact.

## 1. The scorecard, recounted honestly

Sixteen headline pre-registered outcomes exist as of thread 17's close. Literal tally:
**12 falsified as specified, 1 misleading literal pass (thread 16), 1 supported (thread
1), 1 parked after adverse smoke reads (thread 6), 1 still formally deferred (thread 9's
prediction B, pending any future A-pass).** So "all negative results" is ~81% literally true at the verdict level,
and 100% true at the level that matters to the user: **no candidate architecture idea has
ever survived its own test, and the repo is no closer to a new architecture than at
thread 0.**

But the falsifications are not one kind of event. Classifying each by what the post-hoc
review actually found (every classification below is the *review's* conclusion, already
on record in the thread docs — nothing here is reinterpreted after the fact):

| Verdict | Threads | What actually failed |
|---|---|---|
| **Apparatus miscalibration** — the pre-registered bar, not the hypothesis, was wrong (reviews reversed the verdict in substance) | 2, 12, 13, 14, 15A, 15B, 16 | Bands imported from asymptotic theory into regimes the experiment provably wasn't in (13, 15A — the softest calls in this bucket: the finite-width rescue they lean on is itself only *qualitatively* supported, thread 15's own quantitative test of it having failed; see the reconciliation note), a bar contradicting the theory's own documented mechanism (14), review-named task/LR-ceiling/threshold confounds (2) and a heavy-tail-corrupted global fit (12), a test with noise floor larger than the effect (15B), a bar so weak its "pass" was meaningless (16) |
| **Genuine hypothesis kills** | 9, 10, 11, 17a (narrow claim), 17c; 16 refined the mechanism story | Real, mechanism-level negative knowledge — but concentrated on a single task/protocol: the gate-family kills (9/10/11) were literature-pre-answered (Zoology, arXiv:2312.04927, Thm 4.4 + the `m >= N/2p` bound, published before thread 9 was written) and are themselves mixed rather than clean — they used the neutral gate init the literature's working gated architectures deliberately avoid, and the 2026-07-07 review (2.2) already calls their optimization-vs-architecture dichotomy false — and the two genuinely-open kills (17a's narrow stacking claim; 17c's write-gate-never-opens finding, which ran *against* the literature's optimistic prior for delta-rule mechanisms) both sit on the same protocol RC2 shows was never validated as learnable by anything |
| **Unfair tests** — a named construction confound did the killing | 17b; 17a (broad claim) | Default-random conv init + GELU scrambling the gate's input (17b); stacking two saturated-closed gates with no residual/norm (17a) — in both cases the review named the omitted standard component as the cause |
| **Supported** | 1 | With the 2026-07-07 calibration: one genuinely empirical result (cross-parameterization agreement), two near-tautologies |
| **Parked adverse** | 6 | Two toy-scale smoke reads against muP transfer; the pre-registered real-scale run has never been executed |

Restated: of the 13 negative-reading verdicts, **7 were the measurement apparatus
falsifying its own bars**, **2 were constructions falsifying their own omitted
engineering**, and **5 (net of overlap) were real kills — 3 of them (the gate family) of
a hypothesis the literature had already answered (and even those carry a known init
confound: mixed, not clean, kills), and the 2 genuinely open ones (17a-narrow, 17c)
confined to a single protocol that RC2 below shows was never validated as learnable by
anything.** The program has mostly been learning about its own bars,
regimes, and constructions, not about the mathematics it set out to test. That is the
core answer to "what is wrong with the setup."

Two secondary observations that sharpen it:

- **All actual information has come from the post-hoc reviews, not from the verdicts.**
  In 7 of 16 outcomes the literal pass/fail bit was substantively reversed or heavily
  requalified on review. A verdict channel that gets overturned ~half the time carries
  almost no information on its own; the expensive part (independent review) has been
  carrying the whole program.
- **Predicted-pass-rate calibration is broken in the falsifying direction.** If
  pre-registered bands encoded honest uncertainty, some would pass. Twelve falsifications
  out of thirteen negative-reading verdicts (and the thirteenth passed only through a
  mis-set bar) means the band-generation step is systematically overconfident about what
  the theory implies *in the regime actually run* — not necessarily that the theories are
  all wrong (though see section 5's named risk for the reading where they are). Threads 13/15 are the clearest case: the "failures" match the
  *next-order* (finite-width) theory's own predictions, i.e. the physics was right and the
  band was derived from the wrong (leading-order) formula for the regime.

## 2. Root causes

**RC1 — Quantitative bands from asymptotic theory, tested in regimes that violate the
theory's own validity conditions.** The CPU sandbox forces small widths, short budgets,
and moderate seed counts; the bands kept being derived from infinite-width /
leading-order / asymptotic results. Receipts: threads 12/13/15 all run the same fixed
`hidden=32` MLP, so the entire criticality grid reaches `r = depth/width ~ 11` at depth
362 where mean-field needs `r << 1` — the 2026-07-07 review's 2.3 stated `r ~ 1.4` for
threads 12/13, but that figure misread a *depth-grid* value (256) as the width; verified
against `experiments/scripts/thread12_gradient_flow_depth_scale.py:33` and
`thread13_robust_gradient_flow.py:29` during this regroup's independent review, with
dated corrections added to that review doc and thread 13's addendum in this commit
(thread 15's own doc had it right all along). Thread 15's review traced its "failure" to
exactly this out-of-regime testing (`Var[log grad]` convex, not linear, once `r` is this
large); thread 2's "trainable within 150 steps" binary threshold inverted rankings at the
boundary; thread 15B's tested quantity had a bootstrap noise floor at or above the effect
size at 50 seeds. None of these falsifications is evidence against the theories *in their
own regimes* — they measured the mismatch between band and regime. The converse caution
(per the independent review): reclassifying them as apparatus errors does not *vindicate*
the theories quantitatively at reachable scale either — thread 15 was the purpose-built
quantitative test of the finite-width rescue and itself failed both predictions; only its
qualitative diagnostics support the rescue. The quantitative story is open, not settled
in the theory's favor.

**RC2 — No positive controls where they mattered most, and an unvalidated task
protocol.** The measurement threads did include controls (thread 14's SP arm, thread 15's
variance-growth control) — and thread 14, the one thread designed around a
known-should-fail reference arm, produced the single most decisive resolution in the
repo. The architecture threads never did. Concretely: across seven recall experiments
(9, 10, 11, 16, 17a/b/c), **no model of any kind has ever exceeded 0.04 held-out accuracy
on the online `n_pairs=8` protocol** — every trained model scores below even the trivial
"always output the most recent value" heuristic (0.137, per thread 16's review) — and
**no known-sufficient architecture (i.e. a small attention model, the literature's
standard positive control for this exact task) has ever been run on the protocol.** There
is no attention model anywhere in `experiments/models/`. Thread 16's Arm B (fixed-pool
memorization) was a partial control but, per its own review, only rules out catastrophic
dead gradients — it says nothing about whether the *online* protocol is learnable by
anything at this scale/budget. Consequence: the entire recall-negative cluster — the
repo's largest block of "results" — cannot currently distinguish "these mechanisms are
insufficient" from "this protocol/budget is unlearnable by any ~50k-param model, and the
harness has been failing everything indiscriminately." The Zoology literature makes the
first reading likely, but the repo's own evidence cannot separate them. That is a serious
hole for a falsification shop.

**RC3 — The literature screen ran at the end instead of the beginning.** The 2026-07-07
review's own ranking criterion ("a prediction the literature already firmly expects to
fail is low-information as specified" — thread 3's deferral logic) was applied to threads
7/8 only *after* they had sat in the portfolio for weeks, and was never applied to the
gate family at all: Zoology's bound predated thread 9 and already answered it, and the
observed gate-never-opens symptom is the classic, named gate-saturation pathology with
published fixes (chrono init, arXiv:1804.11188; HGRN's gate lower bound, arXiv:2311.04823
— portfolio review 2.2). Five thread-equivalents of budget went into a question whose
answer was on arXiv before the first line of code. The muP smoke-test saga has the same
shape: no published muP validation uses an algorithmic/grokking task (2.5), and the task
was flagged as the prime suspect only at review time.

**RC4 — "Minimal faithful construction" kept stripping engineering the literature says is
load-bearing, so failures indict the omission, not the math.** Thread 17b: a
default-random depthwise conv + GELU in front of the gate — the review measured ~7x
gradient starvation and named the fix (near-identity init) that Mamba's conv1d uses *by
design*. Thread 17a: two saturated-closed gates stacked with no residual/normalization —
the review measured 36x inter-block attenuation and a ~167x gate-gradient deficit and
named the omitted standard components. The gate family itself used neutral gate init
where every working gated architecture in the literature deliberately doesn't (2.2).
Building minimal constructions *up* from broken and reading each failure as information
about the mechanism class re-derives, expensively, why the published recipes contain the
parts they contain. (Thread 17c is the honorable exception — its review hunted for a
rescuing confound and found none — which is exactly why it's the one arm that produced a
genuinely new data point.)

**RC5 — Task monoculture, and both tasks are repeat confound sources.** Two tasks exist
(`modular_arith.py`, `recall.py`). Modular arithmetic doesn't need depth (thread 2's
review: added depth is a pure handicap, regardless of criticality) and has
grokking/weight-decay dynamics that poison width-scaling reads (threads 6/14). The recall
protocol is unvalidated (RC2). `docs/methodology.md`'s own diagnostic-task list is half
unbuilt: no tiny char-LM (named as *required* for thread 6's real verdict), no small
vision task (named for depth/criticality predictions — where the task-needs-depth confound
of thread 2 would not have occurred).

**RC6 — The compute allocation inverts the plan, so the program structurally cannot reach
its own positive product.** The methodology budgets a GPU-day per falsification and gates
"supported" threads into scaling checks (step 5). In practice 100% of runs were CPU-toy;
step 5 has never once triggered; and the one thread whose *entire job* is to establish
that small-scale findings transfer — thread 6, promoted to priority 1 by the draft-v0
advisor review (RESEARCH.md section 5) precisely because the whole "falsify small, trust
the trend at scale" premise rests on transfer holding — is parked with two *adverse*
smoke reads and its pre-registered real run unexecuted.
The program is operating on credit against a keystone assumption for which its only
evidence is currently negative. Nothing built on "falsify small" is secure until thread 6
runs at the scale its own pre-registration demands.

**RC7 — The success criterion drifted.** `docs/methodology.md` step 4 declares "negative
results are the actual product of this repo." That framing is only healthy if the
negatives are *novel* and *load-bearing* (kills of live, open hypotheses). Per section 1,
most weren't. Meanwhile the genuinely novel claims in the portfolio — thread 17's
prediction B (does gradient-flow predictability survive learned selectivity: per the
portfolio review, unanswered anywhere in the SSM literature), thread 4's FLOP-honest
integrator comparison (literature-confirmed open), thread 6's transfer verdict at real
scale — are exactly the items that never got a real shot: B is formally moot (gated on an
A-pass that never came), 4 is untouched, 6 is parked. Execution order has been
cheapest-first, which in this portfolio correlated with least-novel-first.

**Named contributors the record documents repeatedly** (folded under RC1/RC2/RC5 above;
listed separately, per the independent review, so they stop being ambient): (a) **chronic
seed underpowering** — thread 2's re-check found per-seed init noise dominating the depth
trend at one point (~9x off), thread 15B was judged likely never resolvable at 50 seeds;
(b) **the 2000-step budget convention**, inherited unexamined by six recall threads —
thread 16 showed the headline "gate never moves" symptom was partly a step-budget
artifact (the gate moved 4.7x at 6x budget); (c) **per-architecture LR-grid adequacy** —
the fixed 5-point grid saturated at its own ceiling for deep nets (thread 2) and was
mis-centered ~10x for the conv arm (17b); the 2026-07-07 review's caveat that "'no LR in
a 5-point grid worked' is weaker than 'no LR works'" applies to every negative in the
recall cluster; (d) **`n_pairs=8` difficulty was never calibrated to the achievable
budget** — known since thread 9 that the same construction reaches 0.32 at `n_pairs=2`
and collapses by 8, yet 8 stayed the bar for every subsequent mechanism. Thread 18's
design must handle (b)-(d) explicitly (see C2).

## 3. What is *not* wrong (do not fix these)

- **The code.** After the first-week fixes (thread 6 v1's leakage bug, `report.py`'s
  ratio=inf bug), every independent review — ten-plus of them — reproduced every headline
  number, several bit-for-bit. No falsification was ever traced to a harness bug. The
  user's instinct to suspect "the setup" is right, but the setup's *software* layer is
  clean; the defect is in experiment design (bars, regimes, controls, and what gets
  tested).
- **The discipline.** Pre-registration, dated addenda, no-retrofit, independent review:
  this machinery is what makes this diagnosis writable at all, and it repeatedly caught
  motivated reasoning in real time (thread 6's circular effective-LR argument, thread 12's
  wrong causal story, thread 17c's shortcut-contaminated sanity check, thread 16's
  misleading literal pass). Keep all of it. The fix is to aim it better, not to loosen it.
- **The falsification rate per se is not the disease.** It is the symptom. A portfolio of
  genuinely risky, novel, in-regime predictions would still falsify often — but the
  falsifications would be *findings*. The disease is that most of these weren't.
- **Real knowledge was produced, and the bookkeeping under-reports it.** A fair ledger:
  (i) the criticality arc (2→12→13→15) delivered a replicated qualitative positive —
  gradient-flow depth-scale peaks at criticality with the correct sign structure, ordered
  phase quantitatively within ~1.65x, chaotic-phase deviation *qualitatively* consistent
  with predicted finite-width corrections (the quantitative version remains unconfirmed —
  thread 15's own test of it failed) — currently filed as four falsifications; (ii) the recall
  cluster delivered one solid mechanism finding (write gates converge to a recency-copy
  shortcut and re-close under forcing, across three distinct write mechanisms) plus a
  clean capacity-is-not-sufficient data point (17c); (iii) two validated tools (muP
  machinery, mean-field numerics). Not nothing — but roughly one thread's worth of net
  science for seventeen threads of budget, which is the honest quantification of the
  user's complaint.

## 4. Course corrections (decided)

**C1 — Methodology amendment v2 (adopted with this commit, additive, in
`docs/methodology.md`).** Six new pre-registration requirements for every future thread:

1. **Literature screen before pre-registration, with a kill rule.** One section, written
   before bands are set: what does the closest published work say about this exact claim?
   If the literature firmly expects the result (either direction), the claim is
   low-information as specified — reshape it toward what is genuinely open or drop it.
   (The 2026-07-07 review's rank criterion, promoted from review-time to design-time.)
2. **Regime-validity statement.** The theory's own validity conditions (width, depth,
   `r = depth/width`, step-count asymptotics, ...) stated next to the experiment's actual
   values. Where the experiment cannot reach the valid regime, only sign/ordering/shape
   predictions may be pre-registered — no absolute quantitative bands imported from
   out-of-regime formulas.
3. **Noise-floor pilot before bands freeze.** Run the estimator on null/synthetic or
   pilot data to establish the resolvable effect size at the planned seed count; a band
   narrower than the measured noise floor is inadmissible (thread 15B's failure mode,
   made impossible by construction).
4. **Positive-control arm for any capability claim.** Every "mechanism X enables
   capability Y" experiment includes an arm known (from literature or prior threads) to
   achieve Y under the same harness/budget, or — if none exists yet — the thread is
   blocked on first validating the protocol itself (see C2). Measurement threads keep
   doing what threads 14/15 already did.
5. **Stated prior pass-probability + calibration ledger.** Each pre-registered prediction
   carries a one-line "expected p(pass) ~ X" at registration; outcomes accumulate in a
   ledger. If the ledger keeps reading 0%, band-generation is broken again — that is now a
   monitorable failure mode instead of a felt one.
6. **Independent pre-run design review.** The finished pre-registration — literature
   screen, regime-validity statement, noise-floor pilot result, bands, controls, and
   stated priors — gets an independent review *before* the experiment runs. Added at the
   independent reviewer's insistence, and they are right: all 10+ reviews in this repo's
   history audited results after the fact, while the failure mode this regroup diagnoses
   (overconfident band-generation) happens before the run, where no reviewer has ever
   looked — and amendments 1-5 are themselves self-authored checklists written by the same
   process that set the mis-calibrated v1 bands. This gate is the structural answer to
   that circularity.

**C2 — Next experiment: validate the recall protocol itself (new thread 18, cheap,
CPU-feasible).** A minimal 2-layer attention model — the literature's known-sufficient
reference for associative recall — with the *exact* existing protocol (vocab=512,
hidden=64, `n_pairs=8`, 2000 online steps, matched tuning-budget rules) as the primary
arm. Crucially (added per the independent review, which is right that the naive version
of this thread could fail undiagnosably): the control itself must be *fair*, or thread 18
just reproduces RC4 one level up — a failing attention arm would not separate "protocol
unlearnable" from "control under-configured." So the design includes an interpretability
ladder around the primary arm: an LR grid centered for attention (not inherited from the
recurrence sweeps), adequate positional information (attention has no recurrence to
encode order), an `n_pairs=2` arm (known reachable by even the gated recurrence, so any
architecture failing *there* indicts the harness outright), and an extended-budget arm
(so a 2000-step failure separates budget from learnability — named contributor (b)). Two
pre-registrable outcome families, both decisive: the primary arm clears the 0.30 bar →
the protocol is learnable, the recall-negative cluster is confirmed as evidence about
*mechanisms*, and the harness permanently gains its missing positive-control arm; the
full ladder fails → the protocol/budget was the confound all along, the recall cluster
gets a dated protocol-artifact caveat, and every future capability claim gets a budget
bar calibrated by what attention actually needed. This is **not** a
reopening of the gate family (no gated-recurrence variant is involved; thread 11/17
closure rules are untouched) — it is harness validation, i.e. the retroactive application
of C1.4 to the repo's largest existing result cluster. Needs its own thread doc (with the
C1 sections) before code, per standing rules.

**C3 — The keystone experiment: thread 6's real run, at its pre-registered scale.**
Highest-stakes open item in the portfolio: the program's operating premise ("falsify
small, trust the trend") currently has *adverse* toy-scale evidence and zero real-scale
evidence. Sequence: build the tiny char-LM task (`experiments/tasks/` — named by the
thread doc and the 2026-07-07 review as the required setting; kills the grokking-task
confound), smoke-test at CPU toy scale, pin the full protocol (widths well past 16x per
the thread's revised width note, multi-threshold steps-to-target reporting, coordinate
check re-run at the real widths), then execute on the user's own GPU hardware per the
standing compute arrangement. Two execution corrections from the independent review,
both adopted: (i) this thread's CPU prep is independent of thread 18 and runs **in
parallel** with it, not after it — strict sequencing would idle the keystone for no
reason; (ii) the sandbox→user-GPU handoff has never been exercised end-to-end even once
(step 5 never triggered in 17 threads), so the handoff itself is a deliverable to prove,
not an assumption: the concrete output is a packaged, self-contained run script with
bands frozen *before* handoff, plus a results-ingestion template the user runs and
returns. If muP transfer holds there, every future small-scale
verdict gains its bridge to scale; if it fails there too, that is a *major, publishable
negative* about the repo's method itself, and the honest response is to narrow the
program to init-time/mechanistic predictions (the kind that need no transfer assumption).
Either outcome is the most valuable single result currently available to this repo.

**C4 — Portfolio hygiene (standing decisions).** Thread 4 (integrators, with the
2026-07-07 amendment) is the designated next *new-science* thread after C2/C3 — the one
untouched item the literature calls genuinely open. Thread 7 as written stays dead
(symmetry-corrected variant only, and only after the analysis work). Thread 8 stays
deferred until ≥2 genuinely novel layers exist to rank. Threads 3/5 stay blocked on their
derivations. Any future recall-mechanism attempt starts from a *published working recipe
and ablates down* toward the minimal mathematical core (so failures indict the removed
component), never builds up from scratch-minimal again — RC4's lesson as a standing rule.
No new gate variants, unchanged.

**C5 — Reporting: separate "verdict" from "what we learned."** RESEARCH.md section 8
entries and thread addenda now also carry a one-line *net-knowledge* statement alongside
the pass/fail verdict (the calibration ledger of C1.5 lives with it). The criticality
arc's qualitative positive and the recall cluster's mechanism findings should read as
what they are in any future summary — the current summaries systematically undersell the
program to its own author, which is part of why the felt experience is "all negative
results."

## 5. What this does *not* promise

Honesty about the ceiling: the mathematical sources this portfolio drew on (muP,
mean-field criticality, Lyapunov stability, K-FAC geometry, PAC-Bayes) are exactly the
veins the field has mined hardest for architectural consequences — that is *why* so many
of the answers were already in the literature. Under the corrected process, the realistic
near-term wins are sharp, narrow, mechanism-level results (a transfer verdict at scale, a
validated protocol, thread 4's integrator ratio, prediction B if a mechanism ever earns
it) — not a new Mamba. If a new architecture ever comes out of this repo, it will come
from a novel mechanism-level invariant first and an architecture second. Course-correcting
to that expectation is part of the correction.

And one named risk this regroup cannot rule out (elevated to explicit status at the
independent review's insistence): **the weak link may be the thesis-level bet itself, not
the process.** The record is equally consistent with a harder reading — that these
particular, heavily-mined mathematical veins simply do not contain *new*,
*small-scale-falsifiable* architectural signals: their real consequences either live at
scales this method cannot reach, or are already published. If that is what's true,
amendments v2 will produce better-calibrated bars that still mostly falsify or re-derive
— just labeled more honestly. The observable that separates the readings, committed in
advance: if the next batch of threads, run under v2 (honest bars, valid regimes, fair
controls, pre-run review), *still* yields only re-derivations and apparatus-independent
falsifications, the correct response is to change veins (different mathematics) or change
scale (the GPU tier as the default, not the exception) — not to iterate on process a
third time.

## 6. Decision

Adopted with this commit, as reconciled with the independent review (dated note below):
C1 (methodology amendments v2, six items), C4, C5. **Next experimental work, in
parallel: C2 (thread 18, recall-protocol validation with its interpretability ladder)
and C3's CPU prep (tiny char-LM task, protocol pinning, packaged GPU handoff for thread
6).** Thread 6's GPU execution hands off to the user's hardware as soon as its package is
frozen. Thread 18's pre-registration doc is the next artifact to write, and must itself
satisfy all six C1 sections — including being the first pre-registration to pass through
the new pre-run design review gate.

---

## Independent review reconciliation, 2026-07-08

Per the preamble's commitment, this dated note records the independent Opus review's
findings on the committed first draft (`f1e8f0c`) and every change made to the body above
in response — the body was edited, but not silently; this note is the changelog.

**Review verdict: approve-with-corrections.** The reviewer verified the scorecard tally,
the classification-to-thread-doc mapping, and every checked number against the record
(explicitly confirming, among others: the 12/1/1/1/1 tally and 16-outcome enumeration;
the 7/2/5 partition mechanics; all eight recall accuracies and the 0.04 ceiling; the
0.137 recency heuristic; thread 16 Arm B's dead-gradients-only scope; the absence of any
attention model in `experiments/models/`; threads 14/15's controls; 17b's ~7x starvation
and 17a's 36x/~167x figures; the Zoology/chrono-init/HGRN citations and their timing; the
150-step threshold, 1.65→0.57 ordered-phase ratios, and 2.7-4x chaotic undershoot; the
two named early harness bugs). It also confirmed the methodology amendments are
internally consistent (the noise-floor pilot does not violate commit-bands-before-running
because it is firewalled from the treatment comparison; the positive-control requirement
does not conflict with the gate-family closure rules) and that thread 18 is fairly
characterized as protocol validation rather than a gate-family reopening.

**Factual corrections adopted (body edited accordingly):**

1. **The `r ~ 1.4` receipt was wrong — the true value is `r ~ 11`** (RC1). Threads 12/13
   hard-code `hidden=32` (`experiments/scripts/thread12_gradient_flow_depth_scale.py:33`,
   `thread13_robust_gradient_flow.py:29`), so their grid reaches `r ≈ 11.3` at depth 362 —
   identical to thread 15's narrowest cell, not the `r ~ 1.4` the 2026-07-07 portfolio
   review's 2.3 states (it misread a depth-grid value, 256, as the width; thread 15's doc
   line 21 had it right all along, so the committed record was internally contradictory
   and the first draft picked the wrong branch). Verified against the scripts before
   adoption. RC1's conclusion strengthens (~8x further outside validity than claimed).
   Dated correction notes added to the portfolio review doc and thread 13's addendum in
   this commit; CLAUDE.md's summary fixed in place (it is the living orientation file).
2. **"Permanently deferred" overstated 9B's status** — the record says formally deferred
   pending any future A-pass. Section 1 now says so.
3. **RC6's attribution and quotation were loose** — the thread-6 promotion belongs to the
   draft-v0 advisor review (RESEARCH.md section 5), and the quoted premise phrase is now
   verbatim rather than a paraphrase in quote marks.
4. **"Twelve-plus consecutive falsifications" contradicted this doc's own section 1**
   (thread 16 is a literal pass inside that run). Now stated exactly.

**Substantive corrections adopted (body edited accordingly):**

5. **The criticality arc's "apparatus miscalibration, physics fine" framing was the most
   theory-flattering reading available** — thread 15 was the purpose-built quantitative
   test of the finite-width rescue and itself falsified both predictions; only its
   qualitative diagnostics support the rescue. Section 1's table, RC1, and section 3 now
   carry this: the qualitative mechanism reproducibly holds; no quantitative version is
   confirmed at reachable width; open, not vindicated.
6. **The gate-family kills are mixed, not clean** — by RC4's own logic (neutral gate init
   where the literature's working gated architectures deliberately avoid it; portfolio
   review 2.2 already calls the dichotomy false). Scorecard adjusted.
7. **A sixth methodology amendment added — independent pre-run design review.** The
   reviewer's sharpest process point: amendments 1-5 are self-authored checklists written
   by the same process that set the mis-calibrated v1 bands, and all 10+ reviews to date
   ran post-hoc on results while the diagnosed failure happens pre-run. Adopted in C1 and
   `docs/methodology.md`.
8. **Thread 18 as first-drafted could fail undiagnosably** — an exact-protocol-only
   attention arm that failed would not separate "protocol unlearnable" from "control
   under-configured" (RC4 one level up). C2 now specifies the interpretability ladder
   (attention-centered LR grid, positional information, `n_pairs=2` arm, extended-budget
   arm).
9. **Thread 6 sequencing and actionability** — its CPU prep now runs in parallel with
   thread 18 rather than after it, and the never-exercised sandbox→GPU handoff is named
   as a deliverable (packaged run script, bands frozen before handoff, results-ingestion
   template) rather than assumed.
10. **Four named contributors elevated from ambient to explicit** (section 2 coda): seed
    underpowering, the unexamined 2000-step convention, per-architecture LR-grid
    adequacy, and `n_pairs=8` difficulty never calibrated to budget.
11. **The thesis-level alternative elevated to a named risk with a pre-committed
    observable** (section 5): if a v2-run batch still only re-derives and falsifies, the
    response is to change mathematical veins or default to GPU scale — not a third round
    of process iteration.

**Reviewer positions noted but not requiring body changes:** the 7/2/5 partition is
directionally robust with soft edges exactly where flagged above (threads 2 and 13 are
the softest "apparatus" calls — thread 2's headline confound is as much task-design as
estimator, which RC5 owns); thread-18-first vs thread-6-first resolves to
"parallelize," now adopted; and the diagnosis stands as a *process* diagnosis while
remaining agnostic — per item 11 — on whether the deeper problem is the thesis bet.
