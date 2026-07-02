# Full-portfolio review and idea ranking — 2026-07-07

**What this is.** A review pass over everything in the repo as of thread 13's close: all
thread ideas, designs, code, results, and the recorded interpretations — plus a literature
check of each area (four independent Sonnet literature-review agents: gated-recurrence
recall, edge-of-chaos criticality, muP small-scale transfer, and the three untouched
threads 4/7/8) and an independent Opus meta-review of the code/design/interpretations
(static reading + by-hand numerics; it could not execute code because this session's fresh
container has no torch installed — see section 2.6). Output: corrections to the recorded
record (section 2), literature verdicts on the untouched threads (section 3), new candidate
ideas (section 4), and a single ranked list of everything runnable next (section 5).

**Verification level.** arXiv and most primary sources returned HTTP 403 to the agents'
fetches throughout; literature claims below are triangulated from search-snippet quotes of
abstracts/papers, not full-text reads. Confidence is high for claims corroborated across
multiple independent sources (marked normally), lower for single-source items (flagged
inline). Before any new pre-registration leans on a specific literature claim, read the
actual paper.

## 1. Where the portfolio stands (one-paragraph audit verdict)

The code is correct where it matters (the meta-review specifically re-verified the
Bauer-Fike construction, seed/leakage handling in `recall.py` and `modular_arith.py`, the
gradient-flow diagnostic, the mean-field quadrature, the Theil-Sen implementation, and the
muP Adam multiplier table — all solid). The pre-registration discipline is real and
consistently followed, with one process deviation (section 2.6, item 3). The main issues
found are not bugs but *interpretation calibration*: three recorded conclusions are stated
more strongly than the evidence licenses (sections 2.2-2.4), and one compute-matching claim
is wrong as stated (section 2.1). Separately, a portfolio-level observation: nearly all
budget so far went into measurement-refinement of two ideas (threads 2→12→13, 9→10→11);
the ranking in section 5 deliberately rebalances toward cheap closure of live ambiguities
plus genuinely new mechanisms.

## 2. Corrections to the recorded record

### 2.1 Thread 10's "matched compute" is matched *step count*, not matched FLOPs

The curriculum arm `(n_pairs, steps) = (2,700),(4,700),(8,600)` costs, in
sequence-length-weighted steps (per-step cost of the recurrence is proportional to
`seq_len = 2*n_pairs+1`): `700*5 + 700*9 + 600*17 = 20,000`, vs. the direct-training
control's `2000*17 = 34,000` — the curriculum used **~59% of the control's actual
compute** while the doc calls it "same total 2000-step budget ... matching thread 9's
direct-training total exactly." This does not overturn the verdict (the curriculum got
*less* compute and still failed, so falsification stands, arguably more strongly), but the
"failed at matched compute" framing is not literally true, and it means a
FLOP-matched curriculum (≈3400 steps) was never actually tested. More broadly, only thread
6 ever reports FLOPs/wall-clock; threads 9/10/11 make explicit compute-matching claims in
units (steps) that `docs/methodology.md`'s own non-negotiable says are insufficient.
Dated correction added to `docs/threads/10-curriculum-gated-recurrence.md`.

### 2.2 Gate family (9→10→11): the closing "optimization/learnability limit, not a
capacity or architecture one" is half-right and half-overstated

Two independent qualifications:

1. **The "learnability limit" clause was never decisively tested.** The thread-11 review
   itself specified the deciding experiment — a much more generous budget (10k+ steps,
   and/or repeated-batch overfitting of a small fixed set) to separate "no gradient signal
   exists" from "2000 online fresh-batch steps are too few to learn a general lookup
   algorithm" — and it was never run. What is actually established: not capacity
   (hidden 64→256 didn't help; single-batch memorization works), not init (forced-open gate
   got pushed back closed), and no signal *within 2000 online steps*. The third clause is
   the untested one.
2. **The "not architecture" clause is likely wrong, per the literature.** The recall
   literature review found strong convergent evidence that a *single* scalar-gated linear
   recurrence — the exact family tested — is a known-insufficient architecture class for
   multi-pair recall, independent of optimization: Zoology (Arora et al., arXiv:2312.04927)
   proves (Thm 4.4 + an `m >= N/2p` lower bound) that gated-convolution/gated-recurrence
   ("BaseConv") models need state growing with sequence length where attention doesn't, and
   every working small-recurrent solution in the literature adds one of exactly three
   missing primitives: a shift/short-convolution operator (H3, arXiv:2212.14052; Mamba's
   conv1d; "Convolution Augments Attention," arXiv:2407.05591), two-layer composition
   (induction heads, arXiv:2209.11895; formal 1-vs-2-layer separation, arXiv:2508.07208),
   or an explicit outer-product key-value state (fast weight programmers; DeltaNet,
   arXiv:2406.06484). Our construction has none of the three. So the failure is plausibly
   *both* architectural (missing addressing primitive) *and* optimizational (the observed
   gate-never-moves symptom is the classic gate-saturation pathology, documented with fixes
   in Tallec & Ollivier chrono-init arXiv:1804.11188, Gu et al. arXiv:1910.09890, and
   sidestepped by every modern gated architecture via non-neutral gate init — Mamba's dt
   init, HGRN's gate lower bound arXiv:2311.04823). The dichotomy in the closing note is
   therefore a false one.

   What the literature *does* support in the recorded conclusion: the capacity half.
   hidden=64 / vocab=512 / n_pairs=8 is comfortably slack against Zoology's own
   information-theoretic bound, so "not capacity" stands on outside evidence, not just the
   local hidden-size ablation. Also honestly noted: one recent paper (arXiv:2508.19029,
   single-source) reports modern recurrent models are unusually LR-sensitive on exactly
   these benchmarks — consistent with the repo's decision to sweep LR, but a reminder that
   "no LR in a 5-point grid worked" is weaker than "no LR works."

   Dated correction added to `docs/threads/11-dual-gate-spectral-recurrence.md`.

### 2.3 Criticality sub-line (2→12→13): the residual chaotic-phase anomaly is probably
*predicted physics*, not measurement error — and the recorded framing leans the wrong way

Thread 13 closed with "the depth-scale ordering and peak now clearly track theory, but a
real systematic bias in the chaotic-phase backward pass ... prevents a clean magnitude
match" — implicitly a measurement-quality story. Two independent inputs say the residual
is more interesting than that:

1. **Finite-width theory predicts exactly this failure mode, quantitatively.** Hanin
   (arXiv:1801.03744) proves finite-width gradient fluctuations are controlled by the sum
   of inverse layer widths (≈ depth/width here); Hanin & Nica (arXiv:1812.05994) prove
   `log ||grad||` is asymptotically **Gaussian (grad norm log-normal) with variance growing
   ∝ depth/width**; Roberts-Yaida-Hanin (arXiv:2106.10165) frame `r = depth/width` as the
   expansion parameter, with mean-field controlled only for `r << 1`. Our grid reaches
   depth 362 at width ~256 (`r ~ 1.4` at the far end, ~0.5 at depth 128) — *outside* the
   controlled regime by the theory's own accounting. Under a log-normal with
   depth-growing variance, a median-based estimator (Theil-Sen on per-depth medians) tracks
   the log-median slope, while the theory's `xi` is stated for `E[grad^2]` — the two differ
   by half the depth-derivative of the log-variance. Since the observed log-std grows from
   ~0.06 to ~3.4 over the depth grid, the median slope *must* undershoot the theory slope,
   worst in the chaotic phase where variance grows fastest — matching the observed 2.7-4x
   undershoot and plausibly the `sigma_w2=2.05` sign flip. This is a sharp, pre-registrable
   correction term, not a nuisance (see idea I2, section 4).
2. **The Opus meta-review adds a genuinely competing mechanism.** The per-depth *median*
   itself turns over non-monotonically (rises to depth ~90, then sags) — a median is
   already robust to heavy tails, so a turning-over median is evidence of a real change in
   central tendency, e.g. compounding chaotic growth driving pre-activations into tanh
   saturation along individual trajectories so the *asymptotic linearized* `xi` (derived at
   the fixed point with frozen `phi'` statistics) genuinely stops applying at large finite
   depth. Thread 12's review checked forward saturation only at the last layer on average —
   the wrong quantity for a per-seed accumulated backward product.

Both readings agree the verdicts stand as pre-registered; they disagree on *why* the
chaotic branch misses, and the disagreement is directly testable with per-layer gradient
tracking plus explicit variance-vs-depth measurement (idea I2). The sub-line's closure is
unaffected; the framing correction is added as a dated note to
`docs/threads/13-robust-gradient-flow-depth-scale.md`. Also noted in the literature review:
the "6*xi trainable depth" constant from Schoenholz et al. is treated in later literature
as an admittedly ad-hoc fit ("as yet unexplained" — Pretorius et al. line,
arXiv:1910.05725), and the largest published stress-test found critical init gives no
measurable benefit at moderate depth once practical confounds enter — i.e., thread 2's
loss-based falsification is itself consistent with the published record, not an outlier.

### 2.4 Thread 1: "clean small-scale support" should be read as "construction verified,
design principle still untested"

The meta-review's calibration, adopted here: measurement (i) (the closed-form `L*`
boundary) never involves training and is near-tautological given `A = (1-eps)*orthogonal`
by construction (the thread doc itself says "near-identity"); measurement (iii) (the
free-vs-constrained predictability asymmetry) is close to guaranteed for any construction
that pins the spectral radius. The genuinely empirical, novel content is measurement (ii)
(cross-parameterization agreement within 2x) — which holds (~1.3x), but only after the
diag_lowrank init was deliberately re-engineered (tanh-saturating diagonal init) to make
effective and nominal decay coincide; a different-but-valid diagonal init would reopen the
2-5x effective-eps gap the first review caught. All of this is disclosed in the thread doc;
the calibration point is that RESEARCH.md's "clean small-scale support" summary is a bit
stronger than "the spectral identity is reproducible across two hand-constructed
parameterizations, with the init matched by hand." Also still open (noted in the thread
doc): diag_lowrank never got the closed-form boundary-bracketing treatment. No addendum
needed — the thread doc already carries the caveats; this is a summary-level calibration.

### 2.5 Thread 6 (muP): the adverse smoke reads are real data points but the literature
says the setup, not muP, is the prime suspect — and there's a decisive cheap test

Three inputs, in tension worth recording precisely:

1. The Opus meta-review statically verified `models/mlp.py`'s muP Adam multiplier table is
   **correct** (hidden/output LR ∝ 1/width-mult for Adam, input LR flat, output forward
   multiplier 1/width-mult). So the simplest "wrong multiplier" bug is excluded.
2. The muP literature review found the observed *direction* (muP's raw optimum drifting a
   decade while SP sits flat) essentially unreported anywhere as a genuine muP failure
   mode, and identified the highest-risk residual suspects for a hand-rolled MLP setup:
   embedding/readout-layer treatment (arXiv:2605.21486 finds the embedding-layer LR
   handling accounts for most of muP's practical benefit — single-source), missing
   independent weight decay (arXiv:2510.19093), and the task itself — modular arithmetic's
   grokking dynamics are governed by weight decay/phase transitions, not the smooth
   width-scaling muP theory addresses, and no published muP validation uses an algorithmic
   task (the standard minimal setting is a small LM).
3. The standard, cheap, decisive diagnostic exists and was never run: the **coordinate
   check** (log mean-abs activation per layer type across widths for a few steps at
   aggressive LR; flat-vs-width = correct implementation), plus the "wider is always
   better" check. Both are minutes of CPU, no training sweeps.

Verdict stays "parked, inconclusive," but with a concrete unblocking path (idea I1).

### 2.6 Bookkeeping / process (fixed or acknowledged in this commit)

1. **Thread 06 doc splice** (fixed): the 2026-07-03 addendum had been inserted mid-sentence
   of the 2026-07-02 note's final paragraph, orphaning its continuation at the file bottom.
2. **RESEARCH.md staleness** (fixed): the file header still said "nothing in this repo has
   been run yet"; section 8's header said 2026-07-05 while covering 2026-07-07 results; the
   portfolio table's thread-9 row carried no falsified marker where the thread-2 row did.
3. **Thread 12 → 13 stopping-rule deviation** (acknowledged, dated note in thread 13's
   doc): thread 12 pre-registered "if this fails ... logged as falsified **without a
   further immediate follow-up in this sub-line**," and thread 13 is exactly such a
   follow-up. Thread 13 justified itself by the gate-family three-attempt precedent but
   never acknowledged the contradiction. The honest record is: running thread 13 violated
   thread 12's own stopping rule (even though thread 13 was itself properly pre-registered
   and its result — the shape criterion flipping to a clean pass under a robust estimator —
   was informative). Future stopping rules should either be followed or explicitly,
   *before* the follow-up is designed, amended with a dated note.
4. **Doc dates vs. commit dates**: the addenda dated 2026-07-02..07 were all committed on
   2026-07-01 (wall clock) — the in-doc dates run ahead of git history. Since the commit
   history is the repo's actual notebook spine, treat in-doc dates as session labels, not
   timestamps. (This review doc keeps the internal labeling convention for continuity.)
5. **Environment note in CLAUDE.md is stale for fresh containers**: this session's remote
   container has no torch/numpy installed (`pip install -r experiments/requirements.txt`
   needed first; note the meta-review agent's install attempt through the proxy timed
   out). Flagged in CLAUDE.md.

## 3. Literature verdicts on the untouched / deferred threads

**Thread 4 (optimal-control integrators) — genuinely open; the proposed experiment is
novel; one design amendment required.** No published FLOP-honest Euler-vs-RK2-vs-RK4
depth-reduction comparison exists (prior work — LM-ResNet arXiv:1710.10121, Momentum
ResNets arXiv:2102.07870, ContinuousNet arXiv:2008.02389, HO-ResNet arXiv:2103.15244,
RKCNN arXiv:1802.08831 — compares at fixed depth/params or targets memory/stability, never
matched-FLOP depth ratios). The "trained ResNets aren't in the small-step regime" caveat is
real but subtler than the doc states: Sander et al. (arXiv:2205.14612) show the binding
condition is *depth-smoothness of learned weights*, not small residual magnitude per se,
and it's construction-controllable in the synthetic setting — good for the thread's primary
arm. The biggest unpriced risk: no literature says SGD actually finds stage functions that
behave like local Taylor approximants, so a positive depth-ratio result would be ambiguous
between "truncation-order benefit" and "4 stage evaluations = more capacity per block."
**Amendment:** add a post-training step-refinement diagnostic (insert more blocks/smaller
steps *without retraining*; error should drop at the integrator's order if the ODE story is
real) as a pre-registered disambiguator.

**Thread 8 (Fisher/K-FAC condition number as cross-architecture trainability predictor) —
likely falsified as stated; salvageable with controls.** No direct precedent tests the
exact claim, but the nearest literature is discouraging: zero-cost NAS proxies (NASWOT
arXiv:2006.04647, TE-NAS arXiv:2102.11535) achieve only moderate, search-space-dependent
rank correlations, and NAS-Bench-Suite-Zero (arXiv:2210.03230) documents systematic
failures — proxies latch onto params/depth confounds and don't transfer across search
spaces (cross-*family* is strictly harder). Sokół & Park (arXiv:1810.03785) directly report
init-time FIM conditioning failing to predict learning speed even within one family. And
Adam (the baseline optimizer) already absorbs the diagonal part of what K-FAC corrects,
shrinking the predicted effect. **Amendments if run:** params/depth-matched architecture
pairs as controls; make the *K-FAC-benefit-correlates-with-conditioning* claim (closer to
Amari's actual theory) the primary prediction rather than the raw ranking claim.

**Thread 7 (PAC-Bayes/flatness proxy ranking across architecture families) — likely
falsified on convergent structural evidence; notably, a targeted search confirmed *nobody
has actually run* the exact pooled cross-family correlation, so it is not literally
pre-refuted — but the expected information is still low because the mechanisms that
predict failure are themselves already established.** Jiang et al. (arXiv:1912.02178) — the strongest *positive* flatness
evidence — is entirely within one NIN-family CNN and even there the best granulated scores
sit below the thread's 0.7 bar; Dziugaite et al. (arXiv:2010.11924) find every measure
(PAC-Bayes included) fails completely on at least one variation axis, with worst-case
performance no better than a coin flip; Dinh et al. (arXiv:1703.04933) show naive sharpness
isn't reparameterization-invariant, and "Hide & Seek" (arXiv:2505.05409) shows transformer
symmetries specifically obscure sharpness until a symmetry-corrected (geodesic) version is
used; Andriushchenko et al. (arXiv:2302.07011) find weak/unstable/negative correlations in
modern settings. By the same rule that deferred thread 3 ("a falsification case that is
near-guaranteed a priori is low-information"), thread 7 as pre-registered should not be run.
The genuinely open variant: does a **symmetry-corrected** sharpness (geodesic /
Fisher-Rao) rank across families? That would need its own fresh thread doc.

**Threads 3 and 5 — remain blocked, nothing in this review unblocks them.** Thread 3 still
has no derived `n*`; thread 5 still needs the commutant constraint and a gauge-invariant
alignment metric on paper first.

## 4. New candidate ideas (from the results + literature; none implemented, none
pre-registered yet — each needs its own thread doc before any code)

**I1. muP coordinate check (unblocks thread 6).** Implement `mup.coord_check`-style
logging: mean-abs activation per layer type at init and after 2-10 steps, widths
64→2048, aggressive LR; plus the "wider is always better" curve. Pass/fail is visual and
sharp (flat vs. width per layer type). If it passes, the smoke-test anomaly is a task
artifact (grokking/weight-decay dynamics) and thread 6's real run should swap to a tiny LM;
if it fails, it localizes which layer type's scaling is broken. Cost: minutes of CPU.
Distinct value: thread 6 is the *methodological cornerstone* — every other thread's
"falsify small, trust the trend" premise leans on transfer holding, so resolving it is
worth more than its own thread's finding.

**I2. Finite-width fluctuation test for the criticality anomaly (new thread; the
"structurally different measurement" threads 12/13 called for).** Pre-register Hanin-Nica
directly: (a) `Var[log ||grad||]` grows ~linearly with depth with slope ∝ 1/width (test two
widths, e.g. 128/256/512 — the width dependence is the sharp part); (b) the gap between the
mean-based slope (`log E[grad]`) and median-based slope equals half the variance growth
rate (log-normal identity), quantitatively explaining thread 13's chaotic-branch 2.7-4x
undershoot and the 2.05 sign flip; (c) per-layer gradient tracking to locate where the
median turnover starts — if `E[log phi'^2]` per layer stays at its fixed-point value while
per-seed variance explodes, the finite-width story wins; if deep layers' `phi'` statistics
drift (saturation along chaotic trajectories), the meta-review's finite-depth story wins.
Cost: CPU-trivial (single fwd/bwd passes, existing `deep_mlp.py` + a hook). Distinct value:
converts the sub-line's residual anomaly from "unexplained bias" into a test of the
*correct* (finite-width) theory — and it's the rare case where the repo's falsified-at-
mean-field results become confirmations of the next-order theory if the prediction holds.

**I3. Minimal recall mechanism ladder (new thread; the "structurally different mechanism"
thread 11 called for — and the vehicle for thread 9's still-deferred prediction B).**
Three pre-registered arms on the unchanged recall task at n_pairs=8, matched FLOPs and
tuning budget: (a) **two stacked** gated blocks (tests the composition hypothesis — the
1-vs-2-layer separation is formally proven for attention, open for this family); (b) one
gated block with a **short causal conv (k=2-4)** on the input path (supplies the missing
shift/previous-token primitive; cheapest change, literature-predicted to help); (c) a
**DeltaNet-style outer-product state** `S_t = S_{t-1}*decay + beta_t*k_t(v_t - S_t k_t)^T`
with the decay spectrally constrained à la thread 1 (the literature's highest-confidence
mechanism for recall at this scale). The repo-specific novelty is not "does something solve
recall" (literature says (b)/(c) very likely do) but **prediction B, finally testable**:
once an arm actually learns recall, does the trained model's gradient-flow boundary stay
within factor-2 of the ungated spectral bound? That question — does predictability survive
learned selectivity — is not answered anywhere in the SSM literature and is this repo's
distinctive angle. Cost: (a)/(b) are small diffs on existing models, ~same runtime as
thread 9 (~15-25s/run); (c) is a new model file, still CPU-feasible.

**I4. Generous-budget gate check (closes the gate-family loose end; specified by the
thread-11 review, never run).** 10k+ steps and/or repeated-batch training on a small fixed
set, single best config from thread 11, few seeds. If the write gate still never opens, the
"no discoverable gradient signal" clause is earned; if it opens and accuracy climbs, the
gate-family closure gets re-characterized as a budget artifact (and the sub-line reopens
under a fresh thread). Either outcome directly firms up or corrects a bolded conclusion in
RESEARCH.md at ~1-2 CPU-hours. Note: this is *not* a fourth gate variant (no architecture
change) — it tests the recorded claim about the existing ones. Chrono-style gate re-init,
which the literature says would likely work, *is* a fourth variant and stays excluded under
thread 11's closing rule; recorded here only as context.

**I5. Fixed-point-matched inputs / orthogonal-init arm for criticality (companion to I2,
same new thread or a second one).** Draw inputs with `q_0 = q*` (and pairs at `c_0 = c*`)
so the theory's asymptotic regime applies from layer 1 (kills the transient confound
thread 12's review identified), and add a delta-orthogonal-init arm (Xiao et al.
arXiv:1806.05393) — theory says dynamical-isometry init removes the spectral spread, so if
the chaotic-phase heavy tails shrink under orthogonal init, they were a Gaussian-init
finite-width artifact, further confirming I2's story. Cheap; shares all I2 infrastructure.

**I6. Thread 4 with the step-refinement amendment (per section 3).** The one untouched
thread the literature actively recommends running as-designed (with the disambiguator
added). Genuinely open question, novel comparison, small compute.

## 5. Ranked portfolio (all runnable ideas, most promising first)

Ranking criterion: expected information per unit cost, weighted by how directly the result
feeds the repo's thesis (math-derived, falsifiable, small-scale-measurable) and by the
literature prior (a prediction the literature already firmly expects to fail is
low-information *as specified* — thread 3's deferral logic, applied uniformly).

| Rank | Idea | Why here | Cost |
|---|---|---|---|
| 1 | **I1 muP coordinate check** | Cornerstone-of-methodology leverage; decisive either way; trivially cheap | minutes |
| 2 | **I2 (+I5) finite-width fluctuation test** | Turns two recorded anomalies into a sharp test of the next-order theory; all infrastructure exists; distinguishes the two live explanations (finite-width vs. finite-depth) | < 1 CPU-hr |
| 3 | **I4 generous-budget gate check** | Directly tests a bolded recorded conclusion; pre-specified by the thread-11 review; cheap | 1-2 CPU-hr |
| 4 | **I3 recall mechanism ladder (a→b→c)** | The portfolio's first genuinely *new mechanism* since thread 9; carries the deferred, repo-distinctive prediction B (predictability under learned selectivity); literature gives it a real chance of the repo's first big positive | ~thread-9 scale per arm |
| 5 | **I6 thread 4 integrators, amended** | Literature-confirmed novel comparison, honest open question, bounded compute | < GPU-day |
| 6 | **Thread 6 real run (tiny LM), if I1 passes** | The pre-registered verdict still needs a non-grokking task at real width range; do after I1, likely on GPU | GPU-day+ |
| 7 | **Thread 8 Fisher/K-FAC, amended** | Adjacent literature says likely falsified; the amended (K-FAC-benefit) version is worth running only once ≥2 genuinely different novel layers exist (e.g. after I3c) | < GPU-day |
| 8 | **Thread 7 flatness — only as the symmetry-corrected variant** | As written: strongly predicted false on established structural grounds (though the exact pooled test is unrun) → low information. Geodesic/Fisher-Rao variant is open but needs analysis work first | analysis first |
| 9 | **Thread 3** | Still blocked on deriving `n*` | analysis only |
| 10 | **Thread 5** | Still blocked on commutant + gauge metric; highest bitter-lesson risk | analysis only |

Sequencing note: ranks 1-3 are collectively under an afternoon of CPU and each closes or
corrects something already on the record — do them before opening any new sub-line. Rank 4
(I3) is the natural next *thread* (new mechanism, new pre-registration, carries prediction
B). Ranks 5+ are the first calls after that, in order.

## 6. Sources

Agent memos (full text in session transcript, not committed): recall/gated-recurrence
review; edge-of-chaos/finite-width review; muP transfer review; threads-4/7/8 review; Opus
code/design meta-review. Key papers cited above by arXiv ID throughout. Reminder: snippet-
level verification only (403-blocked primary fetches) — read the primary papers before
building a pre-registration on any single-source claim (flagged inline in sections 2-4).

---

**Correction note, 2026-07-08 (from the program regroup's independent review — see
`docs/reviews/2026-07-08-program-regroup.md`):** section 2.3's finite-width receipt
contains a numerical error. It reads threads 12/13's grid as "depth 362 at width ~256
(`r ~ 1.4` at the far end, ~0.5 at depth 128)". The width is actually fixed at
`hidden=32` in both scripts (`experiments/scripts/thread12_gradient_flow_depth_scale.py:33`,
`thread13_robust_gradient_flow.py:29`) — 256 is a value on the *depth* grid, not the
width — so `r = depth/width` reaches ~11.3 at depth 362, identical to thread 15's
narrowest cell (thread 15's own doc states this correctly, so the record was internally
contradictory until now). The section's argument is unaffected and, if anything,
strengthened: the grid sits roughly 8x further outside mean-field's `r << 1` validity
regime than stated. The wrong figure had been imported into thread 13's addendum and
CLAUDE.md's summary; dated/in-place corrections were added there in the same commit as
this note.
