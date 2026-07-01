# Mathematical Foundations for Neural Architecture Search — Research Plan (Draft v1)

Status: draft v1, revised after an adversarial review pass (see section 5). Everything
here is still a hypothesis, not a result — nothing in this repo has been run yet.

## 1. Thesis

Find new neural network architectures by starting from mathematics rather than from
architecture-tweaking. For each candidate idea:

1. State the mathematical structure being imported (a stability theorem, a symmetry group,
   a phase transition, an operator identity, a control-theoretic principle).
2. Derive an architectural consequence — a layer, a parameterization, an init scheme, a
   scaling rule.
3. Derive a **falsifiable, quantitative prediction** from the theory — not "this should
   work better" but "this specific quantity should behave this specific way, and if it
   doesn't, the theory is wrong." Pre-registered before the experiment runs (section 5).
4. Run the cheapest experiment that can break the prediction. Most of the research budget
   is spent here, at small scale.
5. Only if a thread survives (1)-(4) does it earn a scaling check: a small sweep across a
   handful of model sizes to see whether the effect's sign and (roughly) its trend hold as
   compute grows, using scaling-law fitting rather than a single large run.

This is closer to how S4/Mamba (HiPPO + linear control theory), RoPE (rotation group
action on queries/keys), muP (infinite-width mean-field limits), Fixup/ReZero init
(dynamical-systems stability of residual stacks), and low-rank adapters (linear algebra of
weight-update rank) actually originated — a specific piece of math, tested small, that
turned out to transfer.

## 2. Corrections / refinements to the original framing

The starting brief is directionally right. Three sharpenings:

**a. The bitter lesson is not "don't design", it's "don't encode".** Sutton's point is that
general methods which exploit computation beat methods that exploit human-supplied
domain knowledge, as compute grows. Attention itself is a hand-designed mathematical prior
(permutation-equivariant, content-addressed routing) — it just happens to be a *general*
prior about computation, not a *specific* prior about a task. So the operative filter for
every thread here is not "did a human design this" (everything is human-designed) but:

- Does it encode task-specific knowledge (bad — this is what the bitter lesson kills), or a
  general computational/statistical structure (fine)?
- Does its usefulness plausibly *increase*, or at least not decay, with more data/compute?
  A prior that only helps in the low-data regime and gets subsumed by scale is a
  short-term hack, not architecture. Worth building anyway sometimes (efficiency wins are
  real), but label it honestly as such.
- Is it cheap to fit onto current accelerators (dense matmuls, parallel scan, FFT — not
  sequential per-token control flow, not gather/scatter-heavy ops)? And does it actually
  win on measured wall-clock, not just an analytic FLOP count (section 5 tightened this).

**b. "Falsification at small scale" needs an explicit protocol or it's just vibes.** Two
specific failure modes to guard against, both addressed in `docs/methodology.md`:
- *Confound from parameter count instead of compute*: always compare at matched FLOPs
  *and* wall-clock, not just matched params.
- *Confound from tuning effort*: a novel architecture with a hand-tuned LR beating an
  under-tuned baseline is not a finding. Every comparison sweeps LR (and any theory-implied
  hyperparameter) for both sides, with the *number* of tuning trials matched between arms.

**c. Not everything is small-scale-detectable, and we should say so per thread.** Some
effects are genuinely emergent and invisible below a scale threshold. We mitigate this by
preferring theories whose predicted quantity is a *mechanism-level* signal measurable at
small scale (gradient/activation statistics, trainable depth, sample efficiency curve,
long-range recall accuracy, loss-vs-compute exponent) rather than a *capability-level*
claim (e.g. "will reason better"), which is exactly the kind of claim that tends to hide
until scale. Each thread doc states which kind of prediction it's making.

## 3. Guardrails — what we are explicitly not doing

- Not chasing SOTA leaderboard numbers. A thread succeeds by surviving falsification
  attempts and showing a favorable trend, not by beating a benchmark.
- Not training anything that needs more than a single consumer/cloud GPU-day per
  experiment in the falsification phase. If an idea needs more than that to even get a
  first signal, the theory isn't sharp enough yet — sharpen the prediction, don't scale
  the experiment.
- Not hand-coding narrow task priors (e.g., "this layer assumes text is left-to-right and
  English"). Priors must be statements about computation, symmetry, stability, or
  information flow, not about a specific dataset.
- Not treating a single seed / single run as a result. Minimum 3 seeds for anything that
  goes in a thread's "status: supported" column, and clearing a pre-registered effect-size
  bar, not just non-overlapping error bars (section 5).

## 4. Methodology summary

See `docs/methodology.md` for the full loop. Short version: hypothesis → pre-registered
prediction → minimal falsification experiment (matched FLOPs + wall-clock, matched tuning
trial count, multi-seed with a stated effect-size bar) → if it survives, a 4-6 point
scaling-law sweep to check the sign and rough trend of the effect holds as N (params) and
D (data) grow → verdict: falsified / supported-so-far / inconclusive (needs a sharper
prediction).

## 5. Review pass v1 (Opus 4.8 advisor)

Draft v0 of this plan (six threads, no methodology gaps addressed) was reviewed
adversarially by an Opus 4.8-based advisor agent before any code was written. Full memo is
preserved in this repo's history; the concrete changes it produced:

- **Methodology gaps fixed:** added pre-registration (numeric predictions committed before
  the run, not adjusted post hoc), explicit FLOPs-*and*-wall-clock accounting (analytic
  FLOP counts alone were flagged as an exploitable comparison axis), and tuning-budget
  symmetry + a pre-registered effect-size criterion (3 non-overlapping seeds was flagged
  as insufficient evidence of a real difference). All three now live in
  `docs/methodology.md`.
- **Thread 6 promoted to priority 1** (was priority 3) — it's a logical dependency for the
  rest of the repo (the whole "falsify small, trust the trend at scale" premise rests on
  hyperparameter/effect transfer holding), not just another candidate idea. Relabeled from
  "RMT-guided" to "muP-style" since random matrix theory was doing no actual work in the
  derivation — the original math-source label was inaccurate.
- **Thread 1 kept, prediction rescoped** — the clean scaling law only holds in the linear-
  recurrence regime; nonlinear recurrence was quietly folded into the original prediction
  without justification and is now explicitly out of scope. The falsifiable core is now
  "does the depth-vs-spectral-bound relationship hold across different structured
  parameterizations," which is the part that's actually informative rather than close to a
  mathematical identity.
- **Thread 2 demoted to priority 3** and its prediction reworked — "derive `sigma*`
  analytically" was undersold as a cheap step (it's a real analysis task for a genuinely
  novel activation, and doesn't apply cleanly to normalized layers at all); scope
  restricted to pointwise, unnormalized activations, and the pass/fail band replaced with
  the theory's own derived depth-scale instead of an arbitrary 0.8x/1.2x offset.
- **Thread 3 deferred, not cut** — its falsification case was flagged as near-guaranteed a
  priori (low information) and its headline "predictable crossover length `n*`" was never
  actually derived anywhere in the doc. Blocked until someone derives a concrete `n*`
  formula.
- **Thread 4 kept, expectations tempered** — the ODE/integrator argument needs small
  per-step residual updates, which standard trained ResNets don't obviously have; prior
  work on this specific idea shows marginal/inconsistent gains. Doc now leads with the
  synthetic-smooth-target task (where the small-step premise is plausible by construction)
  rather than CIFAR-10, and states the likely-falsified expectation up front.
- **Thread 5 deferred, not cut** — the Lie-algebra/matrix-exponential construction doesn't
  by itself guarantee equivariance (that requires an explicit commutant constraint,
  `[L, G_i] = 0`, that the original doc skipped), and the generator-alignment metric has an
  unresolved gauge-freedom problem. Blocked until both are resolved on paper.
- **Two threads added:** Thread 7 (PAC-Bayes/flatness as a cross-architecture
  generalization-gap design target) and Thread 8 (Fisher/K-FAC condition number as a
  cross-architecture optimization-difficulty design target) — both flagged as genuine gaps
  in the original portfolio, which had nothing targeting generalization or optimization
  geometry directly. Category theory and tropical geometry were considered and rejected —
  neither currently yields a sharp small-scale falsifiable prediction for this repo's
  purposes.

## 6. Candidate research threads (portfolio, in priority order)

| Priority | Thread | Math source | Predicts (falsifiable) | Bitter-lesson risk |
|---|--------|--------------|--------------------------|---------------------|
| 1 | [muP-style hyperparameter transfer](docs/threads/06-mup-hparam-transfer.md) | Tensor Programs / muP (infinite-width limits) | LR optimal at width W stays within 2x of optimal at k*W under the derived scaling rule; naive parameterization drifts by 10x+ | Low — this *is* the scale-transfer methodology, made explicit |
| 2 | [Stability-constrained recurrence](docs/threads/01-stability-constrained-recurrence.md) | Control theory, Lyapunov stability, Koopman operators | Depth-vs-spectral-bound relationship (linear regime) holds across different structured parameterizations, not just one construction | Low — general stability constraint, parallel-scan friendly |
| 3 | [Criticality-guided initialization](docs/threads/02-criticality-guided-init.md) | Mean-field theory / statistical mechanics (pointwise, unnormalized layers only) | Empirical (sigma, depth) trainability boundary matches the theory's derived `xi(sigma)` depth-scale curve | Low — a derivation procedure, not a fixed prior |
| 4 | [Optimal-control integrators for depth](docs/threads/04-optimal-control-integrators.md) | Pontryagin maximum principle, ODE view of ResNets, numerical integrator theory | On a synthetic smooth-target task (small-step regime), required depth drops with integrator order per truncation-error theory, FLOP-honest | Low — still dense matmul stacks; likely falsified outside the constructed small-step regime |
| 5 | [Fisher/K-FAC-preconditioned optimization](docs/threads/08-natural-gradient-preconditioning.md) | Information geometry, natural gradient, K-FAC | Fisher condition number near init predicts steps-to-target-loss ranking across architectures; preconditioning benefit scales with how ill-conditioned the architecture is | Low — general optimization-geometry statement |
| 6 | [PAC-Bayes / flatness as design target](docs/threads/07-pac-bayes-flatness.md) | PAC-Bayes bounds, loss-landscape flatness | A cheap flatness proxy ranks architecturally distinct models (at matched train loss) in the same order as their actual test gap | Low — landscape-geometry measurement, not a layer |
| 2b | [Gated spectral recurrence](docs/threads/09-gated-spectral-recurrence.md) (extends thread 1) | Control theory + linear time-varying systems | A minimal input-dependent retention gate on thread 1's spectrally-constrained core makes associative recall solvable (>=0.30 mean acc vs. chance) while the gated failure boundary stays within 2x of the ungated one at matched eps | Low — same class of update Mamba already runs efficiently |

**Deferred (blocked on unresolved issues, see thread docs for what's needed before building):**

- [Learned/adaptive equivariance](docs/threads/05-learned-equivariance.md) — Lie group
  theory. Blocked: the matrix-exponential construction doesn't by itself guarantee
  equivariance; needs an explicit commutant constraint and a gauge-invariant alignment
  metric before implementation starts. Highest bitter-lesson risk in the portfolio either
  way (must earn its keep vs. "just add more data").
- [Spectral / operator mixing](docs/threads/03-spectral-operator-mixing.md) — Fourier
  analysis / operator learning. Blocked: no derived formula for the claimed crossover
  length `n*`; as written the falsification case is close to guaranteed a priori and
  therefore low-information.

## 7. Shared experiment harness (see `experiments/README.md`)

Harness code now exists and is being built incrementally per-thread rather than all at
once (tasks, models, and harness pieces for threads 1 and 6 are built; see
`experiments/README.md` for the current file-by-file inventory). Diagnostic tasks in use
so far: modular arithmetic (thread 6) and associative recall (thread 1, though its
accuracy metric turned out to be structurally inert for a linear model — see thread 1's
status below). Still not built: tiny char/token-level LM, small vision classification,
the generalized `scaling_sweep.py` and `curvature.py` harness pieces threads 2/4/7/8 will
need.

## 8. Status as of 2026-07-05 and immediate next step

**Thread 6 (muP-style hyperparameter transfer): parked, inconclusive at toy scale.** Two
CPU runs (8x and 32x width ranges) both ran *against* the prediction (muP's raw-LR
transfer drifted more than a naive baseline's), but both runs have real, acknowledged
scale/task limitations (toy modular-arithmetic task, LR grid resolution, width range far
below where muP's advantage is normally demonstrated). Not falsified, not supported —
genuinely unresolved pending a real run on a harder task (tiny LM) at proper scale. See
`docs/threads/06-mup-hparam-transfer.md`'s dated addenda for the full history.

**Thread 1 (stability-constrained recurrence): closed for now, clean small-scale support.**
All three of its originally-scoped measurements came back with mutually consistent,
positive results: the linear-regime scaling law was confirmed to closed-form precision
(predicted failure sequence-length matched measured within a few percent, at two very
different scales); the cross-parameterization claim held (orthogonal and diag_lowrank
constructions agree within the pre-registered factor-of-2 bound); and the
free-vs-constrained predictability asymmetry was well-supported (unconstrained recurrence
fails unpredictably, sometimes catastrophically; constrained variants fail the same way
every time). One scope limitation found and left alone, not fixed: the associative-recall
task is unlearnable by a strictly linear model (proved by construction — recall needs a
nonlinear comparison, which conflicts with this thread's own linear-recurrence
pre-registration), so task accuracy is dropped as a metric here. Full history in
`docs/threads/01-stability-constrained-recurrence.md`'s dated addenda.

**Thread 9 (gated spectral recurrence, extends thread 1): prediction A run, falsified at
the pre-registered depth — likely a fixable undertraining issue, not a structural one.**
Built `GatedLinearRecurrentBlock` (minimal input-dependent retention gate on thread 1's
`orthogonal` core) and ran prediction A exactly as pre-registered (vocab=512, hidden=64,
eps=0.1, n_pairs=8, 2000 steps, 5-point LR grid, 5 seeds, matched budget vs. an ungated
control). Result: gated best-of-grid mean accuracy 0.032, control 0.020, both far below the
0.30 target — **falsified as literally specified.** An independent Opus 4.8 review (re-ran
everything itself) found no bug, confirmed by direct test that the gate does inject
query-dependent content-sensitivity thread 1 proved impossible for the ungated case (ratio
0.02 near-closed vs. 0.47 forced-open), and traced the failure to depth-specific
undertraining — the same construction reaches 0.32 accuracy at n_pairs=2, collapsing
monotonically as n_pairs grows toward 8, a credit-assignment problem for a single
read/write-shared scalar gate, not evidence the mechanism can't work. Not retrofitting the
prediction to an easier depth post-hoc — any curriculum or dual-gate follow-up needs its
own pre-registered protocol. Prediction B deferred (the trained n_pairs=8 gates never
opened meaningfully, so there's no genuinely-gated model yet to test predictability
against). Full account in `docs/threads/09-gated-spectral-recurrence.md`'s dated addendum.

**Thread 10 (curriculum follow-up to thread 9): also falsified as pre-registered.** Tested
whether a 3-stage curriculum (n_pairs 2->4->8, same 2000-step total compute as thread 9's
direct training) recovers depth-8 recall accuracy. Best-of-grid mean accuracy 0.039 vs. the
0.30 target — only marginally above thread 9's direct-training control (0.032), within
per-seed noise. Notable: 1400 of the 2000 steps were spent at depths where the review found
this architecture reaches 0.32 alone, yet the final depth-8 stage still collapsed to near
the direct-training baseline — consistent with a credit-assignment failure specific to the
shared read/write gate at that depth, though two unpre-registered confounds (step budget per
stage) aren't ruled out. Two independent recovery attempts have now failed while the
mechanism-level evidence (gate provably injects content-dependence; shallow depths train
fine) still says the idea isn't structurally dead. Next different-in-kind option, if
pursued: an architectural change (independent read/write gates instead of one shared
scalar) — needs its own fresh pre-registration, not a further training-schedule variant.
Full account in `docs/threads/10-curriculum-gated-recurrence.md`'s dated addendum.

Other threads (2, 4, 5/8 per the priority table) are untouched — still just written up in
`docs/threads/`, no code.
4. Proceed through the rest of the priority table in order — criticality-guided init
   (priority 3), optimal-control integrators (priority 4), Fisher/K-FAC preconditioning
   (priority 5), PAC-Bayes/flatness (priority 6) — revisiting the two deferred threads
   only once their blocking issues are resolved on paper.
