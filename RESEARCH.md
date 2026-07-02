# Mathematical Foundations for Neural Architecture Search — Research Plan (Draft v1)

Status: living plan. Originally draft v1, revised after an adversarial review pass
(section 5); sections 1-7 are the standing plan, section 8 tracks per-thread status as
experiments run (several threads now have recorded verdicts — the "nothing has been run
yet" framing this header once carried is long stale). A full-portfolio review with a
ranked next-step list lives in `docs/reviews/2026-07-07-portfolio-review.md`.

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
| 3 | [Criticality-guided initialization](docs/threads/02-criticality-guided-init.md) — **prediction A falsified 2026-07-07**; [thread 12](docs/threads/12-gradient-flow-depth-scale.md) and [thread 13](docs/threads/13-robust-gradient-flow-depth-scale.md) follow-ups **also falsified 2026-07-07** — measurement-refinement sub-line closed, thread 13 came closest (shape criterion passed) | Mean-field theory / statistical mechanics (pointwise, unnormalized layers only) | Empirical (sigma, depth) trainability boundary matches the theory's derived `xi(sigma)` depth-scale curve | Low — a derivation procedure, not a fixed prior |
| 4 | [Optimal-control integrators for depth](docs/threads/04-optimal-control-integrators.md) | Pontryagin maximum principle, ODE view of ResNets, numerical integrator theory | On a synthetic smooth-target task (small-step regime), required depth drops with integrator order per truncation-error theory, FLOP-honest | Low — still dense matmul stacks; likely falsified outside the constructed small-step regime |
| 5 | [Fisher/K-FAC-preconditioned optimization](docs/threads/08-natural-gradient-preconditioning.md) | Information geometry, natural gradient, K-FAC | Fisher condition number near init predicts steps-to-target-loss ranking across architectures; preconditioning benefit scales with how ill-conditioned the architecture is | Low — general optimization-geometry statement |
| 6 | [PAC-Bayes / flatness as design target](docs/threads/07-pac-bayes-flatness.md) | PAC-Bayes bounds, loss-landscape flatness | A cheap flatness proxy ranks architecturally distinct models (at matched train loss) in the same order as their actual test gap | Low — landscape-geometry measurement, not a layer |
| 2b | [Gated spectral recurrence](docs/threads/09-gated-spectral-recurrence.md) (extends thread 1) — **prediction A falsified 2026-07-06**; [thread 10](docs/threads/10-curriculum-gated-recurrence.md) (curriculum) and [thread 11](docs/threads/11-dual-gate-spectral-recurrence.md) (dual gate) follow-ups **also falsified 2026-07-06** — gate-family sub-line closed as a negative result; prediction B still deferred (no gated model ever learned recall) | Control theory + linear time-varying systems | A minimal input-dependent retention gate on thread 1's spectrally-constrained core makes associative recall solvable (>=0.30 mean acc vs. chance) while the gated failure boundary stays within 2x of the ungated one at matched eps | Low — same class of update Mamba already runs efficiently |

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

## 8. Status as of 2026-07-07 and immediate next step

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

**Thread 11 (dual-gate follow-up to threads 9/10): also falsified — closes the sub-line as
a negative result.** Independent read/write gates (LSTM-style), same direct-training
protocol as thread 9: best-of-grid mean accuracy 0.032, indistinguishable from thread 9
(0.032) and thread 10 (0.039). Three different gate interventions now converge on the same
number. An Opus 4.8 review reproduced this and ruled out two competing explanations
directly: forcing the write gate open at init didn't help (training pushed it back toward
closed), and quadrupling hidden size (64->256) didn't help either (ruling out a
hidden-vs-vocab capacity limit). Direct measurement showed the write-relevant gate barely
moves from init across all three architectures — training finds no signal to open the
content-selective write path at n_pairs=8, regardless of gate design. **Closing this
sub-line (gate-on-spectral-core mechanism, recall task, this depth) as a negative result**,
framed as an optimization/learnability limit (no discoverable gradient signal at this
depth/budget), not a capacity or architecture one. This was the pre-registered last attempt
on this gate family; any further attempt needs a structurally different mechanism (e.g.
explicit key-addressed memory) and its own fresh thread doc. Full account in
`docs/threads/11-dual-gate-spectral-recurrence.md`'s dated addendum.

**Thread 2 (criticality-guided initialization, priority 3): prediction A falsified exactly
as pre-registered, but the loss-based operationalization looks like the wrong test of the
theory — not closed as a negative result.** Built mean-field/edge-of-chaos numerics
(`chi_1(sigma_w2, sigma_b2)`, depth scale `xi=1/|log(chi_1)|`, cross-checked against the
analytically-exact `sigma_b2=0` case and an independent Monte Carlo derivative estimate)
and a plain unnormalized tanh MLP, then ran the pre-registered depth x `sigma_w2` sweep
(13x13 grid, matched LR/seeds, modular arithmetic, "trainable" = loss target within a
150-step budget). Result: the empirical trainable-depth boundary is nearly flat (depth
8-16 across nearly the whole grid) while `xi` spans ~9 orders of magnitude — falsified as
specified. An Opus 4.8 review (re-ran the numerics and several cells itself) traced this to
three named confounds, not a harness bug: the task doesn't need depth (added depth is a
pure handicap regardless of criticality), the LR grid saturates at its own ceiling for deep
nets, and a binary loss threshold inverts the ranking right at the ordered/chaotic boundary
given only 150 steps. When re-measured with the theory-appropriate diagnostic instead
(init-time gradient-flow *decay/growth length*, not raw magnitude or loss-reaching — the
same fix thread 1 needed once already), both the review and my own independent spot-check
found the length peaks at criticality with the correct sign flip, though my own re-check
found the review's "~2x constant factor" characterization is optimistic (one point gave
~9x, traced to per-seed init-noise dominating the depth trend with this seed count).
**Verdict for the pre-registered claim: falsified as specified.** The qualitative
signal-propagation mechanism looks real; the quantitative "small constant factor" claim
needs its own freshly pre-registered follow-up (bigger seed count, per-`sigma_w2`-matched
depth grid) before it counts as supported — not started yet. Full account in
`docs/threads/02-criticality-guided-init.md`'s dated addendum.

**Thread 12 (gradient-flow-depth-scale follow-up to thread 2, freshly pre-registered): also
falsified as specified, with a diagnosed estimator confound.** Tested the theory-
appropriate diagnostic (init-time gradient-flow decay/growth *length*, not loss-reaching)
under its own pre-registered protocol: 9 `sigma_w2` x 16 depths x 30 seeds, one global
`log(grad_norm)` vs. `depth` fit per `sigma_w2`, compared to theory's `xi(sigma_w2)` via a
shape criterion (log-log correlation + peak location) and a magnitude criterion (ratio
band). Both failed: shape correlation 0.524 (need >=0.8), and a 36.7x outlier at
`sigma_w2=2.2` blew the magnitude band. Notably, the ordered phase alone (`sigma_w2` <=
1.9) already matched theory well under this exact fit (ratios 1.65 -> 0.57, monotonic) —
the failure concentrates entirely in the chaotic phase. An Opus 4.8 review reproduced every
number exactly (ruling out a harness bug) and corrected my own working hypothesis for the
chaotic-phase anomaly: not forward tanh-derivative saturation (verified flat and low), but
heavy-tailed backward-pass seed variance at large depth in the chaotic phase, which
corrupts a single global depth-fit. The review also confirmed a genuine transient-vs-
asymptotic confound (this task's near-orthogonal one-hot inputs start far from theory's
fixed point, so a chunk of the depth range is a transient, not the asymptotic regime the
theory's linearized `xi` describes) — restricting the fit window (exploratory only, does
not change the verdict) substantially recovers the pattern (correlation 0.865, correct peak
location). **Verdict: falsified as specified, no do-over under this thread's label** — a
properly different estimator (near-asymptotic-only or piecewise fit), pre-specified before
running, would need its own fresh pre-registration to test the window-restricted signal for
real. Full account in `docs/threads/12-gradient-flow-depth-scale.md`'s dated addendum.

**Thread 13 (second, explicitly-last follow-up to thread 12, freshly pre-registered): also
falsified on the joint criterion, but the closest of the three attempts — sub-line now
closed.** Before designing it, checked and ruled out two candidate "transient" mechanisms
(a correlation-map-based one turned out to use the wrong recursion entirely — `c=1` is a
*repelling* fixed point in the chaotic phase, so trajectories move away from it, not
toward it; the actually-relevant variance-map transient converges in ~6 layers, far too
fast to explain a failure persisting past depth 100). Targeted the review-diagnosed
mechanism directly instead: same `sigma_w2`/depth grid as thread 12 (no window search), 50
seeds (up from 30), Theil-Sen robust regression (median of pairwise slopes on per-depth
medians) instead of ordinary least squares. Result: **shape criterion now passes cleanly**
(correlation 0.872, correct peak at `sigma_w2=2.05`, vs. thread 12's failing 0.524 and
wrong-location peak). **Magnitude criterion still fails**, but only at one interior point
(`sigma_w2=2.2`, ratio 3.98 vs. the 3.0 band) instead of thread 12's 36.7x outlier there. An
Opus 4.8 review reproduced every number exactly, independently verified the Theil-Sen
implementation, and found the `sigma_w2=2.2` miss is the visible edge of a systematic
chaotic-phase bias (empirical slope undershoots theory by 2.7-4x across the whole chaotic
branch, even sign-flips at `sigma_w2=2.05` — which the magnitude window's `[5,60]` interior
band happens to exclude) rather than an isolated fluke; no untuned point estimator tried
(median- or mean-based) cleanly passes every chaotic-phase point. **Verdict: falsified on
the pre-registered joint criterion, with the strongest partial support of the three
attempts in this sub-line.** Per the pre-registered plan, this closes the
measurement-refinement sub-line (thread 2's loss metric -> thread 12's OLS fit -> thread
13's Theil-Sen fit) — a genuine next attempt on this idea needs a structurally different
measurement (e.g. a task whose inputs start closer to theory's fixed point, or per-layer
gradient tracking), not a fourth regression-estimator variant. Full account in
`docs/threads/13-robust-gradient-flow-depth-scale.md`'s dated addendum.

Other threads (4, 5/8 per the priority table) are untouched — still just written up in
`docs/threads/`, no code.

A full-portfolio review (2026-07-07, independent code/design meta-review plus four
literature reviews) is on record in `docs/reviews/2026-07-07-portfolio-review.md`. It
corrects several recorded framings (see dated notes in threads 6/10/11/13), gives
literature verdicts on the untouched threads (4: genuinely open, run amended; 8: likely
falsified as stated, salvage with controls; 7: strongly predicted false as written, only
the symmetry-corrected variant is worth a fresh thread), and ends with a ranked next-step
list: (1) a muP coordinate check to unblock thread 6, (2) a finite-width fluctuation test
(Hanin-Nica) for the criticality sub-line's residual chaotic-phase anomaly, (3) the
generous-budget gate check thread 11's review specified but never ran, (4) a new
recall-mechanism thread carrying thread 9's still-deferred prediction B.

**Thread 14 (rank-1 item, muP coordinate check) built and run 2026-07-07: falsified as
specified, but an Opus review resolved it decisively in muP's favor.** The pre-registered
`|slope|<0.15` flatness bar failed literally (output_layer slope ~-1 at every checkpoint;
hidden layers drifted to 0.3-0.85 by step 10) — but SP failed dramatically as the intended
positive control (loss to ~405 by width 4096 vs. muP's flat ~3.7), and an independent Opus
review (re-ran the code, reproduced every number bit-for-bit) traced both muP "failures" to
a mis-specified bar rather than a bug: the output-layer's ~-1 slope at init is the
arithmetically necessary, intended consequence of muP's documented `base_width/width`
readout multiplier (and relaxes toward 0 under training, exactly as theory predicts); the
hidden-layer drift is an artifact of the deliberately-aggressive pilot LR (0.3) and
vanishes at typical LR (<=0.01, verified by an added robustness sweep). **Verdict: no
implementation bug found — the muP forward/backward scaling machinery is mechanically
sound, positively supporting thread 6's task/metric-artifact hypothesis** (grokking
dynamics, not a broken scaling rule) over an implementation-bug explanation. No fix needed
before further thread-6 work. See `docs/threads/14-mup-coordinate-check.md`'s dated
addendum for the full account.

**Thread 15 (rank-2 item, finite-width fluctuation test) built and run 2026-07-07: both
predictions falsified as specified, but the qualitative finite-width signal survives an
Opus review.** Tested whether the threads 12/13 chaotic-phase gradient-flow undershoot
matches Hanin-Nica finite-width theory quantitatively, across widths {32,64,128} at the
four anomalous `sigma_w2` points. **Prediction A** (variance-growth slope should scale as
`~1/width`) failed its pairwise-ratio band, but the positive control passed (4/4 `sigma_w2`
show growing `Var[log grad]` with depth) and variance-growth slope decreases *monotonically*
with width at every point — an independent Opus review fit the actual width exponent
(-1.4 to -1.8, steeper than Hanin-Nica's leading-order -1) and traced this to `Var[log grad]`
being convex, not linear, in depth once `depth/width` gets large (reaching `r~11` at this
grid's deepest/narrowest cell) — outside the theory's own leading-order-controlled regime,
not evidence against the mechanism. **Prediction B** (mean/median slope-gap should match the
log-normal identity) also failed, but a bootstrap found the tested quantity's sampling noise
is comparable to or larger than the effect itself in most cells — likely never resolvable at
50 seeds. **Prediction C** (informational per-layer diagnostic) independently favors
finite-width over the competing finite-depth-saturation explanation: forward statistics stay
pinned at the theoretical fixed-point value through 362 layers with no saturation drift, even
as gradient-log-variance explodes. **Verdict: falsified as specified — pre-registered bands
were miscalibrated for this depth/width regime (leading-order-only law tested outside its
validity range; underpowered gap test) — but every cleanly-resolvable qualitative signal
points toward finite-width theory, not toward refuting it.** A properly powered
quantitative re-test would need its own fresh pre-registration. See
`docs/threads/15-finite-width-fluctuation-test.md`'s dated addendum for the full account.

**Thread 16 (rank-3 item, generous-budget gate check) built and run 2026-07-07: both arms
technically pass their pre-registered bars, but an Opus review found the honest headline is
"falsified in spirit," not "budget artifact confirmed."** Tested whether thread 11's
gate-family closure ("no discoverable gradient signal") was a genuine optimization limit or
an artifact of its 2000-step budget, via two arms at 6x the budget (12,000 steps): Arm A
(fresh-random-batch, generalization) and Arm B (repeated sampling from a fixed 128-example
pool, pure memorization). **Arm A's held-out accuracy stayed flat at 0.0316** — statistically
indistinguishable from thread 11's 0.032 control despite 6x more compute, far below the 0.30
"solved recall" bar — but the write gate moved 4.70x from init, clearing the pre-registered
weaker OR-clause and producing a literal PASS. **Arm B reached 1.0000 training accuracy**
(near-perfect memorization) on the fixed pool. An independent Opus review (re-ran the code,
reproduced every number, added diagnostics the original driver didn't collect) found: (1) the
`>=2x` gate-growth bar was miscalibrated — it sits in the sigmoid's saturated tail, so 4.7x
relative growth is only +1.63 nats of real logit movement, from "deeply closed" to "still
quite closed," not to "open"; (2) what the gate is opening *toward* is a **recency-weighted
in-context-copy shortcut, not partial recall** — per-position accuracy climbs from 0.000
(oldest pairs) to 0.08-0.13 (most recent pair), and "always output the most recent value"
alone scores 0.137, close to the observed accuracy; (3) Arm B's memorization is a low bar
(~550 params/example) that mainly rules out catastrophic dead gradients — the memorizing
model generalizes at chance (0.0039) on fresh data, so memorization and the recall
*algorithm* are fully decoupled. **Verdict: the budget-artifact reading is not supported —
6x compute bought zero held-out-accuracy gain — but one real correction to the prior record
is earned: the gate does NOT show "no discoverable gradient signal" given enough budget (it
moves substantially, directedly), it just converges on a copy shortcut rather than solving
recall.** This strengthens the architectural-insufficiency reading (Zoology, portfolio
review section 2.2) over a pure "just needed more steps" one. Sub-line does not reopen. See
`docs/threads/16-generous-budget-gate-check.md`'s dated addendum for the full account.

**Thread 17 (rank-4 item, minimal recall mechanism ladder) pre-registered and arm (b)
(short causal conv) run 2026-07-07: falsified as specified, but the shift-primitive claim
was not fairly tested.** Three-arm ladder (composition / short-conv / DeltaNet-style state)
on the unchanged recall task at `n_pairs=8`, carrying thread 9's still-deferred prediction B
(does predictability survive learned selectivity, once an arm actually learns recall). Arm
(b), run first (cheapest): a depthwise causal conv inserted before thread 9's existing gated
recurrence. **Best config reached only 0.0109 mean accuracy — below every existing
gate-family control (0.02-0.032), not just short of the 0.30 target.** An Opus review
(reproduced every number, added gradient/init diagnostics) found this is not evidence
against the literature's shift-primitive prediction: the conv's default random init plus a
GELU starves the downstream gate of training gradient (~7x attenuation), shifting the
model's effective LR optimum up ~10x and capping its ceiling below what the no-conv model
already reached at its own best LR — a genuine, named, fixable confound (a near-identity
conv init would test the primitive fairly; this one tested "an untrained scrambling filter
in front of the gate" instead). Init-time diagnostics ruled out the alternative concern
(the conv does not disturb thread 9's careful near-baseline gate init). **Per the
pre-registered ladder's own design (arms are independent), moving on to arm (a) rather than
retrying arm (b)** — the review also noted a standing reason not to over-invest in a
conv-only fix regardless: a short conv only widens the *input* window feeding a single
scalar gate, and Zoology's capacity lower bound is about *state* capacity, which arms (a)
(composition) and (c) (matrix-valued state) are mechanistically more likely to actually
address. Prediction B not run for arm (b) (per pre-registration, only runs on an A-pass).
See `docs/threads/17-recall-mechanism-ladder.md`'s dated addendum for the full account.

**Arm (a) (two stacked gated blocks) run 2026-07-07: falsified as specified — cleanly
this time, but still not a fair test of the broader composition hypothesis.** Best config
(lr=3e-4) reached 0.0227 mean accuracy — far below 0.30, but (unlike arm (b)) squarely
inside the same noisy ~0.02-0.032 band every single-layer gate-family variant has landed
in, not a below-baseline regression. An Opus review (reproduced the best config exactly,
added per-block gradient/signal diagnostics) found the literal construction — two of
thread 9's exact, unmodified blocks stacked — is a clean, fair falsification as specified.
But it also found a depth-2-specific optimization pathology: block1's near-closed gate
(by design) attenuates its own output 36x before it reaches block2, starving block2's gate
of gradient (~167x weaker than a single-block control at init) and pinning both gates near
their closed init through step 500 of a 2000-step budget — the standard fix (residual
connections, inter-block normalization) is exactly what this minimal, faithful-to-thread-9
construction omits by design. **Verdict: the narrow claim (does naive unmodified stacking
help) is cleanly falsified; the broader composition hypothesis was never given a fair shot
within this budget and would need a residual/normalized variant under its own fresh
pre-registration.** Every failed arm in the sub-line so far (9/10/11/16, 17a, 17b) converges
to a training loss near `ln(512)=6.238` — not even fitting the training distribution,
reinforcing these are optimization/capacity failures, not eval artifacts.

**Arm (c) (DeltaNet-style matrix state) run 2026-07-07: falsified decisively, and unlike
arms (a)/(b), with no rescuing confound found — thread 17's recall-mechanism ladder now
closes with all three arms falsified.** Best config (lr=3e-4) reached only 0.0047 mean
accuracy — the worst result in the entire portfolio, below every prior control including
the mathematically-incapable ungated linear baseline (0.020). A pre-run sanity check had
shown the same architecture memorizes a fixed batch to 100% within 100 steps, seemingly
ruling out a dead-gradient problem — but an Opus review found this was a shortcut, not
evidence the mechanism works: corrupting the context but keeping only the query token still
retained 0.81 memorization accuracy, i.e. memorization rode a direct query-token-to-target
leak in the read-back formula (`o_t=(1-beta)*pred_v+beta*v_t`), not real storage/retrieval —
a shortcut only available with a *fixed* set of queries, unavailable in the actual online
protocol's fresh-random queries. Direct diagnostics confirmed the write gate (`beta`) never
opens toward recall: it falls *monotonically* during online training (0.020→0.0087), a
trained-model retrieval probe found the state carries essentially no recoverable
information about the correct value (retrieved-vs-correct cosine similarity 0.033), and
forcing `beta` open at init doesn't help — training drives it back closed, the same pattern
thread 11 found for a different mechanism. Testing both candidate fixable confounds
directly (closed-`beta` init; read-after-write self-corruption at the query step) found
neither rescues recall. **Verdict: capacity alone is not sufficient — a full matrix state
was provided and never used, because the delta rule's gated write has no discoverable
online gradient toward recall in this budget.** This is a genuinely new, less-confounded
data point than arms (a)/(b) (not a third instance of the same bug), strengthening the
sub-line's optimization/learnability story. Since all three arms failed prediction A,
**prediction B (thread 9's still-deferred question) stays formally untested** — moot per
the pre-registration, since it only runs on an A-pass. See
`docs/threads/17-recall-mechanism-ladder.md`'s dated addenda for the full account of all
three arms.

**Program-level regroup, 2026-07-08 (`docs/reviews/2026-07-08-program-regroup.md`):**
prompted by the user's direct question ("all negative results — what is wrong with the
setup?"), a full audit of *why* the verdict stream is nearly all falsifications. Headline
finding: of 13 negative-reading verdicts, 7 were the apparatus falsifying its own bars
(out-of-regime bands, underpowered tests, mis-specified thresholds — threads 2, 12, 13,
14, 15A, 15B, 16), 2 were construction confounds (17a-broad, 17b), and 5 were real kills —
3 literature-pre-answered (the gate family, 9/10/11) and 2 genuinely open (17a-narrow,
17c) but confined to a protocol never validated as learnable by anything. The code
itself is clean (no falsification was ever traced to a harness bug); the defects are in
experiment design. Root causes and the full argument are in the regroup doc. Adopted with
it: **methodology amendments v2** (`docs/methodology.md`: literature screen with kill
rule, regime-validity statement, noise-floor pilot, mandatory positive-control arm for
capability claims, stated prior pass-probability per prediction) and a reporting rule
separating "verdict" from "net knowledge."

**Immediate next step (decided):** (1) **thread 18 — recall-protocol validation**: a
minimal 2-layer attention model (the literature's known-sufficient reference) under the
exact existing recall protocol, to determine whether the protocol is learnable at all at
this scale/budget — retroactively supplying the positive control the seven-experiment
recall cluster never had (not a gate-family reopening; thread 11/17 closure rules
untouched). Its pre-registration doc, written to the amended methodology, is the next
artifact. (2) Then **thread 6's real run** (the keystone: build the tiny char-LM task,
pin the protocol, execute at pre-registered scale on the user's GPU hardware) — the
program's "falsify small, trust the trend" premise currently rests on adverse toy-scale
evidence and no real-scale evidence. (3) Thread 4 (amended) is the designated next
new-science thread after those. Threads 7-as-written/8/3/5 stay dead/deferred/blocked per
the regroup's C4.

**Calibration ledger** (methodology amendment v2.5; entries begin with thread 18 — past
threads pre-registered no prior pass-probabilities, so they are not retro-scored):

| Thread / prediction | Stated p(pass) | Outcome |
|---|---|---|
| — | — | — |
