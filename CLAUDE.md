# samhitas — orientation for a new/resumed session

Read `RESEARCH.md` first — thesis, methodology, thread portfolio, and (section 8) current
status and the decided-but-not-yet-started next step. This file is working conventions
and environment notes established during development, not a restatement of the plan.

## Environment

- This sandbox is CPU-only, no GPU (`nvidia-smi` fails, `torch.cuda.is_available()` is
  False). 4 cores (`nproc`). torch and numpy are already `pip install`-ed system-wide —
  **but not in every session type**: fresh remote/cloud containers (2026-07-07 observation)
  start without them, and a `pip install torch` through the proxy has been seen to time
  out. Check `python3 -c "import torch"` before assuming smoke tests can run; install via
  `pip install -r experiments/requirements.txt` if missing.
- Real GPU-day-scale runs (per `docs/methodology.md`'s compute budget) happen on the
  user's own hardware elsewhere. Code here just needs to CPU-smoke-test cleanly at toy
  scale — it does not need to hit the methodology's real compute budget itself.
- Only 4 cores: don't run a heavy background sweep and a foreground timing check (or two
  background sweeps) at once — they contend and produce misleading timing numbers (this
  happened once; a "80s per run" reading turned out to be pure contention with a review
  agent's own re-run, and was ~3s uncontended). Check `ps aux` before launching something
  CPU-heavy if anything else might still be running.

## Git workflow

- Work directly on `dev`, push there (`git push -u origin dev`). No per-task feature
  branches — explicit user instruction, given only one person works on this repo. The
  original harness-assigned branch (`claude/neural-arch-math-foundations-ljiv0s`) was
  renamed to `dev`; an old remote ref with the original name may still exist as a stale,
  harmless artifact.
- Commit each validated increment separately (fix → smoke-test → commit; don't batch
  several days of work into one diff). Write commit messages that explain *why*, matching
  the detail level already in `git log` — these commits are also the project's lab notebook.

## The pattern that's worked for every thread so far

1. Build/fix code for one thread at a time.
2. Smoke-test it directly. Do a quick timing check on the most expensive config *before*
   committing to a full sweep size — CPU is slow and cost often doesn't scale the way a
   naive estimate predicts (seen repeatedly: measure, don't just extrapolate). For runs
   too long to wait on synchronously, use `run_in_background` and wait for the
   notification rather than polling.
3. **Before writing up a conclusion or scaling up to a bigger run, send the code and
   results to an independent Opus review** (`Agent` tool, `model: "opus"`,
   `subagent_type: "general-purpose"`, instructed to re-run things itself rather than trust
   the write-up). This has repeatedly caught real bugs, and more than once caught a wrong
   *interpretation* of correct results (motivated reasoning softening an inconvenient
   finding; a wrong causal explanation for a real effect). When asked to review a
   *hypothesis* before implementation (not code), tell the agent explicitly not to write
   implementation code, just evaluate the reasoning.
4. Log results honestly as a dated "Post-hoc note, YYYY-MM-DD" addendum at the bottom of
   the relevant `docs/threads/NN-*.md` file. Never silently edit the original
   prediction/threshold — that's this repo's own pre-registration rule
   (`docs/methodology.md`). If a later review finds the write-up itself was wrong (not
   just the code), add a further dated correction on top rather than rewriting history.
5. Commit + push to `dev`.
6. Update `experiments/README.md`'s per-thread finding section and `RESEARCH.md` section 8
   to stay in sync with the thread docs — don't let them drift stale.

## Before writing code for a new hypothesis

Every new experimental idea — including an extension of an existing thread that changes
its falsifiable claim (e.g., adding nonlinearity to what was a linear-only thread) — gets
its own `docs/threads/NN-*.md` with a pre-registered falsifiable prediction and pass/fail
band *before* implementation starts, per `docs/methodology.md`. Don't retrofit a thread
doc after the fact to match whatever the code happened to do.

## Current status (see RESEARCH.md section 8 for the full version)

As of 2026-07-06: thread 6 (muP transfer) is parked, inconclusive at toy scale. Thread 1
(stability-constrained recurrence) is closed for now with clean small-scale support on all
three of its original measurements. Thread 9 (gated spectral recurrence, extends thread 1):
prediction A run exactly as pre-registered (n_pairs=8) and **falsified** — but an
independent Opus review found the gate mechanism genuinely does inject content-dependence
(unlike thread 1's provably-impossible ungated case) and traced the failure to
depth-specific undertraining (same construction reaches 0.32 acc at n_pairs=2, collapses by
n_pairs=8). Not retrofitting the pre-registered depth after seeing this — a curriculum or
dual-gate follow-up would need its own fresh pre-registration. Prediction B deferred (the
trained gates never opened meaningfully, so there's nothing informative to test yet). See
`docs/threads/09-gated-spectral-recurrence.md`'s dated addendum for the full account.

Thread 10 (curriculum follow-up, pre-registered separately per that rule) also
**falsified as specified**: a 3-stage curriculum (n_pairs 2->4->8, same 2000-step total
compute as thread 9's direct training) reached only 0.039 mean accuracy vs. the 0.30 target
— barely above thread 9's own direct-training control (0.032). See
`docs/threads/10-curriculum-gated-recurrence.md`'s dated addendum.

Thread 11 (dual-gate follow-up, independent read/write gates instead of one shared scalar)
also **falsified — closes the gate-family sub-line as a negative result.** Best-of-grid
mean accuracy 0.032, indistinguishable from threads 9 and 10. Three different gate
interventions now converge on the same number; an Opus review ruled out both a
saturated-init trap (forcing the write gate open didn't help — training pushed it back
toward closed) and a hidden-vs-vocab capacity limit (quadrupling hidden size to 256 didn't
help either) as explanations. The write-relevant gate barely moves from init across all
three architectures — **this is an optimization/learnability limit (no discoverable
gradient signal at this depth/budget), not a capacity or architecture one.** Per the
pre-registered plan, this was the last attempt on this gate family; any further attempt on
recall at this depth needs a structurally different mechanism (e.g. explicit key-addressed
memory) and its own fresh thread doc, not another gate variant. See
`docs/threads/11-dual-gate-spectral-recurrence.md`'s dated addendum.

As of 2026-07-07: moved to the next untouched portfolio item, thread 2 (criticality-guided
initialization, priority 3). Built mean-field/edge-of-chaos numerics (`chi_1(sigma_w2,
sigma_b2)`, depth scale `xi=1/|log(chi_1)|` via Gauss-Hermite quadrature; cross-checked
against the analytically-exact `sigma_b2=0` case and an independent Monte Carlo derivative
estimate before trusting it) and a plain unnormalized tanh MLP, then ran the pre-registered
depth x `sigma_w2` sweep (13x13 grid, matched LR/seeds, modular arithmetic, "trainable" =
loss target within a 150-step budget). **Prediction A falsified exactly as
pre-registered**: the empirical trainable-depth boundary is nearly flat (depth 8-16 across
nearly the whole grid) while `xi` spans ~9 orders of magnitude. An Opus review (re-ran the
numerics and several cells itself, reproduced results to every digit) traced this to three
named confounds rather than a harness bug: the task (modular arithmetic) doesn't need
depth, so added depth is a pure handicap regardless of criticality; the LR grid saturates
at its own ceiling for deep nets; and a binary loss threshold inverts the ranking right at
the boundary given only 150 steps. Re-measured with the theory-appropriate diagnostic
instead (init-time gradient-flow *decay/growth length*, not raw magnitude or
loss-reaching — the same metric swap thread 1 needed once already), both the review and my
own independent spot-check found the length peaks at criticality with the correct
decay-to-growth sign flip — but my own re-check found the review's "~2x constant factor"
framing is optimistic (one point gave ~9x, traced to per-seed init-noise dominating the
depth trend at that `sigma_w2` with this seed count). **Verdict for the pre-registered
claim: falsified as specified. Not closed as a negative result** — the qualitative
signal-propagation mechanism looks real, but the quantitative "small constant factor" claim
needs its own freshly pre-registered follow-up (bigger seed count, per-`sigma_w2`-matched
depth grid) before it counts as supported. Not yet started. See
`docs/threads/02-criticality-guided-init.md`'s dated addendum for the full account.

Thread 12 (that gradient-flow-depth-scale follow-up, freshly pre-registered per the above)
was then built and run — also **falsified as specified**. Protocol: 9 `sigma_w2` x 16
depths x 30 seeds, one global `log(grad_norm)` vs. `depth` fit per `sigma_w2`, compared
against theory's `xi(sigma_w2)` via a shape criterion (log-log correlation >= 0.8 + peak
location) and a magnitude criterion (ratio within 3x). Both failed (correlation 0.524; a
36.7x outlier at `sigma_w2=2.2`) — though the ordered phase alone (`sigma_w2` <= 1.9)
already matched theory well (ratios 1.65 -> 0.57, monotonic); the failure concentrates
entirely in the chaotic phase. An Opus review reproduced every number exactly (no harness
bug) and **corrected my own working hypothesis** for the chaotic-phase anomaly: not
forward tanh-derivative saturation (verified flat/low), but heavy-tailed backward-pass seed
variance at large depth in the chaotic phase, which corrupts a single global depth-fit. It
also confirmed a real transient-vs-asymptotic confound (this task's near-orthogonal inputs
start far from theory's fixed point) — restricting the fit window (exploratory only, does
not change the verdict) substantially recovers the pattern (correlation 0.865, correct
peak). **Verdict: falsified as specified, no do-over under this label.** A properly
different estimator (near-asymptotic-only or piecewise fit), pre-specified before running,
would need its own fresh pre-registration to test the window-restricted signal for real —
not yet started. See `docs/threads/12-gradient-flow-depth-scale.md`'s dated addendum for
the full account.

Thread 13 (second, explicitly-last follow-up, freshly pre-registered) was then built and
run — **also falsified on the joint criterion, but the closest of the three attempts;
sub-line now closed.** Before designing it, ruled out two candidate "transient" mechanisms:
a correlation-map-based one turned out to use the wrong recursion entirely (`c=1` is
*repelling* in the chaotic phase, so trajectories move away from it, not toward it), and
the actually-relevant variance-map transient converges in ~6 layers, far too fast to
explain a failure past depth 100. Targeted the diagnosed mechanism directly instead: same
`sigma_w2`/depth grid as thread 12 (no window search), 50 seeds (up from 30), Theil-Sen
robust regression instead of ordinary least squares. **Shape criterion now passes cleanly**
(correlation 0.872, correct peak at `sigma_w2=2.05`, vs. thread 12's failing 0.524 and
wrong peak). **Magnitude criterion still fails**, but only at one interior point
(`sigma_w2=2.2`, ratio 3.98 vs. the 3.0 band) instead of thread 12's 36.7x outlier there. An
Opus review reproduced every number exactly, independently verified the Theil-Sen
implementation, and found the `sigma_w2=2.2` miss is the edge of a systematic chaotic-phase
bias (empirical slope undershoots theory 2.7-4x across the branch, even sign-flips at
`sigma_w2=2.05`, which the magnitude window happens to exclude) — not an isolated fluke; no
untuned point estimator tried cleanly passes every chaotic-phase point. **Verdict:
falsified on the pre-registered joint criterion, with the strongest partial support of the
three attempts.** Per the pre-registered plan, this closes the criticality-guided-init
measurement-refinement sub-line (thread 2 -> 12 -> 13) — a genuine next attempt needs a
structurally different measurement (e.g. a task whose inputs start closer to theory's fixed
point, or per-layer gradient tracking), not a fourth regression-estimator variant. See
`docs/threads/13-robust-gradient-flow-depth-scale.md`'s dated addendum for the full
account.

As of 2026-07-07 (later the same working session): ran a **full-portfolio review** — all
ideas/design/code/results/interpretations, an independent Opus code/design meta-review,
and four Sonnet literature reviews — written up in
`docs/reviews/2026-07-07-portfolio-review.md`, with dated correction notes added to
threads 6/10/11/13. Headline corrections to keep in mind when reading the summaries above:
thread 10's "same total compute" was matched *steps*, not FLOPs (curriculum actually used
~59% of the control's compute — verdict unchanged); the gate-family closing frame
"optimization/learnability limit, not capacity or architecture" is half-overstated (the
generous-budget check that would earn "learnability limit" was specified but never run,
and the literature says this single-gated-recurrence class is a known-insufficient
architecture for multi-pair recall — the capacity half *is* supported, via Zoology's lower
bound); thread 13's residual chaotic-phase bias is plausibly *predicted* finite-width
physics (Hanin & Nica log-normal gradients; depth/width up to ~1.4 on this grid is outside
mean-field's controlled regime), not mere measurement error; thread 6's muP Adam
multipliers were statically verified correct, shifting suspicion to the task
(grokking/weight-decay dynamics) — a cheap coordinate check is the decisive next step. The
review ends with a ranked next-step list (RESEARCH.md section 8 has the short version):
(1) muP coordinate check, (2) finite-width fluctuation test for the criticality anomaly,
(3) generous-budget gate check, (4) new recall-mechanism thread (composition / short-conv /
DeltaNet-style state, carrying thread 9's deferred prediction B). Each needs its own
pre-registered thread doc before code. None started.

**Reconciliation note, 2026-07-07 (end of session):** this full-portfolio review
(`docs/reviews/2026-07-07-portfolio-review.md`) was produced by a separate agent working on
this repo concurrently with an independent adversarial-review pass run in this session's own
conversation (not separately committed — the two covered overlapping ground and were
reconciled by hand). Where they agreed, confidence is higher: thread 1's "clean support" is
somewhat inflated (two of its three measurements are near-tautological at init; only the
cross-parameterization result is genuinely empirical, and even that needed a hand-tuned
diag_lowrank init to match), and thread 11's closing "not architecture" clause is shaky.
Where they disagreed, **the portfolio review's verdict wins** — it did a deeper literature
dig: this session's own adversarial pass had suggested promoting the untouched
ranking-correlation threads (7 PAC-Bayes/flatness, 8 Fisher/K-FAC) on a pattern-matching
heuristic ("ranking claims are the kind that survive here"), but the portfolio review found
both specific claims are already well-studied in the literature and likely falsified as
stated (NASWOT/TE-NAS/NAS-Bench-Suite-Zero/Sokol & Park for thread 8; Dziugaite et
al./Dinh et al./Andriushchenko et al. for thread 7) — same "near-guaranteed a priori,
low-information" logic that already deferred thread 3. **Decision at end of session: start
the next work with idea I1 (muP coordinate check) from the review's ranked list — cheapest
(minutes of CPU), most decisive, and unblocks thread 6, which the rest of the portfolio's
"falsify small, trust the trend" premise depends on.**

As of 2026-07-07 (next session): idea I1 was built and run as thread 14
(`docs/threads/14-mup-coordinate-check.md`). Pre-registered a standard muP coordinate check
(per-layer-type activation scale vs. width, at init and after up to 10 Adam steps at a fixed
aggressive `base_lr=0.3`, widths 64-4096). Full grid ran in ~9s CPU. **Falsified as
literally specified** — muP failed its own `|slope|<0.15` flatness bar (output_layer ~-1 at
every checkpoint; hidden layers drifted to 0.3-0.85 by step 10) — but SP failed dramatically
as the intended positive control (loss to ~405 by width 4096 vs. muP's flat ~3.7), and an
independent Opus review (re-ran the code, reproduced every number bit-for-bit) traced both
muP "failures" to a mis-specified pre-registered bar, not a bug: the output layer's ~-1
slope at init is the arithmetically necessary, intended consequence of muP's documented
`base_width/width` readout multiplier (and relaxes toward 0 under training, exactly as
theory predicts); the hidden-layer drift is an artifact of the deliberately-aggressive pilot
LR and vanishes at typical LR (`<=0.01`, verified by an added robustness sweep). **Verdict:
no implementation bug found — the muP forward/backward scaling machinery is mechanically
sound, positively supporting thread 6's task/metric-artifact hypothesis** (grokking
dynamics, not a broken scaling rule). No fix needed before further thread-6 work; thread 6
itself stays parked (its real GPU-scale run is still not started), but the implementation-
bug explanation for its adverse smoke reads is now closed off with a decisive, reproduced
answer. See the thread 14 doc's dated addendum for the full account.

As of 2026-07-07 (continued): idea I2 was built and run as thread 15
(`docs/threads/15-finite-width-fluctuation-test.md`). Pre-registered two quantitative
predictions testing whether the threads 12/13 chaotic-phase gradient-flow undershoot
matches Hanin-Nica finite-width theory: (A) `Var[log||grad||]` growth-vs-depth slope should
scale as `~1/width` across widths `{32,64,128}`; (B) the gap between mean-based and
median-based depth-fit slopes should match the log-normal identity
`gap_theory=var_growth_slope/2`. Full grid (4 anomalous `sigma_w2` points x 3 widths x 16
depths x 50 seeds) ran in ~142s CPU. **Both falsified as specified** — but an independent
Opus review (re-ran the grid, reproduced every number bit-for-bit) found the pre-registered
bands were miscalibrated for this regime rather than the mechanism being wrong. Prediction
A's positive control passed cleanly (4/4 `sigma_w2` show growing variance with depth), and
the actual width-scaling exponent, directly fit, is a consistent -1.4 to -1.8 — steeper
than leading-order Hanin-Nica's -1, traced to `Var[log grad]` being convex rather than
linear in depth once `depth/width` reaches ~11, outside the theory's own
leading-order-controlled regime (not evidence against the mechanism). Prediction B's tested
quantity (a difference of two independently-fit slopes) has a bootstrap noise floor
comparable to or larger than the effect itself in most cells — likely unresolvable at 50
seeds, and its theory-side denominator inherited A's convexity bias. The informational
Prediction C independently favors finite-width over the competing finite-depth-saturation
story: per-layer forward statistics (`E[phi'^2]`) stay pinned at the theoretical
fixed-point value through 362 layers with no saturation drift, even as gradient-log-variance
explodes to ~11.7. **Verdict: falsified as specified; every cleanly-resolvable qualitative
signal still points toward finite-width theory, not toward refuting it.** A properly
powered quantitative re-test (convexity-respecting variance estimator, matched regression
estimator on both sides of the gap, many more seeds) would need its own fresh
pre-registration — not pursued now, since the disambiguating question (finite-width vs.
finite-depth-saturation) this thread was built to answer already has a fairly clear
qualitative answer. See the thread 15 doc's dated addendum for the full account.

**Next step: idea I4 from the portfolio review's ranked list** — the generous-budget gate
check thread 11's review specified but never ran (10k+ steps and/or repeated-batch
overfitting on a small fixed set, single best config from thread 11, few seeds), testing
whether the gate-family's "no discoverable gradient signal" clause holds under a much larger
budget or is itself a budget artifact. Needs its own pre-registered thread doc before any
code. Not yet started.
