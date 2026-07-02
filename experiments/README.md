# Experiment harness

First slice is built (thread 6's baseline case): a modular-arithmetic task, an MLP with
both SP and muP parametrizations, a training loop with FLOPs+wall-clock accounting, and
sweep aggregation. Everything else below is still planned, added incrementally as each
thread needs it, per `RESEARCH.md` section 5's incremental-build call. Setup: `pip install
-r experiments/requirements.txt` (torch, numpy; CPU is fine for smoke tests, real runs need
a GPU per `docs/methodology.md`'s compute budget).

## Current structure

```
experiments/
  tasks/
    modular_arith.py   # (a, b) -> (a+b) mod p, one-hot, p-way classification -- built
    recall.py           # associative recall: n key-value pairs + query -- built
  models/
    mlp.py              # MuPMLP: SP vs muP parametrization, per-layer-type LR groups -- built
    linear_recurrence.py  # h_t=Ah_{t-1}+Bx_t, free/orthogonal/diag_lowrank A -- built
    gated_linear_recurrence.py  # thread 9: single shared retention gate -- built
    dual_gate_recurrence.py     # thread 11: independent read/write gates -- built
    deep_mlp.py          # thread 2: unnormalized tanh MLP, configurable init variance -- built
  harness/
    flops.py            # analytic FLOP counting (6*fan_in*fan_out/sample/layer) -- built
    train.py             # train_one(): single (param, width, lr, seed) run -- built
    report.py           # save_run(), summarize_sweep() (LR-drift-across-width) -- built
    gradient_flow.py    # first/last-timestep gradient-norm ratio, 1 fwd+bwd pass -- built
    meanfield.py         # thread 2: chi_1/xi via Gauss-Hermite quadrature -- built
  scripts/
    thread06_mup_sanity.py, thread06_mup_widerange.py  # dev smoke tests -- built
    thread01_stability_sanity.py  # dev smoke test, see finding below -- built
    thread01_orthogonal_boundary.py  # closed-form boundary check, see finding below -- built
    thread09_gate_recall_sanity.py, thread10_curriculum_sanity.py,
    thread11_dual_gate_sanity.py  # gate-family sub-line, see findings below -- built
    thread02_criticality_sanity.py  # depth x sigma_w2 sweep, see finding below -- built
    thread12_gradient_flow_depth_scale.py  # grad-flow length vs xi, see finding below -- built
    thread13_robust_gradient_flow.py  # Theil-Sen version of thread 12, see finding below -- built
    thread14_mup_coordinate_check.py  # idea I1, see finding below -- built
    thread15_finite_width_fluctuation.py  # idea I2, see finding below -- built
    thread16_generous_budget_gate_check.py  # idea I4, see finding below -- built
    thread17_arm_b_shortconv.py  # idea I3 arm (b), see finding below -- built
    thread17_arm_a_composition.py  # idea I3 arm (a), see finding below -- built
  runs/                # experiment outputs, gitignored except .gitkeep

  # planned, not yet built:
    tasks/convdist_task.py (deferred thread 3), symmetry_gen.py (deferred thread 5)
    harness/scaling_sweep.py   # 4-6 point width/depth/data sweep + trend fit
    harness/curvature.py       # Fisher/K-FAC condition number + flatness proxy (threads 7, 8)
    models/ blocks for thread 4
```

## Smoke-test finding v2 (2026-07-02, CPU, `scripts/thread06_mup_sanity.py`)

v1's null result led to an Opus 4.8 code review, which found a real bug (train/test
leakage in `modular_arith.make_dataset`, since fixed) and flagged the final-loss metric as
too weak to discriminate on a saturating task. Both fixed: the split is now disjoint by
construction, and the harness now measures steps-to-reach-a-target-training-loss instead
of final loss, with an effect-size check against the pre-registered >=3x bar added to
`summarize_sweep`. Also had to drop the task from p=97 to p=41 — p=97 with this MLP never
got off the uniform-baseline loss plateau within a CPU-feasible step budget (that's the
"grokking" regime, which needs thousands of steps, not hundreds).

With those fixes, the 126-run sweep (widths 64/256/512 = k in {1, 4, 8}, 3 seeds, ~82s)
now produces a real, non-degenerate signal for the first time — and it runs *against* the
prediction as measured: SP's optimal raw `base_lr` was exactly flat across all three
widths (log10 drift 0.0), muP's shifted a full decade (log10 drift 1.0). Verdict per the
pre-registered bar: **fails** (ratio 0.0x, needs >=3x).

A second Opus 4.8 review pass caught that the first write-up of this result (both here and
in the thread doc) softened it in a way that didn't hold up: the "convert to effective LR,
it's actually fine" argument was circular (it divides out the exact factor muP's mechanism
adds, then calls the result flat) and also arithmetically wrong once the middle width's
data point was included instead of skipped. See the corrected, dated addendum in
`docs/threads/06-mup-hparam-transfer.md` for the full accounting. Honest reading: the
smoke test ran cleanly against the prediction at this scale — not yet the thread's real
verdict (task/scale/width-range all differ from the pre-registered spec), but not
"inconclusive, ignore it" either. `summarize_sweep` also had two latent bugs the review
caught (a both-arms-flat sweep could auto-pass the effect-size bar via `ratio=inf`; the
drift summary silently dropped noise-gated widths without saying so) — both fixed.

Real run needs: an LR grid finer than 2x per step (not ~3.3x, which can't resolve a
2x-tolerance claim), and a width range reaching the pre-registered 16x and ideally beyond.

## Smoke-test finding v2 (2026-07-04, CPU, `scripts/thread01_stability_sanity.py`)

v1 (2026-07-03) caught one bug while building the model (see git history) and produced a
first, inconclusive read. An Opus 4.8 review of v1 found two more real problems --
VOCAB=24 caused constant key collisions in the recall task, making eval_acc chance-level
noise, and the "diag_lowrank reaches half of orthogonal's range" reading was wrongly
attributed to the delta-budget reservation when the actual cause was the diagonal init
spreading eigenvalues across a wide range instead of concentrating near the target -- both
fixed (VOCAB to 512; diagonal init to saturate tanh per-entry), plus an
`effective_decay_rate` metric added and the misleading `max_healthy_seq_len` summary
(assumed monotone degradation) replaced with a per-length healthy-fraction table.

Re-ran at a wider grid (seq lengths 17-385, eps in {0.2..0.002}, 7 seeds, ~945 configs).
See the dated addendum in `docs/threads/01-stability-constrained-recurrence.md` for the
full result -- short version: orthogonal now matches its nominal target almost exactly at
every eps/length tested; diag_lowrank tracks it closely with a small, consistent residual
gap (~1.3x in failure-boundary sequence length) well inside the pre-registered
factor-of-2 bound; and the free/constrained asymmetry is now dramatic and well-supported --
at the longest tested length, 5 of 7 free seeds exploded (one to literal float infinity)
while 2 vanished, vs. near-zero seed variance for both constrained parameterizations.

Task accuracy stayed exactly chance-level even after the vocab fix. A follow-up review
(asked to check the diagnosis, not just the fix) found the original explanation above --
shared embedding table, no positional signal -- was itself wrong: position *is* visible to
a linear recurrence (a token's contribution depends on it through the matrix power `A^k`).
The real obstruction is that the whole model is end-to-end linear, and recall needs a
nonlinear comparison ("does this key match the query?"); proved by construction that two
sequences with identical stored pairs but different queries differ in the final state by
exactly a fixed linear function of the query alone, independent of what was stored. Not
fixable inside this thread without leaving its own pre-registered linear-recurrence scope
-- `eval_acc` is dropped as a tracked metric here rather than "fixed." See the corrected
addendum in the thread doc for the full account.

## Orthogonal boundary finding (2026-07-05, CPU, `scripts/thread01_orthogonal_boundary.py`)

Orthogonal stayed healthy across the *entire* range tested above (up to seq_len=385) at
eps in {0.005, 0.002}, so its actual failure boundary was never observed. Since
`ratio_first_over_last = (1-eps)^(seq_len-1)` exactly for orthogonal, the crossing point
has a closed form (`L* = 1 + ln(0.1)/ln(1-eps)`) and needs no training loop to check --
39 seconds for 90 configs. Measured crossing matched the closed-form prediction almost
exactly at both eps values (460.4 predicted vs. bracketed between 447/475 measured;
1151.1 predicted vs. 1151/1185 measured) -- the cleanest quantitative confirmation of the
linear-regime prediction so far. Full writeup in the thread doc's 2026-07-05 addendum.

## Thread 9 finding (2026-07-06, CPU, `scripts/thread09_gate_recall_sanity.py`)

Ran thread 9's prediction A exactly as pre-registered: a minimal input-dependent retention
gate (`experiments/models/gated_linear_recurrence.py`) on top of thread 1's orthogonal core,
tested at vocab=512, hidden=64, eps=0.1, n_pairs=8, 2000 fresh-batch Adam steps, 5-point LR
grid, 5 seeds, matched budget against the ungated control. **Falsified at this depth**:
gated best-of-grid mean accuracy 0.032, ungated control 0.020, both far below the 0.30
target. An Opus 4.8 review (re-ran everything itself) found no bug, confirmed the gate does
mathematically inject query-dependent content-sensitivity that thread 1 proved impossible
for the ungated case, and found the real bottleneck is depth-specific undertraining, not a
structural limit — the same construction reaches 0.32 at n_pairs=2 in the review's own
re-run, and accuracy collapses monotonically as n_pairs grows (a hard credit-assignment
problem for a single scalar gate shared between "protect old memory" and "write new
content" at 8 pairs deep). Not re-running at easier depths under this same pre-registered
claim — that would retrofit the prediction after seeing the data. Prediction B (does
predictability survive) deferred: the trained n_pairs=8 gates never opened meaningfully
(mean g stayed near its 0.018 init value), so there's no meaningfully-gated model yet to
test it against. Full account, including the review's numbers, in
`docs/threads/09-gated-spectral-recurrence.md`'s dated addendum. Open follow-up questions
(curriculum training, independent read/write gates) would need their own pre-registered
protocol, not a retrofit of this one.

## Thread 10 finding (2026-07-06, CPU, `scripts/thread10_curriculum_sanity.py`)

Follow-up to thread 9's falsified prediction A: tested whether a 3-stage curriculum
(n_pairs 2 -> 4 -> 8, same total 2000-step budget as thread 9's direct training) recovers
depth-8 recall accuracy. **Falsified as pre-registered**: best-of-grid mean accuracy 0.039
vs. the 0.30 target, only marginally above thread 9's direct-training control (0.032,
reused rather than re-run) and within per-seed noise. Notable because the curriculum spends
1400 of its 2000 steps at depths (n_pairs=2, 4) where the review behind thread 9 found this
same architecture reaches 0.32 accuracy alone — yet the final n_pairs=8 stage still
collapses to near the direct-training baseline, consistent with (but not proof of) a
credit-assignment failure specific to the shared read/write gate at that depth, not simply
"needed more warm-up." Two unpre-registered confounds (the 700-step n_pairs=2 stage may be
under the budget the review's 0.32 number used; 600 final steps may be too few to
consolidate) are noted but not chased under this thread's own label. Two independent
recovery attempts (direct training, then curriculum) have now both failed while the
mechanism-level evidence still says the idea isn't structurally dead; the next
different-in-kind thing to try is architectural (independent read/write gates), which would
need its own fresh pre-registration. Full account in
`docs/threads/10-curriculum-gated-recurrence.md`'s dated addendum.

## Thread 11 finding (2026-07-06, CPU, `scripts/thread11_dual_gate_sanity.py`) — closes the gate-family sub-line

Tested independent read/write gates (LSTM-style, instead of thread 9's single shared gate)
under the exact same protocol as thread 9's direct training. **Falsified as pre-registered**:
best-of-grid mean accuracy 0.032, statistically indistinguishable from thread 9's
direct-training control (0.032) and thread 10's curriculum control (0.039). Three
meaningfully different gate interventions now converge on the same ~0.03 number. An Opus 4.8
review reproduced this first-party and ruled out two competing explanations directly rather
than by assumption: forcing the write gate open at init didn't help (training pushed it back
toward closed, 0.50 -> 0.31, so it isn't a saturated-init trap), and quadrupling hidden size
(64 -> 256) didn't help either (still ~0.038 best), ruling out a hidden-vs-vocab capacity
limit. Direct measurement showed the write-relevant gate barely moves from its init value
(~0.021 -> ~0.025-0.026) across all three architectures after the full 2000-step run --
training finds no signal to open the content-selective write path at this depth, regardless
of gate design or initialization. **Closing this sub-line (gate-on-spectral-core mechanism,
recall task, n_pairs=8) as a negative result**, framed as an optimization/learnability
limit, not a capacity or architecture one -- per the pre-registered plan, this was the last
planned gate variant; a further attempt needs a structurally different mechanism and its own
thread doc, not another gate tweak. Full account in
`docs/threads/11-dual-gate-spectral-recurrence.md`'s dated addendum.

## Thread 2 finding (2026-07-07, CPU, `scripts/thread02_criticality_sanity.py`)

Built mean-field/edge-of-chaos numerics (`harness/meanfield.py`: `chi_1(sigma_w2,
sigma_b2)`, depth scale `xi=1/|log(chi_1)|`, via Gauss-Hermite quadrature with a bisection
root-finder for `q*` -- plain fixed-point iteration converges too slowly right at
criticality) and a plain unnormalized tanh MLP (`models/deep_mlp.py`). Ran the
pre-registered protocol: `sigma_b2=0.1`, 13-point `sigma_w2` grid bracketing the theory's
own critical point, 13-point geometric depth grid (4-256), modular arithmetic, matched LR
grid/seeds, "trainable" = loss <= target within a 150-step budget. **Falsified as
pre-registered**: the empirical (sigma_w2, depth) trainable boundary is nearly flat (depth
8-16 across nearly the whole grid) while the theory's `xi` spans ~9 orders of magnitude.
An Opus 4.8 review (re-ran the numerics and 7 cells itself, reproduced results to every
digit) traced this to three named confounds that decouple the *loss-reaching* metric from
the theory's actual claim: the task doesn't need depth (added depth is a pure handicap
regardless of criticality), the LR grid saturates at its own ceiling for deep nets, and the
binary loss threshold inverts the ranking right at the boundary given only 150 steps. When
tested with the theory-appropriate diagnostic instead (init-time gradient-flow *decay/
growth length*, not raw magnitude or loss) -- the same fix thread 1 needed once already --
both the review and my own independent spot-check (5 points, 12 seeds) found the length
peaks at criticality with the correct decay-to-growth sign flip, though my own re-check
found the review's "~2x constant factor" framing is optimistic (one point gave ~9x, traced
to per-seed init-noise dominating the depth trend at that `sigma_w2` with this seed count).
**Verdict for the pre-registered claim: falsified as specified.** Not closed as a negative
result -- the qualitative signal-propagation mechanism looks real, but the quantitative
"small constant factor" claim needs a properly designed, freshly pre-registered follow-up
(bigger seed count, per-`sigma_w2`-matched depth grid) before it counts as supported. Full
account in `docs/threads/02-criticality-guided-init.md`'s dated addendum.

## Thread 12 finding (2026-07-07, CPU, `scripts/thread12_gradient_flow_depth_scale.py`)

Follow-up to thread 2's falsified prediction, testing the theory-appropriate diagnostic
(init-time gradient-flow decay/growth length, not loss-reaching) under its own fresh
pre-registration (9 `sigma_w2` x 16 depths x 30 seeds, single fwd+bwd pass per cell, one
global `log(grad_norm)` vs. `depth` fit per `sigma_w2`). **Falsified as pre-registered**:
shape correlation 0.524 (need >=0.8), magnitude band violated by a 36.7x outlier at
`sigma_w2=2.2`. Notably, the ordered phase alone (`sigma_w2` <= 1.9) already matches theory
well under this exact fit (ratios 1.65 -> 0.57, monotonic) -- the failure concentrates
entirely in the chaotic phase. An Opus 4.8 review reproduced every number exactly (ruling
out a harness bug) and corrected my own working hypothesis for the anomaly: not forward
tanh-derivative saturation (verified flat/low), but heavy-tailed backward-pass seed
variance at large chaotic-phase depth, which corrupts a single global depth-fit. The review
also confirmed a real transient-vs-asymptotic confound (this task's near-orthogonal inputs
start far from theory's fixed point) -- restricting the fit window (exploratory only, not
used to flip the verdict) substantially recovers the pattern (corr 0.865, correct peak
location). **Verdict: falsified as specified, no do-over under this label** -- a properly
different estimator, pre-registered fresh before running, would be needed to test the
window-restricted signal for real. Full account in
`docs/threads/12-gradient-flow-depth-scale.md`'s dated addendum.

## Thread 13 finding (2026-07-07, CPU, `scripts/thread13_robust_gradient_flow.py`) — closes the criticality-measurement-refinement sub-line

Second, explicitly-last follow-up to thread 12: same `sigma_w2`/depth grid, but 50 seeds
and Theil-Sen robust regression (median of pairwise slopes on per-depth medians) instead of
30 seeds and ordinary least squares -- targeting the heavy-tailed backward-pass variance a
review traced thread 12's failure to. **Falsified on the pre-registered joint criterion,
but with the strongest partial support of the three attempts.** Shape criterion now passes
cleanly (correlation 0.872, correct peak at `sigma_w2=2.05`, up from thread 12's 0.524 and
wrong-location peak). Magnitude criterion still fails, but only at one interior point
(`sigma_w2=2.2`, ratio 3.98 vs. the 3.0 band) instead of thread 12's 36.7x outlier at the
same point. An Opus 4.8 review reproduced every number exactly and independently verified
the Theil-Sen implementation, then found the `sigma_w2=2.2` miss isn't an isolated fluke --
it's the visible edge of a systematic chaotic-phase bias (empirical slope undershoots
theory by 2.7-4x across the chaotic branch, even flips sign at `sigma_w2=2.05`, which the
magnitude window happens to exclude). No single untuned point estimator (Theil-Sen on
medians, or on means) cleanly passes every chaotic-phase point. **Closing this
measurement-refinement sub-line (thread 2 -> thread 12 -> thread 13) as pre-registered** --
any further attempt needs a structurally different measurement (e.g. a task whose inputs
start closer to theory's fixed point, or per-layer gradient tracking), not a fourth
regression-estimator variant. Full account in
`docs/threads/13-robust-gradient-flow-depth-scale.md`'s dated addendum.

## Thread 14 finding (2026-07-07, CPU, `scripts/thread14_mup_coordinate_check.py`) — resolves thread 6's implementation-vs-task-artifact question

Portfolio review's rank-1 idea (I1): standard muP coordinate check -- per-layer-type
mean(|activation|) vs. width (64 to 4096), at init and after up to 10 Adam steps at one
fixed aggressive `base_lr=0.3`, for both `sp` and `mup`. Full grid ran in ~9s CPU.
**Falsified as literally specified** (muP failed the pre-registered `|slope|<0.15`
flatness bar), but SP failed dramatically as the intended positive control (loss to ~405 by
width 4096 vs. muP's flat ~3.7), and an independent Opus review (re-ran the code, matched
every number bit-for-bit) found both muP "failures" are explained by a mis-specified bar,
not a bug: the output layer's ~-1 log-log slope at every checkpoint is the arithmetically
necessary, intended consequence of muP's documented `base_width/width` readout multiplier
(and relaxes toward 0 under training, as theory predicts); the hidden-layer drift is an
artifact of the deliberately-aggressive pilot LR and vanishes at typical LR (`<=0.01`, per
an added robustness sweep). **Verdict: no implementation bug found -- the muP scaling
machinery is mechanically sound, positively supporting thread 6's task/metric-artifact
hypothesis** over an implementation-bug explanation. Full account in
`docs/threads/14-mup-coordinate-check.md`'s dated addendum.

## Thread 15 finding (2026-07-07, CPU, `scripts/thread15_finite_width_fluctuation.py`) — chaotic-phase anomaly still open, qualitatively finite-width

Portfolio review's rank-2 idea (I2): does the threads 12/13 chaotic-phase gradient-flow
undershoot match Hanin-Nica finite-width log-normal-gradient theory quantitatively? Swept
width `{32,64,128}` at the four anomalous `sigma_w2` points, 50 seeds, ~142s CPU. **Both
pre-registered predictions failed as specified** -- Prediction A's `slope*width~const`
band (ratios up to 2.9x vs. a 2x band) and Prediction B's log-normal mean/median gap
identity (sign match 7/12, magnitude ratio 0.142) -- but an independent Opus review (re-ran
the grid, reproduced every number bit-for-bit) found both bands were miscalibrated for this
regime rather than the mechanism being wrong: Prediction A's positive control passed
cleanly (4/4 `sigma_w2` show growing variance with depth) and the actual width-scaling
exponent, directly fit, is a consistent -1.4 to -1.8 (steeper than leading-order Hanin-Nica,
traced to `Var[log grad]` being convex rather than linear in depth once `depth/width`
reaches ~11, outside the theory's controlled regime); Prediction B's tested quantity has a
bootstrap noise floor comparable to or larger than the effect itself in most cells --
likely unresolvable at 50 seeds. The informational Prediction C (per-layer forward
statistics) independently favors finite-width over the competing finite-depth-saturation
story: `E[phi'^2]` stays pinned at the theoretical fixed-point value through 362 layers
with no saturation drift, even as gradient-log-variance explodes. **Verdict: falsified as
specified; qualitative signal still points toward finite-width theory, not against it** --
a properly powered magnitude re-test would need its own fresh pre-registration. Full
account in `docs/threads/15-finite-width-fluctuation-test.md`'s dated addendum.

## Thread 16 finding (2026-07-07, CPU, `scripts/thread16_generous_budget_gate_check.py`) — gate-family record corrected, sub-line does not reopen

Portfolio review's rank-3 idea (I4): does the gate-family sub-line's (9/10/11) "no
discoverable gradient signal" closure hold under 6x more budget, or was it a budget
artifact? Two arms at 12,000 steps (vs. thread 11's 2000), 5 seeds, ~17 min CPU. **Arm A**
(fresh-batch, generalization) held-out accuracy stayed flat at 0.0316 -- statistically
indistinguishable from thread 11's 0.032 control -- but the write gate moved 4.70x from
init, clearing the pre-registered OR-criterion's weaker gate-growth clause, producing a
literal PASS. **Arm B** (repeated sampling from a fixed 128-example pool) reached 1.0000
training accuracy. An independent Opus review (re-ran the code, reproduced every number,
added diagnostics the driver didn't collect) found the literal PASS is misleading: the
`>=2x` gate-growth bar sits in the sigmoid's saturated tail (4.7x relative growth is only
+1.63 nats, "deeply closed" to "still quite closed," not "open"), and what the gate opens
*toward* is a recency-weighted in-context-copy shortcut, not partial recall (per-position
accuracy climbs from 0.000 for the oldest pairs to 0.08-0.13 for the most recent, matching
a naive "always output the most recent value" heuristic's 0.137). Arm B's memorization is a
low bar (~550 params/example) that mainly rules out catastrophic dead gradients -- the
memorizing model generalizes at chance (0.0039) on fresh data. **Verdict: budget-artifact
reading not supported (6x compute bought zero held-out-accuracy gain) -- but one real
correction to the prior record is earned: the gate does show a discoverable, directed
gradient signal given enough budget, it just converges on a copy shortcut rather than
solving recall.** Strengthens the architectural-insufficiency reading (Zoology) over a pure
budget-artifact one; sub-line does not reopen. Full account in
`docs/threads/16-generous-budget-gate-check.md`'s dated addendum.

## Thread 17 arm (b) finding (2026-07-07, CPU, `scripts/thread17_arm_b_shortconv.py`) — falsified as specified, shift-primitive claim not fairly tested

First arm of the recall-mechanism ladder (idea I3): a depthwise causal conv (k=2-4)
inserted before thread 9's existing gated recurrence, unmodified. Full grid (3 kernel
sizes x 5 LRs x 5 seeds) ran in ~22 min CPU. **Best config reached only 0.0109 mean
accuracy -- below every existing gate-family control (0.02-0.032), not just short of the
0.30 target.** An Opus review (reproduced every number, added gradient/init diagnostics
the driver didn't collect) found this doesn't refute the literature's shift-primitive
prediction: the conv's default random init plus a GELU starves the downstream gate of
training gradient (~7x attenuation measured directly), shifting the effective LR optimum
up ~10x and capping the reachable accuracy below what the no-conv model already reached at
its own best LR -- a genuine, fixable confound (an untrained scrambling filter in front of
the gate, not a fair test of a shift primitive). Init-time diagnostics separately ruled out
a different concern: the conv does not disturb thread 9's careful near-baseline gate init
(effective decay rates at init are, if anything, closer to ungated orthogonal than the
no-conv gated model's own). **Per the pre-registered ladder's design (arms are
independent), moving on to arm (a) rather than retrying arm (b)** with a corrected init --
the review also noted a short conv only widens the *input* window feeding a single scalar
gate, while Zoology's capacity bound is about *state* capacity, which arms (a)/(c) are
mechanistically more likely to address. Prediction B not run (per pre-registration, A-pass
required). Full account in `docs/threads/17-recall-mechanism-ladder.md`'s dated addendum.

## Thread 17 arm (a) finding (2026-07-07, CPU, `scripts/thread17_arm_a_composition.py`) — cleanly falsified, but composition itself not fairly tested

Second arm of the recall-mechanism ladder: two of thread 9's exact, unmodified gated
blocks stacked (composition hypothesis). Full grid (5 LRs x 5 seeds) ran in ~11 min CPU.
**Best config (lr=3e-4) reached 0.0227 mean accuracy** -- far below 0.30, but this time
squarely inside the same noisy ~0.02-0.032 band every single-layer gate-family variant has
landed in, not a below-baseline regression like arm (b). An Opus review (reproduced the
best config exactly, added per-block gradient/signal diagnostics) found the literal
construction is cleanly and fairly falsified as specified -- but also found a depth-2-
specific optimization pathology: block1's near-closed gate attenuates its own output 36x
before it reaches block2, starving block2's gate of gradient (~167x weaker than a
single-block control at init) and pinning both gates near their closed init through half
the training budget. The standard fix (residual connections, inter-block normalization)
is exactly what this minimal construction omits by faithfully reusing thread 9's block
unmodified, as pre-registered. **Verdict: the narrow claim is cleanly falsified; the
broader composition hypothesis was never given a fair shot within this budget.** Every
failed arm in the sub-line so far converges to a training loss near `ln(512)=6.238` --
optimization/capacity failures, not eval artifacts. Moving to arm (c) next -- both arm
reviews independently recommend it as a single-block change that sidesteps both
confounds and directly targets Zoology's state-capacity bottleneck. Full account in
`docs/threads/17-recall-mechanism-ladder.md`'s dated addendum.

## Non-negotiables carried over from `docs/methodology.md` (tightened after review)

- Every comparison reports **both** FLOPs and measured wall-clock, with the FLOP-counting
  method stated — not parameter count alone, and not FLOPs alone.
- Every comparison sweeps LR (and any theory-implied hyperparameter) for all arms being
  compared, with the *number* of tuning trials matched between arms, not just the LR
  sweep itself.
- Minimum 3 seeds before a result is reported as "supported," and the result must clear
  the thread's pre-registered effect-size criterion, not just show non-overlapping means.
- Every run's config + raw per-seed metrics get saved, not just an aggregate number.
- Every thread's numeric prediction and pass/fail band must be committed to the thread doc
  *before* `harness/train.py` is pointed at it for that thread — see "Pre-registration" in
  `docs/methodology.md`.

## Diagnostic tasks (shared across threads)

- **Associative recall / selective copy** — built (`tasks/recall.py`); thread 1
  (structured recurrence), deferred thread 3 if it's ever unblocked.
- **Modular arithmetic / parity** — built (`tasks/modular_arith.py`); used so far as
  thread 6's smoke-test task, also the intended fast depth-probe task for thread 2.
- **Tiny char/token-level LM** (Shakespeare / TinyStories scale) — thread 6's real run,
  thread 4, and any loss-vs-compute scaling check. Not yet built.
- **Small vision classification** (CIFAR-10/100 subset) — thread 2 (secondary), thread 4
  (secondary/exploratory arm only). Not yet built.
- **Curvature/flatness measurement on top of existing trained models** — threads 7 and 8;
  not a new task, a diagnostic layered on models from other threads once trained.
- **Lie-group-generated synthetic data** — deferred thread 5 only, not built until that
  thread's blocking issues are resolved.

## Next step

The criticality-guided-init measurement-refinement sub-line (thread 2 -> 12 -> 13) is
closed, mirroring the gate-family sub-line (9/10/11). A full-portfolio review
(`docs/reviews/2026-07-07-portfolio-review.md`) ranked the candidate next steps; rank-1
(muP coordinate check, thread 14) is done -- no implementation bug found, thread 6 stays
parked but its adverse reads are now attributed to the task, not the scaling code. Rank-2
(finite-width fluctuation test, thread 15) is also done -- both pre-registered predictions
falsified on miscalibrated bands, but the qualitative signal still favors finite-width
theory over finite-depth-saturation for the criticality sub-line's residual anomaly; a
properly powered quantitative re-test is not currently planned (needs its own fresh
pre-registration if picked up later). Rank-3 (generous-budget gate check, thread 16) is
also done -- 6x budget bought zero held-out-accuracy gain (gate-family sub-line does not
reopen), but corrected the prior record's "no discoverable gradient signal" phrasing (the
gate does move substantially given more budget, just toward a copy shortcut, not recall).
Rank-4 (recall-mechanism ladder, thread 17) is pre-registered (all three arms: composition,
short-conv, DeltaNet-style state, carrying thread 9's deferred prediction B); arms (b) and
(a) are both run and both falsified (arm (b): a confounded conv init, not a clean negative
result; arm (a): cleanly falsified as specified, but the broader composition hypothesis
wasn't fairly tested due to a depth-2 gradient-starvation pathology). Next: build and run
arm (c) (DeltaNet-style outer-product state), already pre-registered in
`docs/threads/17-recall-mechanism-ladder.md`, needs a new model file. See `RESEARCH.md`
section 8 for the full status.
