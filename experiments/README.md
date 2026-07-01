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
  harness/
    flops.py            # analytic FLOP counting (6*fan_in*fan_out/sample/layer) -- built
    train.py             # train_one(): single (param, width, lr, seed) run -- built
    report.py           # save_run(), summarize_sweep() (LR-drift-across-width) -- built
    gradient_flow.py    # first/last-timestep gradient-norm ratio, 1 fwd+bwd pass -- built
  scripts/
    thread06_mup_sanity.py, thread06_mup_widerange.py  # dev smoke tests -- built
    thread01_stability_sanity.py  # dev smoke test, see finding below -- built
    thread01_orthogonal_boundary.py  # closed-form boundary check, see finding below -- built
  runs/                # experiment outputs, gitignored except .gitkeep

  # planned, not yet built:
    tasks/convdist_task.py (deferred thread 3), symmetry_gen.py (deferred thread 5)
    harness/scaling_sweep.py   # 4-6 point width/depth/data sweep + trend fit
    harness/curvature.py       # Fisher/K-FAC condition number + flatness proxy (threads 7, 8)
    models/ blocks for threads 2, 4
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

Fix `gradient_flow.py` to track raw gradient magnitude alongside the ratio (needed to
distinguish "exploded" from "vanished," per thread 1's smoke-test finding above), then
extend thread 1's eps/seq_len grids for a real cross-parameterization read. Before running
thread 6 for real (as opposed to its smoke tests above), address its task/metric caveat.
