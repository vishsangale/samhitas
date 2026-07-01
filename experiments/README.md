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

## Smoke-test finding (2026-07-03, CPU, `scripts/thread01_stability_sanity.py`)

First cut at thread 1. One real bug caught and fixed while building
`models/linear_recurrence.py`: the diag_lowrank parameterization's spectral-norm capping
only ever scaled a matrix *down*, and a near-zero init left it far below its intended
target radius; fixed via a Bauer-Fike-theorem-based bound instead of norm-capping, which
gives a rigorous spectral-*radius* guarantee rather than an approximate spectral-*norm*
one. See the dated addendum in `docs/threads/01-stability-constrained-recurrence.md` for
the full result — short version: orthogonal matches the linear-regime prediction almost
exactly (0.98^257 predicted vs. measured within ~10%), diag_lowrank tracks the same trend
but isn't a clean quantitative match yet, and the unconstrained "free" baseline reached
further than any tested constrained config at these eps values by getting lucky at init,
while separately confirmed to explode catastrophically just past that range — a
distinction the current healthy/unhealthy binary metric doesn't capture and needs fixing
(track raw gradient magnitude, not just the ratio) before this says anything conclusive.

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
