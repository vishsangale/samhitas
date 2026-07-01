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
  models/
    mlp.py              # MuPMLP: SP vs muP parametrization, per-layer-type LR groups -- built
  harness/
    flops.py            # analytic FLOP counting (6*fan_in*fan_out/sample/layer) -- built
    train.py             # train_one(): single (param, width, lr, seed) run -- built
    report.py           # save_run(), summarize_sweep() (LR-drift-across-width) -- built
  scripts/
    thread06_mup_sanity.py  # dev smoke test, see caveat below -- built
  runs/                # experiment outputs, gitignored except .gitkeep

  # planned, not yet built:
    tasks/recall.py, convdist_task.py (deferred thread 3), symmetry_gen.py (deferred thread 5)
    harness/scaling_sweep.py   # 4-6 point width/depth/data sweep + trend fit
    harness/curvature.py       # Fisher/K-FAC condition number + flatness proxy (threads 7, 8)
    models/ blocks for threads 1, 2, 4
```

## Smoke-test finding (2026-07-01, CPU, `scripts/thread06_mup_sanity.py`)

The harness runs end-to-end: dataset -> model -> per-layer-type param groups -> training
-> FLOPs/wall-clock -> multi-seed LR-sweep aggregation, all working. The result itself is
**not informative and should not be read as support or falsification of thread 6**: at toy
scale (widths 32/128/512, ~4K train examples, 150 steps), the task saturates (near-zero
loss) across a wide LR range for both parametrizations, and Adam's own per-parameter
gradient normalization provides enough incidental scale-robustness at this size that the
"best LR" argmin is noisy/tied rather than tracking a real trainability boundary — SP
showed zero drift and muP showed *more* drift in this run, the opposite of the prediction,
which is a sign the metric isn't discriminating, not a real finding. This is expected and
fine for a smoke test (its only job was to prove the code path works), but it also tells
us the real thread-6 run (`docs/threads/06-mup-hparam-transfer.md`) needs: real width
multiples (4x/8x/16x an actually-not-tiny base width, not 32), likely a harder task than
this one, and probably tracking training stability/steps-to-target rather than final-loss
argmin on a task easy enough to saturate. Worth folding into that thread's doc before the
real run.

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

- **Associative recall / selective copy** — thread 1 (structured recurrence), deferred
  thread 3 if it's ever unblocked. Not yet built.
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

Implement thread 1's structured-recurrence layer (`models/`), which is both its own
falsification target and thread 6's first novel-layer test case, and `tasks/recall.py`.
Before running thread 6 for real (as opposed to the smoke test above), address the task/
metric caveat noted above in the thread doc.
