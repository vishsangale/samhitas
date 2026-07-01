# Experiment harness (planned, not yet built)

No code lives here yet. This is the plan for the shared harness that every active thread in
`docs/threads/` will use, so results are comparable across threads and so we're not
re-implementing a training loop per idea. Reflects the revised priority order from the
Opus 4.8 review pass (see `RESEARCH.md` section 5).

## Planned structure

```
experiments/
  tasks/           # synthetic + tiny-real-data task generators, shared across threads
    recall.py         # associative recall / selective copy
    modular_arith.py  # modular arithmetic / parity
    convdist_task.py  # position/distance-structured synthetic task (deferred thread 3)
    symmetry_gen.py   # Lie-group-generated synthetic data (deferred thread 5)
  models/          # small reference implementations, one per thread's block type
  harness/
    train.py          # shared training loop: matched FLOPs+wall-clock, multi-seed, LR sweep
    scaling_sweep.py  # runs the 4-6 point width/depth/data sweep, fits trend, plots
    curvature.py       # Fisher/K-FAC condition-number + flatness proxy estimation (threads 7, 8)
    report.py         # dumps config + metrics + fitted trend into a run directory
  runs/            # experiment outputs (configs, metrics, plots) — gitignored except summaries
```

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
  thread 3 if it's ever unblocked.
- **Modular arithmetic / parity** — thread 2 (fast depth-probe task), general sanity task
  for any new layer.
- **Tiny char/token-level LM** (Shakespeare / TinyStories scale) — thread 6 (muP
  transfer), thread 4, and any loss-vs-compute scaling check.
- **Small vision classification** (CIFAR-10/100 subset) — thread 2 (secondary), thread 4
  (secondary/exploratory arm only, per that thread's tempered expectations).
- **Curvature/flatness measurement on top of existing trained models** — threads 7 and 8;
  not a new task, a diagnostic layered on models from other threads once trained.
- **Lie-group-generated synthetic data** — deferred thread 5 only, not built until that
  thread's blocking issues are resolved.

## Next step

Start with `harness/train.py` (matched FLOPs+wall-clock accounting, multi-seed, LR/trial-
budget-matched sweep) plus a baseline model and `tasks/modular_arith.py`, since **thread 6
(muP-style hyperparameter transfer) is priority 1** — validate it against known muP
results on the plain baseline first. Then implement thread 1's structured-recurrence layer
and `tasks/recall.py`, which serve both thread 1's own falsification experiment and thread
6's first novel-layer test case.
