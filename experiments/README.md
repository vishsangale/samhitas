# Experiment harness (planned, not yet built)

No code lives here yet. This is the plan for the shared harness that every thread in
`docs/threads/` will use, so results are comparable across threads and so we're not
re-implementing a training loop per idea. Code lands once `RESEARCH.md` and the thread
docs have gone through a review pass.

## Planned structure

```
experiments/
  tasks/           # synthetic + tiny-real-data task generators, shared across threads
    recall.py         # associative recall / selective copy
    modular_arith.py  # modular arithmetic / parity
    symmetry_gen.py   # Lie-group-generated synthetic data (thread 5)
    convdist_task.py  # position/distance-structured synthetic task (thread 3)
  models/          # small reference implementations, one per thread's block type
  harness/
    train.py          # shared training loop: matched-FLOPs accounting, multi-seed, LR sweep
    scaling_sweep.py  # runs the 4-6 point width/depth/data sweep, fits trend, plots
    report.py         # dumps config + metrics + fitted trend into a run directory
  runs/            # experiment outputs (configs, metrics, plots) — gitignored except summaries
```

## Non-negotiables carried over from `docs/methodology.md`

- Every comparison reports FLOPs, not just parameter count.
- Every comparison sweeps LR (and any theory-implied hyperparameter) for all arms being
  compared, not just the novel one.
- Minimum 3 seeds before a result is reported as "supported."
- Every run's config + raw per-seed metrics get saved, not just an aggregate number.

## Diagnostic tasks (shared across threads)

- **Associative recall / selective copy** — threads 1, 3.
- **Modular arithmetic / parity** — thread 2 (as a fast depth-probe task alongside vision),
  general sanity task for any new layer.
- **Tiny char/token-level LM** (Shakespeare / TinyStories scale) — threads 4, 6, and any
  loss-vs-compute scaling check.
- **Small vision classification** (CIFAR-10/100 subset) — thread 2, thread 4.
- **Lie-group-generated synthetic data** — thread 5 only.

## Next step

Once the plan above is reviewed, start with `harness/train.py` (the matched-compute,
multi-seed training loop) and `tasks/recall.py` + `tasks/modular_arith.py`, since threads 1
and 2 are the first two slated to run (see `RESEARCH.md` section 7).
