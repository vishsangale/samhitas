# samhitas — orientation for a new/resumed session

Read `RESEARCH.md` first — thesis, methodology, thread portfolio, and (section 8) current
status and the decided-but-not-yet-started next step. This file is working conventions
and environment notes established during development, not a restatement of the plan.

## Environment

- This sandbox is CPU-only, no GPU (`nvidia-smi` fails, `torch.cuda.is_available()` is
  False). 4 cores (`nproc`). torch and numpy are already `pip install`-ed system-wide.
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
