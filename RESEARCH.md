# Mathematical Foundations for Neural Architecture Search — Research Plan (Draft v0)

Status: seed draft, written for review (including adversarial review by another model)
before any thread gets implemented. Everything here is a hypothesis, not a result.

## 1. Thesis

Find new neural network architectures by starting from mathematics rather than from
architecture-tweaking. For each candidate idea:

1. State the mathematical structure being imported (a stability theorem, a symmetry group,
   a phase transition, an operator identity, a control-theoretic principle).
2. Derive an architectural consequence — a layer, a parameterization, an init scheme, a
   scaling rule.
3. Derive a **falsifiable, quantitative prediction** from the theory — not "this should
   work better" but "this specific quantity should behave this specific way, and if it
   doesn't, the theory is wrong."
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
  sequential per-token control flow, not gather/scatter-heavy ops)?

**b. "Falsification at small scale" needs an explicit protocol or it's just vibes.** Two
specific failure modes to guard against, both addressed in `docs/methodology.md`:
- *Confound from parameter count instead of compute*: always compare at matched FLOPs, not
  just matched params (a cheap-op architecture can look worse per-parameter and better
  per-FLOP, or vice versa — report both, decide on FLOPs).
- *Confound from tuning effort*: a novel architecture with a hand-tuned LR beating an
  under-tuned baseline is not a finding. Every comparison sweeps LR (and any theory-implied
  hyperparameter) for both sides.

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
  goes in a thread's "status: supported" column.

## 4. Methodology summary

See `docs/methodology.md` for the full loop. Short version: hypothesis → prediction →
minimal falsification experiment (matched-compute, multi-seed, small models/datasets) →
if it survives, a 4-6 point scaling-law sweep to check the sign and rough trend of the
effect holds as N (params) and D (data) grow → verdict: falsified / supported-so-far /
inconclusive (needs a sharper prediction).

## 5. Candidate research threads (portfolio)

Each row links to a full doc in `docs/threads/`. This is a portfolio on purpose — we
expect most threads to be falsified quickly and cheaply; that's the point.

| # | Thread | Math source | Predicts (falsifiable) | Bitter-lesson risk |
|---|--------|--------------|--------------------------|---------------------|
| 1 | [Stability-constrained recurrence](docs/threads/01-stability-constrained-recurrence.md) | Control theory, Lyapunov stability, Koopman operators | Spectral-radius-constrained state matrices raise max trainable depth/sequence length by a predictable factor vs. unconstrained recurrence, at matched compute | Low — general stability constraint, parallel-scan friendly |
| 2 | [Criticality-guided initialization](docs/threads/02-criticality-guided-init.md) | Mean-field theory / statistical mechanics of deep nets | For a given (activation, normalization, skip topology), theory predicts a critical init variance; off-critical init fails past a theory-predicted depth, on-critical init doesn't | Low — a derivation procedure, not a fixed prior |
| 3 | [Spectral / operator mixing](docs/threads/03-spectral-operator-mixing.md) | Fourier analysis, operator learning (FNO) | There exists a sequence-length crossover n\* below which O(n log n) spectral mixing matches attention's accuracy on long-range recall, predicted from the convolution theorem | Medium — must not degrade into a fixed-basis prior that caps expressivity |
| 4 | [Optimal-control integrators for depth](docs/threads/04-optimal-control-integrators.md) | Pontryagin maximum principle, ODE view of ResNets, numerical integrator theory | Residual blocks built as higher-order integrators (vs. Euler/plain residual) need fewer effective layers for matched approximation error, by a factor predictable from integrator order | Low — still dense matmul stacks |
| 5 | [Learned/adaptive equivariance](docs/threads/05-learned-equivariance.md) | Lie group theory, representation theory | A layer that learns its symmetry generators (instead of a human picking the group) recovers the ground-truth symmetry group on synthetic data and matches a hand-built equivariant net's sample efficiency once converged | Medium-high — must earn its keep vs. "just add more data," the classic bitter-lesson trap |
| 6 | [RMT-guided hyperparameter transfer](docs/threads/06-rmt-hparam-transfer.md) | Random matrix theory, muP / infinite-width limits | A derived per-layer LR/init scaling rule lets hyperparameters tuned at small width transfer zero-shot to larger width, within a theory-predicted tolerance | Low — this *is* the scale-transfer methodology, made explicit |

**Backlog (not yet written up, lower priority):** information-bottleneck-constrained
layers (information theory), optimal-transport / Sinkhorn mixing layers (OT theory),
sparse-coding-derived layers (compressed sensing). Will get thread docs if 1-6 don't fill
the available research capacity.

## 6. Shared experiment harness (see `experiments/README.md`)

All threads should be testable against a common small set of diagnostic tasks so results
are comparable: synthetic algorithmic tasks (associative recall, selective copy, modular
arithmetic — cheap and known to separate architectures, per the S4/Mamba/RWKV line of
work), a tiny char/token-level LM task (Shakespeare / TinyStories scale), and a small
vision classification task (CIFAR-10 or a subset). No code committed yet — harness lands
once the methodology doc and thread docs get a review pass.

## 7. Immediate next steps

1. Review this draft (including the adversarial pass against another model) — is the
   thread selection sane, is anything mathematically sloppy, is anything secretly a
   task-specific hack wearing a math costume?
2. Pick 2 threads to actually build first. Suggest starting with #1 (stability-constrained
   recurrence) and #2 (criticality-guided init) — both have the sharpest, cheapest-to-test
   predictions and the most existing theory to lean on, so they're the fastest way to
   validate the *methodology itself* before investing in the riskier threads (#5 especially).
3. Build the shared harness in `experiments/`.
4. Run thread #1 and #2 falsification experiments.
