# Thread 17 (idea I3): minimal recall mechanism ladder

**Math source:** control theory + LTV systems (same as threads 1/9/10/11) for the
spectral-decay/predictability half; the recall-capability half draws on three literature-
identified primitives for multi-pair associative recall in small recurrent models
(Zoology, arXiv:2312.04927; H3, arXiv:2212.14052; DeltaNet, arXiv:2406.06484) rather than a
new derivation. Proposed by the 2026-07-07 portfolio review (`docs/reviews/
2026-07-07-portfolio-review.md`, idea I3) as the "structurally different mechanism" thread
11 called for when closing the gate-family sub-line, and the vehicle for thread 9's still-
deferred prediction B.

## Why this thread exists

The gate-family sub-line (threads 9/10/11, closed; thread 16 confirmed the closure holds
even under 6x budget) established that a single scalar-gated linear recurrence on top of
thread 1's spectrally-constrained orthogonal core cannot solve associative recall at
`n_pairs=8`, and that this is not primarily a budget/optimization problem -- thread 16
found the write gate moves substantially given more steps, but converges on a
recency-weighted copy shortcut, not real content-addressed recall. The 2026-07-07
literature review (portfolio review section 2.2) found a specific, convergent explanation
in the SSM literature: gated-recurrence models of this exact family are proven
information-theoretically insufficient for multi-pair recall (Zoology's Theorem 4.4 +
`m >= N/2p` lower bound) *unless* they add one of three missing primitives, all absent from
this repo's constructions so far:

1. **A shift/short-convolution operator** (H3, Mamba's `conv1d`, "Convolution Augments
   Attention" arXiv:2407.05591) -- gives the model direct access to a short window of
   recent tokens before gating, supplying a "look at what came just before" primitive no
   scalar gate can approximate on its own.
2. **Two-layer composition** (induction heads, arXiv:2209.11895; a formal 1-vs-2-layer
   separation exists for attention, arXiv:2508.07208, open for this gated-recurrence
   family) -- lets one layer detect "this token equals an earlier key" and a second layer
   act on that signal, a capability no single layer (gated or not) can implement alone if
   the 1-vs-2-layer separation extends to this family.
3. **An explicit outer-product key-value state** (fast weight programmers; DeltaNet,
   arXiv:2406.06484) -- replaces the single vector state `h_t` with a matrix state that can
   store multiple, separately-addressable key-value associations, the literature's
   highest-confidence mechanism for recall at this scale.

This thread tests all three as a ladder of pre-registered arms on the *unchanged* recall
task at `n_pairs=8` (the same depth every gate variant in the prior sub-line failed at).
The literature already predicts (b) and (c) will very likely solve recall -- that alone is
not the interesting, repo-distinctive question. What's actually novel, and unanswered
anywhere in the SSM literature per the portfolio review, is **thread 9's still-deferred
prediction B**: once an arm actually learns recall, does the trained model's gradient-flow
predictability -- the property thread 1 spent its whole falsification budget establishing
for the *linear*, ungated case -- survive within a bounded tolerance of the ungated
spectral bound? Does predictability survive learned selectivity, once selectivity actually
works (not just once it's injected but undertrained, as in thread 9)?

## Architectural hypotheses (three arms, ladder order by cost)

All three arms keep `n_pairs=8`, `vocab=512`, `hidden=64`, `eps=0.1` (the orthogonal core's
spectral-radius target, reused unchanged everywhere it appears) -- the *only* thing that
changes across arms and versus the gate-family sub-line is the mechanism added on top of
thread 1's orthogonal core.

**Arm (b): short causal convolution + single gate (cheapest, run first).** Insert a
depthwise causal 1D convolution (kernel size `k` in `{2, 3, 4}`, swept as a discrete
architectural choice, not a continuous hyperparameter -- report all three, pick the best
per the "report the full surface" rule) over the input embeddings, *before* they reach
thread 9's existing, unmodified `GatedLinearRecurrentBlock`. This gives each timestep's
gate/write decision direct access to a short window of immediately-preceding tokens,
supplying the literature's "shift" primitive without touching the recurrence or gate
mechanism themselves.

**Arm (a): two stacked gated blocks (composition).** Two `GatedLinearRecurrentBlock`
instances in sequence (`h1_t` from block 1 feeds as the input sequence to block 2, matching
thread 9's exact per-block construction unchanged, just composed), testing whether stacking
alone -- with no new primitive, just depth-2 composition of the existing mechanism -- is
enough for one layer to detect key-matches and a second to act on them.

**Arm (c): DeltaNet-style outer-product state (new model file, run last).**
`S_t = S_{t-1} * decay - beta_t * k_t (S_{t-1}^T k_t - v_t)^T` (rank-1 delta update to a
matrix state `S_t` in `R^{hidden x hidden}`, `k_t`/`v_t`/`beta_t` linear/sigmoid functions
of `x_t`, matching the DeltaNet recurrence's structure), with `decay` spectrally
constrained via thread 1's exact `orthogonal` construction (`decay = (1-eps) * expm(skew)`)
so the state-retention half of the update inherits the same provable spectral-radius
property the rest of this repo's constructions use, rather than an unconstrained decay.
Readout at the query position reads `S_T @ query_key` (a single matrix-vector product,
matching DeltaNet's associative-recall readout convention) rather than the vector-state
readout the other arms use.

## Falsifiable predictions (pre-registered)

**Prediction A (per arm): does the arm solve recall at n_pairs=8?** Same protocol as
threads 9/10/11/16 for direct comparability: `vocab=512, hidden=64, eps=0.1, n_pairs=8`
(`seq_len=17`), `batch=32`, 2000 fresh-random-batch Adam steps (online, matching the
gate-family sub-line's harder setting, not curriculum), LR grid
`{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`, >=5 seeds, matched tuning-trial count against the
already-collected ungated `orthogonal` control (0.020, not re-run) and gate-family controls
(thread 9 direct: 0.032; thread 11 dual-gate: 0.032; thread 16's 12k-step run: 0.0316) --
reused, not re-run, since the base task/eps/hidden/vocab/step-budget/LR-grid/seed protocol
is identical and only the mechanism differs.

- **Pass** iff best-of-grid mean held-out accuracy `>= 0.30` (the sub-line's existing bar,
  kept unchanged for continuity) **and** the gap over every already-collected control listed
  above is not within pooled seed-spread of the control (i.e., a real, not noise-level,
  improvement -- same effect-size discipline as every other thread in this repo).
- **Fail** iff accuracy stays at or near the existing ~0.02-0.03 control band.

Arms are run and evaluated independently -- one arm failing A does not block running the
next arm in the ladder (each is testing a structurally distinct missing-primitive
hypothesis, not a refinement of the same idea).

**Prediction B (only for arms that pass A): does predictability survive learned
selectivity?** Directly carries thread 9's deferred prediction B forward, unchanged in
design. For each arm that passes A, take the best-LR, all->=5-seeds trained models, freeze
them, and measure `gradient_norm_ratio` (`experiments/harness/gradient_flow.py`, reused
unchanged -- every arm's model exposes `.embed`, `.recur.forward(x)->states`, `.readout`
specifically so this harness function needs no modification) on fresh random recall batches
across thread 1's full sequence-length grid
(`N_PAIRS_LIST = [8, 16, 24, 32, 48, 64, 96, 128, 192]`, i.e. `seq_len` 17..385 -- note
every arm is only ever trained at `seq_len=17`, so this also tests whether the learned
mechanism's gradient-flow behavior generalizes predictably to lengths never seen in
training). Find each seed's healthy/unhealthy crossing point (first `n_pairs` in the grid
where `failure_mode != "healthy"`), average across seeds.

- **Pass** iff the trained arm's mean healthy/unhealthy boundary is within a **factor of 2**
  (in sequence length) of ungated `orthogonal`'s boundary at the same nominal `eps=0.1`
  (closed-form: `L* = 1 + ln(0.1)/ln(1-eps) ~= 22.85`, i.e. crossing between `n_pairs=8` and
  `n_pairs=16` on the tested grid -- reusing thread 1's own pre-registered
  cross-parameterization tolerance, since this is structurally the same question, "does a
  different way of building on `1-eps` preserve the predicted scaling," now asked of a
  *trained* mechanism instead of a fixed construction).
- **Fail** iff the trained boundary is more than 2x earlier (the learned mechanism collapsed
  effective decay) or more than 2x later (the mechanism solved recall through some route
  that doesn't touch the spectral-decay pathway at all, which would itself be an interesting
  but different finding) than the ungated boundary.

**Interpretation, pre-committed per arm:** A passes + B passes -> predictability survives
learned selectivity for this mechanism -- the strongest, most repo-distinctive positive
result available in this portfolio, not established anywhere in the SSM literature. A
passes + B fails -> the mechanism buys recall at the cost of predictability, a real and
useful but different finding (would motivate asking how much of the added mechanism is
minimally needed for A while preserving B). A fails -> mechanism insufficient at this depth
regardless of B; move to the next arm in the ladder, no claim either way about
predictability for a model that didn't learn the task.

## Explicitly out of scope

- Not re-testing `diag_lowrank` in any arm -- `orthogonal` only, isolating the added
  mechanism's effect from thread 1's already-answered cross-parameterization question.
- Not a curriculum variant of any arm -- thread 10 already tested curriculum against the
  single-gate mechanism and found it doesn't help; not re-litigated here unless a specific
  arm's *online* protocol result is ambiguous enough to warrant it, which would need its own
  fresh pre-registration per this repo's no-retrofitting rule.
- Not claiming these three arms exhaust the literature's known-working mechanisms, or that
  passing arm (b)/(c) constitutes a novel capability result -- the literature already
  expects (b)/(c) to solve small-scale recall; this thread's own distinctive content is
  prediction B, not prediction A.
- Arm (c)'s DeltaNet-style construction does not claim to reproduce the full DeltaNet
  architecture (which typically pairs the delta rule with multi-head structure and specific
  normalization) -- deliberately the minimal single-head, `hidden`-dimensional version to
  keep the comparison to the other arms as apples-to-apples as the mechanism allows.

## Minimal experiment / compute budget

Per-arm timing check (single seed, single LR) before committing to each arm's full 5-seed
x 5-LR grid, per this repo's standard practice -- arms (a)/(b) are small diffs on
thread 9's already-timed model class (thread 9's own protocol ran in low tens of seconds
per (seed, LR) point), so expected in the same range; arm (c) is a new model file with a
matrix-valued state (`O(hidden^2)` per-step cost instead of `O(hidden)`) and needs its own
fresh timing check before any grid commitment. Ladder order (b) -> (a) -> (c) runs cheapest
and most literature-predicted-to-work first, deferring the most novel/riskiest
implementation (c) until the cheaper arms' results are in hand.

## Compute budget

Well under a GPU-day per arm; CPU-smoke-testable here per `CLAUDE.md`'s environment notes,
matched to the gate-family sub-line's demonstrated budget (thread 9's protocol: 2000 steps
x 5 LRs x 5 seeds fit comfortably in this sandbox).

## Bitter-lesson check

All three primitives are decades-to-years-old, general-purpose, hardware-friendly
mechanisms already used at scale in production sequence models (causal conv1d: WaveNet,
now standard in Mamba/H3; two-layer composition: every transformer; delta-rule outer-product
state: fast weight programmers, now DeltaNet) -- none is task-specific machinery built to
solve this repo's toy recall task in particular.

## Known prior work / risk of reinventing

Explicitly not claiming novelty for "does adding a shift/composition/key-value primitive
let a small recurrent model solve associative recall" -- the literature (Zoology, H3,
DeltaNet, induction-heads work) already answers that at real scale. The claimed novelty is
narrower and specific to this repo: does thread 1's provable "trainable range predictable
from a single spectral-radius number" property, once a mechanism has actually learned
content-based selectivity (not just been given the capacity for it, as in thread 9's
undertrained case), still hold within a stated tolerance? This is thread 9's own deferred
prediction B, not a new claim invented for this thread.

## Status

Not yet run. Pre-registered 2026-07-07 (session-label date, per this repo's now-standard
doc-date-vs-commit-date convention). Arms will be run and logged in ladder order (b, then
a, then c); each arm's result gets its own dated post-hoc note below as it completes, per
this repo's incremental-commit discipline -- not all three at once.
