# Thread 9 (extends thread 1): Gated spectral recurrence

**Math source:** control theory (as thread 1) plus linear time-varying (LTV) systems
theory — a recurrence whose transition matrix is allowed to change per timestep as a
function of the *input* (not the state) is still linear in the hidden state at each step,
so gradient-flow-style analysis extends to a product of time-varying transition matrices
rather than a single fixed one raised to a power.

## Motivation

Thread 1 established, at clean small-scale support, that constraining a *linear*
recurrence's transition matrix `A` to a target spectral radius makes its trainable
sequence-length range predictable from that one number (`docs/threads/
01-stability-constrained-recurrence.md`). It also proved, by construction, a hard limit of
that result: an end-to-end linear model (linear recurrence + linear readout) cannot solve
associative recall, because recall requires a content-based comparison ("does this key
equal the query?") and any such comparison is a nonlinear operation. That's not a bug in
thread 1 — it's a correct description of what linearity can't do, and thread 1's own
pre-registered claim is explicitly scoped to stay linear (see its "out of scope" section).

Mamba's selective SSM (Gu & Dao 2023) gets around exactly this limitation not by making the
recurrence's state-update nonlinear in the hidden state, but by making the update's
*parameters* (their `B`, `C`, and discretization step `Delta`) functions of the current
input token. The state-to-state map is still linear at every individual timestep — it's
input-*dependent* linear (an LTV system), not state-dependent nonlinear. That distinction
is what this thread is testing: does the *minimum* version of that idea (a single
input-dependent elementwise gate on top of thread 1's existing spectrally-constrained core)
already buy back content-based routing, and if so, at what cost to the predictability
property thread 1 spent its whole falsification budget establishing?

## Architectural hypothesis

Take thread 1's `orthogonal` parameterization (`A = (1-eps) * expm(skew(theta))`, exact
spectral radius `1-eps` by construction) and add one minimal gate: a per-channel,
input-dependent interpolation between "keep the old state" and "write the new input,"

```
g_t = sigmoid(W_g x_t + b_g)                 # in [0,1]^hidden, a nonlinear function of x_t only
h_t = (1 - g_t) * (A h_{t-1}) + g_t * (B x_t)   # elementwise *, on the hidden dimension
```

`g_t` depends only on the current input token, never on `h_{t-1}` — so for a *fixed* input
sequence, `h_t` is still a linear function of `h_{t-1}` (the map is `diag(1-g_t) @ A`, a
time-varying matrix, but a fixed one once the inputs are fixed). This is the LTV structure
Mamba's selection mechanism actually has. The hypothesis: this minimal addition is enough
nonlinearity (through `g_t`'s dependence on token identity) to let different tokens route
to different state channels, which is what a lookup-style task needs — while the
underlying `A`'s fixed spectral radius still governs the *typical* decay behavior closely
enough that thread 1's predictability finding survives in a bounded (not necessarily exact)
form.

`b_g` is initialized to -4 (`sigmoid(-4) ~= 0.018`), so **at init the gate starts almost
fully closed** and the model reduces to (almost) exactly the ungated `orthogonal` control —
caught as a necessary fix during the pre-implementation smoke test: with `nn.Linear`'s
default zero-bias init, `g_t` averaged ~0.5 regardless of `eps`, which would make the
gate's own init noise dominate decay rather than the spectral bound, failing prediction B
near-trivially before training even starts. Verified directly (`gated_orthogonal`'s
effective decay rate at init, 3 seeds x 4 eps values, matches ungated orthogonal's to
within ~2%; see `experiments/models/gated_linear_recurrence.py`'s docstring for the
numbers) — this is the same "start close to a non-selective baseline, let training turn on
selection where needed" convention Mamba's own selection-parameter init uses, not an
arbitrary numeric hack.

## Falsifiable predictions (pre-registered)

Two predictions, both required for this thread to count as "supported" — they are
deliberately in tension (a gate strong enough to enable recall could easily be strong
enough to break predictability), which is what makes this worth testing rather than
assuming either half.

**A. Content-based routing becomes possible.** On the same associative-recall task thread
1 proved unlearnable for a linear model (`experiments/tasks/recall.py`, `VOCAB=512`,
`HIDDEN=64`), the gated model (`gated_orthogonal`, eps=0.1, matched FLOPs/param count to
thread 1's setup up to the gate's own small parameter cost) reaches **mean held-out
accuracy >= 0.30** at `n_pairs=8` (`seq_len=17`) within a fixed training budget (2000 Adam
steps, LR swept over `{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`, 5 points, best-of-grid reported with
the full sweep surface per `docs/methodology.md`'s tuning-symmetry rule), averaged over
>=5 seeds. The **ungated `orthogonal` control** (identical eps, identical training budget
and LR grid — matched tuning trial count between arms) is expected to stay at
chance-level accuracy (`~1/512 = 0.00195`), reproducing thread 1's proof directly inside
this thread's own harness as the negative control. If the gated model does not clear 0.30
mean accuracy, or if it clears it by such a thin margin that it's within seed noise of the
control, prediction A is **falsified** — the gate as implemented is not enough nonlinearity
for content-based routing, contrary to the hypothesis.

**B. The predictable-training-range property survives, within a stated tolerance.** Because
the gate is initialized near-closed (see above), the *at-init* gradient-flow behavior is
expected to already match ungated orthogonal closely and is not the interesting test of
this claim by itself — it's included only as a sanity check that the construction is
correct (expect within ~5% relative effective-decay-rate match to ungated orthogonal at
init, across the full eps list, before any training; a bigger init-time gap than that would
mean the model isn't built as intended and blocks running anything else). The actual claim
is about the model **after training**: take the model trained under prediction A's protocol
(eps=0.1, n_pairs=8, best LR from that sweep, same seeds), freeze it, and measure
`gradient_norm_ratio` on fresh random recall batches at thread 1's full sequence-length grid
(`N_PAIRS_LIST = [8, 16, 24, 32, 48, 64, 96, 128, 192]`, i.e. seq_len 17..385 — note the
model was only ever trained at seq_len=17; this also tests whether whatever gating behavior
it learned generalizes predictably to lengths it never saw, which is itself informative).
The prediction: **the trained gated model's healthy/unhealthy boundary, averaged over the
>=5 seeds from prediction A, stays within a factor of 2 (in sequence length) of the ungated
orthogonal boundary at the same nominal eps=0.1** — reusing thread 1's own pre-registered
cross-parameterization tolerance as the yardstick, since this is structurally the same kind
of question (does a different way of realizing "spectral radius `1-eps`" preserve the
predicted scaling), now asked of a *trained*, task-shaped gate rather than a fixed
construction. If the trained boundary is more than 2x *earlier* (training pushed the gate
open in a way that collapses effective decay) or more than 2x *later* (gate stayed mostly
closed, meaning whatever let prediction A pass didn't come from a generalizable gating
behavior) than ungated orthogonal, prediction B is **falsified** — training the gate to
solve recall breaks the predictability property thread 1 established.

**Both must hold for "supported so far."** If A holds but B fails: the gate buys recall
capability at the cost of the predictability property — a real, useful, but different
finding than hoped (would motivate a follow-up asking how much gate strength is minimally
needed for A, trading off against B). If B holds but A fails: the gate as built doesn't
carry Mamba's actual mechanism (likely missing the input-dependent projection, not just the
retention gate — see "known prior work" below) and needs a design revision, not more
scale. If neither holds, this specific minimal construction is falsified and any next
attempt needs a different (not just bigger) design.

## Explicitly out of scope for this thread's claim

- Downstream task accuracy beyond the recall diagnostic (no claim about language modeling,
  vision, etc. — same discipline as thread 1).
- Any claim that this specific gate matches Mamba's full mechanism (Mamba also makes `B`,
  `C`, and the discretization step input-dependent, and uses a much larger state expansion
  per channel; this is deliberately a smaller ablation of just the retention/write
  interpolation, chosen to keep the change minimal and isolate what that one piece buys).
- The `diag_lowrank` parameterization is not re-tested with the gate in this thread — only
  `orthogonal`, to isolate the gate's effect from the cross-parameterization question
  thread 1 already answered.

## Minimal experiment

- Model: `GatedRecallModel` — same `Embedding -> readout` shell as thread 1's
  `RecallModel`, with a new `GatedLinearRecurrentBlock` replacing `LinearRecurrentBlock`.
  Reuses thread 1's exact `_A()` construction for `orthogonal` mode (no changes to that
  code); adds one `nn.Linear(input_dim, hidden)` for `W_g` and the elementwise interpolation
  above. Extra params: `hidden * (input_dim + 1)` for `W_g` — reported explicitly, not
  waved away, per the FLOP/param-accounting non-negotiable in `experiments/README.md`.
- Task: `experiments/tasks/recall.py`, unchanged.
- Arms: `gated_orthogonal` (eps=0.1) vs. `orthogonal` (eps=0.1, ungated control) for
  prediction A; the same trained `gated_orthogonal` models (frozen post-training) vs.
  `orthogonal` at eps=0.1 for prediction B's actual claim, plus an at-init sanity check
  across the full eps list before training. No `free`/`diag_lowrank` arms needed for this
  thread's own claims (thread 1 already covers those; re-litigating them here would blur
  what's new).
- Compute: recall training loop at `n_pairs=8` (seq_len=17), 2000 steps, batch 32, is
  small — expect low tens of seconds per (seed, LR) point on this sandbox's CPU based on
  thread 1's timing experience; measure before committing to the full 5-seed x 5-LR grid,
  per this repo's own "measure, don't extrapolate" rule.

## Compute budget

Well under a GPU-day; CPU-smoke-testable here per `CLAUDE.md`'s environment notes. Real
compute-budget accounting only matters once/if this graduates past toy scale.

## Bitter-lesson check

- The gate is a *general* mechanism (input-dependent linear interpolation), not
  task-specific machinery built to solve recall in particular — same justification
  Mamba's own paper makes for the selection mechanism's generality.
- Elementwise gating and a fixed matrix `A` both stay parallel-scan friendly in spirit
  (Mamba's real implementation runs this class of recurrence efficiently on accelerators
  today) — unlike, say, a hand-built lookup table keyed on token identity, which would not
  generalize or scale.

## Known prior work / risk of reinventing

This is a deliberately shrunk ablation of Mamba's selective SSM (Gu & Dao, "Mamba: Linear-
Time Sequence Modeling with Selective State Spaces," 2023) — the actual paper already
established that input-dependent `B`/`C`/`Delta` solves induction-head/associative-recall-
style synthetic tasks that plain (non-selective) S4 cannot. This thread is not claiming
that as a novel result; it is testing a narrower, mechanism-level question the original
paper doesn't isolate on its own: *how much* of that capability comes from just the
retention/write interpolation gate (this thread's construction) versus the fuller
input-dependent `B`/`C`/`Delta` parameterization, and whether the resulting system keeps
thread 1's specific "predictable range from spectral bound" property, which is not a
question the Mamba paper asks (it doesn't have thread 1's constrained-spectral-radius
starting point to test predictability against). If prediction A already holds with just
this minimal gate, that's the interesting, reportable part of this thread — the paper's
result would suggest it shouldn't, since Mamba's ablations imply the fuller mechanism is
needed; if A holds anyway with less machinery, that's worth being surprised by (and worth
re-checking hard before trusting) rather than assuming it as expected.

## Status

Not yet implemented. This doc exists to satisfy `docs/methodology.md`'s pre-registration
rule before `experiments/models/`/`experiments/scripts/` code for this thread is written.
