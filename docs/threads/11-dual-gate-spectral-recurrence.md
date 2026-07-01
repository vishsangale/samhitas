# Thread 11 (follow-up to threads 9/10): Independent read/write gates for spectral recurrence

**Math source:** same as threads 9/10 (control theory + linear time-varying systems). This
thread changes the *architecture* (an added degree of freedom in the gate), not the task or
the underlying spectrally-constrained core — the third and last planned attempt on this
specific sub-line before treating it as a closed negative result.

## Motivation

Thread 9 found that a minimal, single, shared scalar gate
(`h_t = (1-g_t)*(A h_{t-1}) + g_t*(B x_t)`) on top of thread 1's spectrally-constrained
orthogonal core provably injects the content-dependence a linear model cannot have, but
fails to learn recall at the pre-registered depth (n_pairs=8): accuracy collapses
monotonically from 0.32 (n_pairs=2) to ~0.03 (n_pairs=8), and the learned gate barely opens
at depth 8 regardless of LR, hidden size, or bias init. Thread 10 tested whether this was a
training-schedule problem (curriculum) rather than an architectural one — it wasn't:
curriculum training over the same total compute reached only 0.039, indistinguishable from
direct training's 0.032, even though 1400 of its 2000 steps were spent at depths where the
same architecture is known to work.

The shared-gate construction forces a specific coupling: `g_t` simultaneously controls how
much of the old state is *kept* (`1-g_t`) and how much new content is *written* (`g_t`) —
they are complementary by construction, `(1-g_t) + g_t = 1`. This means a channel cannot
both "write strongly now" and "retain strongly going forward" at the same timestep; the gate
has to trade one off against the other every step. For associative recall specifically, the
model needs to write a new key-value association at one timestep, then *fully retain* that
exact association (not just "reasonably retain," but retain against however many more
pairs get written afterward) for up to `2*(n_pairs-1)` further steps until the query. A
shared gate makes "write now" and "protect forever after" the same knob, evaluated at
different timesteps by different tokens with no direct credit-assignment path connecting
them — plausibly the actual mechanism behind threads 9/10's failure. Splitting the gate into
independent retain/write components is the standard fix for exactly this coupling (this is,
architecturally, moving from a GRU-style interpolation gate toward an LSTM-style
independent forget/input gate) — well-precedented, not a novel proposal, and the natural
next thing to isolate before concluding the underlying idea doesn't work at this depth.

## Architectural hypothesis

Replace the single shared gate with two independent per-channel gates, each a separate
`sigmoid(W x_t + b)`:

```
f_t = sigmoid(W_f x_t + b_f)     # retain/forget gate, independent of w_t
w_t = sigmoid(W_w x_t + b_w)     # write gate, independent of f_t
h_t = f_t * (A h_{t-1}) + w_t * (B x_t)      # elementwise, no constraint f_t + w_t = 1
```

`f_t` and `w_t` both remain functions of `x_t` alone (not `h_{t-1}`), so the LTV structure
and gradient-flow analysis from threads 1/9 still applies — this is still "linear given a
fixed input sequence," just with two data-dependent scalars per channel per step instead of
one. `b_f` initialized to +4 (`sigmoid(4) ~= 0.982`, gate starts almost fully *open* — i.e.
"retain by default," matching orthogonal's un-gated behavior at init) and `b_w` initialized
to -4 (`sigmoid(-4) ~= 0.018`, write gate starts almost fully *closed*) — together this
means the model starts at init behaving almost exactly like the ungated orthogonal
baseline (mostly retaining, rarely writing), same design rationale as thread 9's single-gate
bias init, just applied to two gates instead of one with complementary target values instead
of one shared value.

## Falsifiable prediction (pre-registered)

**A. Dual gate recovers depth-8 performance under the same direct-training protocol thread
9 used.** vocab=512, hidden=64, eps=0.1, n_pairs=8 (seq_len=17), 2000 fresh-random-batch
Adam steps (direct training, not curriculum — this thread isolates the architecture change
from the schedule change thread 10 already tested), LR grid `{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}`,
5 seeds. **Pass:** best-of-grid mean held-out accuracy >= 0.30 (thread 9/10's bar,
maintained for continuity). **Fail:** stays at or near thread 9/10's already-collected
results (0.032 direct, 0.039 curriculum) — reused as controls, not re-run, since architecture
is the only thing this thread changes and both already used this exact eps/vocab/hidden/
step-budget/LR-grid/seed protocol.

**B. If A holds, does predictability survive?** Only evaluated if prediction A passes (no
point testing predictability of a model that never learned anything, as thread 9 found).
Same design as thread 9's prediction B: measure `gradient_norm_ratio` on the trained,
frozen dual-gate model across thread 1's full eps list and sequence-length grid, and check
whether the healthy/unhealthy boundary stays within a factor of 2 of ungated orthogonal at
matched eps.

**This is the last planned attempt on this specific sub-line (single-core-plus-gate
mechanism, recall task, this depth) before treating the sub-line as closed.** If A fails
here too, the honest conclusion becomes: this family of minimal gates (shared or
independent) does not solve associative recall at n_pairs=8/vocab=512/hidden=64 within a
~2000-step CPU-toy budget, despite the gate mechanism provably being capable of injecting
content-dependence in principle. Any further attempt at that point should not be another
small variant of the same construction — it would need a genuinely different mechanism
(e.g. explicit key-addressed memory, larger hidden size relative to vocab, or accepting this
sub-line as a negative result and returning to it only if some other thread's tooling makes
it cheaper to test) and, per this repo's own discipline, its own fresh thread doc with a
different falsifiable claim, not a fourth variant of "same gate family, tweak the details."

## Explicitly out of scope for this thread's claim

- Not testing curriculum + dual gate together (would conflate two changes; if A fails here,
  that combination is a plausible but separate follow-up, not assumed).
- Not testing `diag_lowrank` (same scope limit as threads 9/10 — isolate one variable).
- No claim beyond the recall diagnostic at this specific scale.

## Minimal experiment

- Model: `DualGateRecallModel` in a new file, reusing thread 1's `orthogonal` `_A()`
  unchanged (same discipline as thread 9). Extra params vs. thread 9's single-gate model:
  one more `nn.Linear(input_dim, hidden)` (doubles the gating parameter count — reported
  explicitly, not waved away, per the FLOP/param-accounting non-negotiable).
- Task: `experiments/tasks/recall.py`, unchanged.
- Compute: same total step budget and grid size as thread 9's already-timed protocol
  (2000 steps, ~15-25s/run observed for this class of model on this sandbox); time the
  first run before committing to the full 5-seed x 5-LR grid.

## Compute budget

Well under a GPU-day; CPU-smoke-testable here, matched to threads 9/10's demonstrated
budget.

## Bitter-lesson check

Independent forget/input gates are exactly LSTM's mechanism (Hochreiter & Schmidhuber 1997)
— a decades-precedented, hardware-friendly, general-purpose gating pattern, not
task-specific machinery built to solve recall in particular.

## Known prior work / risk of reinventing

This is architecturally an LSTM-style gate applied on top of a spectrally-constrained
linear core instead of LSTM's usual unconstrained recurrent matrix — the novelty claim (if
any) is entirely in combining it with thread 1's provable spectral-radius control, not in
the gating mechanism itself, which is well-established.

## Status

Not yet implemented. This doc exists to satisfy `docs/methodology.md`'s pre-registration
rule before any dual-gate code is written.

**Post-hoc note, 2026-07-06 (`scripts/thread11_dual_gate_sanity.py`): falsified as
pre-registered — closes the gate-family sub-line as a negative result.**

Ran the exact pre-registered protocol (direct training, not curriculum, so this isolates
the architecture change from thread 10's schedule change): vocab=512, hidden=64, eps=0.1,
n_pairs=8, 2000 fresh-random-batch Adam steps, same LR grid and 5 seeds as threads 9/10.

- **Best-of-grid mean accuracy: 0.032** (lr=3e-4) — statistically indistinguishable from
  thread 9's direct-training control (0.032, not re-run) and thread 10's curriculum control
  (0.039, not re-run). **Falsified.**

Three meaningfully different interventions on the gate mechanism — a single shared gate
(thread 9), the same shared gate under a curriculum schedule (thread 10), and independent
read/write gates (this thread) — all converged to the same ~0.03 accuracy. As pre-registered,
this is the last planned attempt on this sub-line, so before closing it, sent the full
three-thread picture (including my own hypothesis that this convergence means the bottleneck
isn't the gate mechanism, but possibly hidden size relative to vocab instead) to an
independent Opus 4.8 review, which reproduced the numbers first-party and ran two targeted
follow-up checks rather than trusting the hypothesis:

- **Not a saturated-init trap.** Forcing the write gate open at init (bias=0, ~0.50 instead
  of ~0.02) did not help — accuracy stayed ~0.01, and *training actively pushed the gate
  back down* toward closed (0.50 -> 0.31) over the course of training. The optimizer isn't
  stuck behind a sigmoid saturation wall; it is choosing to keep the gate closed because
  opening it doesn't reduce the loss at this depth/budget.
- **Not a capacity limit.** Quadrupling hidden size (64 -> 256, i.e. from 1/8th to half of
  vocab=512) on the dual-gate model bought essentially nothing (0.038 best, still far below
  0.30) — ruling out my own "hidden too small relative to vocab" hypothesis directly, not
  just by assumption.
- **Direct measurement of how much the gates moved from init:** across all three
  architectures, the write-relevant gate value moved from ~0.021 to only ~0.025-0.026 after
  the full 2000-step training run (every seed) — i.e. training does essentially nothing to
  the content-selective write path at this depth, regardless of which of the three gate
  designs is used or whether the gate starts open or closed.

**Corrected framing from the review, adopted here:** this is an **optimization/learnability**
finding, not a capacity one — there is no usable gradient signal pushing the write path to
open at n_pairs=8 within this budget, across every gate design and initialization tried. The
gate mechanism *can* inject content-dependence in principle (thread 9's construction test:
query-swap response ratio 0.02 near-closed vs. 0.47 forced-open) and the model *can* learn
recall at shallower depths (thread 9's review: 0.32 at n_pairs=2) — but nothing tried so far
finds a training signal that carries that capability to n_pairs=8.

**Closing this sub-line (single-gate-family-on-spectral-core mechanism, recall task, this
depth) as a negative result, per the pre-registered plan.** Three independent gate
variants plus a forced-open initialization all fail identically; a fourth gate tweak is not
expected to behave differently given this evidence. If this line is revisited, it needs a
genuinely different kind of intervention — the review's own suggestion, not chased further
here since it's optional polish rather than a new finding: confirm under a much more
generous optimization budget (10k+ steps, or repeated-batch overfitting on a small fixed
training set) that even a very forgiving setup never opens the gate at depth 8, which would
further support "no gradient signal exists here" over "just needed more steps." Otherwise,
any real next attempt on recall at this depth belongs to a structurally different mechanism
(e.g. explicit key-addressed memory) and its own fresh thread doc — not another variant of
"gate on top of thread 1's spectral core."

**Correction, 2026-07-07 (full-portfolio review, `docs/reviews/
2026-07-07-portfolio-review.md`): the closing framing "optimization/learnability limit,
not a capacity or architecture one" needs two qualifications.** The verdict (falsified as
pre-registered; sub-line closed) is unchanged.

1. **The "learnability limit" clause is not yet earned.** The generous-budget check the
   paragraph above describes (10k+ steps / repeated-batch) was specified but never run, so
   what's actually established is: not capacity, not init, and no signal *within 2000
   online fresh-batch steps*. "No discoverable gradient signal" (period) remains untested —
   see the review doc's idea I4 for the cheap experiment that would settle it.
2. **The "not architecture" clause is likely wrong per the literature.** A dedicated
   literature review found convergent evidence that a *single* scalar-gated linear
   recurrence — this exact family — is a known-insufficient architecture class for
   multi-pair recall regardless of optimization: Zoology (arXiv:2312.04927) proves
   gated-conv/gated-recurrence ("BaseConv") models need state growing with sequence length
   for MQAR-type recall, and every working small-recurrent solution adds one of three
   primitives this construction lacks — a shift/short-conv operator (H3, arXiv:2212.14052;
   "Convolution Augments Attention," arXiv:2407.05591), two-layer composition (induction
   heads, arXiv:2209.11895; formal 1-vs-2-layer separation, arXiv:2508.07208), or explicit
   outer-product key-value state (DeltaNet, arXiv:2406.06484). The observed
   gate-never-moves symptom is also the classic, named gate-saturation pathology (Tallec &
   Ollivier chrono-init, arXiv:1804.11188; Gu et al., arXiv:1910.09890) — so the failure is
   plausibly architectural *and* optimizational at once, a false dichotomy in the original
   framing. The capacity half of the claim, by contrast, gains outside support:
   hidden=64/vocab=512/n_pairs=8 is comfortably slack against Zoology's information-
   theoretic lower bound, so "not capacity" now rests on published theory, not only the
   local hidden-size ablation. The structurally-different-mechanism follow-up this doc
   calls for is sketched as the review doc's idea I3 (two-layer composition / short-conv /
   DeltaNet-style state, carrying thread 9's still-deferred prediction B).
