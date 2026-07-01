# Thread 1 (priority 2): Stability-constrained recurrence

**Math source:** control theory (linear systems, spectral radius / eigenvalue placement),
Lyapunov stability, Koopman operator theory (linearizing nonlinear dynamics via a lifted
linear operator).

## Motivation

A recurrent state update `h_{t} = f(h_{t-1}, x_t)` is a discrete-time dynamical system.
Classical control theory gives exact conditions for when such a system is stable (bounded
state, non-exploding/non-vanishing sensitivity to past inputs): for a linear update
`h_t = A h_{t-1} + B x_t`, stability and long-range memory are governed entirely by the
eigenvalue spectrum of `A`. This is precisely the mechanism S4/Mamba-style state-space
models exploit (HiPPO-derived `A` matrices with a controlled spectrum), but it's usually
presented as "this specific matrix works well" rather than as a general design rule:
*parameterize `A` so its spectrum is constrained by construction, and predict trainability
from the constraint, rather than discovering it by trial and error per architecture.*

## Architectural hypothesis

If we parameterize a *linear* recurrent state update so its transition matrix has a
spectral radius provably in a target band (e.g. via a structured parameterization —
orthogonal/unitary, or diagonal-plus-low-rank with bounded diagonal), the network should
be trainable at proportionally greater depth/sequence length than an unconstrained update
of the same parameter count, and the relationship should be predictable from the spectral
bound itself — and, more importantly, that predictability should hold *across different
structured parameterizations that enforce the same bound*, not just for one specific
construction (this is the part that would actually be new; the linear-regime scaling law
itself is close to a known identity — see prediction below).

## Falsifiable prediction (pre-registered, revised after review)

Two predictions, scoped separately because they carry different risk:

1. **Linear-regime prediction (low risk, near-identity).** For a strictly *linear*
   recurrence `h_t = A h_{t-1} + B x_t` with spectral radius constrained to `1 - eps`, the
   maximum depth (or sequence length) at which the model trains without gradient
   explosion/vanishing (gradient norm ratio between first and last layer staying within a
   fixed band, e.g. [0.1x, 10x]) scales like `O(1/eps)`. This follows almost directly from
   contraction/mixing-time arguments and is expected to hold — it's included mainly as a
   harness sanity check, not as the interesting claim.
2. **Cross-parameterization prediction (the actual falsifiable core).** The `O(1/eps)`
   relationship holds *regardless of which structured parameterization enforces the
   spectral bound* — i.e., it is not an artifact specific to HiPPO's construction.
   Concretely: orthogonal/unitary parameterizations and diagonal-plus-low-rank
   parameterizations, both constrained to the same `eps`, should show max-trainable-depth
   curves that agree with each other and with the `O(1/eps)` prediction within a factor of
   2, at matched FLOPs. If the two parameterization families diverge from each other by
   more than that factor, the claim that "trainable depth is predictable from the spectral
   bound alone, independent of construction" is falsified, even though the linear-regime
   math above would still be technically correct for each construction individually.

**Nonlinear recurrence is explicitly out of scope for this thread's falsifiable claim.**
For a nonlinear update, the Jacobian spectrum is state- and time-dependent, so "the
spectral radius" is not a single well-defined number, and the clean mixing-time argument
does not transfer without a much more careful (uniform-in-state) stability analysis. If a
nonlinear variant is tried, it is exploratory and must not be reported under this thread's
pre-registered prediction — it would need its own falsifiable statement (e.g., a stability
margin computed along the actual training trajectory, not the linearization at init).

An unconstrained baseline (same parameter count, free spectrum) is expected to plateau
well below both constrained variants and show no such scaling relationship with any single
tunable knob — this is the comparison arm, not the main claim.

This is a *mechanism-level* prediction (gradient-flow behavior, trainable depth) — it does
not claim anything about downstream capability, which keeps it testable at small scale.

## Minimal experiment

- Tiny recurrent/SSM blocks (<= a few M params), depths swept from 4 to 128 (or sequence
  lengths from 64 to 8192 for a fixed-depth model).
- Task: associative recall / selective copy (see `docs/methodology.md`).
- Compare: (a) unconstrained linear recurrence, (b) spectral-constrained recurrence at 2-3
  different `eps` values, matched FLOPs and matched tuning sweep.
- Measure: max trainable depth/length, gradient norm ratio curves, task accuracy.

## Compute budget

Well under a GPU-day: all models are small, sequence lengths modest, no large dataset
needed (synthetic task).

## Bitter-lesson check

- Encodes a general dynamical-systems property (stability), not task-specific knowledge. Low risk.
- Structured spectral constraints (orthogonal/diagonal-plus-low-rank) are parallel-scan
  friendly — this is exactly the class of update S4/Mamba already run efficiently on
  current accelerators, so hardware plausibility is well precedented, not speculative.

## Known prior work / risk of reinventing

This is close to S4/HiPPO/Mamba's actual mechanism, and to unitary/orthogonal RNNs
(uRNN, expRNN). The contribution here isn't a new matrix family — it's testing whether the
*general rule* ("trainable depth is predictable from the spectral bound") holds across
different structured parameterizations, not just the one HiPPO happened to pick. If it
doesn't generalize beyond HiPPO's specific construction, that's a real (and useful)
falsification.

## Status

Not yet run for real. Priority 2.

**Post-hoc note, 2026-07-03 (`thread01_stability_sanity.py`, first smoke test):** built
`experiments/models/linear_recurrence.py` (free/orthogonal/diag_lowrank parameterizations)
and `experiments/tasks/recall.py` (associative recall), plus a gradient_norm_ratio
diagnostic that needs only a single forward+backward pass per config (much cheaper than
thread 6's training sweeps). One real bug caught and fixed while building: the
diag_lowrank parameterization's spectral-norm capping only ever scaled a matrix *down*,
never up, so a near-zero init left it far below its intended target radius; fixed, and
in fixing it found that capping spectral *norm* doesn't rigorously bound spectral *radius*
for a non-normal matrix like diag+low-rank anyway (they can differ a lot) -- redid it via
the Bauer-Fike theorem (diagonal is normal; bounding the low-rank perturbation's norm by a
small eps-scaled budget delta guarantees every eigenvalue of diag+perturbation stays
within delta of some diagonal entry, so spectral radius <= (1-eps-delta)+delta = 1-eps,
exactly, not just in spirit).

First smoke-test read (5 seq lengths 17-257, eps in {0.5, 0.1, 0.02}, 3 seeds, 60 training
steps -- small, meant to validate the code path and get a first signal, not a verdict):

- **Orthogonal parameterization matches the linear-regime prediction almost exactly.** At
  eps=0.02 (spectral radius 0.98), seq_len=257: predicted gradient ratio 0.98^257 = 0.0056,
  measured 0.0057-0.0062 across seeds. That's strong evidence the mechanism and the
  harness are implemented correctly, independent of anything about diag_lowrank.
- **diag_lowrank tracks the same order of magnitude and the same qualitative trend as
  orthogonal, but isn't a clean quantitative match yet.** Its effective decay rate runs a
  bit faster than orthogonal's at the same nominal eps, which is the expected, intentional
  consequence of reserving part of the spectral budget (delta = 0.1*eps) for the low-rank
  term rather than a bug -- but the current eps grid (3 points) and seq_len grid (5 points,
  spaced by ~2x) are too coarse to say with confidence whether it clears or misses the
  thread's "within a factor of 2" cross-parameterization bar.
- **The unconstrained "free" baseline reached further (healthy to seq_len=129) than any
  tested constrained config**, including orthogonal at the smallest tested eps (healthy to
  65 only) -- on its face this looks like it cuts against the thread's premise. It
  shouldn't be read that way yet: free's random init happened to land near spectral radius
  1 by luck (small-Gaussian matrices scaled 1/sqrt(n) cluster near radius 1 per the
  circular law), which can reach further than a *deliberately contracting* eps=0.02 system
  before decaying -- but earlier ad hoc testing (not part of this sweep) showed that same
  free config exploding catastrophically (gradient ratio ~2x10^10, not just "unhealthy") at
  seq_len=257. The current healthy/unhealthy binary metric doesn't distinguish "exploded
  wildly" from "decayed gracefully," which is exactly the distinction this thread's premise
  is actually about (predictable, bounded behavior vs. uncontrolled tail risk) -- the metric
  needs to track raw gradient-norm magnitude, not just the ratio, to make that comparison
  fairly. Also worth testing eps values small enough that a constrained variant can reach
  into free's range (129+) to see whether it does so *without* the explosion risk, which is
  the actual claim worth checking, not just "does it reach as far."

Not a verdict either way. Concrete next steps before this counts as anything: track raw
gradient-norm magnitude alongside the ratio to separate explosion from graceful decay,
extend the eps grid smaller (so constrained variants are tested at ranges comparable to
where free happens to reach), and use a finer seq_len grid for an honest factor-of-2 read
on the cross-parameterization question.

**Post-hoc note, 2026-07-04 (Opus 4.8 review of the above, then a wider CPU run
incorporating its findings -- still not the pre-registered run, but the strongest read so
far).** The review confirmed the Bauer-Fike bound is implemented correctly (checked
directly via `torch.linalg.eigvals`, not just the derivation) and found two real problems
with the prior smoke test: VOCAB=24 caused constant key collisions in the recall task at
the tested lengths, making every `eval_acc` number chance-level noise regardless of the
model; and the "diag_lowrank reaches half of orthogonal's range" reading was *not*
explained by the delta-budget reservation (negligible, ~0.0005 effect) as the previous
note claimed -- the real cause was the diagonal's init spreading eigenvalue magnitudes
across a wide range (mean |d_i| ~= 0.72x the cap, only the max entry near it), so decay
was governed by the eigenvalue *bulk*, not the nominal target, giving diag_lowrank an
*effective* eps 2-5x its nominal one and silently invalidating the matched-eps comparison.

Both fixed (VOCAB raised to 512; diagonal init changed to saturate tanh for nearly every
entry, not just the max), plus an `effective_decay_rate` metric added
(`ratio^(1/(seq_len-1))`, directly comparable to a parameterization's nominal `1-eps`) and
the misleading `max_healthy_seq_len` summary (assumed monotone degradation, false for
`free`) replaced with a per-length healthy-fraction table. Re-ran at the review's
recommended wider grid (seq lengths 17-385, eps in {0.2, ..., 0.002}, 7 seeds -- trimmed
from the review's suggested 10 to fit a ~15-minute CPU budget):

- **Orthogonal matches its nominal target almost exactly at every eps and every sequence
  length tested** -- e.g. eps=0.1: measured effective decay rate 0.900-0.901 across all 9
  lengths (nominal 0.9); eps=0.02: 0.980-0.981 (nominal 0.98). Essentially zero seed
  variance. This is now about as clean a confirmation of the linear-regime prediction as a
  smoke test can give.
- **diag_lowrank now tracks orthogonal closely, and the residual gap is small, consistent,
  and within the pre-registered factor-of-2 bound.** At eps=0.02: orthogonal healthy up to
  seq_len=97, fails at 129; diag_lowrank healthy up to 65, fails at 97 -- one grid step
  earlier, a ~1.3x gap in the failure boundary. At eps=0.01: orthogonal fails at 257,
  diag_lowrank at 193 -- again ~1.3x. At eps=0.005 and 0.002, orthogonal stayed healthy
  across the *entire* tested range (up to 385), so its true failure boundary is unmeasured
  and at least as far out as diag_lowrank's (257 and ~385 respectively) -- consistent with,
  not contradicting, the same small gap. This is the cross-parameterization prediction
  holding up under its own pre-registered tolerance, for the first time with a fair
  (matched-effective-radius) comparison.
- **The free/constrained asymmetry is now dramatic and well-supported (7 seeds, not 3).**
  At seq_len=385, free's 7 seeds split: 5 exploded (`exploding_absolute`, gradient norms
  from ~200 up to literal float infinity on one seed), 2 vanished to near-zero. Orthogonal
  and diag_lowrank, across every eps and length where they're healthy, show effective decay
  rates varying only in the 3rd-4th decimal place seed to seed -- i.e. free's failure is
  wild and unpredictable in both direction and magnitude, while the constrained variants'
  behavior is essentially deterministic given eps, exactly the "predictable vs. uncontrolled
  tail risk" distinction this thread's premise is about.
- **Task accuracy is still exactly chance-level (0.00198 measured vs. 1/512=0.00195
  theoretical) even after the vocab fix** -- a separate, deeper task-design issue, not
  corrected here: keys and values share one embedding table with no positional/role
  signal, and a *linear time-invariant* recurrence applies the identical per-step update
  regardless of position, so it has no mechanism to distinguish "this token is a key" from
  "this token is a value" from dynamics alone. This doesn't affect the gradient-flow result
  (a property of the matrix, independent of whether the task is solvable), but means
  `eval_acc` in this repo's results so far should be read as inert, not as evidence either
  way, until the task is fixed (separate key/value embedding tables, or an explicit
  positional/parity signal).

Still not the pre-registered verdict (fixed HIDDEN=64, single task family, 7 not the
methodology's usual >=3-but-more-is-better seed count, single held-fixed low_rank=4), but
this is the first genuinely clean, mutually-consistent result across both falsifiable
predictions in this thread's doc. Next real gaps: fix the recall task's learnability
issue, and extend orthogonal's eps grid smaller / seq_len grid longer to actually find its
failure boundary at eps<=0.005 rather than running off the end of the tested range.
