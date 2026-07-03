# Thread 6 (priority 1): muP-style hyperparameter transfer for novel layers

**Math source:** infinite-width mean-field limits / Tensor Programs (muP). Originally
labeled "random matrix theory" — that was inaccurate. RMT (Marchenko-Pastur spectra etc.)
does no load-bearing work in this derivation; the actual machinery is muP's
forward/backward Jacobian scaling program. Naming corrected after review.

## Motivation

muP (maximal update parameterization) shows that a specific per-layer scaling of init and
learning rate lets hyperparameters tuned at small width transfer, in principle exactly, to
much larger width — this is the closest existing precedent to the entire methodology this
repo is built around (falsify/tune cheaply at small scale, trust the theory to carry the
result to larger scale). The open question worth treating as its own thread: does the same
style of derivation extend cleanly to the *other* architectural mechanisms proposed in this
portfolio (structured recurrence from thread 1, integrator blocks from thread 4), or does
each new mechanism need its own scaling derivation because it changes the relevant
forward/backward statistics in a way vanilla muP doesn't cover?

This thread is promoted to priority 1 (was priority 3 in the original draft) because it is
a **logical dependency for the rest of the repo**, not just another candidate idea: the
entire premise of "falsify at small scale, trust the trend at larger scale" rests on
hyperparameter (and, implicitly, effect) transfer actually holding. If it doesn't hold for
the novel layers this repo proposes, every other thread's small-scale verdict needs an
explicit caveat. Build and run this first, on the plain baseline and on Thread 1's layer,
before trusting any other thread's small-scale result at face value.

## Architectural hypothesis

For any of this repo's proposed layer types, a per-layer learning-rate/init scaling rule
can be derived from the layer's forward/backward Jacobian statistics (following the same
program as muP) such that hyperparameters swept and selected at a small width transfer,
within a pre-registered tolerance on final loss, to a width several multiples larger —
without re-tuning.

## Falsifiable prediction (pre-registered)

For a chosen layer type (start with the plain baseline to validate the protocol against
known muP results, then apply to Thread 1's structured-recurrence layer once that thread
has a stable implementation), the learning rate that is optimal at width `W` under the
derived scaling rule should also land within a factor of 2x of optimal at width `k*W`
(k in {4, 8, 16}), measured by directly sweeping LR at each width and comparing where the
sweep minimum lands. A naive (unscaled) parameterization, swept the same way, should show
the optimal LR drifting by an order of magnitude or more across the same width range.

Effect-size criterion (per `docs/methodology.md`): "transfer holds" requires the derived
rule's LR-drift-in-log-space across widths to be at least 3x smaller than the naive
parameterization's drift, not merely non-overlapping across 3 seeds. If the derived rule's
transfer tolerance is no better than the naive parameterization's by that margin, the
derivation for that specific layer type is falsified (even if muP itself remains valid for
the layers it was originally derived for).

## Minimal experiment

- Pick one baseline layer type (sanity check against known muP results) and Thread 1's
  structured-recurrence layer as the first novel case.
- Widths swept across at least 3 multiples (e.g. base, 4x, 8x), LR swept log-spaced at
  each width, small model/dataset (tiny LM or small classification task).
- Measure: location of the LR sweep minimum at each width, under both the derived scaling
  rule and an unscaled control parameterization. Report the full sweep surface, not just
  the argmax (methodology requirement).

## Compute budget

This is the one thread that most directly needs a genuine, if small, scaling sweep (by
construction — the prediction is about transfer across width). Still bounded: small
widths, small dataset, LR sweep at each of ~3 widths. Should stay within a small multiple
of a GPU-day, not require anything resembling a large training run.

## Bitter-lesson check

- Lowest risk in the portfolio: this thread's entire content *is* the "test cheap, confirm
  the trend survives scale" methodology, applied to hyperparameters instead of to a novel
  mechanism. Its main value is instrumental — if it works for the novel layers, it justifies
  trusting small-scale results from the other threads more; if it doesn't transfer for a
  given layer type, that's an important warning that the layer's small-scale falsification
  results in this repo may not be scale-stable, and should be flagged as such wherever they
  appear.

## Known prior work / risk of reinventing

Directly extends Tensor Programs / muP (Yang & Hu et al.). Novelty is applying the
derivation as a required checklist item for every other thread's novel layer, and treating
transfer failure as a first-class, reportable outcome rather than skipping the check. This
thread is honestly a re-validation-plus-extension of known theory, not a new hypothesis —
its value is instrumental (validating the repo's core methodology), not novelty.

## Status

Not yet run for real. Priority 1 — build first.

**Post-hoc note, 2026-07-01 (harness smoke test, not the pre-registered run):** a CPU-scale
smoke test of the harness (`experiments/scripts/thread06_mup_sanity.py`, widths 32/128/512,
~4K train examples, 150 steps) confirmed the code path works end to end but produced a
non-informative LR-transfer signal — the task saturates across most of the LR grid at this
size, and Adam's own per-parameter normalization appears to mask width-scaling effects at
such small width, so the "best LR" argmin was noisy rather than tracking a real
trainability boundary (SP showed *less* drift than muP in this run, opposite the
prediction, which reads as a metric/scale problem, not evidence against the theory). Per
this repo's pre-registration rule, this does not change the prediction above or count as a
verdict. It does inform how the real run should be designed: use width multiples large
enough to be unambiguous (the pre-registered k in {4, 8, 16} should be applied to a
not-tiny base width, not 32), consider a harder task than modular arithmetic that doesn't
saturate across the whole LR grid, and consider tracking training stability / steps-to-
target-loss rather than final-loss argmin, which is a weak signal on any task easy enough
to reach near-zero loss under many configurations.

**Post-hoc note, 2026-07-02 (harness smoke test v2, still not the pre-registered run):**
fixed three issues an Opus 4.8 code review found in v1 (train/test leakage in the dataset
split, a final-loss metric too weak to discriminate, and no effect-size check in
`summarize_sweep`). With those fixed and switched to a steps-to-target-loss metric on a
disjoint split, the harness now produces a real, non-degenerate signal for the first time
(126-run sweep, widths 64/256/512 = k in {1, 4, 8}, 3 seeds, ~82s on CPU) — and the signal
runs *against* the prediction as measured: SP's optimal raw `base_lr` was exactly flat
across all three widths (0.01, log10 drift = 0.0), while muP's optimal raw `base_lr`
shifted a full decade (0.01 -> 0.1, log10 drift = 1.0). Verdict per the pre-registered bar:
**fails** (drift ratio 0.0x, needs >=3x).

**Correction, 2026-07-02, after a second Opus 4.8 review of this addendum itself:** the
paragraph originally here softened that result in a way the reviewer correctly called out
as motivated reasoning, on two counts, and it's worth recording both since getting this
wrong is itself informative about how easy it is to explain away an inconvenient result.

First, the grid-coarseness argument (spacing ~3.3x vs. a 2x tolerance) doesn't actually
apply: muP's optimum moved a full decade, roughly two grid steps, which the grid resolves
just fine. That objection argues against precision the result doesn't depend on.

Second, and more importantly, the original addendum claimed converting to *effective* LR
(`base_lr * base_width/width`) showed muP staying flat ("~0.01 -> ~0.0125, well under
2x") and used that to argue the underlying theory was fine even though the raw-`base_lr`
claim failed. This doesn't hold up two ways. (a) It's circular: the `base_width/width`
factor *is* muP's mechanism, so dividing it back out and then observing "flatness" is
recovering the input, not evidence about anything. The thread's actual pre-registered
prediction is stated in terms of "the learning rate that is optimal ... under the derived
scaling rule" — i.e. the raw `base_lr` a user sets and reuses, specifically because that's
what "transfer without re-tuning" has to mean in practice. (b) It was also arithmetically
wrong: it quoted only the width=64 and width=512 endpoints and dropped the width=256 point
(0.025), which the sanity gate had flagged as noise-tied, not as invalid — including it,
the effective-LR sequence is 0.010 -> 0.025 -> 0.0125, a 2.5x spread, which fails the very
&lt;2x bar the paragraph claimed it cleared.

The honest reading: **the smoke test ran cleanly against the prediction.** SP's raw
LR-optimum was flat where muP's moved a full decade — at this scale, muP made the exact
quantity the thread cares about worse, not better, than doing nothing. This is still not
the thread's real verdict (wrong scale, wrong task, and the width range is too narrow for
muP's usually-asymptotic advantage to plausibly appear at all — that caveat is legitimate
and is the one to keep), but it should not be read as "inconclusive, ignore it." Two
related bugs this exposed in `experiments/harness/report.py` are now fixed: a
both-arms-flat sweep no longer silently auto-passes the effect-size bar (it previously
divided by a zero drift and reported `ratio=inf`, which technically clears any threshold),
and the drift summary now reports which widths were actually used versus gated out instead
of silently dropping a noise-tied point the way this addendum's first draft implicitly
did.

What the real run needs, revised: width range genuinely matters and should reach well past
the pre-registered 16x (30-60x+, since 8x already looks insufficient) to give muP a fair
shot at its asymptotic regime — but extend it to find out whether muP starts winning, not
as a reason to discount the current adverse reading. The step-budget/threshold setup
(`target_train_loss` + `max_steps`) is also a real confounder the first draft of this note
didn't flag: which configs count as "converged" is sensitive to one arbitrary cutoff and
step ceiling, so the real run should report steps-to-target across several thresholds (or
fit the loss curve) rather than hinge everything on one target value. LR grid resolution
was a secondary concern, not the binding one.

**Post-hoc note, 2026-07-03 (`thread06_mup_widerange.py`, k up to 32x, dual thresholds,
still CPU/toy scale, last smoke-test iteration for now):** extending to widths
64/256/1024/2048 (k=1,4,16,32) with a finer LR grid (~2.3x spacing) and two loss
thresholds gives the same qualitative answer as the narrower run, not a different one:
muP's raw `base_lr` optimum keeps moving (0.015 -> 0.08 -> 0.4 -> 0.9 across widths at the
looser threshold), and its log10 drift (1.43 decades, using the two widths that cleared
the sanity gate) is still larger than SP's (0.33 decades) — ratio 0.23x, still failing the
>=3x bar, in the same direction as before. One new, honest wrinkle: at this wider range SP
is no longer perfectly flat either (its optimum drifts down: 0.015 -> 0.015 -> 0.007 ->
0.003), and *both* arms hit the edge of the LR grid at width=2048, which the sanity gate
correctly flags as inconclusive rather than reporting a misleading number — so the grid
itself is now too narrow in both directions, on top of everything else.

Stopping smoke-test iteration on this thread here rather than continuing to chase a
cleaner number: two independent runs at two different width ranges now agree that muP's
raw-LR transfer looks worse than SP's naive robustness at small/toy scale, and the
remaining open question (does this flip once the width range, task, and step budget match
what real muP validation actually needs — thousands of steps, a harder task, much larger
widths) can only be answered by the pre-registered GPU run, not by further CPU tuning.
Moving on to thread 1 (the first thread that's an actual candidate architecture, not a
methodology check) rather than continuing to refine this smoke test.

**Post-hoc note, 2026-07-07 (full-portfolio review, `docs/reviews/
2026-07-07-portfolio-review.md`):** two updates from the review, no change to the
parked/inconclusive verdict. (1) A doc-splice error was fixed above — the 2026-07-03 note
had been inserted mid-sentence into the 2026-07-02 note's final paragraph; the orphaned
fragment is now restored to its sentence. Nothing substantive changed. (2) An independent
static code review verified `models/mlp.py`'s muP Adam multiplier table is correct, ruling
out the simplest implementation-bug explanation for the adverse smoke reads; a literature
review found the observed failure direction essentially unreported as a genuine muP
failure mode, with the leading remaining suspects being the task itself (modular
arithmetic's grokking dynamics are weight-decay-governed, not width-governed; no published
muP validation uses an algorithmic task) and subtler embedding/readout-layer or
weight-decay handling. The cheap decisive next step, before any GPU run: a standard muP
*coordinate check* (per-layer-type activation scale vs. width over a few steps) plus the
"wider is always better" check — see the review doc's idea I1.

**Post-hoc note, 2026-07-03 (real-run PREP, per `docs/reviews/2026-07-08-program-regroup.md`
item C3 — "the keystone experiment"): task, both arms, coordinate-check re-run, frozen
protocol, and a packaged GPU handoff. NO GPU RUN HAS HAPPENED. This note records
preparation only — it contains no verdict, partial or otherwise, on the pre-registered
prediction above.** (Program-regroup context: this repo went through a program-level
regroup diagnosing why so many threads read as negative; thread 6 was named the single
highest-stakes open item, since the whole "falsify small, trust the trend" methodology
currently rests on adverse toy-scale-only evidence for its own transfer premise.)

*What was built* (all under `experiments/`, CPU-smoke-tested, none of it the pre-registered
run itself):

- `tasks/char_lm.py` + committed `tasks/data/tinyshakespeare.txt` — the non-algorithmic task
  named as required, to kill the modular-arithmetic grokking-dynamics confound the three
  smoke tests above are suspected of actually measuring. Corpus is the canonical Karpathy
  char-rnn/nanoGPT tiny-shakespeare file (1,115,394 chars, 65-char ASCII vocab, public
  domain), fetched once and committed verbatim rather than downloaded at run time, so the
  GPU run has no network dependency and the source text can't drift under the experiment.
  Contiguous (not random-window) train/test split, same leakage lesson as the first
  modular-arith smoke test.
- `models/char_lm_mlp.py` (`CharLMMLP`) — baseline arm, a fixed-context neural LM
  (Bengio et al. style) reusing `MuPMLP`'s exact per-layer-type muP table. `d_embed` is a
  FIXED featurizer dimension that does not scale with width (mirrors `MuPMLP`'s fixed
  `input_dim`), so `proj` (the first width-scaling layer) is a genuine muP *input* layer,
  not a hidden layer with a context-scaled fan-in.
- `models/char_lm_recurrence.py` (`CharLMRecurrence`) — thread 6's declared "first novel
  case": embedding -> thread 1's `LinearRecurrentBlock("orthogonal")` (reused **unmodified**)
  -> readout. See "the theta-init question" below — this is where the one genuine open
  design fork lived.
- `harness/train_charlm.py` — training loop producing the **same `RunResult` dataclass**
  `harness/train.py`/`report.py` already use, so `save_run`/`summarize_sweep` consume its
  output with zero format translation. States its FLOP-counting method per
  `docs/methodology.md` and flags the `matrix_exp` term as an order-of-magnitude estimate
  (wall-clock, not this count, is the load-bearing hardware number).
- `scripts/thread06_charlm_smoke.py` — plumbing check (both arms x both parametrizations x
  widths {64, 256}, 500 steps): all 8 configs finite, decreasing, learning (test_acc ~0.31
  vs. 0.015 chance; muP init loss sits exactly at log(65)=4.17, the correct
  vanishing-readout-at-init signature). Not a verdict, same framing as the three smoke tests
  above.
- `scripts/thread06_charlm_coord_check.py` + `scripts/thread06_gpu_run.py` — the coordinate
  check and the frozen-protocol GPU handoff package, both described in detail below.

*The theta-init question (resolved, high confidence in the resolution itself; one residual
uncertainty flagged, not resolved):* `LinearRecurrentBlock`'s `theta` (the free generator of
`A = (1-eps)*expm(skew(theta))`) was originally initialized at a **fixed** `std=0.01`
regardless of width — an SP-like choice for a parameter muP's table says should be
`1/sqrt(width)`-scaled. But `A`'s spectral radius is pinned at exactly `1-eps` by
*construction*, for any theta, so the norm/decay quantity muP normally protects for a hidden
weight is already protected here by `eps` alone — for that one quantity, theta looked
possibly exempt from the usual treatment. Checked directly rather than assumed, two ways:

1. *Numerically (standalone, before touching the real model):* `skew(theta)`'s spectral
   radius (its largest rotation angle, governing `Q=expm(skew(theta))`'s mixing structure —
   the thing `eps` does *not* pin) grows as `~sqrt(width)` under a fixed `std=0.01` (0.213 rad
   @ width=64 -> 1.803 @ width=4096, matching the closed-form prediction
   `2*sqrt(2)*0.01*sqrt(width)` almost exactly at every width), and stays flat (~0.21-0.23)
   under `std=0.08/sqrt(width)` scaling. Separately confirmed `gradient_flow.py`'s existing
   decay diagnostic is *blind* to theta's scale by construction either way
   (`effective_decay_rate` ~0.950 regardless, since `||A^t||=(1-eps)^t*||Q^t||=(1-eps)^t`
   for orthogonal `Q`) — so that diagnostic cannot arbitrate this question; only an
   activation-level coordinate check can.
2. *On the real `CharLMRecurrence` model, via the coordinate-check re-run (below):* the
   coordinate check directly measures `skew(theta)`'s spectral radius vs. width at the
   pinned widths (64/256/1024) under each parametrization's actual init. Result: muP's
   `1/sqrt(width)`-scaled theta gives a log-log slope of **+0.026** (flat); SP's fixed-`0.01`
   theta gives **+0.514** (matches the ~sqrt(width) prediction, exponent 0.5, almost exactly).

**Resolution:** `theta`'s *init* is scaled hidden-like (`THETA_BASE_STD * sqrt(base_width /
width)`), confirmed both by a standalone check and by direct measurement on the real model —
high confidence this is the right call for init. `theta`'s Adam *LR multiplier* is also set
hidden-like (`base_width/width`) as the principled default, but — flagged explicitly, per
the pre-registered instruction to say so rather than guess confidently — **the exact LR
exponent for a matrix-exponential-parameterized generator is not derived in closed form
here**; the expm nonlinearity is exactly the kind of thing this thread exists to ask whether
vanilla muP covers, and the coordinate check below found one signal consistent with (not
proof of) this specific residual uncertainty mattering in practice. If a future real-run
reading looks adverse specifically on the recurrence arm, theta's LR exponent is the first
thing to re-derive, not the init scaling (which has direct numerical support).

*Coordinate-check re-run (thread 14's method, applied to the real char-LM models/widths, not
the original toy MLP-on-modular-arith setup — the pre-flight step the prep instructions
called for before freezing any band):* `scripts/thread06_charlm_coord_check.py`, moderate
read LR (0.01, per thread 14's corrected lesson that its own original aggressive pilot LR
induced a benign transient rather than measuring a defect), checkpoints
`{0,1,2,5,10}` Adam steps. MLP ran the full pinned width grid (64 through 4096, 64x); the
recurrence arm was capped at 1024 (16x) for this pre-flight check specifically — a
recurrence coordinate-check step at width >=2048 is dominated by `matrix_exp` backward on a
`W x W` matrix plus BPTT through the 32-step scan (~minutes/step), which would have pushed
this sanity check itself into GPU-run-scale cost; the muP wiring is width-identical, so a
flat 64->1024 read validates it, and the full grid to 4096 gets exercised for real on the
GPU run. Raw results: `experiments/runs/thread06_charlm_coord_check/{raw_results,
slopes}.json`.

Findings, using thread 14's own corrected pass criteria (flat at init; output/readout
post-multiplier slope ~= -1 at init as the *intended* signature, relaxing toward 0 under
training; positive-control SP expected to fail):

- **At init (t=0), the least noisy and most theory-relevant reading: clean pass for both
  arms.** Every non-output/readout layer type sits at `|slope| < 0.04` for both arms under
  muP. `output_post_mult`/`readout_post_mult` sit at -0.97 / -0.99 respectively — matching
  the expected arithmetically-necessary `-1` signature, not a defect.
- **Positive control: SP fails dramatically for both arms**, more dramatically than thread
  14's original MLP-on-modular-arith reading. MLP-SP at width 4096 spikes to loss=**2743.6**
  at step 2 (from 4.16 at step 1) before partially, erratically recovering (final loss@10
  3.95, vs. muP's smooth monotonic 4.17->3.32); recurrence-SP at width 1024 blows from 4.93
  (step 1) to **45.65** (step 10) while recurrence-muP over the identical steps stays
  tightly, monotonically bounded 4.18->3.80. The check clearly has power on both new models.
- **Theta-init resolution, directly validated:** as reported above, muP's scaled theta gives
  a flat (+0.026) skew-spectral-radius slope vs. SP's +0.514 (matching the ~sqrt(width)
  prediction) — this is the cleanest, most decisive single number produced by the whole
  re-run for the question this thread's prep asked about most directly.
- **Under training, most (not all) layers show the same LR-transient signature thread 14
  diagnosed as benign, confirmed by an LR-robustness sweep here too** (not originally
  required by thread 14's own protocol, run anyway since the at-`0.01` reading exceeded the
  flatness bar in several places and the honest next step, per thread 14's own precedent, is
  to check whether it shrinks with LR before calling it settled). At read LR
  `{0.03, 0.01, 0.003, 0.001}`: MLP's `hidden_0` worst-slope-at-t10 goes `0.679 -> 0.612 ->
  0.125 -> 0.047` (clears the 0.15 bar by LR=0.003, same non-monotonic-at-the-high-end shape
  thread 14's own sweep showed between its 0.3 and 0.1 points); `hidden_1` and `proj` show
  the same pattern. Recurrence's `recur_final` shows the same clean shrink (`0.244 -> 0.119
  -> 0.011`, fully flat by LR=0.001).
- **One genuine, unresolved anomaly, flagged rather than smoothed over:** recurrence's
  `readout_pre_mult` does **not** show this shrinking pattern — it stays at log-log slope
  `0.415 -> 0.448 -> 0.301` across the same tenfold LR range, never trending toward the 0.15
  bar the way every other signal (including the MLP's directly analogous `output_pre_mult`,
  which shrank 0.860 -> 0.351 -> 0.253 over the same LR range) does. This is a real,
  measured difference between the two models' otherwise-structurally-similar output-type
  layers, not noise — repeated at all three LR points. Leading hypothesis (not confirmed):
  this connects to the theta-LR-exponent uncertainty flagged above, since the readout sits
  downstream of the recurrence's evolving state, which depends on theta specifically. An
  isolating test was designed (train normally vs. freeze theta's LR to exactly 0, compare
  `readout_pre_mult`'s slope) but did not complete — it hit severe CPU contention from
  thread 18's concurrently-running arm-rerun processes on this session's shared 4-core
  sandbox (load average ~9-10 observed), and per `CLAUDE.md`'s own logged lesson about
  contention producing misleading numbers, it was not worth forcing through under those
  conditions rather than re-queuing it properly. **This is reported honestly as untested,
  not resolved** — a leading hypothesis with one piece of indirect corroborating evidence
  (the readout-specific, theta-downstream pattern), not a finding. It does not block
  freezing the protocol below (thread 6's real prediction is about steps-to-target-loss
  under a multi-hundred/thousand-step LR sweep, not a 10-step activation-flatness reading),
  but it is the first thing to revisit if the recurrence arm's real-run reading looks odd
  specifically at small width or specifically differs from the MLP arm's pattern.

**Net judgment on "does the muP wiring check out for the new models":** yes, on the
decisive init-time evidence and the theta-init resolution, for both arms — no implementation
defect serious enough to block freezing the protocol. The under-training drift mostly
follows the same benign-transient shape thread 14 already characterized, with one named,
unresolved exception (recurrence's `readout_pre_mult`) carried forward explicitly rather
than swept in with the rest.

**Frozen protocol** (baked into `scripts/thread06_gpu_run.py`, calibrated from the smoke
test above plus a separate 2000-step floor check at width=256 across LRs `{1e-3, 3e-3, 1e-2,
3e-2, 1e-1}` for both arms — MLP bottoms around 2.0-2.2 nats, best LR ~3e-3, diverges by
1e-1; recurrence bottoms around 2.3-2.4 nats at the same best LR):

- Widths `{64, 256, 1024, 2048, 4096}` — k in `{1, 4, 16, 32, 64}` off `base_width=64`,
  reaching **64x**, past both the pre-registered 16x and the revised "30-60x+" note.
- LRs: 8-point log grid `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1}`, ~3.16x spacing.
- Seeds `{0, 1, 2, 3, 4}` — 5 seeds, this repo's preferred floor.
- Both arms (`mlp`, `recurrence`) x both parametrizations (`sp`, `mup`).
- `context_len=32`, `d_embed=64` (MLP arm), `depth=4`, `eps=0.05` (recurrence arm),
  `batch_size=128`, `max_steps=2000`.
- Loss thresholds `{3.0, 2.7, 2.4, 2.2, 2.0}` nats (init CE = log(65) = 4.17), multiple
  thresholds tracked per the existing steps-to-target convention, sized from the floor
  check above so at least the looser thresholds are reachable by both arms within budget.
- Total configs: 2 arms x 2 params x 5 widths x 8 LRs x 5 seeds = **800 runs.**

**These bands are frozen and must not be edited in place after seeing GPU results** — this
is this repo's own pre-registration rule (`docs/methodology.md`). A miscalibration found
after running gets a fresh dated addendum and a re-run under a revised, re-committed
protocol, exactly like the three CPU smoke tests above.

**GPU handoff package:** `experiments/scripts/thread06_gpu_run.py` — self-contained, the
frozen bands above are baked into the script (not re-decided at run time), auto-detects CUDA
(falls back to CPU with an explicit warning that CPU is only sane for re-aggregating an
existing result set — the recurrence arm at width 4096 costs ~98s per training *step* on
this sandbox's CPU, dominated by `matrix_exp` backward on a 4096x4096 matrix, timed directly
before writing this). Writes one JSON per run, in the exact `RunResult` schema
`harness/report.py` already consumes, under `experiments/runs/thread06_gpu_run/<arm>/` —
resumable (skips any config whose output file already exists). After the sweep it writes
`summary.json` via the existing `summarize_sweep`. Ingestion path (no CI/cloud storage in
this repo): copy the whole `experiments/runs/thread06_gpu_run/` directory back into a
checkout of this repo, commit it, and re-run the same script once (it finds every JSON
already present, runs nothing new, and rebuilds `summary.json`) — this round trip (save ->
reload -> `summarize_sweep`, including the JSON string-key-to-float fix `steps_to_target`
needs) was exercised end-to-end at toy scale (2 params x 2 widths x 2 LRs x 2 seeds, 60
steps, temp directory) before this note was written: 16/16 runs saved and reloaded, slopes
and file-naming matched exactly, `summarize_sweep` consumed the reloaded results without
modification, and re-saving an existing config overwrote in place rather than duplicating
(confirms resumability). This is the sandbox->GPU->ingest handoff the program regroup named
as never having been exercised even once in 17 prior threads — now exercised, at toy scale,
for the first time. The script's own docstring repeats the frozen-bands warning and gives
brief run/ingestion instructions for the user.

**What this note does not do:** run the pre-registered experiment. No claim of "supported,"
"falsified," or "inconclusive" attaches to the numeric prediction stated at the top of this
file from anything above — that verdict can only come from `thread06_gpu_run.py` executed
on real GPU hardware at the frozen protocol above, which has not happened yet.
