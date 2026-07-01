# Thread 7 (priority 6, added after review): PAC-Bayes / flatness as a design target

**Math source:** PAC-Bayes generalization bounds, loss-landscape sharpness/flatness
measures (Hessian trace/top eigenvalues, SAM-style perturbation sharpness).

## Motivation

Added after the Opus 4.8 review flagged a gap: every other thread in this portfolio
targets trainability (depth, stability, hyperparameter transfer) or compute efficiency —
none of them targets *generalization* directly. PAC-Bayes bounds give a computable
quantity (a KL-divergence-weighted perturbation robustness of the trained solution) that
is provably related to the test/train gap, and flatness measures (Hessian trace, SAM
sharpness) are a cheap proxy for the same thing. This is squarely a "general statistical
structure, not task-specific knowledge" prior in the sense of section 2 of `RESEARCH.md` —
it says nothing about what the data is, only about the shape of the loss landscape the
architecture induces.

This thread differs from the others in kind: it doesn't propose a new layer. It proposes a
**design target** — a way to score any layer/architecture choice (including ones from
other threads) by predicted generalization gap, computable at small scale, before caring
about accuracy at all.

## Architectural hypothesis

For a fixed task and matched training loss, architectural variants that a PAC-Bayes bound
(or a cheap flatness proxy computed from it) ranks as "flatter" will show a smaller
train/test gap in practice, and this ranking should be predictive *across* architecture
families (e.g. it should correctly order two different threads' layer types against each
other), not just within one family where flatness-generalization correlations are already
well documented (e.g. within plain MLPs/CNNs).

## Falsifiable prediction (pre-registered)

Take at least 3 architecturally distinct small models (e.g., a plain MLP/CNN baseline plus
two variants from other threads in this repo, once implemented) trained to the *same*
target training loss on the same small dataset. Compute a cheap flatness/PAC-Bayes proxy
for each (e.g. Hessian trace estimated via Hutchinson's estimator, or SAM-style worst-case
perturbation loss at a fixed radius) immediately after training. The proxy's ranking of
the models should match the ranking of their actual test-set gap (train loss minus test
loss, or train acc minus test acc) with correlation above a pre-registered threshold
(e.g. Spearman rho >= 0.7 across a sweep of >=6 model/seed combinations). If the proxy's
ranking doesn't track the actual generalization-gap ranking across architecturally
different models (as opposed to just across random seeds/hyperparameters of one
architecture, which is the easier and already-well-established result), the "flatness
proxy is a useful cross-architecture design target" hypothesis is falsified — even though
within-architecture flatness-generalization correlations are expected to hold and are not
themselves the interesting claim here.

## Minimal experiment

- Take the plain baseline plus 2 already-implemented novel layers from other threads
  (e.g. thread 1's structured recurrence, thread 4's integrator block, once those exist).
- Train each to matched training loss on a small classification/LM task, multiple seeds.
- Compute the flatness/PAC-Bayes proxy for each trained model; compute actual test gap;
  correlate across the full set of (architecture x seed) runs.

## Compute budget

Dominated by whatever the other threads' models already cost to train (this thread is a
measurement layered on top of them), plus the cost of a Hessian-trace/SAM-perturbation
estimate per trained model, which is cheap (a handful of extra forward/backward passes,
not a new training run). Should add negligible compute budget on top of threads already
being run.

## Bitter-lesson check

- Purely a measurement of loss-landscape geometry, no task-specific knowledge. Low risk.
- Does not itself change model architecture or add inference-time cost — it's a design-
  time diagnostic, which is compute-scaling-neutral by construction (it doesn't compete
  with scale, it's orthogonal to it).

## Known prior work / risk of reinventing

Flatness-generalization correlation is a large, sometimes contested literature (Keskar et
al. on sharp minima, subsequent work questioning whether the correlation is causal or an
artifact of reparameterization-sensitive sharpness measures — Dinh et al. 2017 is the key
caution here). The prediction above is written to be robust to that caution by requiring
the correlation to hold *across architecturally different models at matched training loss*
specifically, and by pre-registering the correlation threshold, rather than reporting
whichever flatness measure happens to correlate post hoc.

## Status

Not yet run. Priority 6 — depends on other threads' layers existing first; treat as a
cross-cutting measurement to layer onto threads 1/4 once they have working
implementations, not a standalone build.
