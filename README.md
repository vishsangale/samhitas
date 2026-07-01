# samhitas

*samhitā* (संहिता) — "collection, compilation, systematic arrangement." This repo is a
collection of mathematically-derived hypotheses about neural network architecture, each
paired with the smallest experiment that could falsify it.

## Thesis

Neural architecture design has mostly proceeded by trial-and-error at large scale. We want
to invert that: start from mathematics — dynamical systems, control theory, statistical
mechanics, information theory, operator theory, group theory — derive an architectural
hypothesis with a *falsifiable, quantitative prediction*, and test that prediction at the
smallest scale that could break it. Scale only enters to check the theory's prediction still
holds along a trend line, not to brute-force a result into existence.

Start here:

- [`RESEARCH.md`](./RESEARCH.md) — the thesis, methodology, guardrails, and the current
  portfolio of research threads. This is the draft under review.
- [`docs/methodology.md`](./docs/methodology.md) — the falsification loop and the
  matched-compute / scaling-law protocol used to grade every thread.
- [`docs/threads/`](./docs/threads) — one doc per candidate architectural idea: motivating
  math, hypothesis, falsifiable prediction, minimal experiment, compute budget.
- [`experiments/README.md`](./experiments/README.md) — the shared experiment harness plan
  (tasks, datasets, sweep protocol). Code lands here once a thread is greenlit.

## Status

Seed stage. Nothing has been run yet. The immediate next step is review, then picking 1-2
threads to actually implement and try to falsify.
