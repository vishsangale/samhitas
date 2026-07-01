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
  portfolio of research threads (draft v1, revised after an adversarial review pass).
- [`docs/methodology.md`](./docs/methodology.md) — the falsification loop and the
  matched-compute / scaling-law protocol used to grade every thread.
- [`docs/threads/`](./docs/threads) — one doc per candidate architectural idea: motivating
  math, hypothesis, falsifiable prediction, minimal experiment, compute budget.
- [`experiments/README.md`](./experiments/README.md) — the shared experiment harness plan
  (tasks, datasets, sweep protocol). Code lands here once a thread is greenlit.

## Status

Seed stage. Nothing has been run yet. Draft v0 went through an adversarial review pass
(`RESEARCH.md` section 5); the plan is now revised (v1) with a tightened methodology,
reordered thread priority, two threads deferred pending unresolved issues, and two new
threads added. Next step: build the shared harness (`experiments/README.md`) and run the
priority-1 thread.
