# Thread 4: Optimal-control integrators for depth

**Math source:** Pontryagin's Maximum Principle (PMP), the ODE view of residual networks
(a ResNet is a discretized `dh/dt = f(h, t; theta)`), numerical integration theory (Euler
vs. Runge-Kutta vs. symplectic integrators, local/global truncation error orders).

## Motivation

A stack of residual blocks `h_{l+1} = h_l + f(h_l; theta_l)` is exactly a forward-Euler
discretization of an ODE. Numerical analysis has precise, well-understood statements about
how integrator choice trades off per-step compute for approximation error order: Euler is
first-order accurate, RK4 is fourth-order, symplectic integrators preserve certain
invariants (useful when the underlying "dynamics" should conserve something, e.g. norm).
If a residual stack is really approximating a continuous transformation, then replacing the
plain-Euler residual block with a higher-order integrator block should let a shallower
stack reach the same approximation quality — a concrete, numerically-derivable depth-efficiency
prediction, not just "RK4 sounds fancier."

## Architectural hypothesis

Residual blocks structured as a single step of a higher-order integrator (e.g., a 2-stage
or 4-stage Runge-Kutta step, reusing the same per-stage function `f` to control extra
compute) need fewer effective layers/steps to reach a target approximation error than
plain residual (Euler) blocks, and the *ratio* of required depth between integrator orders
should roughly track the integrators' known truncation-error order difference.

## Falsifiable prediction

For a fixed target task loss, the number of blocks required for an RK4-style block to
match a plain-residual (Euler) stack's loss should be smaller by a factor consistent with
the difference in local truncation error order (a specific, pre-registered numeric
ballpark from integrator theory, not an arbitrary "should be better"), when compute is
matched by charging RK-style blocks their true extra per-step FLOP cost (a 4-stage block
costs ~4x an Euler block's per-step compute, so this must be reflected in the matched-FLOPs
comparison, not just matched-block-count). If the required-depth ratio doesn't track the
predicted order difference (even loosely) once FLOPs are matched fairly, the "residual
stack behaves like an ODE discretization" framing is weaker than claimed for this
architecture and the thread is falsified.

## Minimal experiment

- Implement plain-residual, 2-stage, and 4-stage integrator blocks reusing the same
  per-stage function, matched parameter count per stage.
- Small task (CIFAR-10 subset or synthetic regression with a known smooth target function,
  which is arguably the fairer test since it's actually close to "approximating a
  continuous map").
- Sweep depth for each integrator order, find minimum depth to hit a fixed loss threshold,
  compute true FLOPs at that depth, compare ratios against theory.

## Compute budget

Small models/datasets, modest depth sweep — well under a GPU-day.

## Bitter-lesson check

- Still dense matmul stacks underneath; no exotic ops, no task-specific priors. Low risk.
- The "charge RK blocks their true extra FLOP cost" rule in the prediction is there
  specifically to prevent the classic self-deception of comparing matched depth instead of
  matched compute — an RK4 block only "wins" if it wins per-FLOP, not per-layer.

## Known prior work / risk of reinventing

Neural ODEs (Chen et al. 2018) and the PMP-based training view (Li, Weinan E, et al.) are
the direct ancestors; some prior work has explored higher-order integrator blocks for
ResNets. The contribution here is the falsification framing: a pre-registered, FLOP-honest
depth-reduction ratio to check against, rather than a general claim that "structured
residual blocks are better."

## Status

Not yet run.
