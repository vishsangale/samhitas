# Thread 4 (priority 4): Optimal-control integrators for depth

**Math source:** Pontryagin's Maximum Principle (PMP), the ODE view of residual networks
(a ResNet is a discretized `dh/dt = f(h, t; theta)`), numerical integration theory (Euler
vs. Runge-Kutta vs. symplectic integrators, local/global truncation error orders).

> **Caveat added after review, read before building.** The ODE/truncation-error argument
> is an asymptotic statement as the step size `Δt → 0`, which requires the residual
> branch's contribution to be *small* relative to the state (`h_{l+1} = h_l + Δt·f(h_l)`
> with small `Δt`). Standard *trained* ResNets typically do not satisfy this — residual
> branches are usually O(1), not small — so the premise may simply not hold for ordinary
> trained depth, and prior work on higher-order/RK-style ResNet blocks reports gains that
> are marginal and inconsistent at best. The honest prior going in is that this thread is
> likely to get falsified on a standard depth/vision setup. The synthetic-smooth-target-
> function task is the one setting where the small-step premise is actually plausible by
> construction (you control the target's smoothness and can keep per-step updates small),
> so that experiment is promoted to the primary test and the CIFAR-10 variant is secondary
> exploration, not the load-bearing evidence.

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

- **Primary test:** synthetic regression with a known smooth target function, where the
  per-step update can be kept small by construction (matching the small-`Δt` premise the
  theory actually needs). Implement plain-residual, 2-stage, and 4-stage integrator blocks
  reusing the same per-stage function, matched parameter count per stage.
- **Secondary/exploratory:** CIFAR-10 subset, same block variants — useful for seeing
  whether the effect survives outside the constructed small-step regime, but a null result
  here should not be treated as falsifying the theory itself (see caveat above), only as
  evidence the effect doesn't help at standard trained-ResNet depth.
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

Not yet run. Priority 4 — run after threads 6, 1, and 2. Expected outcome (stated up
front, per the pre-registration rule in `docs/methodology.md`): likely falsified on the
CIFAR-10 arm, possibly supported on the synthetic-smooth-target arm.
