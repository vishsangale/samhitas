"""Mean-field / edge-of-chaos numerics for thread 2
(docs/threads/02-criticality-guided-init.md).

Implements the Poole et al. 2016 / Schoenholz et al. 2017 forward-signal-propagation
recursion for a pointwise-activation, unnormalized MLP with per-layer weight/bias
variance (sigma_w^2, sigma_b^2): the single-input variance map q -> q' and its fixed
point q*, and the correlation-map slope at that fixed point chi_1(sigma_w^2, sigma_b^2).
chi_1 < 1 is the ordered phase (correlations/gradients collapse with depth), chi_1 > 1 is
chaotic (correlations decorrelate / gradients blow up), chi_1 = 1 is the critical edge.
The theory's own predicted depth scale is xi = 1 / |log(chi_1)|.

Expectations over z ~ N(0,1) are computed by Gauss-Hermite quadrature (exact for
polynomials up to degree 2*n_quad-1, and accurate to float64 precision well below that
degree for smooth bounded activations like tanh) rather than Monte Carlo, since the
recursion runs inside a fixed-point iteration and a bisection search and needs to be both
fast and deterministic.
"""

import numpy as np


def _gauss_hermite_nodes(n_quad: int):
    x, w = np.polynomial.hermite.hermgauss(n_quad)
    # hermgauss integrates against exp(-x^2); convert to E_{z~N(0,1)}[f(z)] via z = sqrt(2)*x
    # and the standard-normal-density change of variables (weights renormalized by sqrt(pi)).
    z = np.sqrt(2.0) * x
    weights = w / np.sqrt(np.pi)
    return z, weights


def expectation(f, q: float, n_quad: int = 100) -> float:
    """E_{z~N(0,1)}[f(sqrt(q) * z)] via Gauss-Hermite quadrature."""
    z, w = _gauss_hermite_nodes(n_quad)
    return float(np.sum(w * f(np.sqrt(max(q, 0.0)) * z)))


def variance_map(q: float, sigma_w2: float, sigma_b2: float, phi, n_quad: int = 100) -> float:
    """q_{l+1} = sigma_w^2 * E_z[phi(sqrt(q_l) z)^2] + sigma_b^2."""
    return sigma_w2 * expectation(lambda z: phi(z) ** 2, q, n_quad) + sigma_b2


def fixed_point_q(sigma_w2: float, sigma_b2: float, phi, n_quad: int = 100,
                   q_lo: float = 1e-10, q_hi: float = None, q_width_tol: float = 1e-13,
                   max_iter: int = 200) -> float:
    """Root-finds q* = variance_map(q*, ...) - q via bisection on g(q) = map(q) - q,
    rather than plain fixed-point iteration of the map itself.

    Plain iteration's convergence rate is governed by the map's derivative at q*, which is
    exactly chi_1 -- i.e. it converges *slowest* right at the critical point (chi_1 = 1,
    "critical slowing down"), which is precisely the regime this thread needs precision in
    for the sigma_w2* bisection below. Bisection on g(q) instead converges geometrically
    at a fixed rate independent of chi_1, since it's solving the algebraic equation
    directly rather than physically simulating the (slow, near-critical) forward map.

    Stopping on the *interval width* (|hi-lo| < q_width_tol) rather than on |g(mid)| <
    some absolute tolerance: right at criticality q* itself is tiny (the map's slope at
    q=0 is ~1, so g is nearly flat near the root), so a fixed absolute tolerance on g
    stops the search too early relative to q*'s own scale -- caught by a first pass of
    this cross-check showing spurious non-monotonicity of chi_1(sigma_w2) right at the
    transition, traced to a handful of bisection steps' worth of under-resolved q* rather
    than a real effect. Interval-width stopping instead directly bounds q*'s absolute
    error regardless of how flat g is there, at the cost of always running close to
    max_iter steps (cheap: each step is one quadrature evaluation).

    variance_map is bounded above by sigma_w2 + sigma_b2 (since phi^2 <= 1 for tanh) and
    increasing in q, so g(q) < 0 for q > sigma_w2 + sigma_b2 always -- a bracket is
    guaranteed to exist. If g(q_lo) <= 0 already (e.g. sigma_b2=0 and sigma_w2<=1, where 0
    is the unique non-negative fixed point), the attracting fixed point is 0 (or below the
    resolution of q_lo), returned directly rather than bisected.
    """
    if q_hi is None:
        q_hi = max(10.0, 4.0 * (sigma_w2 + sigma_b2) + 10.0)

    def g(q):
        return variance_map(q, sigma_w2, sigma_b2, phi, n_quad) - q

    g_lo = g(q_lo)
    if g_lo <= 0:
        return 0.0
    g_hi = g(q_hi)
    while g_hi > 0 and q_hi < 1e6:
        q_hi *= 2.0
        g_hi = g(q_hi)
    assert g_hi <= 0, f"could not bracket a root: g({q_hi})={g_hi}"

    lo, hi = q_lo, q_hi
    for _ in range(max_iter):
        if hi - lo < q_width_tol:
            break
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if gm > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chi_1(sigma_w2: float, sigma_b2: float, phi, phi_prime, n_quad: int = 100,
          **fp_kwargs) -> tuple:
    """Returns (chi_1, q_star). chi_1 = sigma_w^2 * E_z[phi'(sqrt(q*) z)^2] is the
    correlation-map derivative at the fixed point c=1 -- the quantity whose value vs. 1
    determines the ordered/chaotic/critical phase."""
    q_star = fixed_point_q(sigma_w2, sigma_b2, phi, n_quad, **fp_kwargs)
    val = sigma_w2 * expectation(lambda z: phi_prime(z) ** 2, q_star, n_quad)
    return val, q_star


def depth_scale(chi: float) -> float:
    """xi = 1 / |log(chi_1)|, the theory's own predicted number of layers signal
    survives. Undefined (infinite) exactly at criticality."""
    if chi == 1.0:
        return float("inf")
    if chi <= 0:
        return 0.0
    return 1.0 / abs(np.log(chi))


def find_critical_sigma_w2(sigma_b2: float, phi, phi_prime, lo: float = 0.05, hi: float = 10.0,
                            n_quad: int = 100, width_tol: float = 1e-9, max_iter: int = 100,
                            **fp_kwargs) -> float:
    """Bisection on sigma_w^2 for chi_1(sigma_w^2, sigma_b^2) == 1, assuming chi_1 is
    monotonically increasing in sigma_w^2 at fixed sigma_b^2 (true for tanh: more weight
    variance always pushes the network toward chaos).

    Stops on interval width, not |f(mid)| < tol, for the same reason fixed_point_q does:
    right at the sigma_b2=0 degenerate transition, chi_1 stays extremely flat (within
    ~1e-8 of 1) over a wide range of sigma_w2 just above the true critical point, so an
    absolute-f tolerance triggers a premature, imprecise stop. Interval-width stopping
    keeps refining regardless of how flat f is near the root.
    """
    def f(sw2):
        c, _ = chi_1(sw2, sigma_b2, phi, phi_prime, n_quad, **fp_kwargs)
        return c - 1.0

    flo, fhi = f(lo), f(hi)
    assert flo < 0 < fhi, (
        f"bracket [{lo},{hi}] doesn't bracket chi_1=1 (f(lo)={flo:.4f}, f(hi)={fhi:.4f})"
    )
    for _ in range(max_iter):
        if hi - lo < width_tol:
            break
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tanh_phi(x):
    return np.tanh(x)


def tanh_phi_prime(x):
    return 1.0 - np.tanh(x) ** 2
