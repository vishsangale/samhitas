"""Pre-registered run for thread 13 (docs/threads/13-robust-gradient-flow-depth-scale.md),
the second follow-up to thread 2 / thread 12: same protocol as thread 12's gradient-flow
depth-scale test, but with 50 seeds (up from 30) and a Theil-Sen robust regression (median
of pairwise slopes on per-depth medians) instead of ordinary least squares, targeting the
heavy-tailed backward-pass seed variance at large chaotic-phase depth that a review traced
thread 12's failure to.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import itertools

import numpy as np
import torch
import torch.nn.functional as F

from experiments.harness import meanfield as mf
from experiments.models.deep_mlp import DeepMLP
from experiments.tasks import modular_arith

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread13_robust_gradient_flow"

P = 17
HIDDEN = 32
SIGMA_B2 = 0.1
N_TRAIN = 200
N_TEST = 89
BATCH = 64

# Same grid as thread 12 -- deliberately unchanged, so this is a fair test of a different
# estimator on the same data-generating design, not a search over depth windows.
SIGMA_W2_GRID = [1.3, 1.4, 1.6, 1.8, 1.9, 2.05, 2.2, 2.4, 2.8]
DEPTH_GRID = [2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256, 362]
SEEDS = list(range(50))

SHAPE_CORR_MIN = 0.8
PEAK_ADJACENT_TO = {1.9, 2.05}
MAGNITUDE_RATIO_BAND = 3.0
MAGNITUDE_XI_INTERIOR = (5.0, 60.0)


def xi_at(sigma_w2: float) -> float:
    chi, _ = mf.chi_1(sigma_w2, SIGMA_B2, mf.tanh_phi, mf.tanh_phi_prime)
    return mf.depth_scale(chi)


def grad_norm(sigma_w2, depth, seed, x_batch, y_batch):
    torch.manual_seed(seed)
    m = DeepMLP(modular_arith.input_dim(P), HIDDEN, depth, modular_arith.num_classes(P), sigma_w2, SIGMA_B2)
    logits = m(x_batch)
    loss = F.cross_entropy(logits, y_batch)
    m.zero_grad(set_to_none=True)
    loss.backward()
    return m.layers[0].weight.grad.norm().item()


def theil_sen_slope(xs, ys):
    """Median of all pairwise slopes (xs, ys) -- robust to up to ~29% corrupted points."""
    slopes = []
    for (x1, y1), (x2, y2) in itertools.combinations(zip(xs, ys), 2):
        if x2 != x1:
            slopes.append((y2 - y1) / (x2 - x1))
    return float(np.median(slopes))


def fit_length_robust(sigma_w2, x_batch, y_batch):
    """Per-depth median log(grad_norm) across 50 seeds, then Theil-Sen slope across the
    16 depth-median points."""
    depth_medians = []
    for depth in DEPTH_GRID:
        vals = [grad_norm(sigma_w2, depth, seed, x_batch, y_batch) for seed in SEEDS]
        log_vals = [np.log(v) for v in vals if v > 0 and np.isfinite(v)]
        depth_medians.append(float(np.median(log_vals)))
    slope = theil_sen_slope(DEPTH_GRID, depth_medians)
    length = 1.0 / abs(slope) if slope != 0 else float("inf")
    return length, slope, depth_medians


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, _, _ = modular_arith.make_dataset(P, seed=0, n_train=N_TRAIN, n_test=N_TEST)
    x_batch, y_batch = x_train[:BATCH], y_train[:BATCH]

    t0 = time.time()
    rows = []
    for sigma_w2 in SIGMA_W2_GRID:
        xi = xi_at(sigma_w2)
        length, slope, depth_medians = fit_length_robust(sigma_w2, x_batch, y_batch)
        ratio = length / xi if xi not in (0, float("inf")) else float("nan")
        rows.append({
            "sigma_w2": sigma_w2, "xi_theory": xi, "empirical_length": length,
            "slope": slope, "ratio": ratio, "depth_medians": depth_medians,
        })
        print(f"sigma_w2={sigma_w2:.4f}  xi={xi:8.2f}  L={length:8.2f}  ratio={ratio:6.3f}  "
              f"elapsed={time.time()-t0:.1f}s")

    (RUN_DIR / "results.json").write_text(json.dumps(rows, indent=2))

    log_xi = [np.log(r["xi_theory"]) for r in rows]
    log_L = [np.log(r["empirical_length"]) for r in rows]
    corr = float(np.corrcoef(log_xi, log_L)[0, 1])
    peak_row = max(rows, key=lambda r: r["empirical_length"])
    peak_sw2 = peak_row["sigma_w2"]
    shape_pass = corr >= SHAPE_CORR_MIN and peak_sw2 in PEAK_ADJACENT_TO

    interior = [r for r in rows if MAGNITUDE_XI_INTERIOR[0] <= r["xi_theory"] <= MAGNITUDE_XI_INTERIOR[1]]
    mag_pass = all(1.0 / MAGNITUDE_RATIO_BAND <= r["ratio"] <= MAGNITUDE_RATIO_BAND for r in interior) if interior else False

    print("\n--- Prediction verdict (pre-registered in docs/threads/13-robust-gradient-flow-depth-scale.md) ---")
    print(f"Shape: log(L) vs log(xi) correlation = {corr:.4f} (need >= {SHAPE_CORR_MIN}); "
          f"peak at sigma_w2={peak_sw2} (need in {PEAK_ADJACENT_TO}) -> {'PASS' if shape_pass else 'FAIL'}")
    print(f"Magnitude: {len(interior)} interior points (xi in {MAGNITUDE_XI_INTERIOR}), "
          f"ratios={[round(r['ratio'], 3) for r in interior]} "
          f"(need all in [1/{MAGNITUDE_RATIO_BAND}, {MAGNITUDE_RATIO_BAND}]) -> {'PASS' if mag_pass else 'FAIL'}")
    print(f"\nOverall: {'PASS' if (shape_pass and mag_pass) else 'FAIL'}")


if __name__ == "__main__":
    main()
