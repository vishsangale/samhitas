"""Pre-registered run for thread 12 (docs/threads/12-gradient-flow-depth-scale.md), a
follow-up to thread 2's falsified loss-based prediction: does the init-time gradient-flow
decay/growth *length* -- not raw magnitude, not loss-reaching within a training budget --
match the mean-field theory's derived depth scale xi(sigma_w2)?

Protocol exactly as pre-registered: sigma_b2=0.1, sigma_w2 grid of 9 points spanning
xi roughly 5-75 on both sides of the theoretical critical point, depth grid of 16
geometric points (2 to 362), 30 seeds per cell. No training loop -- a single
forward+backward pass per (sigma_w2, depth, seed) cell, using the first hidden layer's
weight gradient norm on a fixed 64-example batch of the modular-arithmetic task (p=17,
same task thread 2 used).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import numpy as np
import torch
import torch.nn.functional as F

from experiments.harness import meanfield as mf
from experiments.models.deep_mlp import DeepMLP
from experiments.tasks import modular_arith

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread12_gradient_flow_depth_scale"

P = 17
HIDDEN = 32
SIGMA_B2 = 0.1
N_TRAIN = 200
N_TEST = 89
BATCH = 64

SIGMA_W2_GRID = [1.3, 1.4, 1.6, 1.8, 1.9, 2.05, 2.2, 2.4, 2.8]
DEPTH_GRID = [2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256, 362]
SEEDS = list(range(30))

# Pass/fail band, pre-registered:
SHAPE_CORR_MIN = 0.8
PEAK_ADJACENT_TO = {1.9, 2.05}  # grid points adjacent to theoretical critical sigma_w2*=1.9861
MAGNITUDE_RATIO_BAND = 3.0
MAGNITUDE_XI_INTERIOR = (5.0, 60.0)  # only check ratio band for xi well inside the resolvable range


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


def fit_length(sigma_w2, x_batch, y_batch):
    """Fits log(grad_norm) vs depth across all (depth, seed) points for this sigma_w2.
    Returns (length, slope, n_points_used, n_zero_or_nan_dropped)."""
    depths, log_norms = [], []
    dropped = 0
    for depth in DEPTH_GRID:
        for seed in SEEDS:
            gn = grad_norm(sigma_w2, depth, seed, x_batch, y_batch)
            if gn > 0 and np.isfinite(gn):
                depths.append(depth)
                log_norms.append(np.log(gn))
            else:
                dropped += 1
    slope, intercept = np.polyfit(depths, log_norms, 1)
    length = 1.0 / abs(slope) if slope != 0 else float("inf")
    return length, slope, len(depths), dropped


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, _, _ = modular_arith.make_dataset(P, seed=0, n_train=N_TRAIN, n_test=N_TEST)
    x_batch, y_batch = x_train[:BATCH], y_train[:BATCH]

    t0 = time.time()
    rows = []
    for sigma_w2 in SIGMA_W2_GRID:
        xi = xi_at(sigma_w2)
        length, slope, n_used, n_dropped = fit_length(sigma_w2, x_batch, y_batch)
        ratio = length / xi if xi not in (0, float("inf")) else float("nan")
        rows.append({
            "sigma_w2": sigma_w2, "xi_theory": xi, "empirical_length": length,
            "slope": slope, "ratio": ratio, "n_points_used": n_used, "n_dropped": n_dropped,
        })
        print(f"sigma_w2={sigma_w2:.4f}  xi={xi:8.2f}  L={length:8.2f}  ratio={ratio:6.3f}  "
              f"(dropped {n_dropped} non-finite points)  elapsed={time.time()-t0:.1f}s")

    (RUN_DIR / "results.json").write_text(json.dumps(rows, indent=2))

    # --- shape criterion ---
    log_xi = [np.log(r["xi_theory"]) for r in rows]
    log_L = [np.log(r["empirical_length"]) for r in rows]
    corr = float(np.corrcoef(log_xi, log_L)[0, 1])
    peak_row = max(rows, key=lambda r: r["empirical_length"])
    peak_sw2 = peak_row["sigma_w2"]
    shape_pass = corr >= SHAPE_CORR_MIN and peak_sw2 in PEAK_ADJACENT_TO

    # --- magnitude criterion ---
    interior = [r for r in rows if MAGNITUDE_XI_INTERIOR[0] <= r["xi_theory"] <= MAGNITUDE_XI_INTERIOR[1]]
    mag_pass = all(1.0 / MAGNITUDE_RATIO_BAND <= r["ratio"] <= MAGNITUDE_RATIO_BAND for r in interior) if interior else False

    print("\n--- Prediction verdict (pre-registered in docs/threads/12-gradient-flow-depth-scale.md) ---")
    print(f"Shape: log(L) vs log(xi) correlation = {corr:.4f} (need >= {SHAPE_CORR_MIN}); "
          f"peak at sigma_w2={peak_sw2} (need in {PEAK_ADJACENT_TO}) -> {'PASS' if shape_pass else 'FAIL'}")
    print(f"Magnitude: {len(interior)} interior points (xi in {MAGNITUDE_XI_INTERIOR}), "
          f"ratios={[round(r['ratio'], 3) for r in interior]} "
          f"(need all in [1/{MAGNITUDE_RATIO_BAND}, {MAGNITUDE_RATIO_BAND}]) -> {'PASS' if mag_pass else 'FAIL'}")
    print(f"\nOverall: {'PASS' if (shape_pass and mag_pass) else 'FAIL'}")


if __name__ == "__main__":
    main()
