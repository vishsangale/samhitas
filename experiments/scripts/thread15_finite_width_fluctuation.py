"""Pre-registered run for thread 15 (docs/threads/15-finite-width-fluctuation-test.md,
idea I2): does the threads 12/13 chaotic-phase gradient-flow anomaly match finite-width
(Hanin-Nica) log-normal-gradient theory, quantitatively?

No training loop -- single forward+backward pass per (sigma_w2, width, depth, seed) cell,
first hidden layer's weight gradient norm on a fixed 64-example batch of the same
modular-arithmetic task (p=17) threads 2/12/13 used. Restricted to the four chaotic-phase
sigma_w2 points thread 13 flagged as anomalous, but adds width as a swept dimension (fixed
at 32 in threads 12/13) to test the width-scaling prediction directly.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import itertools
import json

import numpy as np
import torch
import torch.nn.functional as F

from experiments.harness import meanfield as mf
from experiments.models.deep_mlp import DeepMLP
from experiments.tasks import modular_arith

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread15_finite_width_fluctuation"

P = 17
SIGMA_B2 = 0.1
N_TRAIN = 200
N_TEST = 89
BATCH = 64

SIGMA_W2_GRID = [2.05, 2.2, 2.4, 2.8]  # the four chaotic-phase anomaly points from thread 13
WIDTH_GRID = [32, 64, 128]
DEPTH_GRID = [2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256, 362]  # same as threads 12/13
SEEDS = list(range(50))

# Pass/fail bands, pre-registered:
SLOPE_WIDTH_RATIO_BAND = 2.0          # Prediction A: slope*width pairwise ratio band
POSITIVE_CONTROL_MIN_SIGMA_W2 = 3     # Prediction A: >=3 of 4 sigma_w2 must show growing variance at width=32
GAP_SIGN_MIN_FRACTION = 10.0 / 12.0   # Prediction B: sign match fraction
GAP_MAGNITUDE_BAND = 3.0              # Prediction B: median ratio band
GAP_THEORY_FLOOR = 0.001              # Prediction B: exclude near-zero-gap-theory cells from ratio

# Prediction C (informational only)
DIAG_SIGMA_W2 = 2.2
DIAG_WIDTH = 32
DIAG_DEPTH = 362
DIAG_SEEDS = list(range(10))


def grad_norm(width, sigma_w2, depth, seed, x_batch, y_batch):
    torch.manual_seed(seed)
    m = DeepMLP(modular_arith.input_dim(P), width, depth, modular_arith.num_classes(P), sigma_w2, SIGMA_B2)
    logits = m(x_batch)
    loss = F.cross_entropy(logits, y_batch)
    m.zero_grad(set_to_none=True)
    loss.backward()
    return m.layers[0].weight.grad.norm().item()


def theil_sen_slope(xs, ys):
    slopes = []
    for (x1, y1), (x2, y2) in itertools.combinations(zip(xs, ys), 2):
        if x2 != x1:
            slopes.append((y2 - y1) / (x2 - x1))
    return float(np.median(slopes))


def measure_cell(width, sigma_w2, x_batch, y_batch):
    """Returns per-depth arrays needed for predictions A and B."""
    var_log_per_depth = []
    median_log_per_depth = []
    mean_val_per_depth = []
    for depth in DEPTH_GRID:
        vals = np.array([grad_norm(width, sigma_w2, depth, s, x_batch, y_batch) for s in SEEDS])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        log_vals = np.log(vals)
        var_log_per_depth.append(float(np.var(log_vals)))
        median_log_per_depth.append(float(np.median(log_vals)))
        mean_val_per_depth.append(float(np.mean(vals)))

    var_growth_slope, _ = np.polyfit(DEPTH_GRID, var_log_per_depth, 1)
    slope_median = theil_sen_slope(DEPTH_GRID, median_log_per_depth)
    log_mean_per_depth = [np.log(m) for m in mean_val_per_depth]
    slope_mean, _ = np.polyfit(DEPTH_GRID, log_mean_per_depth, 1)

    gap_emp = slope_mean - slope_median
    gap_theory = var_growth_slope / 2.0

    return {
        "var_growth_slope": float(var_growth_slope),
        "slope_median": float(slope_median),
        "slope_mean": float(slope_mean),
        "gap_emp": float(gap_emp),
        "gap_theory": float(gap_theory),
        "var_log_per_depth": var_log_per_depth,
        "median_log_per_depth": median_log_per_depth,
    }


def run_diagnostic_c(x_batch, y_batch):
    """Prediction C: per-layer E[phi'(pre-activation)^2], averaged over DIAG_SEEDS, at the
    single most anomalous cell -- compared against the fixed-point-predicted value."""
    q_star = mf.fixed_point_q(DIAG_SIGMA_W2, SIGMA_B2, mf.tanh_phi)
    theory_phi_prime_sq = mf.expectation(lambda z: mf.tanh_phi_prime(z) ** 2, q_star, n_quad=100)

    per_layer_sums = None
    n_layers = None
    for seed in DIAG_SEEDS:
        torch.manual_seed(seed)
        m = DeepMLP(modular_arith.input_dim(P), DIAG_WIDTH, DIAG_DEPTH, modular_arith.num_classes(P),
                    DIAG_SIGMA_W2, SIGMA_B2)
        pre_acts = []
        handles = []

        def make_hook(store):
            def hook(module, inp, out):
                store.append(out.detach())
            return hook

        for layer in m.layers:
            handles.append(layer.register_forward_hook(make_hook(pre_acts)))
        with torch.no_grad():
            m(x_batch)
        for h in handles:
            h.remove()

        phi_prime_sq_per_layer = [float((1.0 - torch.tanh(pa) ** 2).pow(2).mean().item()) for pa in pre_acts]
        if per_layer_sums is None:
            n_layers = len(phi_prime_sq_per_layer)
            per_layer_sums = [0.0] * n_layers
        for i, v in enumerate(phi_prime_sq_per_layer):
            per_layer_sums[i] += v

    per_layer_mean = [s / len(DIAG_SEEDS) for s in per_layer_sums]
    return {
        "theory_phi_prime_sq_at_qstar": float(theory_phi_prime_sq),
        "empirical_phi_prime_sq_per_layer": per_layer_mean,
        "n_layers": n_layers,
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, _, _ = modular_arith.make_dataset(P, seed=0, n_train=N_TRAIN, n_test=N_TEST)
    x_batch, y_batch = x_train[:BATCH], y_train[:BATCH]

    t0 = time.time()
    cells = {}  # (sigma_w2, width) -> measure_cell(...) dict
    for sigma_w2 in SIGMA_W2_GRID:
        for width in WIDTH_GRID:
            result = measure_cell(width, sigma_w2, x_batch, y_batch)
            cells[(sigma_w2, width)] = result
            print(f"sigma_w2={sigma_w2:.2f} width={width:4d}  "
                  f"var_growth_slope={result['var_growth_slope']:+.5f}  "
                  f"slope_median={result['slope_median']:+.5f}  slope_mean={result['slope_mean']:+.5f}  "
                  f"gap_emp={result['gap_emp']:+.5f}  gap_theory={result['gap_theory']:+.5f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    dump = {f"{sw2}|{w}": v for (sw2, w), v in cells.items()}
    (RUN_DIR / "results.json").write_text(json.dumps(dump, indent=2))

    # --- Prediction A: variance-growth slope scales as 1/width ---
    print("\n--- Prediction A: variance-growth slope * width should be ~constant across widths ---")
    a_ratio_pass = True
    for sigma_w2 in SIGMA_W2_GRID:
        scaled = {w: cells[(sigma_w2, w)]["var_growth_slope"] * w for w in WIDTH_GRID}
        vals = list(scaled.values())
        pairwise_ok = all(
            (1.0 / SLOPE_WIDTH_RATIO_BAND) <= (v1 / v2) <= SLOPE_WIDTH_RATIO_BAND
            for v1, v2 in itertools.combinations(vals, 2) if v2 != 0
        ) if all(v != 0 for v in vals) else False
        if not pairwise_ok:
            a_ratio_pass = False
        print(f"  sigma_w2={sigma_w2}: slope*width = {scaled} -> {'OK' if pairwise_ok else 'FAIL'}")

    n_positive_at_32 = sum(1 for sw2 in SIGMA_W2_GRID if cells[(sw2, 32)]["var_growth_slope"] > 0)
    positive_control_pass = n_positive_at_32 >= POSITIVE_CONTROL_MIN_SIGMA_W2
    print(f"  positive control: {n_positive_at_32}/4 sigma_w2 show growing variance at width=32 "
          f"(need >= {POSITIVE_CONTROL_MIN_SIGMA_W2}) -> {'OK' if positive_control_pass else 'FAIL'}")
    a_pass = a_ratio_pass and positive_control_pass
    print(f"Prediction A: {'PASS' if a_pass else 'FAIL'}")

    # --- Prediction B: mean/median slope gap matches log-normal identity ---
    print("\n--- Prediction B: gap_emp vs gap_theory (log-normal identity) ---")
    all_cells = [(sw2, w) for sw2 in SIGMA_W2_GRID for w in WIDTH_GRID]
    n_sign_match = sum(1 for k in all_cells if cells[k]["gap_emp"] > 0)
    sign_pass = (n_sign_match / len(all_cells)) >= GAP_SIGN_MIN_FRACTION
    print(f"  sign match: {n_sign_match}/{len(all_cells)} cells have gap_emp>0 "
          f"(need >= {GAP_SIGN_MIN_FRACTION:.2f} fraction) -> {'OK' if sign_pass else 'FAIL'}")

    ratios = []
    for k in all_cells:
        gt = cells[k]["gap_theory"]
        if gt > GAP_THEORY_FLOOR:
            ratios.append(cells[k]["gap_emp"] / gt)
    if ratios:
        median_ratio = float(np.median(ratios))
        magnitude_pass = (1.0 / GAP_MAGNITUDE_BAND) <= median_ratio <= GAP_MAGNITUDE_BAND
    else:
        median_ratio = float("nan")
        magnitude_pass = False
    print(f"  magnitude: median(gap_emp/gap_theory) over {len(ratios)} cells (gap_theory>{GAP_THEORY_FLOOR}) "
          f"= {median_ratio:.3f} (need in [1/{GAP_MAGNITUDE_BAND}, {GAP_MAGNITUDE_BAND}]) -> "
          f"{'OK' if magnitude_pass else 'FAIL'}")
    b_pass = sign_pass and magnitude_pass
    print(f"Prediction B: {'PASS' if b_pass else 'FAIL'}")

    print(f"\nOverall (A and B): {'PASS' if (a_pass and b_pass) else 'FAIL'}")

    # --- Prediction C: informational per-layer diagnostic ---
    print("\n--- Prediction C (informational, non-gating): per-layer E[phi'^2] diagnostic ---")
    diag = run_diagnostic_c(x_batch, y_batch)
    print(f"  theory E[phi'^2] at q*: {diag['theory_phi_prime_sq_at_qstar']:.5f}")
    print(f"  empirical per-layer (first 5, last 5 of {diag['n_layers']}):")
    layers = diag["empirical_phi_prime_sq_per_layer"]
    print(f"    first: {[round(v, 4) for v in layers[:5]]}")
    print(f"    last:  {[round(v, 4) for v in layers[-5:]]}")
    (RUN_DIR / "diagnostic_c.json").write_text(json.dumps(diag, indent=2))

    verdict = {
        "prediction_a_pass": a_pass,
        "prediction_b_pass": b_pass,
        "n_sign_match": n_sign_match,
        "median_gap_ratio": median_ratio,
    }
    (RUN_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print(f"Results written to {RUN_DIR}")


if __name__ == "__main__":
    main()
