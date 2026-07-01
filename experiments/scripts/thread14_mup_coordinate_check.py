"""Thread 14: muP coordinate check -- per-layer-type activation scale vs. width, at init
and after a handful of Adam steps at one fixed aggressive LR. No convergence sweep; this is
a forward/backward-pass structural diagnostic, not a stochastic-outcome claim. See
docs/threads/14-mup-coordinate-check.md for the pre-registered prediction and pass/fail
bands.

base_lr=0.3 was picked via a quick pilot (not committed as a separate script -- see the
thread doc's "Minimal experiment" section and the commit message for the pilot numbers):
at width=64 both arms show a destabilizing-but-recovering transient (loss spikes ~20x then
settles back near its start), and at width=4096 SP diverges catastrophically (loss to
~1.2e7) while muP stays essentially flat -- exactly the width-dependent blowup this check
is designed to surface, and comfortably far from both a no-signal LR and a NaN-producing
one at the smallest width.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import math

import torch
import torch.nn.functional as F

from experiments.models.mlp import MuPMLP
from experiments.tasks import modular_arith

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread14_coord_check"

P = 41
DEPTH = 4  # 2 hidden layers, matches thread 6's RunConfig default
BATCH = 128
BASE_WIDTH = 64
WIDTH_GRID = [64, 128, 256, 512, 1024, 2048, 4096]  # k = 1,2,4,8,16,32,64
CHECKPOINT_STEPS = [0, 1, 2, 5, 10]
BASE_LR = 0.3
N_TRAIN, N_TEST = 1200, 400
EVAL_BATCH_SEED = 999  # fixed across every (parametrization, width, step) for comparability

SLOPE_TOL_MUP = 0.15
SLOPE_TOL_SP_FAIL = 0.3


def measure_layer_scales(model: MuPMLP, x_eval: torch.Tensor) -> dict:
    acts = {}
    handles = []

    def make_hook(name, kind):
        def hook(module, inp, out):
            val = torch.relu(out) if kind == "relu" else out * model._output_mult
            acts[name] = val.detach().abs().mean().item()
        return hook

    handles.append(model.input_layer.register_forward_hook(make_hook("input_layer", "relu")))
    for i, layer in enumerate(model.hidden_layers):
        handles.append(layer.register_forward_hook(make_hook(f"hidden_{i}", "relu")))
    handles.append(model.output_layer.register_forward_hook(make_hook("output_layer", "output")))

    with torch.no_grad():
        model(x_eval)
    for h in handles:
        h.remove()
    return acts


def run_one(parametrization: str, width: int, x_train, y_train, x_eval):
    torch.manual_seed(width)  # same seed across parametrizations at a given width, distinct per width
    model = MuPMLP(
        input_dim=modular_arith.input_dim(P), width=width, num_classes=modular_arith.num_classes(P),
        depth=DEPTH, base_width=BASE_WIDTH, parametrization=parametrization,
    )
    optimizer = torch.optim.Adam(model.param_groups(BASE_LR))
    g = torch.Generator().manual_seed(1000 + width)

    checkpoints = {}
    losses = {}
    step_count = 0
    checkpoints[0] = measure_layer_scales(model, x_eval)
    for target in CHECKPOINT_STEPS[1:]:
        while step_count < target:
            idx = torch.randint(0, x_train.shape[0], (BATCH,), generator=g)
            xb, yb = x_train[idx], y_train[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_count += 1
            losses[step_count] = loss.item()
        checkpoints[target] = measure_layer_scales(model, x_eval)
    return checkpoints, losses


def log_log_slope(widths: list, ys: list) -> float:
    log_w = [math.log(w) for w in widths]
    log_y = [math.log(max(y, 1e-12)) for y in ys]
    n = len(widths)
    mean_x = sum(log_w) / n
    mean_y = sum(log_y) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_w, log_y))
    den = sum((x - mean_x) ** 2 for x in log_w)
    return num / den if den > 0 else float("nan")


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = modular_arith.make_dataset(P, seed=0, n_train=N_TRAIN, n_test=N_TEST)
    g_eval = torch.Generator().manual_seed(EVAL_BATCH_SEED)
    eval_idx = torch.randint(0, x_test.shape[0], (BATCH,), generator=g_eval)
    x_eval = x_test[eval_idx]

    all_results = {}  # (param, width) -> {"checkpoints": ..., "losses": ...}
    t_start = time.perf_counter()
    for parametrization in ("sp", "mup"):
        for width in WIDTH_GRID:
            t0 = time.perf_counter()
            checkpoints, losses = run_one(parametrization, width, x_train, y_train, x_eval)
            dt = time.perf_counter() - t0
            all_results[(parametrization, width)] = {"checkpoints": checkpoints, "losses": losses}
            print(f"{parametrization} width={width}: time={dt:.2f}s "
                  f"loss@10={losses.get(10, 'n/a')}", flush=True)
    print(f"\nTotal elapsed: {time.perf_counter() - t_start:.1f}s")

    dump = {f"{p}_{w}": v for (p, w), v in all_results.items()}
    (RUN_DIR / "raw_results.json").write_text(json.dumps(dump, indent=2))

    layer_names = ["input_layer", "hidden_0", "hidden_1", "output_layer"]
    slopes = {}  # (param, layer, step) -> slope
    for parametrization in ("sp", "mup"):
        for layer in layer_names:
            for step in CHECKPOINT_STEPS:
                ys = [all_results[(parametrization, w)]["checkpoints"][step][layer] for w in WIDTH_GRID]
                slopes[(parametrization, layer, step)] = log_log_slope(WIDTH_GRID, ys)

    print("\n--- log-log slope of mean(|activation|) vs width (pre-registered bar: "
          f"muP pass |slope|<{SLOPE_TOL_MUP}, SP positive-control fail |slope|>={SLOPE_TOL_SP_FAIL} "
          "for >=1 layer at t>=1) ---")
    mup_pass = True
    mup_worst = (None, None, 0.0)
    sp_fails_as_expected = False
    sp_worst_fail = (None, None, 0.0)
    for parametrization in ("sp", "mup"):
        print(f"\n{parametrization}:")
        for layer in layer_names:
            row = []
            for step in CHECKPOINT_STEPS:
                s = slopes[(parametrization, layer, step)]
                row.append(f"t={step}:{s:+.3f}")
                if parametrization == "mup":
                    if abs(s) >= SLOPE_TOL_MUP and abs(s) > abs(mup_worst[2]):
                        mup_worst = (layer, step, s)
                    if abs(s) >= SLOPE_TOL_MUP:
                        mup_pass = False
                elif parametrization == "sp" and step >= 1:
                    if abs(s) >= SLOPE_TOL_SP_FAIL:
                        sp_fails_as_expected = True
                        if abs(s) > abs(sp_worst_fail[2]):
                            sp_worst_fail = (layer, step, s)
            print(f"  {layer}: " + " ".join(row))

    print(f"\nmuP passes flatness bar: {mup_pass} (worst offender: {mup_worst})")
    print(f"SP fails as expected positive control: {sp_fails_as_expected} (worst: {sp_worst_fail})")

    print("\n--- 'wider is always better' (loss@10 vs width, informational only) ---")
    for parametrization in ("sp", "mup"):
        row = [f"w={w}:{all_results[(parametrization, w)]['losses'].get(10, float('nan')):.3f}"
               for w in WIDTH_GRID]
        print(f"  {parametrization}: " + " ".join(row))

    verdict = {
        "mup_passes_flatness_bar": mup_pass,
        "mup_worst_offender": mup_worst,
        "sp_fails_as_expected_positive_control": sp_fails_as_expected,
        "sp_worst_fail": sp_worst_fail,
        "slopes": {f"{p}|{l}|{s}": v for (p, l, s), v in slopes.items()},
    }
    (RUN_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"\nRaw results and verdict written to {RUN_DIR}")


if __name__ == "__main__":
    main()
