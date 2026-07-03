"""Thread 6 pre-flight: muP coordinate check on the ACTUAL char-LM models and widths being
pinned for the real GPU run -- the same diagnostic thread 14
(docs/threads/14-mup-coordinate-check.md) already validated once for the modular-arith
MuPMLP, re-run here for the two NEW models (CharLMMLP, CharLMRecurrence) so the muP wiring is
checked before any GPU budget is spent. This is a wiring sanity check, not a re-registered
pre-registered thread; it uses thread 14's *corrected* reading of the output layer (post-
forward-multiplier slope ~= -1 at init is the intended, arithmetically-necessary consequence
of the base_width/width readout multiplier, NOT a defect; the pre-multiplier activation is
the quantity that should be width-flat) and a moderate read LR (thread 14's review found the
aggressive pilot LR induces a benign width-correlated transient in the hidden layers).

What "wired correctly" looks like here:
  - muP: token_embed/proj/hidden (MLP) and embed/recur/B (recurrence) mean|activation| slopes
    vs log(width) are ~flat (|slope| small); readout/output PRE-mult slope ~flat, POST-mult
    slope ~= -1 (expected). Loss stays width-stable.
  - SP (positive control): at least one layer type drifts with width (the check discriminates).
  - recurrence-specific: skew(theta) spectral radius is ~width-flat under muP's 1/sqrt(W)
    theta init (the theta-init resolution, see the char_lm_recurrence.py docstring and the
    thread 6 doc addendum), and would grow ~sqrt(W) under a fixed-std (SP) theta.

The recurrence arm's coordinate check is capped at width 1024 (16x) on CPU: a recurrence
step at w>=2048 is dominated by matrix_exp backward on a WxW matrix plus BPTT through the
32-step scan at large hidden (~minutes/step), beyond a sane pre-flight budget. The muP wiring
is width-identical, so a flat 64->1024 read validates it; the full grid up to 4096 is
exercised for real on the GPU run. The MLP arm runs the full pinned grid.

Run: python experiments/scripts/thread06_charlm_coord_check.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import math

import torch
import torch.nn.functional as F

from experiments.models.char_lm_mlp import CharLMMLP
from experiments.models.char_lm_recurrence import CharLMRecurrence
from experiments.models.linear_recurrence import _skew_symmetric
from experiments.tasks import char_lm

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread06_charlm_coord_check"

BASE_WIDTH = 64
MLP_WIDTHS = [64, 256, 1024, 2048, 4096]        # k = 1, 4, 16, 32, 64 (full pinned grid)
REC_WIDTHS = [64, 256, 1024]                    # capped at 16x on CPU: a recurrence coord-check
# step at w>=2048 is dominated by matrix_exp backward on a WxW matrix plus BPTT through the
# 32-step scan at large hidden (~minutes/step at 2048, times 10 steps times widths times
# params = beyond a sane pre-flight budget). The muP wiring is width-identical, so a flat
# 64->1024 (16x) read validates it; the full grid up to 4096 is exercised on the GPU run.
CHECKPOINT_STEPS = [0, 1, 2, 5, 10]
# Moderate Adam LR for the muP flatness read (thread 14's corrected lesson: the aggressive
# pilot LR induces a benign width-correlated transient). Override with env READ_LR to run an
# aggressive positive-control pass (e.g. READ_LR=0.3) confirming the check discriminates.
READ_LR = float(os.environ.get("READ_LR", "0.01"))
CONTEXT_LEN = 32
BATCH = 128
EPS = 0.05
EVAL_SEED = 999
SLOPE_TOL_MUP = 0.15    # thread 14's bar (applies to non-output layers / output pre-mult)


def mlp_scales(model, x_eval):
    acts, handles = {}, []

    def hook(name, kind):
        def _h(mod, inp, out):
            if kind == "relu":
                acts[name] = torch.relu(out).detach().abs().mean().item()
            elif kind == "embed":
                acts[name] = out.detach().abs().mean().item()
            else:  # output: record pre-mult and post-mult
                pre = out.detach().abs().mean().item()
                acts["output_pre_mult"] = pre
                acts["output_post_mult"] = pre * model._output_mult
        return _h

    handles.append(model.token_embed.register_forward_hook(hook("token_embed", "embed")))
    handles.append(model.proj.register_forward_hook(hook("proj", "relu")))
    for i, layer in enumerate(model.hidden_layers):
        handles.append(layer.register_forward_hook(hook(f"hidden_{i}", "relu")))
    handles.append(model.output_layer.register_forward_hook(hook("output", "output")))
    with torch.no_grad():
        model(x_eval)
    for h in handles:
        h.remove()
    return acts


def rec_scales(model, x_eval):
    acts, handles = {}, []

    def embed_hook(mod, inp, out):
        acts["embed"] = out.detach().abs().mean().item()

    def recur_hook(mod, inp, out):        # out: (batch, seq, width) -> final position
        acts["recur_final"] = out[:, -1, :].detach().abs().mean().item()

    def readout_hook(mod, inp, out):
        pre = out.detach().abs().mean().item()
        acts["readout_pre_mult"] = pre
        acts["readout_post_mult"] = pre * model._output_mult

    handles.append(model.embed.register_forward_hook(embed_hook))
    handles.append(model.recur.register_forward_hook(recur_hook))
    handles.append(model.readout.register_forward_hook(readout_hook))
    with torch.no_grad():
        model(x_eval)
    for h in handles:
        h.remove()
    return acts


def run_one(arm, param, width, train_data, x_eval, y_eval):
    torch.manual_seed(width)  # same init seed across params at a width, distinct per width
    if arm == "mlp":
        model = CharLMMLP(char_lm.vocab_size(), width, CONTEXT_LEN, base_width=BASE_WIDTH,
                          parametrization=param)
        scales_fn = mlp_scales
    else:
        model = CharLMRecurrence(char_lm.vocab_size(), width, EPS, BASE_WIDTH, param)
        scales_fn = rec_scales
    optimizer = torch.optim.Adam(model.param_groups(READ_LR))
    g = torch.Generator().manual_seed(1000 + width)

    extra = {}
    if arm == "recurrence":
        with torch.no_grad():
            sv = torch.linalg.svdvals(_skew_symmetric(model.recur.theta))
        extra["skew_spectral_radius"] = sv[0].item()

    checkpoints = {0: scales_fn(model, x_eval)}
    losses, theta_grad = {}, None
    step_count = 0
    for target in CHECKPOINT_STEPS[1:]:
        while step_count < target:
            xb, yb = char_lm.make_batch(train_data, CONTEXT_LEN, BATCH, g)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            if arm == "recurrence" and step_count == 0:
                theta_grad = model.recur.theta.grad.norm().item()
            optimizer.step()
            step_count += 1
            losses[step_count] = loss.item()
        checkpoints[target] = scales_fn(model, x_eval)
    if theta_grad is not None:
        extra["theta_grad_norm_step0"] = theta_grad
    return checkpoints, losses, extra


def log_log_slope(widths, ys):
    log_w = [math.log(w) for w in widths]
    log_y = [math.log(max(y, 1e-12)) for y in ys]
    n = len(widths)
    mx, my = sum(log_w) / n, sum(log_y) / n
    num = sum((x - mx) * (y - my) for x, y in zip(log_w, log_y))
    den = sum((x - mx) ** 2 for x in log_w)
    return num / den if den > 0 else float("nan")


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    train_data, test_data = char_lm.make_split()
    g_eval = torch.Generator().manual_seed(EVAL_SEED)
    x_eval, y_eval = char_lm.make_batch(test_data, CONTEXT_LEN, BATCH, g_eval)

    layer_names = {
        "mlp": ["token_embed", "proj", "hidden_0", "hidden_1", "output_pre_mult", "output_post_mult"],
        "recurrence": ["embed", "recur_final", "readout_pre_mult", "readout_post_mult"],
    }
    widths_for = {"mlp": MLP_WIDTHS, "recurrence": REC_WIDTHS}

    all_results, verdict = {}, {}
    t_start = time.perf_counter()
    for arm in ("mlp", "recurrence"):
        widths = widths_for[arm]
        for param in ("mup", "sp"):
            for width in widths:
                t0 = time.perf_counter()
                checkpoints, losses, extra = run_one(arm, param, width, train_data, x_eval, y_eval)
                all_results[(arm, param, width)] = {"checkpoints": checkpoints, "losses": losses, "extra": extra}
                print(f"{arm}/{param} w={width}: {time.perf_counter() - t0:.1f}s "
                      f"loss@10={losses.get(10, float('nan')):.3f} extra={extra}", flush=True)

        print(f"\n=== {arm}: log-log slope of mean|activation| vs width "
              f"(muP flat |slope|<{SLOPE_TOL_MUP} for non-output & output_pre_mult; "
              f"output_post_mult ~= -1 expected) ===")
        for param in ("mup", "sp"):
            print(f"\n{param}:")
            for layer in layer_names[arm]:
                slopes = []
                row = []
                for step in CHECKPOINT_STEPS:
                    ys = [all_results[(arm, param, w)]["checkpoints"][step][layer] for w in widths]
                    s = log_log_slope(widths, ys)
                    slopes.append(s)
                    row.append(f"t={step}:{s:+.3f}")
                    verdict[f"{arm}|{param}|{layer}|t{step}"] = s
                print(f"  {layer:18s}: " + " ".join(row))
        if arm == "recurrence":
            print("\n  skew(theta) spectral radius vs width (muP 1/sqrt(W) init -> flat; "
                  "SP fixed-std -> grows ~sqrt(W)):")
            for param in ("mup", "sp"):
                srs = [all_results[(arm, param, w)]["extra"]["skew_spectral_radius"] for w in widths]
                sl = log_log_slope(widths, srs)
                print(f"    {param}: " + " ".join(f"w{w}:{sr:.3f}" for w, sr in zip(widths, srs))
                      + f"  (log-log slope {sl:+.3f})")
                verdict[f"recurrence|{param}|skew_spectral_radius_slope"] = sl

    print("\n--- loss@10 vs width (informational 'wider is not worse' check) ---")
    for arm in ("mlp", "recurrence"):
        for param in ("mup", "sp"):
            row = [f"w{w}:{all_results[(arm, param, w)]['losses'].get(10, float('nan')):.3f}"
                   for w in widths_for[arm]]
            print(f"  {arm}/{param}: " + " ".join(row))

    print(f"\nTotal: {time.perf_counter() - t_start:.1f}s")
    dump = {f"{a}_{p}_{w}": v for (a, p, w), v in all_results.items()}
    (RUN_DIR / "raw_results.json").write_text(json.dumps(dump, indent=2))
    (RUN_DIR / "slopes.json").write_text(json.dumps(verdict, indent=2))
    print(f"Written to {RUN_DIR}")


if __name__ == "__main__":
    main()
