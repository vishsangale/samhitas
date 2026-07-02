"""Thread 17 (docs/threads/17-recall-mechanism-ladder.md), arm (b): short causal conv +
single gate. Prediction A: does adding a depthwise causal conv (k in {2,3,4}) before
thread 9's existing gated recurrence make n_pairs=8 associative recall solvable? Prediction
B (only run if A passes): does the trained model's gradient-flow healthy/unhealthy boundary
stay within 2x of ungated orthogonal's closed-form boundary at eps=0.1?

Protocol matches threads 9/10/11/16 exactly for direct comparability: vocab=512, hidden=64,
eps=0.1, n_pairs=8 (seq_len=17), 2000 fresh-random-batch Adam steps, LR grid
{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}, 5 seeds.
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import torch
import torch.nn.functional as F

from experiments.harness.gradient_flow import gradient_norm_ratio
from experiments.models.shortconv_gated_recurrence import ShortConvGatedRecallModel
from experiments.tasks import recall

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread17_arm_b_shortconv"

VOCAB = 512
HIDDEN = 64
EPS = 0.1
N_PAIRS = 8
BATCH = 32
TRAIN_STEPS = 2000
LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
KERNEL_SIZES = [2, 3, 4]
SEEDS = list(range(5))
EVAL_BATCH = 512
TARGET_ACC = 0.30

# Already-collected controls (not re-run): thread 9 direct (0.032), thread 11 dual-gate
# (0.032), thread 16 12k-step (0.0316), ungated orthogonal control (0.020).
CONTROL_ACCS = {"thread9_gated": 0.032, "thread11_dual_gate": 0.032,
                 "thread16_12k": 0.0316, "orthogonal_ungated": 0.020}

# Prediction B setup
N_PAIRS_LIST = [8, 16, 24, 32, 48, 64, 96, 128, 192]
B_TOLERANCE = 2.0


def predicted_orthogonal_boundary(eps: float) -> float:
    """L* = 1 + ln(0.1)/ln(1-eps), the closed-form healthy/unhealthy crossing (seq_len)
    for ungated orthogonal (thread 1's exact result, reused here as the yardstick)."""
    return 1 + math.log(0.1) / math.log(1 - eps)


def train_and_eval(kernel_size, lr, seed):
    torch.manual_seed(seed)
    model = ShortConvGatedRecallModel(VOCAB, HIDDEN, EPS, kernel_size=kernel_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(TRAIN_STEPS):
        tokens, targets = recall.make_batch(N_PAIRS, VOCAB, BATCH, seed=seed * 100_000 + step)
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    eval_tokens, eval_targets = recall.make_batch(N_PAIRS, VOCAB, EVAL_BATCH, seed=seed + 999_999)
    with torch.no_grad():
        acc = (model(eval_tokens).argmax(-1) == eval_targets).float().mean().item()
    return model, acc, loss.item()


def find_boundary(model, seed):
    """First n_pairs in N_PAIRS_LIST where gradient_norm_ratio's failure_mode != 'healthy',
    on a fresh recall batch. Returns the corresponding seq_len, or None if healthy across
    the whole tested grid (boundary is at least as far out as the grid's end)."""
    for n_pairs in N_PAIRS_LIST:
        seq_len = recall.seq_len_for(n_pairs)
        tokens, targets = recall.make_batch(n_pairs, VOCAB, BATCH, seed=seed + 555_555)
        r = gradient_norm_ratio(model, tokens, targets)
        if not r["healthy"]:
            return seq_len
    return None


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(KERNEL_SIZES) * len(LR_GRID) * len(SEEDS)
    done = 0
    t0 = time.time()
    trained_models = {}  # (kernel_size, lr, seed) -> model, kept only for the best kernel/lr combo post-hoc
    for kernel_size in KERNEL_SIZES:
        for lr in LR_GRID:
            for seed in SEEDS:
                model, acc, final_loss = train_and_eval(kernel_size, lr, seed)
                results.append({"kernel_size": kernel_size, "lr": lr, "seed": seed,
                                 "eval_acc": acc, "final_loss": final_loss})
                trained_models[(kernel_size, lr, seed)] = model
                done += 1
                elapsed = time.time() - t0
                print(f"[{done}/{total}] k={kernel_size} lr={lr} seed={seed} "
                      f"acc={acc:.4f} loss={final_loss:.3f} elapsed={elapsed:.0f}s", flush=True)

    (RUN_DIR / "results.json").write_text(json.dumps(
        [{k: v for k, v in r.items()} for r in results], indent=2))

    print("\n--- Prediction A: best-of-grid mean accuracy per kernel size ---")
    best_overall = {"mean_acc": -1.0}
    for kernel_size in KERNEL_SIZES:
        by_lr = {}
        for r in results:
            if r["kernel_size"] == kernel_size:
                by_lr.setdefault(r["lr"], []).append(r["eval_acc"])
        best_lr = max(by_lr, key=lambda lr: sum(by_lr[lr]) / len(by_lr[lr]))
        mean_acc = sum(by_lr[best_lr]) / len(by_lr[best_lr])
        print(f"kernel_size={kernel_size}: best_lr={best_lr} mean_acc={mean_acc:.4f}")
        for lr, accs in sorted(by_lr.items()):
            print(f"  lr={lr}: mean={sum(accs)/len(accs):.4f} per_seed={[round(a,4) for a in accs]}")
        if mean_acc > best_overall["mean_acc"]:
            best_overall = {"kernel_size": kernel_size, "lr": best_lr, "mean_acc": mean_acc}

    print(f"\nBest overall: kernel_size={best_overall['kernel_size']} lr={best_overall['lr']} "
          f"mean_acc={best_overall['mean_acc']:.4f} (target={TARGET_ACC})")
    print(f"Controls (not re-run): {CONTROL_ACCS}")
    max_control = max(CONTROL_ACCS.values())
    a_pass = best_overall["mean_acc"] >= TARGET_ACC and best_overall["mean_acc"] > max_control * 1.5
    print(f"Prediction A: {'PASS' if a_pass else 'FAIL'} "
          f"(need >= {TARGET_ACC} and clearly above controls, max control = {max_control})")

    if not a_pass:
        print("\nPrediction A failed -- prediction B not run (per pre-registration, only "
              "run for arms that pass A).")
        return

    print("\n--- Prediction B: gradient-flow boundary for the best (kernel_size, lr) arm ---")
    l_star_theory = predicted_orthogonal_boundary(EPS)
    print(f"ungated orthogonal theoretical boundary (seq_len): {l_star_theory:.2f}")
    boundaries = []
    for seed in SEEDS:
        model = trained_models[(best_overall["kernel_size"], best_overall["lr"], seed)]
        boundary = find_boundary(model, seed)
        boundaries.append(boundary)
        print(f"  seed={seed}: boundary seq_len={boundary}")

    resolved = [b for b in boundaries if b is not None]
    if resolved:
        mean_boundary = sum(resolved) / len(resolved)
        ratio = mean_boundary / l_star_theory
        b_pass = (1.0 / B_TOLERANCE) <= ratio <= B_TOLERANCE
        print(f"\nmean boundary (seq_len, {len(resolved)}/{len(boundaries)} seeds resolved) "
              f"= {mean_boundary:.2f}, ratio to theory = {ratio:.3f} "
              f"(need in [1/{B_TOLERANCE}, {B_TOLERANCE}]) -> {'PASS' if b_pass else 'FAIL'}")
    else:
        print("\nNo seed's boundary resolved within the tested grid -- inconclusive.")
        b_pass = None

    (RUN_DIR / "verdict.json").write_text(json.dumps({
        "a_pass": a_pass, "best_overall": best_overall, "boundaries": boundaries,
        "l_star_theory": l_star_theory, "b_pass": b_pass,
    }, indent=2))
    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
