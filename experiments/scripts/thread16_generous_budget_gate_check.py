"""Pre-registered run for thread 16 (docs/threads/16-generous-budget-gate-check.md, idea
I4): does a 6x more generous budget (Arm A, 12000 online fresh-batch steps vs. thread 11's
2000) or pure memorization capacity (Arm B, repeated-batch training on a small fixed
128-example pool) reveal a gradient signal the gate-family sub-line's 2000-step online
protocol missed?

Reuses DualGateRecallModel and the recall task unchanged -- no architecture change, per the
pre-registration. Single LR (thread 11's best, 3e-4), 5 seeds, matching threads 9/10/11.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import torch
import torch.nn.functional as F

from experiments.models.dual_gate_recurrence import DualGateRecallModel
from experiments.tasks import recall

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread16_generous_budget_gate_check"

VOCAB = 512
HIDDEN = 64
EPS = 0.1
N_PAIRS = 8
BATCH = 32
LR = 3e-4  # thread 11's best-of-grid choice, reused directly
STEPS = 12000
CHECKPOINT_EVERY = 1000
SEEDS = list(range(5))
EVAL_BATCH = 512
POOL_SIZE = 128

TARGET_ACC_A = 0.30
GATE_GROWTH_MIN_A = 2.0
TARGET_ACC_B = 0.90
FAIL_ACC_B = 0.10

DIAG_TOKENS_SEED = 777_777  # fixed diagnostic batch for gate-value tracking, same across seeds/arms


def mean_write_gate(model, tokens):
    with torch.no_grad():
        x = model.embed(tokens)
        return torch.sigmoid(model.recur.write_gate(x)).mean().item()


def run_arm_a(seed, diag_tokens):
    """Fresh-random-batch-per-step, 12000 steps -- generous online budget."""
    torch.manual_seed(seed)
    model = DualGateRecallModel(VOCAB, HIDDEN, EPS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gate_checkpoints = {0: mean_write_gate(model, diag_tokens)}
    for step in range(1, STEPS + 1):
        tokens, targets = recall.make_batch(N_PAIRS, VOCAB, BATCH, seed=seed * 1_000_000 + step)
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % CHECKPOINT_EVERY == 0:
            gate_checkpoints[step] = mean_write_gate(model, diag_tokens)

    eval_tokens, eval_targets = recall.make_batch(N_PAIRS, VOCAB, EVAL_BATCH, seed=seed + 999_999)
    with torch.no_grad():
        acc = (model(eval_tokens).argmax(-1) == eval_targets).float().mean().item()
    return {"eval_acc": acc, "final_loss": loss.item(), "gate_checkpoints": gate_checkpoints}


def run_arm_b(seed, diag_tokens):
    """Repeated sampling from a small fixed 128-example pool, 12000 steps -- pure
    memorization/overfitting capacity, isolated from generalization."""
    pool_tokens, pool_targets = recall.make_batch(N_PAIRS, VOCAB, POOL_SIZE, seed=0)  # same pool every seed
    torch.manual_seed(seed)
    model = DualGateRecallModel(VOCAB, HIDDEN, EPS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed * 1_000_000 + 1)
    gate_checkpoints = {0: mean_write_gate(model, diag_tokens)}
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,), generator=g)
        tokens, targets = pool_tokens[idx], pool_targets[idx]
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % CHECKPOINT_EVERY == 0:
            gate_checkpoints[step] = mean_write_gate(model, diag_tokens)

    with torch.no_grad():
        train_acc = (model(pool_tokens).argmax(-1) == pool_targets).float().mean().item()
    return {"train_acc": train_acc, "final_loss": loss.item(), "gate_checkpoints": gate_checkpoints}


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    diag_tokens, _ = recall.make_batch(N_PAIRS, VOCAB, 64, seed=DIAG_TOKENS_SEED)

    t0 = time.time()
    results_a, results_b = [], []
    for seed in SEEDS:
        r = run_arm_a(seed, diag_tokens)
        results_a.append({"seed": seed, **r})
        print(f"[Arm A] seed={seed} eval_acc={r['eval_acc']:.4f} "
              f"gate_final={r['gate_checkpoints'][STEPS]:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
    for seed in SEEDS:
        r = run_arm_b(seed, diag_tokens)
        results_b.append({"seed": seed, **r})
        print(f"[Arm B] seed={seed} train_acc={r['train_acc']:.4f} "
              f"gate_final={r['gate_checkpoints'][STEPS]:.4f} elapsed={time.time()-t0:.0f}s", flush=True)

    (RUN_DIR / "results.json").write_text(json.dumps({"arm_a": results_a, "arm_b": results_b}, indent=2))

    mean_acc_a = sum(r["eval_acc"] for r in results_a) / len(results_a)
    gate0 = results_a[0]["gate_checkpoints"][0]
    mean_gate_final_a = sum(r["gate_checkpoints"][STEPS] for r in results_a) / len(results_a)
    gate_growth_a = mean_gate_final_a / gate0 if gate0 > 0 else float("inf")
    a_pass = mean_acc_a >= TARGET_ACC_A or gate_growth_a >= GATE_GROWTH_MIN_A

    mean_acc_b = sum(r["train_acc"] for r in results_b) / len(results_b)
    b_pass = mean_acc_b >= TARGET_ACC_B
    b_fail_confirmed = mean_acc_b < FAIL_ACC_B

    print("\n--- Prediction verdict (pre-registered in docs/threads/16-generous-budget-gate-check.md) ---")
    print(f"Arm A: mean eval_acc={mean_acc_a:.4f} (need >= {TARGET_ACC_A}) OR "
          f"gate growth={gate_growth_a:.2f}x (need >= {GATE_GROWTH_MIN_A}x, gate0={gate0:.4f}, "
          f"gate_final={mean_gate_final_a:.4f}) -> {'PASS' if a_pass else 'FAIL'}")
    print(f"Arm B: mean train_acc (fixed 128-pool)={mean_acc_b:.4f} "
          f"(pass >= {TARGET_ACC_B}, fail-confirmed < {FAIL_ACC_B}) -> "
          f"{'PASS' if b_pass else ('FAIL (confirmed)' if b_fail_confirmed else 'FAIL (ambiguous)')}")

    if a_pass:
        verdict = "Arm A passes: gate-family closure was likely a budget artifact -- sub-line needs reopening."
    elif b_pass:
        verdict = ("Arm B passes, Arm A fails: write-gate pathway CAN move under optimization "
                   "(memorization works) but fresh-batch generalization remains unsolved at this budget.")
    else:
        verdict = ("Both fail: 'no discoverable gradient signal' is now earned with actual evidence "
                   "(6x budget + pure memorization both fail), not merely asserted.")
    print(f"\nOverall: {verdict}")

    (RUN_DIR / "verdict.json").write_text(json.dumps({
        "mean_acc_a": mean_acc_a, "gate_growth_a": gate_growth_a, "a_pass": a_pass,
        "mean_acc_b": mean_acc_b, "b_pass": b_pass, "b_fail_confirmed": b_fail_confirmed,
        "verdict": verdict,
    }, indent=2))
    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
