"""Pre-registered run for thread 18 (docs/threads/18-recall-protocol-validation.md):
does TinyAttentionModel -- the literature's known-sufficient reference architecture for
associative recall -- solve the exact online recall protocol threads 9-17 all failed on?

Three arms (design exactly as pre-registered, including the post-design-review
corrections):
  Arm 1 (primary):    n_pairs=8,  2000 steps, 6-point LR grid, 5 seeds
  Arm 2 (floor check): n_pairs=2,  2000 steps, 6-point LR grid, 5 seeds
  Arm 3 (extended):    n_pairs=8, 12000 steps, Arm 1's best LR + the adjacent grid point
                        (next-lower, or next-higher if the best LR is already the grid
                        minimum), 5 seeds each

Near-boundary escalation rule (design-review finding D1): if an arm's best-config 5-seed
mean falls within one seed-SEM of the 0.30 pass bar, 5 more seeds are run at that same
config before a verdict row is assigned -- the null-regime noise-floor pilot doesn't
bound variance there, and induction-head formation is a training-time phase change
(Olsson et al. 2022) that could land unpredictably close to the bar.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import statistics

import torch
import torch.nn.functional as F

from experiments.models.tiny_attention import TinyAttentionModel
from experiments.tasks import recall

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread18_recall_protocol_validation"

VOCAB = 512
HIDDEN = 64
BATCH = 32
EVAL_BATCH = 512
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
SEEDS = list(range(5))
ESCALATION_SEEDS = list(range(5, 10))
TARGET_ACC = 0.30
ARM1_STEPS = 2000
ARM2_N_PAIRS = 2
ARM3_STEPS = 12000


def train_and_eval(n_pairs, steps, lr, seed):
    """Implementation contract per the pre-registration: fresh model per call, seq_len
    matched to n_pairs, seed formulas reused unchanged from the noise-floor pilot and
    every gate-family thread before it."""
    seq_len = recall.seq_len_for(n_pairs)
    torch.manual_seed(seed)
    model = TinyAttentionModel(VOCAB, HIDDEN, seq_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_val = None
    for step in range(steps):
        tokens, targets = recall.make_batch(n_pairs, VOCAB, BATCH, seed=seed * 100_000 + step)
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = loss.item()
    eval_tokens, eval_targets = recall.make_batch(n_pairs, VOCAB, EVAL_BATCH, seed=seed + 999_999)
    with torch.no_grad():
        acc = (model(eval_tokens).argmax(-1) == eval_targets).float().mean().item()
    return acc, loss_val


def run_grid(n_pairs, steps, lrs, seeds, label, t0):
    by_lr = {}
    for lr in lrs:
        accs = []
        for seed in seeds:
            acc, loss_val = train_and_eval(n_pairs, steps, lr, seed)
            accs.append(acc)
            print(f"[{label}] n_pairs={n_pairs} lr={lr:.0e} seed={seed} steps={steps} "
                  f"acc={acc:.4f} loss={loss_val:.3f} elapsed={time.time()-t0:.0f}s", flush=True)
        by_lr[lr] = accs
    return by_lr


def best_lr_of(by_lr):
    return max(by_lr, key=lambda lr: statistics.mean(by_lr[lr]))


def mean_and_sem(accs):
    mean = statistics.mean(accs)
    sem = (statistics.stdev(accs) / (len(accs) ** 0.5)) if len(accs) > 1 else float("inf")
    return mean, sem


def resolve_arm(n_pairs, steps, lr, seeds_run, label, t0):
    """Near-boundary escalation rule: if the current mean is within one seed-SEM of the
    0.30 bar, run 5 more seeds at this same config before finalizing the arm's result."""
    mean, sem = mean_and_sem(seeds_run)
    escalated = False
    if abs(mean - TARGET_ACC) < sem:
        escalated = True
        print(f"[{label}] near-boundary (mean={mean:.4f} sem={sem:.4f} vs bar={TARGET_ACC}) "
              f"-- escalating to 10 seeds", flush=True)
        for seed in ESCALATION_SEEDS:
            acc, loss_val = train_and_eval(n_pairs, steps, lr, seed)
            seeds_run.append(acc)
            print(f"[{label}] (escalation) n_pairs={n_pairs} lr={lr:.0e} seed={seed} "
                  f"steps={steps} acc={acc:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
        mean, sem = mean_and_sem(seeds_run)
    return mean, sem, escalated, seeds_run


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== Arm 1: primary (n_pairs=8, 2000 steps) ===", flush=True)
    arm1_by_lr = run_grid(8, ARM1_STEPS, LR_GRID, SEEDS, "Arm1", t0)
    arm1_best_lr = best_lr_of(arm1_by_lr)
    r1_mean, r1_sem, r1_escalated, r1_seeds = resolve_arm(
        8, ARM1_STEPS, arm1_best_lr, arm1_by_lr[arm1_best_lr], "Arm1", t0
    )
    print(f"Arm 1 result: best_lr={arm1_best_lr} R1={r1_mean:.4f} (sem={r1_sem:.4f}, "
          f"escalated={r1_escalated}, n={len(r1_seeds)})", flush=True)

    print("\n=== Arm 2: floor check (n_pairs=2, 2000 steps) ===", flush=True)
    arm2_by_lr = run_grid(ARM2_N_PAIRS, ARM1_STEPS, LR_GRID, SEEDS, "Arm2", t0)
    arm2_best_lr = best_lr_of(arm2_by_lr)
    r2_mean, r2_sem, r2_escalated, r2_seeds = resolve_arm(
        ARM2_N_PAIRS, ARM1_STEPS, arm2_best_lr, arm2_by_lr[arm2_best_lr], "Arm2", t0
    )
    print(f"Arm 2 result: best_lr={arm2_best_lr} R2={r2_mean:.4f} (sem={r2_sem:.4f}, "
          f"escalated={r2_escalated}, n={len(r2_seeds)})", flush=True)

    print("\n=== Arm 3: extended budget (n_pairs=8, 12000 steps) ===", flush=True)
    sorted_lrs = sorted(LR_GRID)
    best_idx = sorted_lrs.index(arm1_best_lr)
    second_idx = best_idx - 1 if best_idx > 0 else best_idx + 1
    arm3_lrs = [arm1_best_lr, sorted_lrs[second_idx]]
    print(f"Arm 3 LRs (Arm 1's best + adjacent grid point): {arm3_lrs}", flush=True)
    arm3_by_lr = run_grid(8, ARM3_STEPS, arm3_lrs, SEEDS, "Arm3", t0)
    arm3_best_lr = best_lr_of(arm3_by_lr)
    r3_mean, r3_sem, r3_escalated, r3_seeds = resolve_arm(
        8, ARM3_STEPS, arm3_best_lr, arm3_by_lr[arm3_best_lr], "Arm3", t0
    )
    print(f"Arm 3 result: best_lr={arm3_best_lr} R3={r3_mean:.4f} (sem={r3_sem:.4f}, "
          f"escalated={r3_escalated}, n={len(r3_seeds)})", flush=True)

    all_results = {
        "arm1_by_lr": arm1_by_lr, "arm1_best_lr": arm1_best_lr,
        "arm1_final": {"mean": r1_mean, "sem": r1_sem, "escalated": r1_escalated, "seeds": r1_seeds},
        "arm2_by_lr": arm2_by_lr, "arm2_best_lr": arm2_best_lr,
        "arm2_final": {"mean": r2_mean, "sem": r2_sem, "escalated": r2_escalated, "seeds": r2_seeds},
        "arm3_by_lr": arm3_by_lr, "arm3_lrs_tested": arm3_lrs, "arm3_best_lr": arm3_best_lr,
        "arm3_final": {"mean": r3_mean, "sem": r3_sem, "escalated": r3_escalated, "seeds": r3_seeds},
    }
    (RUN_DIR / "results.json").write_text(json.dumps(all_results, indent=2))

    r1_pass = r1_mean >= TARGET_ACC
    r2_pass = r2_mean >= TARGET_ACC
    r3_pass = r3_mean >= TARGET_ACC

    print("\n=== Interpretation matrix (pre-committed, docs/threads/18-recall-protocol-validation.md) ===")
    print(f"R1 (primary)   = {r1_mean:.4f} -> {'PASS' if r1_pass else 'FAIL'} (bar {TARGET_ACC})")
    print(f"R2 (n_pairs=2) = {r2_mean:.4f} -> {'PASS' if r2_pass else 'FAIL'} (bar {TARGET_ACC})")
    print(f"R3 (extended)  = {r3_mean:.4f} -> {'PASS' if r3_pass else 'FAIL'} (bar {TARGET_ACC})")

    if r1_pass:
        verdict = ("Protocol validated at standard budget. Recall-negative cluster (9-17) "
                   "confirmed as evidence about mechanism insufficiency, not harness/protocol "
                   "failure. Thread 18 supplies the missing positive control retroactively.")
    elif r2_pass and r3_pass:
        verdict = ("Budget was the confound, not the protocol. The 2000-step convention was "
                   "too tight for any architecture at n_pairs=8 here, not evidence about "
                   "mechanism sufficiency at 2000 steps specifically.")
    elif r2_pass and not r3_pass:
        verdict = ("Depth-8 recall is genuinely hard at this scale for any tested architecture "
                   "within this budget range, but the harness/task-generator is not broken "
                   "(n_pairs=2 is learnable). Recall-cluster verdicts stand with a dated "
                   "difficulty/budget-calibration caveat.")
    else:
        verdict = ("Harness/protocol defect indicated: a literature-known-sufficient "
                   "architecture failed even the n_pairs=2 floor. Blocks trusting any prior "
                   "recall-cluster verdict until found and fixed.")

    print(f"\nVerdict: {verdict}")
    (RUN_DIR / "verdict.json").write_text(json.dumps({
        "r1_mean": r1_mean, "r1_pass": r1_pass, "r1_escalated": r1_escalated,
        "r2_mean": r2_mean, "r2_pass": r2_pass, "r2_escalated": r2_escalated,
        "r3_mean": r3_mean, "r3_pass": r3_pass, "r3_escalated": r3_escalated,
        "verdict": verdict,
    }, indent=2))
    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
