"""Pre-registered run for thread 10 (docs/threads/10-curriculum-gated-recurrence.md):
does a 3-stage curriculum (n_pairs 2 -> 4 -> 8), same total step budget as thread 9's
already-run direct-training protocol, recover depth-8 recall accuracy?

Protocol exactly as pre-registered: vocab=512, hidden=64, eps=0.1, curriculum stages
(n_pairs=2, 700 steps) -> (n_pairs=4, 700 steps) -> (n_pairs=8, 600 steps) = 2000 total
steps, matching thread 9's direct-training total exactly. Same LR grid and seeds as
thread 9. Evaluated on held-out n_pairs=8 batches. The comparison arm (direct training at
n_pairs=8 for 2000 steps) is NOT re-run here -- it's thread 9's already-collected result
(best-of-grid mean 0.032), reused as the matched-compute control per the thread doc.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import torch
import torch.nn.functional as F

from experiments.models.gated_linear_recurrence import GatedRecallModel
from experiments.tasks import recall

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread10_curriculum_sanity"

VOCAB = 512
HIDDEN = 64
EPS = 0.1
BATCH = 32
CURRICULUM = [(2, 700), (4, 700), (8, 600)]  # (n_pairs, n_steps), total 2000
LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
SEEDS = list(range(5))
EVAL_N_PAIRS = 8
EVAL_BATCH = 512
TARGET_ACC = 0.30
DIRECT_TRAINING_CONTROL_MEAN_ACC = 0.032  # thread 9's already-collected result, not re-run


def train_curriculum_and_eval(lr, seed):
    torch.manual_seed(seed)
    model = GatedRecallModel(VOCAB, HIDDEN, EPS)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0
    final_loss = None
    for n_pairs, n_steps in CURRICULUM:
        for _ in range(n_steps):
            tokens, targets = recall.make_batch(
                n_pairs, VOCAB, BATCH, seed=seed * 100_000 + global_step)
            logits = model(tokens)
            loss = F.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            final_loss = loss.item()
    eval_tokens, eval_targets = recall.make_batch(
        EVAL_N_PAIRS, VOCAB, EVAL_BATCH, seed=seed + 999_999)
    with torch.no_grad():
        acc = (model(eval_tokens).argmax(-1) == eval_targets).float().mean().item()
    return acc, final_loss


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(LR_GRID) * len(SEEDS)
    done = 0
    t0 = time.time()
    for lr in LR_GRID:
        for seed in SEEDS:
            acc, final_loss = train_curriculum_and_eval(lr, seed)
            results.append({"lr": lr, "seed": seed, "eval_acc": acc, "final_loss": final_loss})
            done += 1
            elapsed = time.time() - t0
            print(f"[{done}/{total}] lr={lr} seed={seed} acc={acc:.4f} "
                  f"loss={final_loss:.3f} elapsed={elapsed:.0f}s")

    (RUN_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print("\n--- best-of-grid mean accuracy (curriculum arm) ---")
    by_lr = {}
    for r in results:
        by_lr.setdefault(r["lr"], []).append(r["eval_acc"])
    best_lr = max(by_lr, key=lambda lr: sum(by_lr[lr]) / len(by_lr[lr]))
    mean_acc = sum(by_lr[best_lr]) / len(by_lr[best_lr])
    print(f"best_lr={best_lr} mean_acc={mean_acc:.4f} "
          f"(target={TARGET_ACC}, {'PASS' if mean_acc >= TARGET_ACC else 'FAIL'})")
    for lr, accs in sorted(by_lr.items()):
        print(f"  lr={lr}: mean={sum(accs)/len(accs):.4f} per_seed={[round(a,4) for a in accs]}")
    print(f"\ncomparison: thread 9 direct-training control (not re-run) = "
          f"{DIRECT_TRAINING_CONTROL_MEAN_ACC}")


if __name__ == "__main__":
    main()
