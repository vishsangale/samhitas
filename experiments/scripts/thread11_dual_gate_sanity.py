"""Pre-registered run for thread 11 (docs/threads/11-dual-gate-spectral-recurrence.md),
prediction A: do independent read/write gates (LSTM-style) recover depth-8 associative
recall, where thread 9's single shared gate (direct training) and thread 10's curriculum
(same architecture, staged schedule) both failed?

Protocol exactly as pre-registered: vocab=512, hidden=64, eps=0.1, n_pairs=8 (seq_len=17),
2000 fresh-random-batch Adam steps (direct training, not curriculum), LR grid
{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}, 5 seeds. Controls (thread 9 direct: 0.032, thread 10
curriculum: 0.039) are reused, not re-run -- same protocol, only the architecture differs.
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

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread11_dual_gate_sanity"

VOCAB = 512
HIDDEN = 64
EPS = 0.1
N_PAIRS = 8
BATCH = 32
TRAIN_STEPS = 2000
LR_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
SEEDS = list(range(5))
EVAL_BATCH = 512
TARGET_ACC = 0.30
THREAD9_DIRECT_CONTROL = 0.032   # already collected, not re-run
THREAD10_CURRICULUM_CONTROL = 0.039  # already collected, not re-run


def train_and_eval(lr, seed):
    torch.manual_seed(seed)
    model = DualGateRecallModel(VOCAB, HIDDEN, EPS)
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
    return acc, loss.item()


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(LR_GRID) * len(SEEDS)
    done = 0
    t0 = time.time()
    for lr in LR_GRID:
        for seed in SEEDS:
            acc, final_loss = train_and_eval(lr, seed)
            results.append({"lr": lr, "seed": seed, "eval_acc": acc, "final_loss": final_loss})
            done += 1
            elapsed = time.time() - t0
            print(f"[{done}/{total}] lr={lr} seed={seed} acc={acc:.4f} "
                  f"loss={final_loss:.3f} elapsed={elapsed:.0f}s")

    (RUN_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print("\n--- best-of-grid mean accuracy (dual-gate arm) ---")
    by_lr = {}
    for r in results:
        by_lr.setdefault(r["lr"], []).append(r["eval_acc"])
    best_lr = max(by_lr, key=lambda lr: sum(by_lr[lr]) / len(by_lr[lr]))
    mean_acc = sum(by_lr[best_lr]) / len(by_lr[best_lr])
    print(f"best_lr={best_lr} mean_acc={mean_acc:.4f} "
          f"(target={TARGET_ACC}, {'PASS' if mean_acc >= TARGET_ACC else 'FAIL'})")
    for lr, accs in sorted(by_lr.items()):
        print(f"  lr={lr}: mean={sum(accs)/len(accs):.4f} per_seed={[round(a,4) for a in accs]}")
    print(f"\ncomparison: thread 9 direct-training control (not re-run) = {THREAD9_DIRECT_CONTROL}")
    print(f"comparison: thread 10 curriculum control (not re-run) = {THREAD10_CURRICULUM_CONTROL}")


if __name__ == "__main__":
    main()
