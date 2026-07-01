"""Pre-registered sanity run for thread 9 (docs/threads/09-gated-spectral-recurrence.md),
prediction A: does a minimal input-dependent retention gate on thread 1's orthogonal core
make associative recall solvable?

Protocol exactly as pre-registered: vocab=512, hidden=64, eps=0.1, n_pairs=8 (seq_len=17),
2000 Adam steps with a FRESH random batch every step (no repeated exposure -- deliberately
the harder, more standard "online" setting, not curriculum or repeated-epoch training),
LR grid {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}, >=5 seeds, matched tuning-trial count between the
gated arm and the ungated `orthogonal` control (same grid, same seeds, same step budget).

An Opus 4.8 review of the pre-implementation smoke tests (not this script) found: no bug
in the harness; the gate provably does inject content-dependence into the query response
(directly verified, distinct from thread 1's proven-impossible linear case); but at this
exact depth (n_pairs=8) the model reliably undertrains -- the learned gate barely opens
(mean g ~ 0.01-0.16) regardless of LR/init/hidden size in the review's own re-runs, and
retrieval accuracy collapses monotonically with n_pairs (2 -> 0.32, 4 -> 0.11, 8 -> 0.01 in
the review's runs) while easier depths clear the pre-registered 0.30 bar easily. This script
runs the ORIGINAL pre-registered protocol in full (not a curriculum-adjusted one) to get an
honest first-party verdict on record before any protocol change -- expected, based on the
review, to fail prediction A's 0.30 bar. See the thread doc's dated post-hoc note for the
full account and for the separate, explicitly-labeled exploratory curriculum follow-up.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import torch
import torch.nn.functional as F

from experiments.models.gated_linear_recurrence import GatedRecallModel
from experiments.models.linear_recurrence import RecallModel
from experiments.tasks import recall

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread09_gate_recall_sanity"

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


def train_and_eval(model_ctor, lr, seed):
    torch.manual_seed(seed)
    model = model_ctor()
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
    arms = {
        "gated_orthogonal": lambda: GatedRecallModel(VOCAB, HIDDEN, EPS),
        "orthogonal_control": lambda: RecallModel(VOCAB, HIDDEN, "orthogonal", EPS),
    }
    total = len(arms) * len(LR_GRID) * len(SEEDS)
    done = 0
    t0 = time.time()
    for arm_name, ctor in arms.items():
        for lr in LR_GRID:
            for seed in SEEDS:
                acc, final_loss = train_and_eval(ctor, lr, seed)
                results.append({"arm": arm_name, "lr": lr, "seed": seed, "eval_acc": acc,
                                 "final_loss": final_loss})
                done += 1
                elapsed = time.time() - t0
                print(f"[{done}/{total}] arm={arm_name} lr={lr} seed={seed} "
                      f"acc={acc:.4f} loss={final_loss:.3f} elapsed={elapsed:.0f}s")

    (RUN_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print("\n--- best-of-grid mean accuracy per arm (across seeds, at each arm's best LR) ---")
    for arm_name in arms:
        arm_results = [r for r in results if r["arm"] == arm_name]
        by_lr = {}
        for r in arm_results:
            by_lr.setdefault(r["lr"], []).append(r["eval_acc"])
        best_lr = max(by_lr, key=lambda lr: sum(by_lr[lr]) / len(by_lr[lr]))
        mean_acc = sum(by_lr[best_lr]) / len(by_lr[best_lr])
        print(f"arm={arm_name} best_lr={best_lr} mean_acc={mean_acc:.4f} "
              f"(target={TARGET_ACC}, {'PASS' if mean_acc >= TARGET_ACC else 'FAIL'})")
        for lr, accs in sorted(by_lr.items()):
            print(f"  lr={lr}: mean={sum(accs)/len(accs):.4f} per_seed={[round(a,4) for a in accs]}")


if __name__ == "__main__":
    main()
