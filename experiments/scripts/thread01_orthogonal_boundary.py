"""Focused follow-up to thread01_stability_sanity.py's wider run: orthogonal at
eps in {0.005, 0.002} stayed healthy across the *entire* tested range (up to seq_len=385),
so its actual failure boundary was never observed -- this finds it.

No training loop needed: for orthogonal, ratio_first_over_last = (1-eps)^(seq_len-1)
exactly (all eigenvalues share magnitude 1-eps by construction, verified in the wider run
to hold to 3-4 decimal places), so the healthy/unhealthy boundary is a closed-form
prediction: it crosses the lower band edge (ratio=0.1) at
    L* = 1 + ln(0.1) / ln(1-eps)
This script brackets L* for each eps with a grid computed from that formula and checks
whether the measured crossing lands where predicted -- a direct, cheap, quantitative test
of the linear-regime prediction, not just a qualitative "did it fail eventually" check.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

from experiments.harness.gradient_flow import gradient_norm_ratio
from experiments.models.linear_recurrence import RecallModel
from experiments.tasks import recall
import torch

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread01_orthogonal_boundary"

VOCAB = 512
HIDDEN = 64
BATCH = 32
EPS_LIST = [0.005, 0.002]
SEEDS = [0, 1, 2, 3, 4]
MULTIPLIERS = [0.75, 0.85, 0.92, 0.97, 1.0, 1.03, 1.08, 1.15, 1.25]


def predicted_boundary(eps: float) -> float:
    return 1 + math.log(0.1) / math.log(1 - eps)


def seq_len_grid(eps: float) -> list:
    l_star = predicted_boundary(eps)
    seq_lens = set()
    for m in MULTIPLIERS:
        n_pairs = max(round((m * l_star - 1) / 2), 1)
        seq_lens.add(recall.seq_len_for(n_pairs))
    return sorted(seq_lens)


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    combos = [(eps, l) for eps in EPS_LIST for l in seq_len_grid(eps)]
    total = len(combos) * len(SEEDS)
    done = 0
    for eps, seq_len in combos:
        n_pairs = (seq_len - 1) // 2
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = RecallModel(VOCAB, HIDDEN, "orthogonal", eps)
            tokens, targets = recall.make_batch(n_pairs, VOCAB, BATCH, seed)
            r = gradient_norm_ratio(model, tokens, targets)
            results.append({"eps": eps, "seq_len": seq_len, "seed": seed, **r})
            done += 1
            print(f"[{done}/{total}] eps={eps} seq_len={seq_len} seed={seed} "
                  f"ratio={r['ratio_first_over_last']:.4g} healthy={r['healthy']}")

    (RUN_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print("\n--- measured vs. predicted boundary ---")
    for eps in EPS_LIST:
        l_star = predicted_boundary(eps)
        rs = [r for r in results if r["eps"] == eps]
        by_len = sorted(set(r["seq_len"] for r in rs))
        healthy_lens = [l for l in by_len if all(r["healthy"] for r in rs if r["seq_len"] == l)]
        unhealthy_lens = [l for l in by_len if not any(r["healthy"] for r in rs if r["seq_len"] == l)]
        last_healthy = max(healthy_lens) if healthy_lens else None
        first_unhealthy = min(unhealthy_lens) if unhealthy_lens else None
        print(f"eps={eps}: predicted L*={l_star:.1f} | "
              f"last fully-healthy length={last_healthy} | "
              f"first fully-unhealthy length={first_unhealthy}")


if __name__ == "__main__":
    main()
