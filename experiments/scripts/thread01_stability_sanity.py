"""Dev smoke test for thread 1 (docs/threads/01-stability-constrained-recurrence.md), NOT
the pre-registered experiment. First cheap read on the two falsifiable predictions:

  1. linear-regime: max "trainable" sequence length (gradient-norm ratio within the
     [0.1x, 10x] band) scales like O(1/eps) for the spectrally-constrained variants.
  2. cross-parameterization (the actual falsifiable core): that relationship holds for
     BOTH the orthogonal and diag_lowrank parameterizations, not just one.

Measures gradient_norm_ratio twice per config: once at init, once after a short training
run (fixed LR, same for every config -- this is a stress test of relative stability, not
a tuned comparison, so the usual LR-sweep/tuning-symmetry protocol doesn't apply the same
way it does for thread 6's training-loss comparisons). The "free" baseline is initialized
without any special spectral properties (a plain small-Gaussian init), specifically so it
starts on equal footing with the constrained variants rather than being handed a
coincidentally well-conditioned spectrum at init.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from experiments.harness.gradient_flow import gradient_norm_ratio
from experiments.models.linear_recurrence import RecallModel
from experiments.tasks import recall

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread01_sanity"

VOCAB = 24
HIDDEN = 64
BATCH = 32
N_PAIRS_LIST = [8, 16, 32, 64, 128]  # seq_len = 2n+1: 17, 33, 65, 129, 257
# Added 0.005 (spectral radius 0.995): v1 only went down to eps=0.02, whose constrained
# variants topped out well short of where the "free" baseline happened to reach (129) --
# too coarse to tell whether a constrained config can reach that range too, and if so,
# whether it does so without free's confirmed explosion risk just beyond it.
EPS_LIST = [0.5, 0.1, 0.02, 0.005]
SEEDS = [0, 1, 2]
TRAIN_STEPS = 60
TRAIN_LR = 1e-3


def configs():
    yield ("free", None)
    for mode in ("orthogonal", "diag_lowrank"):
        for eps in EPS_LIST:
            yield (mode, eps)


def run_one(mode, eps, n_pairs, seed):
    torch.manual_seed(seed)
    model = RecallModel(VOCAB, HIDDEN, mode, eps)
    tokens, targets = recall.make_batch(n_pairs, VOCAB, BATCH, seed)

    at_init = gradient_norm_ratio(model, tokens, targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    for step in range(TRAIN_STEPS):
        tr_tokens, tr_targets = recall.make_batch(n_pairs, VOCAB, BATCH, seed * 10_000 + step)
        logits = model(tr_tokens)
        loss = F.cross_entropy(logits, tr_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    eval_tokens, eval_targets = recall.make_batch(n_pairs, VOCAB, BATCH * 4, seed + 999)
    with torch.no_grad():
        eval_logits = model(eval_tokens)
        eval_acc = (eval_logits.argmax(-1) == eval_targets).float().mean().item()

    after_train = gradient_norm_ratio(model, tokens, targets)

    return {
        "mode": mode, "eps": eps, "n_pairs": n_pairs,
        "seq_len": recall.seq_len_for(n_pairs), "seed": seed,
        "ratio_at_init": at_init["ratio_first_over_last"],
        "failure_mode_at_init": at_init["failure_mode"],
        "healthy_at_init": at_init["healthy"],
        "first_grad_norm_after_train": after_train["first_grad_norm"],
        "last_grad_norm_after_train": after_train["last_grad_norm"],
        "ratio_after_train": after_train["ratio_first_over_last"],
        "failure_mode_after_train": after_train["failure_mode"],
        "healthy_after_train": after_train["healthy"],
        "eval_acc": eval_acc,
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    combos = list(configs())
    total = len(combos) * len(N_PAIRS_LIST) * len(SEEDS)
    done = 0
    for mode, eps in combos:
        for n_pairs in N_PAIRS_LIST:
            for seed in SEEDS:
                r = run_one(mode, eps, n_pairs, seed)
                results.append(r)
                done += 1
                print(f"[{done}/{total}] mode={mode} eps={eps} seq_len={r['seq_len']} "
                      f"seed={seed} ratio_init={r['ratio_at_init']:.3g} "
                      f"ratio_trained={r['ratio_after_train']:.3g} "
                      f"mode_trained={r['failure_mode_after_train']} acc={r['eval_acc']:.2f}")

    (RUN_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print("\n--- max healthy (post-training) seq_len per (mode, eps), and how it fails past that ---")
    by_config = defaultdict(list)
    for r in results:
        by_config[(r["mode"], r["eps"])].append(r)
    for (mode, eps), rs in sorted(by_config.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
        healthy_lens = [r["seq_len"] for r in rs if r["healthy_after_train"]]
        max_healthy = max(healthy_lens) if healthy_lens else None
        mean_acc_at_max = None
        if max_healthy is not None:
            accs = [r["eval_acc"] for r in rs if r["seq_len"] == max_healthy]
            mean_acc_at_max = sum(accs) / len(accs)
        # First seq_len beyond max_healthy (or the shortest tested, if nothing was
        # healthy) and how it actually failed -- vanished gracefully vs. exploded.
        longer = sorted(set(r["seq_len"] for r in rs if max_healthy is None or r["seq_len"] > max_healthy))
        failure_modes_next = None
        if longer:
            next_len = longer[0]
            modes_at_next = [r["failure_mode_after_train"] for r in rs if r["seq_len"] == next_len]
            failure_modes_next = f"at seq_len={next_len}: {modes_at_next}"
        print(f"mode={mode} eps={eps}: max_healthy_seq_len={max_healthy} "
              f"(acc there={mean_acc_at_max}) | first failure {failure_modes_next}")


if __name__ == "__main__":
    main()
