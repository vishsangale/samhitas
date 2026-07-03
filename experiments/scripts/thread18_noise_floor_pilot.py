"""Noise-floor pilot for thread 18 (docs/threads/18-recall-protocol-validation.md),
per methodology.md's amendment v2 item 3: run the planned estimator (held-out recall
accuracy under TinyAttentionModel) on null data and on a short training budget BEFORE
freezing the pre-registered bands, to check the >=0.30-vs-~0.02-0.03-control gap is
resolvable at 5 seeds and isn't swamped by seed-to-seed noise.

Exploratory only -- firewalled from the treatment comparison per the amendment's own rule:
this script's LR-vs-accuracy numbers are NOT used to hand-pick which LR "wins" in the real
pre-registered grid (that grid is swept in full and reported in full, unchanged by what's
seen here). This pilot only answers "is 5 seeds enough resolution," not "which LR is best."
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import statistics

import torch
import torch.nn.functional as F

from experiments.models.tiny_attention import TinyAttentionModel
from experiments.tasks import recall

VOCAB = 512
HIDDEN = 64
N_PAIRS = 8
SEQ_LEN = recall.seq_len_for(N_PAIRS)
BATCH = 32
EVAL_BATCH = 512
PILOT_STEPS = 400
PILOT_LRS = [3e-4, 1e-3, 3e-3]
PILOT_SEEDS = [0, 1, 2]


def chance_level_check():
    """Untrained model, held-out eval -- should read close to 1/VOCAB, confirming the
    eval harness (batching, argmax, accuracy) is wired correctly before trusting anything
    downstream of it."""
    torch.manual_seed(12345)
    model = TinyAttentionModel(VOCAB, HIDDEN, SEQ_LEN)
    eval_tokens, eval_targets = recall.make_batch(N_PAIRS, VOCAB, EVAL_BATCH, seed=555_555)
    with torch.no_grad():
        acc = (model(eval_tokens).argmax(-1) == eval_targets).float().mean().item()
    return acc


def short_train_and_eval(lr, seed):
    torch.manual_seed(seed)
    model = TinyAttentionModel(VOCAB, HIDDEN, SEQ_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_val = None
    for step in range(PILOT_STEPS):
        tokens, targets = recall.make_batch(N_PAIRS, VOCAB, BATCH, seed=seed * 100_000 + step)
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = loss.item()
    eval_tokens, eval_targets = recall.make_batch(N_PAIRS, VOCAB, EVAL_BATCH, seed=seed + 999_999)
    with torch.no_grad():
        acc = (model(eval_tokens).argmax(-1) == eval_targets).float().mean().item()
    return acc, loss_val


def main():
    print(f"seq_len={SEQ_LEN} (n_pairs={N_PAIRS}), hidden={HIDDEN}, vocab={VOCAB}")

    chance = chance_level_check()
    print(f"\n--- Chance-level check (untrained model, held-out eval) ---")
    print(f"acc={chance:.4f} (nominal chance = 1/{VOCAB} = {1/VOCAB:.5f})")

    print(f"\n--- Short-budget pilot ({PILOT_STEPS} steps, {len(PILOT_LRS)} LRs x "
          f"{len(PILOT_SEEDS)} seeds) ---")
    t0 = time.time()
    by_lr = {}
    for lr in PILOT_LRS:
        accs, losses = [], []
        for seed in PILOT_SEEDS:
            acc, loss_val = short_train_and_eval(lr, seed)
            accs.append(acc)
            losses.append(loss_val)
            print(f"  lr={lr:.0e} seed={seed} acc={acc:.4f} final_loss={loss_val:.3f} "
                  f"elapsed={time.time()-t0:.1f}s")
        by_lr[lr] = accs
        mean_acc = statistics.mean(accs)
        spread = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  lr={lr:.0e}: mean_acc={mean_acc:.4f} stdev={spread:.4f} "
              f"(n={len(accs)} seeds, {PILOT_STEPS} steps)")

    print(f"\n--- Resolvability check ---")
    all_accs = [a for accs in by_lr.values() for a in accs]
    max_seed_spread = max(
        statistics.stdev(accs) for accs in by_lr.values() if len(accs) > 1
    )
    print(f"Max per-LR seed stdev observed (short budget): {max_seed_spread:.4f}")
    print("Reference gap this needs to resolve: pass bar 0.30 vs. existing gate-family "
          "control band ~0.02-0.03 (gap ~0.27-0.28), and vs. chance ~0.002.")
    print(f"Full run uses 2000 steps (5x this pilot's budget) and the same 5-seed count "
          f"as every other recall-family thread; if this short-budget pilot's spread is "
          f"already small relative to the ~0.27 target gap, 5 seeds at the full budget is "
          f"expected to resolve it comfortably -- narrower bands than this would be "
          f"inadmissible per methodology.md amendment v2 item 3.")
    print(f"\nTotal pilot elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
