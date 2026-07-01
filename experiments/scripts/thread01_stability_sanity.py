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

v2, after an Opus 4.8 review of the v1 smoke test: VOCAB was 24, which for up to 128
key-value pairs meant constant key collisions (most queries were structurally ambiguous),
making eval_acc pure chance -- raised to 512. The summary used to report a single
"max_healthy_seq_len" per config, which silently assumes degradation is monotonic in
sequence length; true for the constrained variants but not for `free` (healthy, then
vanishing, then healthy again, then exploding, in that order as length grows) -- replaced
with a per-length healthy-fraction-across-seeds table plus the dominant failure mode at
each length, which doesn't hide non-monotonic behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from experiments.harness.gradient_flow import gradient_norm_ratio
from experiments.models.linear_recurrence import RecallModel
from experiments.tasks import recall

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread01_sanity"

VOCAB = 512  # was 24 -- caused constant key collisions at n_pairs > ~12, see module docstring
HIDDEN = 64
BATCH = 32
# Grid widened per an Opus 4.8 review's recommendation after the v2 smoke test (which used
# 5 seq lengths / 4 eps / 3 seeds): denser near the failure boundary, and eps points kept
# in full (the review's advice: don't cut eps resolution, it's what an O(1/eps) fit needs).
N_PAIRS_LIST = [8, 16, 24, 32, 48, 64, 96, 128, 192]  # seq_len = 2n+1: 17..385
EPS_LIST = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
# Review suggested 10 seeds; timed at TRAIN_STEPS=15 this grid's dominant cost (the O(seq_len)
# python-loop training steps) put 10 seeds over the ~15min CPU budget, so trimmed to 7 --
# above the methodology floor of 3, short of the review's ideal, a compute-constrained
# compromise, noted here rather than silently done.
SEEDS = list(range(7))
TRAIN_STEPS = 15  # was 60 -- ratio_after_train tracked ratio_at_init closely in the v2
# smoke test for the constrained variants (structurally expected: training moves theta/d/
# U/V but the parameterization keeps enforcing the spectral bound regardless), so the
# training loop's main remaining job is showing whether "free" drifts -- 15 steps still
# does that far more cheaply than 60.
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
        "effective_decay_rate_after_train": after_train["effective_decay_rate"],
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
                      f"seed={seed} ratio_trained={r['ratio_after_train']:.3g} "
                      f"eff_rate={r['effective_decay_rate_after_train']:.4f} "
                      f"mode_trained={r['failure_mode_after_train']} acc={r['eval_acc']:.2f}")

    (RUN_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print("\n--- per (mode, eps, seq_len): healthy fraction across seeds, dominant failure "
          "mode, mean effective decay rate vs. nominal target ---")
    by_config = defaultdict(list)
    for r in results:
        by_config[(r["mode"], r["eps"], r["seq_len"])].append(r)
    for (mode, eps, seq_len), rs in sorted(
        by_config.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1, kv[0][2])
    ):
        healthy_frac = sum(r["healthy_after_train"] for r in rs) / len(rs)
        modes = Counter(r["failure_mode_after_train"] for r in rs)
        dominant_mode = modes.most_common(1)[0][0]
        mean_eff_rate = sum(r["effective_decay_rate_after_train"] for r in rs) / len(rs)
        nominal = 1 - eps if eps is not None else None
        print(f"mode={mode} eps={eps} seq_len={seq_len}: healthy_frac={healthy_frac:.2f} "
              f"dominant_failure={dominant_mode} mean_eff_decay_rate={mean_eff_rate:.4f} "
              f"nominal_target={nominal}")


if __name__ == "__main__":
    main()
