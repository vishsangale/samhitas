"""Dev smoke test for thread 6 (docs/threads/06-mup-hparam-transfer.md), NOT the
pre-registered experiment.

v2, after review caught three problems with v1: a train/test leakage bug in the dataset
(fixed in experiments/tasks/modular_arith.py), a final-loss metric that saturates and
can't discriminate configs (switched to steps-to-target in experiments/harness/train.py),
and no effect-size check against the pre-registered bar (added to
experiments/harness/report.py). This version also uses a deeper model (2 hidden layers,
not 1) so the width-scaling machinery actually gets exercised.

Widths/depth are still scaled down from what docs/threads/06-mup-hparam-transfer.md's real
run should use (k in {4, 8, 16} off a real base width) -- this sandbox is CPU-only, and
width=1024+ with 2-3 hidden layers gets slow fast. k = 1, 4, 8 here instead of 1, 4, 8, 16;
the real GPU run should extend to k=16 and probably a larger base width.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.harness.report import save_run, summarize_sweep
from experiments.harness.train import RunConfig, train_one

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread06_sanity"

# p=97 with target_train_loss=0.5/max_steps=200 (first cut at this fix) never converged
# for *any* config, at any width or LR -- 97-way modular arithmetic from one-hot inputs is
# close to the "grokking" setup (Power et al. 2022), which needs thousands of steps, not
# hundreds, to get off the uniform-baseline loss plateau. Dropped to p=41 and a looser
# target (well below the ~3.7 uniform baseline but short of full convergence) so the
# metric actually discriminates within a CPU-smoke-test step budget. Real run should
# either keep p small like this or budget enough steps for the harder task.
P = 41
N_TRAIN, N_TEST = 1200, 400  # 1600 of 41*41=1681 unique pairs
BASE_WIDTH = 64
WIDTHS = [64, 256, 512]  # k = 1, 4, 8 relative to base_width
LRS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
SEEDS = [0, 1, 2]
MAX_STEPS = 400
TARGET_TRAIN_LOSS = 1.0


def main():
    results = []
    total = len(WIDTHS) * len(LRS) * len(SEEDS) * 2
    done = 0
    for parametrization in ("sp", "mup"):
        for width in WIDTHS:
            for lr in LRS:
                for seed in SEEDS:
                    cfg = RunConfig(
                        parametrization=parametrization,
                        width=width,
                        base_width=BASE_WIDTH,
                        base_lr=lr,
                        seed=seed,
                        p=P,
                        n_train=N_TRAIN,
                        n_test=N_TEST,
                        max_steps=MAX_STEPS,
                        target_train_loss=TARGET_TRAIN_LOSS,
                    )
                    result = train_one(cfg)
                    save_run(result, RUN_DIR)
                    results.append(result)
                    done += 1
                    stt = result.steps_to_target
                    stt_str = str(stt) if stt is not None else "DNC"  # did not converge
                    print(f"[{done}/{total}] {parametrization} w={width} lr={lr:.1e} "
                          f"seed={seed} steps_to_target={stt_str} "
                          f"wall_clock={result.wall_clock_seconds:.2f}s")

    summary = summarize_sweep(results)
    print("\n--- LR-transfer summary (smoke test, not a pre-registered verdict) ---")
    for param in ("sp", "mup"):
        s = summary[param]
        print(f"\n{param}:")
        for w in sorted(s["per_width"]):
            print(f"  width={w}: {s['per_width'][w]}")
        print(f"  log10_drift_decades={s['log10_drift_decades']}")
    print(f"\nverdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
