"""Dev smoke test for thread 6 (docs/threads/06-mup-hparam-transfer.md), NOT the
pre-registered experiment.

This is a CPU-scale, few-seed, few-step run meant only to exercise the harness code path
(dataset -> model -> param groups -> training loop -> FLOPs/wall-clock -> LR-sweep
aggregation) and sanity-check that the muP scaling rule in experiments/models/mlp.py at
least points the right direction. It is deliberately too small (narrow widths, few steps,
few seeds) to stand in for the real falsification run described in the thread doc, which
needs widths spanning k in {4, 8, 16} at real training scale and the pre-registered
effect-size bar (>=3x smaller log-drift for muP vs SP). Do not cite this script's numbers
as a thread verdict.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.harness.report import save_run, summarize_sweep
from experiments.harness.train import RunConfig, train_one

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread06_sanity"

BASE_WIDTH = 32
WIDTHS = [32, 128, 512]  # k = 1, 4, 16 relative to base_width
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]
SEEDS = [0, 1, 2]
N_STEPS = 150


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
                        n_steps=N_STEPS,
                    )
                    result = train_one(cfg)
                    save_run(result, RUN_DIR)
                    results.append(result)
                    done += 1
                    print(f"[{done}/{total}] {parametrization} w={width} lr={lr:.1e} "
                          f"seed={seed} test_loss={result.final_test_loss:.3f} "
                          f"wall_clock={result.wall_clock_seconds:.2f}s")

    summary = summarize_sweep(results)
    print("\n--- LR-transfer summary (smoke test, not a verdict) ---")
    for param, s in summary.items():
        print(f"{param}: best_lr_per_width={s['best_lr_per_width']} "
              f"log10_drift_decades={s['log10_drift_decades']:.2f}")
    print("\nExpectation per thread doc: mup drift << sp drift. "
          "This is a code-path check, not the pre-registered result.")


if __name__ == "__main__":
    main()
