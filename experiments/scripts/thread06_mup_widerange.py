"""Wider/finer CPU exploratory sweep for thread 6, still NOT the pre-registered GPU run --
just a bigger step up from thread06_mup_sanity.py's quick smoke test.

Extends smoke-test v2/v3 in the two directions a second Opus 4.8 review flagged as the
real gaps: width range (was k in {1,4,8}, 8x total -- known to be too narrow for muP's
usually-asymptotic advantage to plausibly show; this reaches k in {1,4,16,32}) and LR grid
resolution (was ~3.3x spacing; this is ~2.3x, closer to the 2x tolerance the pre-registered
prediction is actually stated in). Also reports both tracked loss thresholds so the result
isn't hostage to one arbitrary cutoff, per the same review.

Sized to fit a single CPU run in roughly 10-15 minutes (see timing notes in commit history)
-- width=4096 was tried and rejected: a single non-convergent run there costs ~90s+ on this
sandbox's 4 cores, which alone blows the time budget across a full LR grid. 2048 (k=32) is
the practical ceiling here; the real run still needs a GPU to go further, per
docs/methodology.md's compute budget.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.harness.report import save_run, summarize_sweep
from experiments.harness.train import RunConfig, train_one

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread06_widerange"

P = 41
N_TRAIN, N_TEST = 1200, 400
BASE_WIDTH = 64
WIDTHS = [64, 256, 1024, 2048]  # k = 1, 4, 16, 32 relative to base_width
LRS = [3e-3, 7e-3, 1.5e-2, 3.5e-2, 8e-2, 1.8e-1, 4e-1, 9e-1]  # ~2.3x spacing
SEEDS = [0, 1, 2]
MAX_STEPS = 400
TARGET_TRAIN_LOSSES = [1.5, 1.0]


def main():
    results = []
    total = len(WIDTHS) * len(LRS) * len(SEEDS) * 2
    done = 0
    t_start = time.perf_counter()
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
                        target_train_losses=TARGET_TRAIN_LOSSES,
                    )
                    result = train_one(cfg)
                    save_run(result, RUN_DIR)
                    results.append(result)
                    done += 1
                    stt = result.steps_to_target[TARGET_TRAIN_LOSSES[-1]]
                    stt_str = str(stt) if stt is not None else "DNC"
                    elapsed = time.perf_counter() - t_start
                    print(f"[{done}/{total}] {parametrization} w={width} lr={lr:.2e} "
                          f"seed={seed} steps_to_target={stt_str} "
                          f"wall_clock={result.wall_clock_seconds:.2f}s "
                          f"elapsed={elapsed:.0f}s", flush=True)

    print(f"\nTotal elapsed: {time.perf_counter() - t_start:.0f}s")
    print("\n--- LR-transfer summary (wider CPU sweep, not a pre-registered verdict) ---")
    for threshold in TARGET_TRAIN_LOSSES:
        summary = summarize_sweep(results, threshold)
        print(f"\n=== threshold={threshold} ===")
        for param in ("sp", "mup"):
            s = summary[param]
            print(f"{param}:")
            for w in sorted(s["per_width"]):
                print(f"  width={w}: {s['per_width'][w]}")
            print(f"  log10_drift_decades={s['log10_drift_decades']} "
                  f"(widths_used={s['widths_used_in_drift']}, gated_out={s['widths_gated_out']})")
        print(f"verdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
