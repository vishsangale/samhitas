"""CPU smoke test for thread 6's char-LM real-run arms -- a plumbing check that both arms
(baseline MLP-muP/SP, thread-1 recurrence-muP/SP) run end-to-end on the char-LM task and
produce sane (finite, non-exploding, decreasing) loss curves at a couple of small widths and
a short step budget. NOT the pre-registered run and NOT a verdict on transfer -- same framing
as thread 6's three prior CPU smoke tests. Its second job is to expose the char-LM loss scale
so thread06_gpu_run.py's steps-to-target thresholds can be picked from data, not guessed.

Run: python experiments/scripts/thread06_charlm_smoke.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.harness.train_charlm import CharLMConfig, train_one_charlm

WIDTHS = [64, 256]
ARMS = ["mlp", "recurrence"]
PARAMS = ["mup", "sp"]
SEED = 0
MAX_STEPS = 500
BASE_WIDTH = 64
BASE_LR = 3e-3  # a middle-of-the-road LR just to see learning happen; not a tuned value


def main():
    t_all = time.perf_counter()
    for arm in ARMS:
        for param in PARAMS:
            for width in WIDTHS:
                cfg = CharLMConfig(
                    arm=arm, parametrization=param, width=width, base_width=BASE_WIDTH,
                    base_lr=BASE_LR, seed=SEED, max_steps=MAX_STEPS,
                )
                t0 = time.perf_counter()
                r = train_one_charlm(cfg)
                dt = time.perf_counter() - t0
                curve = " ".join(f"{s}:{l:.2f}" for s, l in r.loss_curve)
                finite = all(torch.isfinite(torch.tensor(l)).item() for _, l in r.loss_curve)
                print(f"\n[{arm}/{param}/w{width}] {dt:.1f}s ({dt / (r.config['max_steps']):.3f}s/step) "
                      f"finite={finite} final_train={r.final_train_loss:.3f} "
                      f"final_test={r.final_test_loss:.3f} test_acc={r.final_test_acc:.3f}")
                print(f"    steps_to_target={r.steps_to_target}")
                print(f"    curve: {curve}")
    print(f"\nTotal: {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
