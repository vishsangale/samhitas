"""Thread 6 -- the pre-registered REAL RUN, packaged for execution on the user's own GPU.

This is the keystone experiment (docs/reviews/2026-07-08-program-regroup.md, item C3): the
first real-scale test of muP-style hyperparameter transfer, on a non-algorithmic task
(char-level tiny-shakespeare) chosen to kill the modular-arithmetic grokking-dynamics
confound that the three prior CPU smoke tests were suspected to be measuring. Read the
pre-registered prediction and pass/fail band in docs/threads/06-mup-hparam-transfer.md
(and its dated addenda) before running or interpreting this.

================================  THE BANDS BELOW ARE FROZEN  ================================
WIDTHS, LRS, SEEDS, the loss thresholds, and the model/task config were pinned and committed
BEFORE this run executed (they were calibrated on CPU smoke tests: thread06_charlm_smoke.py
and a 2000-step floor check; the muP wiring was validated by thread06_charlm_coord_check.py).
Per docs/methodology.md's pre-registration rule, whoever runs this MUST NOT change any frozen
band and then report the output as thread 6's verdict. If a band turns out to be
mis-calibrated, add a fresh dated "Post-hoc note" addendum to the thread doc explaining the
change and re-run under the revised (re-committed) protocol -- never silently edit a band
after seeing data. This is the same discipline the whole repo runs on.
=============================================================================================

HOW TO RUN
  1. pip install -r experiments/requirements.txt   (needs a CUDA build of torch)
  2. python experiments/scripts/thread06_gpu_run.py
     - Auto-detects CUDA (falls back to CPU with a warning -- CPU is only sane for
       re-aggregating an existing result set, not for the real run: the recurrence arm at
       width 4096 is ~minutes PER STEP on CPU because of the matrix_exp(W x W) each step).
     - Writes one JSON per run under experiments/runs/thread06_gpu_run/<arm>/, in the exact
       RunResult schema harness/report.py already uses. Resumable: a re-run skips any config
       whose JSON already exists, so an interrupted sweep just continues.
     - After the sweep it prints and writes the LR-transfer summary
       (experiments/runs/thread06_gpu_run/summary.json).

HOW TO GET RESULTS BACK INTO THIS REPO (no CI/cloud here -- manual copy)
  - Copy the whole experiments/runs/thread06_gpu_run/ directory back into a checkout of this
    repo at the same path, commit it, then run this script once more on any machine with
    torch installed: it finds every JSON already present, skips all runs, and rebuilds
    summary.json from them. Ingestion is therefore: copy dir back -> re-run script (it only
    aggregates) -> read summary.json. No bespoke format -- the per-run files ARE the
    RunResult schema, so summarize_sweep() consumes them directly.

WHAT THE PREDICTION NEEDS (from the thread doc, for the person reading summary.json)
  For each arm, summarize_sweep reports muP's and SP's log10 LR-drift-across-width. The
  pre-registered bar: "transfer holds" for an arm requires muP's drift to be >= 3x SMALLER
  than SP's (report.py's EFFECT_SIZE_BAR). muP's raw base_lr optimum should stay within ~2x
  across the width grid; SP's should drift an order of magnitude. The three CPU smokes ran
  AGAINST this (muP drifted more than SP) -- the open question this run answers is whether
  that flips on a real task at real width, where muP's asymptotic advantage can appear.

COMPUTE (rough; the script prints real per-config wall-clock so you see actuals)
  The MLP (baseline) arm is minutes total on any modern GPU. The recurrence (thread-1) arm
  is the cost driver -- dominated by matrix_exp(W x W) once per step -- and its width-4096
  cells are ~minutes/config. Whole sweep target: within thread 6's stated budget (a small
  multiple of a GPU-day at worst; likely a few GPU-hours on an A100-class card).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.harness.report import save_run, summarize_sweep
from experiments.harness.train import RunResult
from experiments.harness.train_charlm import CharLMConfig, train_one_charlm

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "thread06_gpu_run"

# ============================= FROZEN PROTOCOL -- do not edit ================================
BASE_WIDTH = 64
WIDTHS = [64, 256, 1024, 2048, 4096]                       # k = 1, 4, 16, 32, 64 (up to 64x)
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]     # 8 pts, ~3.16x log-spaced
SEEDS = [0, 1, 2, 3, 4]                                    # 5 seeds (repo convention)
ARMS = ["mlp", "recurrence"]                               # baseline + thread-1 novel case
PARAMS = ["sp", "mup"]
CONTEXT_LEN = 32
D_EMBED = 64          # mlp arm: fixed featurizer dim (does NOT scale with width)
DEPTH = 4             # mlp arm: proj + 2 hidden + output
EPS = 0.05            # recurrence arm: spectral radius = 1 - eps
BATCH_SIZE = 128
MAX_STEPS = 2000
TARGET_TRAIN_LOSSES = [3.0, 2.7, 2.4, 2.2, 2.0]            # nats; init CE = log(65) ~= 4.17
# ============================================================================================


def cfg_for(arm, param, width, lr, seed):
    return CharLMConfig(
        arm=arm, parametrization=param, width=width, base_width=BASE_WIDTH, base_lr=lr,
        seed=seed, context_len=CONTEXT_LEN, d_embed=D_EMBED, depth=DEPTH, eps=EPS,
        batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
        target_train_losses=list(TARGET_TRAIN_LOSSES),
    )


def run_name(param, width, lr, seed):
    # Must match harness/report.py:save_run's naming exactly, so skip-if-exists works.
    return f"{param}_w{width}_lr{lr:.2e}_s{seed}.json"


def load_results(arm_path: Path):
    """Load saved runs for one arm, floatifying steps_to_target keys (JSON stringifies dict
    keys, but summarize_sweep indexes them with the float threshold)."""
    out = []
    if not arm_path.exists():
        return out
    for f in sorted(arm_path.glob("*.json")):
        raw = json.loads(f.read_text())
        raw["steps_to_target"] = {float(k): v for k, v in raw["steps_to_target"].items()}
        out.append(RunResult(**raw))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. CPU is fine ONLY for re-aggregating an existing "
              "result set -- the real recurrence-arm run at large width is prohibitively slow "
              "on CPU (matrix_exp each step).", flush=True)
    print(f"device={device}  total runs={len(ARMS)*len(PARAMS)*len(WIDTHS)*len(LRS)*len(SEEDS)}",
          flush=True)

    total = len(ARMS) * len(PARAMS) * len(WIDTHS) * len(LRS) * len(SEEDS)
    done = 0
    t0 = time.perf_counter()
    for arm in ARMS:
        arm_path = RUN_DIR / arm
        arm_path.mkdir(parents=True, exist_ok=True)
        for param in PARAMS:
            for width in WIDTHS:
                for lr in LRS:
                    for seed in SEEDS:
                        done += 1
                        out_path = arm_path / run_name(param, width, lr, seed)
                        if out_path.exists():
                            continue  # resumable
                        try:
                            r = train_one_charlm(cfg_for(arm, param, width, lr, seed), device=device)
                        except Exception as e:  # keep the sweep alive; the config retries on restart
                            print(f"[{done}/{total}] FAILED {arm}/{param} w={width} lr={lr:.0e} "
                                  f"s={seed}: {type(e).__name__}: {e}", flush=True)
                            continue
                        save_run(r, arm_path)
                        el = time.perf_counter() - t0
                        print(f"[{done}/{total}] {arm}/{param} w={width} lr={lr:.0e} s={seed} "
                              f"final_train={r.final_train_loss:.3f} "
                              f"wall={r.wall_clock_seconds:.1f}s elapsed={el:.0f}s", flush=True)

    print("\n==== LR-transfer summary (per arm, per threshold) ====")
    full_summary = {}
    for arm in ARMS:
        results = load_results(RUN_DIR / arm)
        full_summary[arm] = {}
        print(f"\n#### arm={arm} ({len(results)} runs loaded) ####")
        for threshold in TARGET_TRAIN_LOSSES:
            summary = summarize_sweep(results, threshold)
            full_summary[arm][str(threshold)] = summary
            print(f"\n== threshold={threshold} ==")
            for param in PARAMS:
                s = summary.get(param)
                if s:
                    print(f"  {param}: log10_drift={s['log10_drift_decades']} "
                          f"widths_used={s['widths_used_in_drift']} "
                          f"gated_out={s['widths_gated_out']}")
            print(f"  verdict: {summary['verdict']}")

    (RUN_DIR / "summary.json").write_text(json.dumps(full_summary, indent=2, default=str))
    print(f"\nSummary written to {RUN_DIR / 'summary.json'}")
    print("\nThis counts as thread 6's pre-registered verdict ONLY if the FROZEN bands above "
          "were unchanged. Pass/fail per the thread doc: muP's log10 LR-drift across width "
          ">= 3x smaller than SP's, per arm.")


if __name__ == "__main__":
    main()
