"""Dumps run results to disk and aggregates a (parametrization, width, lr, seed) sweep
into the LR-transfer summary thread 6's prediction is about: does the LR-sweep minimum
stay put across width?
"""

import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from experiments.harness.train import RunResult


def save_run(result: RunResult, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    name = f"{cfg['parametrization']}_w{cfg['width']}_lr{cfg['base_lr']:.2e}_s{cfg['seed']}.json"
    path = run_dir / name
    path.write_text(json.dumps(asdict(result), indent=2))
    return path


def summarize_sweep(results: list[RunResult]) -> dict:
    """Groups by (parametrization, width, lr), averages final_test_loss across seeds,
    finds the lr with the lowest mean loss per (parametrization, width), and reports the
    log10-space spread of that best lr across widths for each parametrization -- the
    quantity thread 6's falsifiable prediction is actually about.
    """
    by_group = defaultdict(list)
    for r in results:
        key = (r.config["parametrization"], r.config["width"], r.config["base_lr"])
        by_group[key].append(r.final_test_loss)

    mean_loss = {k: statistics.mean(v) for k, v in by_group.items()}

    best_lr_per_width = defaultdict(dict)  # parametrization -> width -> best_lr
    for (param, width, lr), loss in mean_loss.items():
        current = best_lr_per_width[param].get(width)
        if current is None or loss < current[1]:
            best_lr_per_width[param][width] = (lr, loss)

    summary = {}
    for param, per_width in best_lr_per_width.items():
        widths = sorted(per_width)
        best_lrs = [per_width[w][0] for w in widths]
        log_lrs = [math.log10(lr) for lr in best_lrs]
        drift_decades = max(log_lrs) - min(log_lrs) if len(log_lrs) > 1 else 0.0
        summary[param] = {
            "widths": widths,
            "best_lr_per_width": {w: per_width[w][0] for w in widths},
            "log10_drift_decades": drift_decades,
        }
    return summary
