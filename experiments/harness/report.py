"""Dumps run results to disk and aggregates a (parametrization, width, lr, seed) sweep
into the LR-transfer summary thread 6's prediction is about: does the LR-sweep minimum
stay put across width, and does it clear the pre-registered effect-size bar?
"""

import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from experiments.harness.train import RunResult

# docs/threads/06-mup-hparam-transfer.md's pre-registered pass/fail band: muP's log10 LR
# drift across width must be at least 3x smaller than SP's.
EFFECT_SIZE_BAR = 3.0


def save_run(result: RunResult, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    name = f"{cfg['parametrization']}_w{cfg['width']}_lr{cfg['base_lr']:.2e}_s{cfg['seed']}.json"
    path = run_dir / name
    path.write_text(json.dumps(asdict(result), indent=2))
    return path


def _mean_steps(steps_to_target_values):
    """None (did-not-converge) is excluded from the mean but tracked in the count -- a
    group where nothing converged has no valid mean at all."""
    converged = [v for v in steps_to_target_values if v is not None]
    if not converged:
        return None, 0.0, 0
    spread = statistics.stdev(converged) if len(converged) > 1 else 0.0
    return statistics.mean(converged), spread, len(converged)


def _best_lr_for_width(group_stats: dict, param: str, width: int, lrs: list) -> dict:
    """Picks the lr with the lowest mean steps-to-target for (param, width), gated against
    two ways a "best" lr can be meaningless: sitting at the edge of the swept grid (the
    real optimum might be outside the grid), or tied with the next-best lr within seed
    noise (the loss surface is too flat/noisy here to say anything)."""
    candidates = [
        (lr, *group_stats[(param, width, lr)])
        for lr in sorted(lrs)
        if group_stats[(param, width, lr)][0] is not None
    ]
    if not candidates:
        return {"lr": None, "note": "no lr in grid converged -- inconclusive"}

    candidates.sort(key=lambda c: c[1])  # ascending mean steps-to-target
    best_lr, best_mean, best_spread, best_n = candidates[0]
    sorted_lrs = sorted(lrs)
    result = {
        "lr": best_lr, "mean_steps": best_mean, "spread": best_spread,
        "n_seeds_converged": best_n, "note": "ok",
    }
    if best_lr in (sorted_lrs[0], sorted_lrs[-1]):
        result["note"] = "best lr at grid edge -- widen the grid, treat as inconclusive"
    elif len(candidates) > 1:
        _, second_mean, second_spread, _ = candidates[1]
        if abs(second_mean - best_mean) <= (best_spread + second_spread):
            result["note"] = "tied with next-best lr within seed noise -- inconclusive"
    return result


def summarize_sweep(results: list[RunResult], threshold: float) -> dict:
    """threshold selects which entry of each run's steps_to_target dict to summarize --
    callers should run this once per tracked threshold and compare, rather than trusting
    a single cutoff (see train.py's module docstring for why)."""
    by_group = defaultdict(list)
    lrs_by_param_width = defaultdict(set)
    for r in results:
        param, width, lr = r.config["parametrization"], r.config["width"], r.config["base_lr"]
        by_group[(param, width, lr)].append(r.steps_to_target[threshold])
        lrs_by_param_width[(param, width)].add(lr)

    group_stats = {k: _mean_steps(v) for k, v in by_group.items()}

    per_param_width = defaultdict(dict)
    for (param, width), lrs in lrs_by_param_width.items():
        per_param_width[param][width] = _best_lr_for_width(group_stats, param, width, list(lrs))

    summary = {}
    for param, per_width in per_param_width.items():
        widths = sorted(per_width)
        valid_widths = [w for w in widths if per_width[w]["note"] == "ok"]
        drift = None
        if len(valid_widths) >= 2:
            log_lrs = [math.log10(per_width[w]["lr"]) for w in valid_widths]
            drift = max(log_lrs) - min(log_lrs)
        # Surfaced explicitly per review: a drift computed from fewer than all swept
        # widths (because some got gated out as edge/tied-in-noise) should say so rather
        # than silently reporting a number as if every width contributed.
        summary[param] = {
            "per_width": per_width,
            "log10_drift_decades": drift,
            "widths_used_in_drift": valid_widths,
            "widths_gated_out": [w for w in widths if w not in valid_widths],
        }

    verdict = None
    sp_drift = summary.get("sp", {}).get("log10_drift_decades")
    mup_drift = summary.get("mup", {}).get("log10_drift_decades")
    if sp_drift is not None and mup_drift is not None:
        if sp_drift == 0 and mup_drift == 0:
            # Both arms flat means no discrimination happened at all -- not a muP win.
            # (Previously this fell through to ratio=inf, which auto-passed the bar; a
            # review caught that a genuinely-flat muP run would falsely "pass" this way.)
            verdict = {"inconclusive": "both arms show zero drift -- grid/task isn't discriminating"}
        else:
            ratio = math.inf if mup_drift == 0 else sp_drift / mup_drift
            verdict = {
                "sp_drift_decades": sp_drift,
                "mup_drift_decades": mup_drift,
                "drift_ratio": ratio,
                "passes_preregistered_bar (>=3x)": ratio >= EFFECT_SIZE_BAR,
            }
    else:
        verdict = {"inconclusive": "not enough widths cleared the sanity gate for both arms"}
    summary["verdict"] = verdict
    return summary
