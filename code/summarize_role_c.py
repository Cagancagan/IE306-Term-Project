from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_json(path: str) -> dict:
    """Load PowerShell-created JSON files safely, including UTF-8 BOM."""
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def metric_mean_std(data: dict, metric: str) -> tuple[float, float | None]:
    """
    Return mean and standard deviation.

    Rollout JSON files contain:
        {"mean": {...}, "per_seed": [...]}

    Greedy JSON files created by run_eval contain:
        {"cost_per_order": ..., ...}
    """
    if "per_seed" in data:
        values = [float(row[metric]) for row in data["per_seed"]]
        return float(np.mean(values)), float(np.std(values, ddof=1))

    if "mean" in data and metric in data["mean"]:
        return float(data["mean"][metric]), None

    if metric in data:
        return float(data[metric]), None

    raise KeyError(f"Metric '{metric}' was not found in the JSON file.")


def fmt(mean: float, std: float | None, decimals: int = 3) -> str:
    """Format metrics consistently for the terminal summary."""
    if std is None:
        return f"{mean:.{decimals}f} (mean)"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    files = {
        "Greedy nearest (10 seeds)": "logs/greedy_nearest_full_seed0to9.json",
        "Role C rollout depth 1 (10 seeds)": (
            "logs/rollout_depth1_weight1_seed0to9.json"
        ),
        "Rollout depth 1 (3 seeds)": (
            "logs/rollout_depth1_weight1_seed012.json"
        ),
        "Rollout depth 2 (3 seeds)": (
            "logs/rollout_depth2_weight1_seed012.json"
        ),
        "Rollout depth 3 (3 seeds)": (
            "logs/rollout_depth3_weight1_seed012.json"
        ),
    }

    print("\nROLE C RESULTS SUMMARY\n")
    print(
        f"{'Method':<35}"
        f"{'Cost/order':>20}"
        f"{'Delivered':>20}"
        f"{'Dropped':>20}"
        f"{'On-time rate':>20}"
    )
    print("-" * 115)

    for label, filename in files.items():
        path = Path(filename)

        if not path.exists():
            print(f"{label:<35} MISSING FILE: {filename}")
            continue

        data = load_json(filename)

        cost_mean, cost_std = metric_mean_std(data, "cost_per_order")
        delivered_mean, delivered_std = metric_mean_std(data, "n_delivered")
        dropped_mean, dropped_std = metric_mean_std(data, "n_dropped")
        ontime_mean, ontime_std = metric_mean_std(data, "ontime_rate")

        print(
            f"{label:<35}"
            f"{fmt(cost_mean, cost_std):>20}"
            f"{fmt(delivered_mean, delivered_std, 2):>20}"
            f"{fmt(dropped_mean, dropped_std, 2):>20}"
            f"{fmt(ontime_mean, ontime_std):>20}"
        )


if __name__ == "__main__":
    main()