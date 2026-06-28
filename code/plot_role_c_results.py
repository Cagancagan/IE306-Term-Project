from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def values(data: dict, metric: str) -> list[float]:
    if "per_seed" in data:
        return [float(row[metric]) for row in data["per_seed"]]

    if metric in data:
        return [float(data[metric])]

    if "mean" in data:
        return [float(data["mean"][metric])]

    raise KeyError(metric)


def mean_std(data: dict, metric: str) -> tuple[float, float]:
    x = values(data, metric)
    if len(x) == 1:
        return x[0], 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))


def main():
    Path("figures").mkdir(exist_ok=True)

    greedy = load_json("logs/greedy_nearest_full_seed0to9.json")
    rollout = load_json("logs/rollout_depth1_weight1_seed0to9.json")

    methods = ["Greedy nearest", "Role C look-ahead"]
    cost_means = [
        mean_std(greedy, "cost_per_order")[0],
        mean_std(rollout, "cost_per_order")[0],
    ]
    cost_stds = [
        mean_std(greedy, "cost_per_order")[1],
        mean_std(rollout, "cost_per_order")[1],
    ]

    plt.figure()
    plt.bar(methods, cost_means, yerr=cost_stds, capsize=6)
    plt.ylabel("Mean cost per delivered order")
    plt.title("Role C vs Greedy Nearest (10 Seeds)")
    plt.tight_layout()
    plt.savefig("figures/role_c_vs_greedy_cost.png", dpi=300)
    plt.close()

    ablation_files = {
        "Depth 1": "logs/rollout_depth1_weight1_seed012.json",
        "Depth 2": "logs/rollout_depth2_weight1_seed012.json",
        "Depth 3": "logs/rollout_depth3_weight1_seed012.json",
    }

    labels = list(ablation_files.keys())
    means = []
    stds = []

    for label in labels:
        data = load_json(ablation_files[label])
        mean, std = mean_std(data, "cost_per_order")
        means.append(mean)
        stds.append(std)

    plt.figure()
    plt.bar(labels, means, yerr=stds, capsize=6)
    plt.ylabel("Mean cost per delivered order")
    plt.title("Role C Planning-Depth Ablation (3 Seeds)")
    plt.tight_layout()
    plt.savefig("figures/role_c_depth_ablation.png", dpi=300)
    plt.close()

    print("Saved figures/role_c_vs_greedy_cost.png")
    print("Saved figures/role_c_depth_ablation.png")


if __name__ == "__main__":
    main()