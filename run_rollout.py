from __future__ import annotations

import argparse
import json
import yaml

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate
from code.rollout_planner import ShortHorizonRolloutPlanner


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Role C short-horizon rollout planner."
    )

    parser.add_argument(
        "--config",
        default="configs/eval_standard.yaml",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
    )
    parser.add_argument(
        "--planning-depth",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--forecast-weight",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--planner-config",
        default=None,
        help="Optional YAML file containing Role C rollout planner parameters.",
    )

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    planner_config = {}
    if args.planner_config is not None:
        with open(args.planner_config, "r", encoding="utf-8") as file:
            planner_config = yaml.safe_load(file) or {}

    planning_depth = int(
        planner_config.get("planning_depth", args.planning_depth)
    )
    forecast_weight = float(
        planner_config.get("forecast_weight", args.forecast_weight)
    )

    seeds_text = planner_config.get("evaluation_seeds", args.seeds)
    seeds = [
        int(seed)
        for seed in str(seeds_text).split(",")
        if str(seed).strip()
    ]

    policy = ShortHorizonRolloutPlanner(
        cfg=cfg,
        planning_depth=planning_depth,
        forecast_weight=forecast_weight,
    )

    results = evaluate(policy, cfg, seeds)

    print(
        json.dumps(
            {
                "planning_depth": planning_depth,
                "forecast_weight": forecast_weight,
                "mean": results["mean"],
                "per_seed": results["per_seed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()