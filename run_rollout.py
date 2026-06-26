from __future__ import annotations

import argparse
import json

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

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    seeds = [
        int(seed)
        for seed in args.seeds.split(",")
        if seed.strip()
    ]

    policy = ShortHorizonRolloutPlanner(
        cfg=cfg,
        planning_depth=args.planning_depth,
        forecast_weight=args.forecast_weight,
    )

    results = evaluate(policy, cfg, seeds)

    print(
        json.dumps(
            {
                "planning_depth": args.planning_depth,
                "forecast_weight": args.forecast_weight,
                "mean": results["mean"],
                "per_seed": results["per_seed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()