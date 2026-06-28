from __future__ import annotations

import argparse
import json

from drone_dispatch_env.baselines import make_baseline
from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline policy and save mean plus per-seed metrics."
    )
    parser.add_argument(
        "--config",
        default="configs/eval_standard.yaml",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4,5,6,7,8,9",
    )
    parser.add_argument(
        "--policy",
        default="greedy_nearest",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    seeds = [
        int(seed)
        for seed in args.seeds.split(",")
        if seed.strip()
    ]

    policy = make_baseline(args.policy, cfg)
    results = evaluate(policy, cfg, seeds)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()