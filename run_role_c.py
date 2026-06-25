from __future__ import annotations

import argparse
import json

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate
from code.role_c_planner import RoleCPlanner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]

    policy = RoleCPlanner(cfg)
    results = evaluate(policy, cfg, seeds)

    print(json.dumps(results["mean"], indent=2))


if __name__ == "__main__":
    main()