from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drone_dispatch_env import Config, RandomPolicy, GreedyNearest, MILPRolling, evaluate


ROLE_A_SCRIPT = ROOT / "code" / "Role A DQN family" / "role_a_dqn_fixed.py"


def load_role_a_module():
    spec = importlib.util.spec_from_file_location("role_a_dqn_fixed", ROLE_A_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Role A module from {ROLE_A_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_seeds(seed_string: str) -> List[int]:
    return [int(s.strip()) for s in seed_string.split(",") if s.strip()]


def print_result(name: str, result: dict) -> None:
    mean = result["mean"]
    print(
        f"{name:<24} "
        f"cost/order={mean['cost_per_order']:.4f} "
        f"return={mean['episode_return']:.2f} "
        f"delivered={mean['n_delivered']:.1f} "
        f"depletion={mean['depletion_events']:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Role A DQN-family policies.")
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--include-ablation",
        action="store_true",
        help="Also evaluate target-network ON/OFF ablation checkpoints.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(str(ROOT / args.config))
    seeds = parse_seeds(args.seeds)
    role_a = load_role_a_module()

    print("\n=== Role A evaluation ===")
    print(f"Config: {args.config}")
    print(f"Seeds:  {args.seeds}\n")

    baselines = [
        ("random", RandomPolicy(cfg, seed=0)),
        ("greedy_nearest", GreedyNearest(cfg)),
        ("milp_rolling", MILPRolling(cfg)),
    ]

    for name, policy in baselines:
        print_result(name, evaluate(policy, cfg, seeds=seeds))

    policies = [
        ("DQN seed0", ROOT / "weights" / "role_a_dqn_seed0.pt"),
        ("DQN seed1", ROOT / "weights" / "role_a_dqn_seed1.pt"),
        ("DQN seed2", ROOT / "weights" / "role_a_dqn_seed2.pt"),
        ("Double DQN seed0", ROOT / "weights" / "role_a_double_seed0.pt"),
        ("Double DQN seed1", ROOT / "weights" / "role_a_double_seed1.pt"),
        ("Double DQN seed2", ROOT / "weights" / "role_a_double_seed2.pt"),
        ("Dueling DQN seed0", ROOT / "weights" / "role_a_dueling_seed0.pt"),
        ("Dueling DQN seed1", ROOT / "weights" / "role_a_dueling_seed1.pt"),
        ("Dueling DQN seed2", ROOT / "weights" / "role_a_dueling_seed2.pt"),
    ]

    if args.include_ablation:
        policies.extend(
            [
                ("Dueling target ON", ROOT / "weights" / "role_a_dueling_seed0_target_on.pt"),
                ("Dueling target OFF", ROOT / "weights" / "role_a_dueling_seed0_target_off.pt"),
            ]
        )

    print()

    for name, weight_path in policies:
        if not weight_path.exists():
            print(f"{name:<24} missing weights: {weight_path}")
            continue

        policy = role_a.DQNPolicy.load(str(weight_path), cfg)
        print_result(name, evaluate(policy, cfg, seeds=seeds))


if __name__ == "__main__":
    main()