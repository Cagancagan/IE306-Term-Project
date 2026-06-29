from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from drone_dispatch_env import Config, GreedyNearest, MILPRolling, RandomPolicy, evaluate
from pairwise_features import build_pairwise_features
from pairwise_model import PairwisePolicy
from rollout_planner import ShortHorizonRolloutPlanner

ROLE_A_SCRIPT = ROOT / "code" / "Role A DQN family" / "role_a_dqn_fixed.py"


def parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PairwiseAdapter:
    """Role B Pairwise MILP-guided dispatcher adapter."""

    def __init__(self, weight_path: Path):
        self.model = PairwisePolicy()
        state = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def act(self, obs) -> int:
        pair_features, charge_features = build_pairwise_features(obs)
        pair_tensor = torch.tensor(pair_features, dtype=torch.float32)
        charge_tensor = torch.tensor(charge_features, dtype=torch.float32)
        mask_tensor = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(pair_tensor, charge_tensor)
            scores = scores.masked_fill(~mask_tensor, -1e9)
            action = torch.argmax(scores, dim=1)
        return int(action.item())

    def action_probs(self, obs):
        pair_features, charge_features = build_pairwise_features(obs)
        pair_tensor = torch.tensor(pair_features, dtype=torch.float32)
        charge_tensor = torch.tensor(charge_features, dtype=torch.float32)
        mask_tensor = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(pair_tensor, charge_tensor)
            scores = scores.masked_fill(~mask_tensor, -1e9)
            probs = torch.softmax(scores, dim=1)
        return probs.squeeze(0).numpy()


def load_role_a_policy(weight_path: Path, cfg: Config):
    role_a = load_module_from_path("role_a_dqn_fixed", ROLE_A_SCRIPT)
    return role_a.DQNPolicy.load(str(weight_path), cfg)


def load_policies(cfg: Config) -> List[Tuple[str, Any]]:
    """Centralized DroneDispatch-v0 policies included in the final table."""
    policies: List[Tuple[str, Any]] = [
        ("random", RandomPolicy(cfg, seed=0)),
        ("greedy_nearest", GreedyNearest(cfg)),
        ("milp_rolling", MILPRolling(cfg)),
    ]

    learned_specs: List[Tuple[str, str, Path]] = [
        ("Role A DQN seed0", "role_a", ROOT / "weights" / "role_a_dqn_seed0.pt"),
        ("Role A Double DQN seed0", "role_a", ROOT / "weights" / "role_a_double_seed0.pt"),
        ("Role A Dueling DQN seed2", "role_a", ROOT / "weights" / "role_a_dueling_seed2.pt"),
        ("Role B Pairwise MILP seed0", "pairwise", ROOT / "weights" / "pairwise_milp_seed0.pt"),
        ("Role C short-horizon rollout", "role_c", ROOT / "configs" / "rollout_role_c.yaml"),
    ]

    for label, kind, path in learned_specs:
        if not path.exists():
            print(f"[WARN] Skipping {label}: missing {path}")
            continue
        if kind == "role_a":
            policies.append((label, load_role_a_policy(path, cfg)))
        elif kind == "pairwise":
            policies.append((label, PairwiseAdapter(path)))
        elif kind == "role_c":
            # configs/rollout_role_c.yaml currently stores depth=1 and forecast_weight=1.0.
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                params = yaml.safe_load(f) or {}
            policies.append(
                (
                    label,
                    ShortHorizonRolloutPlanner(
                        cfg=cfg,
                        planning_depth=int(params.get("planning_depth", 1)),
                        forecast_weight=float(params.get("forecast_weight", 1.0)),
                    ),
                )
            )
    return policies


def summarize_per_seed(per_seed: List[Dict[str, float]], metric: str) -> Tuple[float, float]:
    values = np.array([float(row[metric]) for row in per_seed], dtype=np.float64)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), std


def print_table(rows: List[Tuple[str, Dict[str, Any]]]) -> None:
    print("\n=== Final evaluation table ===")
    print(
        f"{'Policy':<32} {'cost/order':>11} {'return':>11} "
        f"{'delivered':>11} {'dropped':>9} {'depletion':>10}"
    )
    print("-" * 92)
    for name, result in rows:
        m = result["mean"]
        print(
            f"{name:<32} "
            f"{m['cost_per_order']:>11.4f} "
            f"{m['episode_return']:>11.2f} "
            f"{m['n_delivered']:>11.1f} "
            f"{m['n_dropped']:>9.1f} "
            f"{m['depletion_events']:>10.1f}"
        )

    print("\n=== Markdown table ===")
    print("| Policy | cost/order | return | delivered | dropped | depletion |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, result in rows:
        m = result["mean"]
        print(
            f"| {name} | {m['cost_per_order']:.4f} | {m['episode_return']:.2f} | "
            f"{m['n_delivered']:.1f} | {m['n_dropped']:.1f} | {m['depletion_events']:.1f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final centralized-policy evaluation table.")
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--json-out", default=None, help="Optional path to save full JSON results.")
    args = parser.parse_args()

    cfg = Config.from_yaml(str(ROOT / args.config))
    seeds = parse_seeds(args.seeds)
    policies = load_policies(cfg)

    rows: List[Tuple[str, Dict[str, Any]]] = []
    print("\n=== IE306 centralized dispatch evaluation ===")
    print(f"Config: {args.config}")
    print(f"Seeds:  {','.join(map(str, seeds))}")

    for name, policy in policies:
        print(f"\nEvaluating {name}...")
        result = evaluate(policy, cfg, seeds=seeds)
        rows.append((name, result))
        m = result["mean"]
        c_mean, c_std = summarize_per_seed(result["per_seed"], "cost_per_order")
        print(
            f"  cost/order={m['cost_per_order']:.4f} "
            f"(per-seed std={c_std:.4f}), return={m['episode_return']:.2f}"
        )

    print_table(rows)

    if args.json_out:
        out_path = ROOT / args.json_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: result for name, result in rows}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON results to {out_path}")


if __name__ == "__main__":
    main()
