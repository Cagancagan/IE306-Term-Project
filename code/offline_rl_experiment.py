from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from drone_dispatch_env import Config, GreedyNearest, RandomPolicy, MILPRolling, evaluate
from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from pairwise_features import build_pairwise_features
from pairwise_model import PairwisePolicy
from rollout_planner import ShortHorizonRolloutPlanner

ROLE_A_SCRIPT = ROOT / "code" / "Role A DQN family" / "role_a_dqn_fixed.py"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_obs(obs: Dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    """Offline-RL observation representation.

    It matches the Role A DQN representation: normalized drones, orders, grid, and time.
    """
    scale_xy = float(max(cfg.H - 1, cfg.W - 1, 1))
    drones = np.asarray(obs["drones"], dtype=np.float32).copy()
    drones[:, 0:2] /= scale_xy

    orders = np.asarray(obs["orders"], dtype=np.float32).copy()
    orders[:, 0:4] /= scale_xy
    orders[:, 4] /= float(max(cfg.T_max, 1))

    grid = np.asarray(obs["grid"], dtype=np.float32).reshape(-1) / 3.0
    time = np.asarray(obs["time"], dtype=np.float32).reshape(-1)
    return np.concatenate([drones.reshape(-1), orders.reshape(-1), grid, time]).astype(np.float32)


class PairwiseAdapter:
    def __init__(self, weight_path: Path):
        self.model = PairwisePolicy()
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        self.model.eval()

    def act(self, obs) -> int:
        pair_features, charge_features = build_pairwise_features(obs)
        pair_tensor = torch.tensor(pair_features, dtype=torch.float32)
        charge_tensor = torch.tensor(charge_features, dtype=torch.float32)
        mask_tensor = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(pair_tensor, charge_tensor).masked_fill(~mask_tensor, -1e9)
            return int(torch.argmax(scores, dim=1).item())


class EpsilonPolicy:
    """Add random valid-action noise around another behavior policy."""

    def __init__(self, base_policy, cfg: Config, seed: int, epsilon: float = 0.15):
        self.base_policy = base_policy
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.epsilon = float(epsilon)

    def act(self, obs) -> int:
        valid = np.flatnonzero(np.asarray(obs["action_mask"], dtype=bool))
        if len(valid) == 0:
            return int(self.cfg.noop_index)
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))
        return int(self.base_policy.act(obs))


def load_role_a_policy(weight_path: Path, cfg: Config):
    role_a = load_module_from_path("role_a_dqn_fixed", ROLE_A_SCRIPT)
    return role_a.DQNPolicy.load(str(weight_path), cfg)


def make_role_c_policy(cfg: Config):
    cfg_path = ROOT / "configs" / "rollout_role_c.yaml"
    planning_depth = 1
    forecast_weight = 1.0
    if cfg_path.exists():
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        planning_depth = int(params.get("planning_depth", planning_depth))
        forecast_weight = float(params.get("forecast_weight", forecast_weight))
    return ShortHorizonRolloutPlanner(cfg, planning_depth=planning_depth, forecast_weight=forecast_weight)


def make_behavior_pool(cfg: Config, seed: int, names: Iterable[str]) -> List[Tuple[str, Any]]:
    pool: List[Tuple[str, Any]] = []
    for name in names:
        key = name.strip().lower()
        if not key:
            continue
        if key == "random":
            pool.append(("random", RandomPolicy(cfg, seed=seed)))
        elif key == "greedy":
            pool.append(("greedy", GreedyNearest(cfg)))
        elif key == "milp":
            pool.append(("milp", MILPRolling(cfg)))
        elif key == "role_a":
            path = ROOT / "weights" / "role_a_dueling_seed2.pt"
            if path.exists():
                pool.append(("role_a", load_role_a_policy(path, cfg)))
            else:
                print(f"[WARN] Missing Role A weights at {path}; skipping.")
        elif key == "role_b":
            path = ROOT / "weights" / "pairwise_milp_seed0.pt"
            if path.exists():
                pool.append(("role_b", PairwiseAdapter(path)))
            else:
                print(f"[WARN] Missing Role B weights at {path}; skipping.")
        elif key == "role_c":
            pool.append(("role_c", make_role_c_policy(cfg)))
        elif key.startswith("noisy_"):
            base_name = key.replace("noisy_", "", 1)
            base_pool = make_behavior_pool(cfg, seed=seed, names=[base_name])
            if base_pool:
                base_label, base_policy = base_pool[0]
                pool.append((f"noisy_{base_label}", EpsilonPolicy(base_policy, cfg, seed=seed, epsilon=0.20)))
        else:
            raise ValueError(f"Unknown behavior policy name: {name}")
    if not pool:
        raise ValueError("Behavior pool is empty. Check --behaviors and weight files.")
    return pool


def generate_dataset(
    cfg: Config,
    out_path: Path,
    min_transitions: int,
    base_seed: int,
    behaviors: List[str],
) -> Dict[str, Any]:
    env = DroneDispatchEnv(cfg)
    obs_l: List[np.ndarray] = []
    act_l: List[int] = []
    rew_l: List[float] = []
    nobs_l: List[np.ndarray] = []
    term_l: List[bool] = []
    tout_l: List[bool] = []
    mask_l: List[np.ndarray] = []
    nmask_l: List[np.ndarray] = []
    ep_ret_l: List[float] = []
    behavior_l: List[str] = []

    seed = int(base_seed)
    behavior_names = list(behaviors)
    pool = make_behavior_pool(cfg, seed=base_seed, names=behavior_names)
    rng = np.random.default_rng(base_seed)
    episode = 0

    while len(act_l) < min_transitions:
        label, base_policy = pool[episode % len(pool)]
        # Rebuild stochastic policies by seed so their random streams vary by episode.
        if label == "random":
            policy = RandomPolicy(cfg, seed=seed)
        elif label.startswith("noisy_"):
            base_label = label.replace("noisy_", "", 1)
            base_policy = make_behavior_pool(cfg, seed=seed, names=[base_label])[0][1]
            policy = EpsilonPolicy(base_policy, cfg, seed=seed, epsilon=0.20)
        else:
            policy = base_policy

        obs, _ = env.reset(seed=seed)
        done = False
        ep_return = 0.0
        while not done and len(act_l) < min_transitions:
            action = int(policy.act(obs))
            next_obs, reward, term, trunc, _info = env.step(action)
            obs_l.append(flatten_obs(obs, cfg))
            act_l.append(action)
            rew_l.append(float(reward))
            nobs_l.append(flatten_obs(next_obs, cfg))
            term_l.append(bool(term))
            tout_l.append(bool(trunc))
            mask_l.append(np.asarray(obs["action_mask"], dtype=bool))
            nmask_l.append(np.asarray(next_obs["action_mask"], dtype=bool))
            ep_return += float(reward)
            obs = next_obs
            done = bool(term or trunc)
        ep_ret_l.append(ep_return)
        behavior_l.append(label)
        seed += 1
        episode += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=np.asarray(obs_l, dtype=np.float32),
        actions=np.asarray(act_l, dtype=np.int64),
        rewards=np.asarray(rew_l, dtype=np.float32),
        next_observations=np.asarray(nobs_l, dtype=np.float32),
        terminals=np.asarray(term_l, dtype=bool),
        timeouts=np.asarray(tout_l, dtype=bool),
        action_masks=np.asarray(mask_l, dtype=bool),
        next_action_masks=np.asarray(nmask_l, dtype=bool),
        episode_returns=np.asarray(ep_ret_l, dtype=np.float32),
        behavior_labels=np.asarray(behavior_l),
    )
    return {
        "path": str(out_path),
        "n_transitions": int(len(act_l)),
        "n_episodes": int(len(ep_ret_l)),
        "behaviors": behavior_l,
        "episode_return_mean": float(np.mean(ep_ret_l)) if ep_ret_l else 0.0,
    }


class OfflineQNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OfflinePolicy:
    def __init__(self, net: nn.Module, cfg: Config, use_action_mask: bool = True):
        self.net = net
        self.cfg = cfg
        self.use_action_mask = use_action_mask
        self.net.eval()

    def act(self, obs) -> int:
        x = torch.tensor(flatten_obs(obs, self.cfg), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.net(x).squeeze(0).cpu().numpy()
        if self.use_action_mask:
            mask = np.asarray(obs["action_mask"], dtype=bool)
            q[~mask] = -1e9
        return int(np.argmax(q))


class DatasetTensors:
    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.obs = torch.tensor(arrays["observations"], dtype=torch.float32)
        self.actions = torch.tensor(arrays["actions"], dtype=torch.long).unsqueeze(1)
        self.rewards = torch.tensor(arrays["rewards"], dtype=torch.float32).unsqueeze(1)
        self.next_obs = torch.tensor(arrays["next_observations"], dtype=torch.float32)
        done_bool = np.asarray(arrays["terminals"], dtype=bool) | np.asarray(arrays["timeouts"], dtype=bool)
        self.dones = torch.tensor(done_bool.astype(np.float32), dtype=torch.float32).unsqueeze(1)
        self.masks = torch.tensor(arrays.get("action_masks", np.ones((len(arrays["actions"]), 169), dtype=bool)), dtype=torch.bool)
        self.next_masks = torch.tensor(arrays.get("next_action_masks", np.ones((len(arrays["actions"]), 169), dtype=bool)), dtype=torch.bool)
        self.n = int(len(arrays["actions"]))
        self.obs_dim = int(self.obs.shape[1])
        self.action_dim = int(max(int(self.actions.max().item()) + 1, self.masks.shape[1]))

    def sample_indices(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.n, (batch_size,))


@dataclass
class TrainResult:
    name: str
    metrics: Dict[str, float]
    loss_last: float
    weight_path: str


def train_bc(data: DatasetTensors, cfg: Config, steps: int, batch_size: int, lr: float, out_path: Path) -> TrainResult:
    net = OfflineQNet(data.obs_dim, cfg.n_actions)
    opt = optim.Adam(net.parameters(), lr=lr)
    loss_value = 0.0
    for step in range(steps):
        idx = data.sample_indices(batch_size)
        logits = net(data.obs[idx]).masked_fill(~data.masks[idx], -1e9)
        loss = nn.CrossEntropyLoss()(logits, data.actions[idx].squeeze(1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        loss_value = float(loss.item())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path)
    return TrainResult("bc", {}, loss_value, str(out_path))


def train_offline_dqn(
    data: DatasetTensors,
    cfg: Config,
    steps: int,
    batch_size: int,
    lr: float,
    gamma: float,
    cql_alpha: float,
    out_path: Path,
    conservative: bool,
) -> TrainResult:
    net = OfflineQNet(data.obs_dim, cfg.n_actions)
    target = OfflineQNet(data.obs_dim, cfg.n_actions)
    target.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=lr)
    loss_value = 0.0
    tau = 0.01

    for step in range(steps):
        idx = data.sample_indices(batch_size)
        obs = data.obs[idx]
        actions = data.actions[idx]
        rewards = data.rewards[idx]
        next_obs = data.next_obs[idx]
        dones = data.dones[idx]
        next_masks = data.next_masks[idx]

        q = net(obs)
        q_sa = q.gather(1, actions)

        with torch.no_grad():
            next_q = target(next_obs)
            if conservative:
                next_q = next_q.masked_fill(~next_masks, -1e9)
            # For naive offline DQN, deliberately leave next_q unmasked to expose
            # over-optimistic OOD/invalid action values.
            next_max = next_q.max(dim=1, keepdim=True).values
            bellman_target = rewards + gamma * next_max * (1.0 - dones)

        bellman_loss = nn.SmoothL1Loss()(q_sa, bellman_target)
        if conservative:
            cql_loss = torch.logsumexp(q, dim=1, keepdim=True).mean() - q_sa.mean()
            loss = bellman_loss + cql_alpha * cql_loss
        else:
            loss = bellman_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

        with torch.no_grad():
            for tp, qp in zip(target.parameters(), net.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * qp.data)
        loss_value = float(loss.item())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path)
    return TrainResult("cql" if conservative else "naive_offline_dqn", {}, loss_value, str(out_path))


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def evaluate_net_policy(weight_path: Path, cfg: Config, seeds: List[int], use_action_mask: bool) -> Dict[str, Any]:
    # Observation dimension is known from the saved dataset-independent Role A representation.
    # Build using one reset to infer obs_dim.
    env = DroneDispatchEnv(cfg)
    obs, _ = env.reset(seed=0)
    obs_dim = len(flatten_obs(obs, cfg))
    net = OfflineQNet(obs_dim, cfg.n_actions)
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    policy = OfflinePolicy(net, cfg, use_action_mask=use_action_mask)
    return evaluate(policy, cfg, seeds=seeds)


def print_eval_row(name: str, result: Dict[str, Any]) -> None:
    m = result["mean"]
    print(
        f"{name:<20} cost/order={m['cost_per_order']:.4f} "
        f"return={m['episode_return']:.2f} delivered={m['n_delivered']:.1f} "
        f"depletion={m['depletion_events']:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal offline RL experiment: BC, naive offline DQN, and CQL.")
    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--dataset", default="logs/offline_dataset.npz")
    parser.add_argument("--min-transitions", type=int, default=30000)
    parser.add_argument("--base-seed", type=int, default=5000)
    parser.add_argument("--behaviors", default="random,greedy,noisy_greedy,role_a,role_b,role_c")
    parser.add_argument("--train-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cql-alpha", type=float, default=0.2)
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument("--mode", choices=["all", "generate", "train-eval"], default="all")
    parser.add_argument("--summary", default="logs/offline_rl_results.json")
    args = parser.parse_args()

    set_seed(args.base_seed)
    cfg = Config.from_yaml(str(ROOT / args.config))
    dataset_path = ROOT / args.dataset
    eval_seeds = parse_seeds(args.eval_seeds)
    summary: Dict[str, Any] = {"config": args.config, "eval_seeds": eval_seeds}

    if args.mode in {"all", "generate"} or not dataset_path.exists():
        print("\n=== Generating offline dataset ===")
        dataset_info = generate_dataset(
            cfg=cfg,
            out_path=dataset_path,
            min_transitions=args.min_transitions,
            base_seed=args.base_seed,
            behaviors=[x.strip() for x in args.behaviors.split(",") if x.strip()],
        )
        summary["dataset"] = dataset_info
        print(json.dumps(dataset_info, indent=2))

    if args.mode == "generate":
        out = ROOT / args.summary
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {out}")
        return

    print("\n=== Loading dataset ===")
    arrays = load_dataset(dataset_path)
    data = DatasetTensors(arrays)
    summary.setdefault("dataset", {})
    summary["dataset"].update({
        "path": str(dataset_path),
        "n_transitions": int(data.n),
        "obs_dim": int(data.obs_dim),
        "action_dim": int(cfg.n_actions),
    })
    print(json.dumps(summary["dataset"], indent=2))

    weight_dir = ROOT / "weights"
    results: Dict[str, Any] = {}

    print("\n=== Training offline policies ===")
    bc = train_bc(data, cfg, args.train_steps, args.batch_size, args.lr, weight_dir / "offline_bc.pt")
    naive = train_offline_dqn(
        data, cfg, args.train_steps, args.batch_size, args.lr, args.gamma,
        args.cql_alpha, weight_dir / "offline_naive_dqn.pt", conservative=False,
    )
    cql = train_offline_dqn(
        data, cfg, args.train_steps, args.batch_size, args.lr, args.gamma,
        args.cql_alpha, weight_dir / "offline_cql.pt", conservative=True,
    )
    summary["training"] = {
        "bc_loss_last": bc.loss_last,
        "naive_loss_last": naive.loss_last,
        "cql_loss_last": cql.loss_last,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "cql_alpha": args.cql_alpha,
    }

    print("\n=== Evaluating offline policies ===")
    evals = {
        "behavioral_cloning": evaluate_net_policy(weight_dir / "offline_bc.pt", cfg, eval_seeds, use_action_mask=True),
        "naive_offline_dqn": evaluate_net_policy(weight_dir / "offline_naive_dqn.pt", cfg, eval_seeds, use_action_mask=False),
        "cql": evaluate_net_policy(weight_dir / "offline_cql.pt", cfg, eval_seeds, use_action_mask=True),
    }
    for name, result in evals.items():
        print_eval_row(name, result)
        results[name] = result

    summary["results"] = results
    out = ROOT / args.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved offline RL summary to {out}")


if __name__ == "__main__":
    main()
