"""Unified, mask-aware DQN / Double DQN / Dueling DQN for DroneDispatch-v0.

Usage examples from repo root:
  python 'code/Role A DQN family/role_a_dqn_fixed.py' train --algo double --seed 42 --episodes 300
  python 'code/Role A DQN family/role_a_dqn_fixed.py' eval --algo double --weights weights/role_a_double_seed42.pt --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import deque
from dataclasses import asdict
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Important: run this script from its own folder or by file path. We import torch
# before adding the repo root, because the repo has code/__init__.py which shadows
# Python's standard-library code module and can break torch/pytest from repo root.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import gymnasium as gym
import drone_dispatch_env  # registers envs
from drone_dispatch_env import Config, GreedyNearest, RandomPolicy, MILPRolling, evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_obs(obs: Dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    """Single source of truth for train/act/eval preprocessing.

    Normalize coordinates/age/grid so the MLP does not have to learn scale
    differences between x/y positions, one-hot status, age, and grid codes.
    """
    scale_xy = float(max(cfg.H - 1, cfg.W - 1, 1))
    drones = np.asarray(obs["drones"], dtype=np.float32).copy()
    drones[:, 0:2] /= scale_xy  # x,y
    # soc/alive/status one-hot/has_order are already small-scale

    orders = np.asarray(obs["orders"], dtype=np.float32).copy()
    orders[:, 0:4] /= scale_xy  # origin/destination coords
    orders[:, 4] /= float(max(cfg.T_max, 1))  # age

    grid = np.asarray(obs["grid"], dtype=np.float32).reshape(-1) / 3.0
    time = np.asarray(obs["time"], dtype=np.float32).reshape(-1)
    return np.concatenate([drones.reshape(-1), orders.reshape(-1), grid, time]).astype(np.float32)


def valid_mask(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(obs["action_mask"], dtype=np.float32)

def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1])))


def nearest_hub_distance(pos: np.ndarray, grid: np.ndarray) -> float:
    hubs = np.argwhere(grid == 3)
    if len(hubs) == 0:
        return 0.0

    # np.argwhere returns [row, col]. In this simulator x/y are used as grid-like coordinates.
    return min(
        abs(float(pos[0]) - float(hub[0])) + abs(float(pos[1]) - float(hub[1]))
        for hub in hubs
    )


def policy_mask(
    obs: Dict[str, np.ndarray],
    cfg: Config,
    safety_threshold: float = 0.35,
    mission_safety: bool = True,
    battery_reserve: float = 0.15,
    e_move: float = 0.01,
) -> np.ndarray:
    """Env action_mask + battery-safety rules.

    Rule 1:
        Low-SoC idle drones are not allowed to accept new assignments.

    Rule 2:
        If enabled, an assignment is allowed only when the drone appears to have
        enough battery for:
            drone -> pickup -> dropoff -> nearest charger
        plus a reserve margin.

    This is a conservative safety prior, not a simulator rule.
    """
    mask = valid_mask(obs).copy()
    drones = np.asarray(obs["drones"], dtype=np.float32)
    orders = np.asarray(obs["orders"], dtype=np.float32)
    grid = np.asarray(obs["grid"])

    for d in range(cfg.n_drones):
        soc = float(drones[d, 2])
        drone_pos = drones[d, 0:2]

        # Rule 1: simple low-battery threshold.
        if safety_threshold is not None and safety_threshold > 0 and soc < safety_threshold:
            start = d * cfg.k_max
            end = start + cfg.k_max
            mask[start:end] = 0.0
            continue

        # Rule 2: distance-aware mission safety.
        if mission_safety:
            for k in range(cfg.k_max):
                action = d * cfg.k_max + k

                # If env says the action is already invalid, leave it invalid.
                if mask[action] <= 0:
                    continue

                order = orders[k]
                pickup = order[0:2]
                dropoff = order[2:4]

                dist_to_pickup = manhattan(drone_pos, pickup)
                dist_delivery = manhattan(pickup, dropoff)
                dist_to_hub = nearest_hub_distance(dropoff, grid)

                required_soc = e_move * (dist_to_pickup + dist_delivery + dist_to_hub) + battery_reserve

                if soc < required_soc:
                    mask[action] = 0.0

    # no-op remains available as fallback
    mask[cfg.noop_index] = 1.0
    return mask

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.asarray(next_masks), dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256, dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.feature = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
            self.advantage = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, action_dim))
        else:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(x)
        z = self.feature(x)
        v = self.value(z)
        a = self.advantage(z)
        return v + (a - a.mean(dim=1, keepdim=True))


class DQNPolicy:
    def __init__(self, cfg: Config, obs_dim: int, action_dim: int, algo: str = "double",
                 lr: float = 5e-4, gamma: float = 0.99, batch_size: int = 64,
                 buffer_size: int = 100_000, tau: float = 0.005, hidden: int = 256,
                 epsilon_decay: float = 0.9995, safety_threshold: float = 0.35,
                 mission_safety: bool = True, battery_reserve: float = 0.15):
        assert algo in {"dqn", "double", "dueling"}
        self.cfg = cfg
        self.algo = algo
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        self.safety_threshold = safety_threshold
        self.mission_safety = mission_safety
        self.battery_reserve = battery_reserve
        dueling = algo == "dueling"
        self.q_net = QNetwork(obs_dim, action_dim, hidden=hidden, dueling=dueling)
        self.target_net = QNetwork(obs_dim, action_dim, hidden=hidden, dueling=dueling)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    def act(self, obs: Dict[str, np.ndarray]) -> int:
        mask = policy_mask(
            obs,
            self.cfg,
            safety_threshold=self.safety_threshold,
            mission_safety=self.mission_safety,
            battery_reserve=self.battery_reserve,
        )
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return int(self.cfg.noop_index)
        if random.random() < self.epsilon:
            return int(random.choice(valid))
        state = torch.as_tensor(flatten_obs(obs, self.cfg), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(state).squeeze(0).cpu().numpy()
        q[mask == 0] = -1e9
        return int(q.argmax())

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(self.batch_size)
        q_sa = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            if self.algo == "dqn":
                next_q = self.target_net(next_states).masked_fill(~next_masks, -1e9)
                next_q_max = next_q.max(dim=1, keepdim=True).values
            else:
                online_next_q = self.q_net(next_states).masked_fill(~next_masks, -1e9)
                next_actions = online_next_q.argmax(dim=1, keepdim=True)
                next_q_max = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q_max * (1.0 - dones)
        loss = nn.SmoothL1Loss()(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        with torch.no_grad():
            for tp, qp in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * qp.data)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.learn_steps += 1
        return float(loss.item())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "algo": self.algo,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "cfg": asdict(self.cfg),
            "state_dict": self.q_net.state_dict(),
            "epsilon_decay": self.epsilon_decay,
            "safety_threshold": self.safety_threshold,
            "mission_safety": self.mission_safety,
            "battery_reserve": self.battery_reserve,
        }, path)

    @classmethod
    def load(cls, path: str, cfg: Config):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(
            cfg,
            ckpt["obs_dim"],
            ckpt["action_dim"],
            algo=ckpt.get("algo", "double"),
            epsilon_decay=ckpt.get("epsilon_decay", 0.9995),
            safety_threshold=ckpt.get("safety_threshold", 0.35),
            mission_safety=ckpt.get("mission_safety", True),
            battery_reserve=ckpt.get("battery_reserve", 0.15),
        )
        agent.q_net.load_state_dict(ckpt["state_dict"])
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        agent.epsilon = 0.0
        return agent


def metrics_from_env(env, ep_return: float) -> Dict[str, float]:
    s = env.unwrapped.stats if hasattr(env, "unwrapped") else env.stats
    delivered = max(s["delivered"], 0)
    total_orders = delivered + s["dropped"]
    denom = delivered if delivered > 0 else 1
    cost = s["energy"] + s["late_cost"] + s["drop_cost"] + s["depletion_cost"]
    return {
        "cost_per_order": cost / denom,
        "success_rate": delivered / total_orders if total_orders else 0.0,
        "ontime_rate": s["ontime"] / delivered if delivered else 0.0,
        "energy_per_order": s["energy"] / denom,
        "n_delivered": float(delivered),
        "n_dropped": float(s["dropped"]),
        "depletion_events": float(s.get("depletion_events", 0)),
        "episode_return": float(ep_return),
    }


def make_agent(algo: str, seed: int, args) -> Tuple[DQNPolicy, Config]:
    cfg = Config.from_yaml(str(ROOT / args.config))
    env = gym.make("DroneDispatch-v0", config=cfg)
    obs, _ = env.reset(seed=seed)
    obs_dim = len(flatten_obs(obs, cfg))
    action_dim = env.action_space.n
    env.close()
    return DQNPolicy(
        cfg, obs_dim, action_dim,
        algo=algo,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        tau=args.tau,
        hidden=args.hidden,
        epsilon_decay=args.epsilon_decay,
        safety_threshold=args.safety_threshold,
        mission_safety=not args.no_mission_safety,
        battery_reserve=args.battery_reserve,
    ), cfg


def train(args) -> None:
    set_seed(args.seed)
    agent, cfg = make_agent(args.algo, args.seed, args)
    env = gym.make("DroneDispatch-v0", config=cfg)
    os.makedirs(ROOT / "logs", exist_ok=True)
    os.makedirs(ROOT / "weights", exist_ok=True)
    log_path = str(ROOT / f"logs/role_a_{args.algo}_seed{args.seed}.csv")
    weight_path = args.out or str(ROOT / f"weights/role_a_{args.algo}_seed{args.seed}.pt")
    best_cost = float("inf")
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed * 100000 + ep)
        done = False
        ep_return = 0.0
        losses = []
        action_counts = {"assign": 0, "charge": 0, "noop": 0}
        while not done:
            action = agent.act(obs)
            decoded = cfg.decode(action)[0]
            action_counts[decoded] = action_counts.get(decoded, 0) + 1
            next_obs, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            agent.memory.push(
                flatten_obs(obs, cfg), action, reward,
                flatten_obs(next_obs, cfg), done,
                policy_mask(
                    next_obs,
                    cfg,
                    safety_threshold=agent.safety_threshold,
                    mission_safety=agent.mission_safety,
                    battery_reserve=agent.battery_reserve,
                ),
            )
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            ep_return += reward
            obs = next_obs
        m = metrics_from_env(env, ep_return)
        row = {
            "episode": ep,
            "epsilon": agent.epsilon,
            "loss": float(np.mean(losses)) if losses else np.nan,
            **action_counts,
            **m,
        }
        rows.append(row)
        if m["n_delivered"] > 0 and m["cost_per_order"] < best_cost:
            best_cost = m["cost_per_order"]
            agent.save(weight_path)
        if ep % args.print_every == 0:
            print(f"{args.algo} seed={args.seed} ep={ep:4d} return={ep_return:8.2f} "
                  f"cost/order={m['cost_per_order']:6.3f} delivered={m['n_delivered']:4.0f} "
                  f"eps={agent.epsilon:.3f} best_cost={best_cost:.3f}")
    if rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if not os.path.exists(weight_path):
        agent.save(weight_path)
    print(f"saved log: {log_path}")
    print(f"saved weights: {weight_path}")


def eval_policy(args) -> None:
    cfg = Config.from_yaml(str(ROOT / args.config))
    seeds = [int(s) for s in args.seeds.split(",") if s]
    agent = DQNPolicy.load(args.weights, cfg)
    res = evaluate(agent, cfg, seeds=seeds)
    print(f"\nRole A {args.algo}:")
    for k, v in res["mean"].items():
        print(f"  {k}: {v:.4f}")
    print("\nBaselines:")
    for name, pol in [
        ("random", RandomPolicy(cfg, seed=0)),
        ("greedy_nearest", GreedyNearest(cfg)),
        ("milp_rolling", MILPRolling(cfg)),
    ]:
        r = evaluate(pol, cfg, seeds=seeds)["mean"]
        print(f"  {name:<15} cost_per_order={r['cost_per_order']:.4f} return={r['episode_return']:.2f} delivered={r['n_delivered']:.1f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/eval_standard.yaml")
    common.add_argument("--algo", choices=["dqn", "double", "dueling"], default="double")
    tr = sub.add_parser("train", parents=[common])
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--episodes", type=int, default=300)
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--buffer-size", type=int, default=100_000)
    tr.add_argument("--tau", type=float, default=0.005)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--epsilon-decay", type=float, default=0.9995)
    tr.add_argument("--safety-threshold", type=float, default=0.35)
    tr.add_argument("--battery-reserve", type=float, default=0.15)
    tr.add_argument("--no-mission-safety", action="store_true")
    tr.add_argument("--print-every", type=int, default=10)
    tr.add_argument("--out", default=None)
    ev = sub.add_parser("eval", parents=[common])
    ev.add_argument("--weights", required=True)
    ev.add_argument("--seeds", default="0,1,2,3,4")
    args = p.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        eval_policy(args)


if __name__ == "__main__":
    main()
