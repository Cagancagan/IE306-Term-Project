from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drone_dispatch_env import Config, DroneDispatchMAEnv, GreedyNearest, RandomPolicy, MILPRolling, evaluate

ACCEPT, MOVE, CHARGE, STAY = 0, 1, 2, 3


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.data: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.data.append((
            np.asarray(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            float(done),
        ))

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.data, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.as_tensor(np.stack(obs), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1),
            torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.stack(next_obs), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )


class IDQNPolicy:
    """Shared-parameter Independent DQN policy for DroneDispatchMA-v0."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        hidden: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        buffer_size: int = 100_000,
        target_tau: float = 0.01,
        device: Optional[str] = None,
        safety_threshold: float = 0.30,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_tau = float(target_tau)
        self.safety_threshold = float(safety_threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_net = QNetwork(self.obs_dim, self.action_dim, hidden=hidden).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.action_dim, hidden=hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.learn_steps = 0
        self.hidden = int(hidden)
        self.lr = float(lr)

    @staticmethod
    def _status(obs: np.ndarray) -> int:
        # env observation uses status / 4.0
        return int(round(float(obs[3]) * 4.0))

    def heuristic_action(self, obs: np.ndarray) -> int:
        """Simple decentralized heuristic used as baseline and safety fallback."""
        soc = float(obs[2])
        status = self._status(obs)
        near_order = obs[4:8]
        has_near_order = bool(np.linalg.norm(near_order) > 1e-6)

        if soc < self.safety_threshold:
            return CHARGE
        if status in (1, 2, 3):  # to_pickup, to_dropoff, to_charger
            return MOVE
        if status == 4:  # charging
            return STAY
        if has_near_order:
            return ACCEPT
        return STAY

    def act(self, obs: np.ndarray, epsilon: float = 0.0, use_safety: bool = True) -> int:
        if use_safety:
            soc = float(obs[2])
            status = self._status(obs)
            if soc < self.safety_threshold:
                return CHARGE
            if status in (1, 2, 3):
                return MOVE
            if status == 4:
                return STAY

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(x).squeeze(0).cpu().numpy()
        return int(np.argmax(q))

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)
        q_sa = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        with torch.no_grad():
            for tp, qp in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - self.target_tau).add_(self.target_tau * qp.data)

        self.learn_steps += 1
        return float(loss.item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden": self.hidden,
            "lr": self.lr,
            "gamma": self.gamma,
            "target_tau": self.target_tau,
            "safety_threshold": self.safety_threshold,
            "state_dict": self.q_net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> "IDQNPolicy":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        policy = cls(
            obs_dim=int(ckpt["obs_dim"]),
            action_dim=int(ckpt.get("action_dim", 4)),
            hidden=int(ckpt.get("hidden", 128)),
            lr=float(ckpt.get("lr", 5e-4)),
            gamma=float(ckpt.get("gamma", 0.99)),
            target_tau=float(ckpt.get("target_tau", 0.01)),
            safety_threshold=float(ckpt.get("safety_threshold", 0.30)),
            device=device,
        )
        policy.q_net.load_state_dict(ckpt["state_dict"])
        policy.target_net.load_state_dict(policy.q_net.state_dict())
        policy.q_net.eval()
        policy.target_net.eval()
        return policy


def make_env(config_path: str) -> DroneDispatchMAEnv:
    cfg = Config.from_yaml(config_path)
    return DroneDispatchMAEnv(cfg)


def run_ma_episode(env: DroneDispatchMAEnv, policy: Optional[IDQNPolicy], seed: int, epsilon: float = 0.0,
                   mode: str = "learned") -> Dict[str, float]:
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    while True:
        actions: Dict[str, int] = {}
        for agent_id, agent_obs in obs.items():
            if mode == "heuristic" or policy is None:
                # Build a small temporary policy wrapper only for heuristic thresholds.
                if policy is not None:
                    actions[agent_id] = policy.heuristic_action(agent_obs)
                else:
                    # Equivalent heuristic without learned network.
                    soc = float(agent_obs[2])
                    status = int(round(float(agent_obs[3]) * 4.0))
                    near_order = agent_obs[4:8]
                    if soc < 0.30:
                        actions[agent_id] = CHARGE
                    elif status in (1, 2, 3):
                        actions[agent_id] = MOVE
                    elif status == 4:
                        actions[agent_id] = STAY
                    elif np.linalg.norm(near_order) > 1e-6:
                        actions[agent_id] = ACCEPT
                    else:
                        actions[agent_id] = STAY
            elif mode == "random":
                actions[agent_id] = random.randrange(4)
            else:
                actions[agent_id] = policy.act(agent_obs, epsilon=epsilon)

        next_obs, rewards, terms, truncs, _infos = env.step(actions)
        total_reward += float(sum(rewards.values()))
        steps += 1
        obs = next_obs
        if all(terms.values()) or all(truncs.values()):
            break

    lost_drones = int(sum(1 for d in env.drones if d.lost))
    pending_orders = int(len(env.pending))
    avg_soc = float(np.mean([d.soc for d in env.drones])) if env.drones else 0.0
    return {
        "episode_return": total_reward,
        "steps": float(steps),
        "lost_drones": float(lost_drones),
        "pending_orders": float(pending_orders),
        "avg_soc": avg_soc,
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.config)
    obs_dim = int(env.single_observation_space.shape[0])
    policy = IDQNPolicy(
        obs_dim=obs_dim,
        hidden=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_tau=args.target_tau,
        safety_threshold=args.safety_threshold,
    )

    logs_path = Path(args.log_path)
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path = Path(args.weights_path)

    epsilon = args.epsilon_start
    rows: List[Dict[str, float]] = []
    best_return = -1e18

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        losses: List[float] = []
        steps = 0

        while True:
            actions = {agent_id: policy.act(agent_obs, epsilon=epsilon) for agent_id, agent_obs in obs.items()}
            next_obs, rewards, terms, truncs, _infos = env.step(actions)

            for agent_id, agent_obs in obs.items():
                done = bool(terms[agent_id] or truncs[agent_id])
                policy.memory.push(agent_obs, actions[agent_id], rewards[agent_id], next_obs[agent_id], done)

            for _ in range(args.updates_per_step):
                loss = policy.learn()
                if loss is not None:
                    losses.append(loss)

            ep_return += float(sum(rewards.values()))
            epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
            steps += 1
            obs = next_obs

            if all(terms.values()) or all(truncs.values()):
                break

        lost_drones = int(sum(1 for d in env.drones if d.lost))
        avg_loss = float(np.mean(losses)) if losses else 0.0
        row = {
            "episode": ep,
            "episode_return": ep_return,
            "epsilon": epsilon,
            "loss": avg_loss,
            "lost_drones": lost_drones,
            "steps": steps,
            "buffer_size": len(policy.memory),
        }
        rows.append(row)

        if ep_return > best_return:
            best_return = ep_return
            policy.save(weights_path)

        if ep % args.print_every == 0:
            print(
                f"idqn-ma ep={ep:4d} return={ep_return:9.2f} "
                f"lost={lost_drones} eps={epsilon:.3f} loss={avg_loss:.4f} best={best_return:.2f}"
            )

    with logs_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved log: {logs_path}")
    print(f"saved weights: {weights_path}")


def mean_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    keys = items[0].keys()
    return {k: float(np.mean([x[k] for x in items])) for k in keys}


def eval_ma(args: argparse.Namespace) -> None:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    learned_policy: Optional[IDQNPolicy] = None
    if args.weights and Path(args.weights).exists():
        learned_policy = IDQNPolicy.load(Path(args.weights))

    results: Dict[str, Dict[str, float]] = {}
    for mode in ["random", "heuristic", "learned"]:
        if mode == "learned" and learned_policy is None:
            continue
        metrics = []
        for seed in seeds:
            env = make_env(args.config)
            metrics.append(run_ma_episode(env, learned_policy, seed=seed, epsilon=0.0, mode=mode))
        results[f"ma_{mode}"] = mean_metrics(metrics)

    # Centralized reference on the standard dispatch env. This is not the same env API,
    # but gives the required centralized-vs-decentralized comparison point.
    cfg = Config.from_yaml(args.config)
    central = {
        "central_random": evaluate(RandomPolicy(cfg, seed=0), cfg, seeds=seeds)["mean"],
        "central_greedy_nearest": evaluate(GreedyNearest(cfg), cfg, seeds=seeds)["mean"],
        "central_milp_rolling": evaluate(MILPRolling(cfg), cfg, seeds=seeds)["mean"],
    }

    print("\n=== Multi-agent DroneDispatchMA-v0 ===")
    for name, m in results.items():
        print(
            f"{name:<16} return={m['episode_return']:.2f} "
            f"lost_drones={m['lost_drones']:.2f} pending={m['pending_orders']:.1f} avg_soc={m['avg_soc']:.3f}"
        )

    print("\n=== Centralized reference on DroneDispatch-v0 ===")
    for name, m in central.items():
        print(
            f"{name:<24} cost/order={m['cost_per_order']:.4f} "
            f"return={m['episode_return']:.2f} delivered={m['n_delivered']:.1f}"
        )

    out = {
        "config": args.config,
        "seeds": seeds,
        "multi_agent": results,
        "centralized_reference": central,
        "note": (
            "DroneDispatchMA-v0 uses decentralized per-drone actions and exposes different "
            "metrics from the centralized dispatcher. The comparison is therefore qualitative: "
            "centralized policies are shown as reference scores on DroneDispatch-v0, while IDQN/heuristic "
            "are evaluated end-to-end on DroneDispatchMA-v0."
        ),
    }
    out_path = Path(args.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved results: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shared-parameter IDQN for DroneDispatchMA-v0.")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--config", default="configs/eval_standard.yaml")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--episodes", type=int, default=100)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--buffer-size", type=int, default=100_000)
    tr.add_argument("--target-tau", type=float, default=0.01)
    tr.add_argument("--updates-per-step", type=int, default=1)
    tr.add_argument("--epsilon-start", type=float, default=1.0)
    tr.add_argument("--epsilon-min", type=float, default=0.05)
    tr.add_argument("--epsilon-decay", type=float, default=0.9995)
    tr.add_argument("--safety-threshold", type=float, default=0.30)
    tr.add_argument("--print-every", type=int, default=10)
    tr.add_argument("--weights-path", default="weights/idqn_ma.pt")
    tr.add_argument("--log-path", default="logs/idqn_ma_training.csv")
    tr.set_defaults(func=train)

    ev = sub.add_parser("eval")
    ev.add_argument("--config", default="configs/eval_standard.yaml")
    ev.add_argument("--weights", default="weights/idqn_ma.pt")
    ev.add_argument("--seeds", default="0,1,2,3,4")
    ev.add_argument("--results-path", default="logs/multi_agent_results.json")
    ev.set_defaults(func=eval_ma)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
