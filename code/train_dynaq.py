from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from code.dynaq_planner import DynaQConfig, DynaQPlanner


def episode_cost(stats: dict) -> float:
    """Compute the project primary cost before normalizing by deliveries."""
    return (
        stats["energy"]
        + stats["late_cost"]
        + stats["drop_cost"]
        + stats["depletion_cost"]
    )


def train(
    cfg: Config,
    episodes: int,
    seed: int,
    planning_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
):
    env = DroneDispatchEnv(cfg)

    dynaq_cfg = DynaQConfig(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        planning_steps=planning_steps,
    )

    agent = DynaQPlanner(
        cfg=cfg,
        dynaq_cfg=dynaq_cfg,
        seed=seed,
    )

    history = []

    for episode in range(episodes):
        # Exploration starts higher and gradually decreases as the Q-table learns.
        agent.dynaq_cfg.epsilon = max(
            0.02,
            epsilon * (0.995 ** episode),
        )
        obs, _ = env.reset(seed=seed + episode)
        done = False
        ep_return = 0.0

        while not done:
            action = agent.act(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            agent.update(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            obs = next_obs
            ep_return += reward

        delivered = max(int(env.stats["delivered"]), 1)
        total_cost = episode_cost(env.stats)
        cost_per_order = total_cost / delivered

        history.append(
            {
                "episode": episode + 1,
                "epsilon": agent.dynaq_cfg.epsilon,
                "episode_return": ep_return,
                "cost_per_order": cost_per_order,
                "delivered": env.stats["delivered"],
                "dropped": env.stats["dropped"],
                "depletion_events": env.stats["depletion_events"],
            }
        )

        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"Episode {episode + 1:>4}/{episodes} | "
                f"return={ep_return:8.2f} | "
                f"cost/order={cost_per_order:6.3f} | "
                f"delivered={env.stats['delivered']:3d} | "
                f"dropped={env.stats['dropped']:3d}"
            )

    return agent, history


def save_history(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_agent(agent: DynaQPlanner, path: Path) -> None:
    """
    Save plain dictionaries rather than the defaultdict object itself.
    This makes the trained table portable and easy to reload later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "q": dict(agent.q),
        "model": agent.model,
        "dynaq_config": {
            "alpha": agent.dynaq_cfg.alpha,
            "gamma": agent.dynaq_cfg.gamma,
            "epsilon": agent.dynaq_cfg.epsilon,
            "planning_steps": agent.dynaq_cfg.planning_steps,
        },
    }

    with path.open("wb") as file:
        pickle.dump(payload, file)


def main():
    parser = argparse.ArgumentParser(
        description="Train the Role C tabular Dyna-Q planner."
    )

    parser.add_argument("--config", default="configs/eval_standard.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planning-steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.10)

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    agent, history = train(
        cfg=cfg,
        episodes=args.episodes,
        seed=args.seed,
        planning_steps=args.planning_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
    )

    log_path = Path(
        f"logs/dynaq_seed{args.seed}_plan{args.planning_steps}.csv"
    )
    weight_path = Path(
        f"weights/dynaq_seed{args.seed}_plan{args.planning_steps}.pkl"
    )

    save_history(history, log_path)
    save_agent(agent, weight_path)

    print("\nTraining completed.")
    print(f"Saved learning log: {log_path}")
    print(f"Saved Q-table/model: {weight_path}")


if __name__ == "__main__":
    main()