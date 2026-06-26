import gymnasium as gym
import drone_dispatch_env
import numpy as np
import torch

from ddpg_model import Actor


def evaluate_actor(actor, seeds):
    rewards = []
    successes = []

    for seed in seeds:
        env = gym.make("DroneControl-v0")
        state, info = env.reset(seed=seed)

        total_reward = 0.0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                action = actor(state_tensor).squeeze(0).numpy()

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        successes.append(float(terminated))

        print(
            f"Seed {seed} | "
            f"Reward: {total_reward:.2f} | "
            f"Info: {info}"
        )

    print()
    print("Mean reward:", float(np.mean(rewards)))
    print("Std reward:", float(np.std(rewards)))
    print("Success rate:", float(np.mean(successes)))


actor = Actor()
actor.load_state_dict(torch.load("ddpg_actor.pt", map_location="cpu"))
actor.eval()

evaluate_actor(actor, [100, 101, 102, 103, 104])