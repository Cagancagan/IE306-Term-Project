import gymnasium as gym
import drone_dispatch_env
import numpy as np
import torch

from b_utils import flatten_obs
from b_model import PolicyNetwork, choose_action


env = gym.make("DroneDispatch-v0")

model = PolicyNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

gamma = 0.99
episodes = 100

for episode in range(episodes):
    obs, info = env.reset(seed=episode)

    log_probs = []
    rewards = []

    terminated = False
    truncated = False

    while not terminated and not truncated:
        state = flatten_obs(obs)

        action, log_prob = choose_action(
            model,
            state,
            obs["action_mask"]
        )

        obs, reward, terminated, truncated, info = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

    returns = []
    discounted_return = 0.0

    for reward in reversed(rewards):
        discounted_return = reward + gamma * discounted_return
        returns.insert(0, discounted_return)

    returns = torch.tensor(returns, dtype=torch.float32)

    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = torch.stack([
        -log_prob * return_value
        for log_prob, return_value in zip(log_probs, returns)
    ]).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 10 == 0:
        print(
            f"Episode {episode + 1} | "
            f"Reward: {sum(rewards):.2f} | "
            f"Steps: {len(rewards)} | "
            f"Loss: {loss.item():.2f}"
        )

torch.save(model.state_dict(), "reinforce_test.pt")

print("Training finished.")