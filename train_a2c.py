import gymnasium as gym
import drone_dispatch_env
import numpy as np
import torch
import torch.nn.functional as F

from b_utils import flatten_obs
from b_a2c_model import ActorCritic, choose_a2c_action


torch.manual_seed(0)
np.random.seed(0)

env = gym.make("DroneDispatch-v0")

model = ActorCritic()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

gamma = 0.99
gae_lambda = 0.95
episodes = 300
value_coef = 0.5
entropy_coef = 0.01

for episode in range(episodes):
    obs, info = env.reset(seed=episode)

    log_probs = []
    values = []
    rewards = []
    entropies = []

    terminated = False
    truncated = False

    while not terminated and not truncated:
        state = flatten_obs(obs)

        action, log_prob, value, entropy = choose_a2c_action(
            model,
            state,
            obs["action_mask"]
        )

        obs, reward, terminated, truncated, info = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy)

    advantages = []
    returns = []
    gae = 0.0
    next_value = 0.0

    for index in reversed(range(len(rewards))):
        value = values[index].detach()
        delta = rewards[index] + gamma * next_value - value
        gae = delta + gamma * gae_lambda * gae

        advantages.insert(0, gae)
        returns.insert(0, gae + value)

        next_value = value

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    values_tensor = torch.stack(values)
    log_probs_tensor = torch.stack(log_probs)
    entropies_tensor = torch.stack(entropies)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = -(log_probs_tensor * advantages.detach()).mean()
    value_loss = F.mse_loss(values_tensor, returns.detach())
    entropy_bonus = entropies_tensor.mean()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    if (episode + 1) % 10 == 0:
        print(
            f"Episode {episode + 1} | "
            f"Reward: {sum(rewards):.2f} | "
            f"Steps: {len(rewards)} | "
            f"Loss: {loss.item():.2f}"
        )

torch.save(model.state_dict(), "a2c_test.pt")

print("A2C training finished.")