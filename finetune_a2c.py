import csv
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
model.load_state_dict(torch.load("a2c_bc.pt", map_location="cpu"))

optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)

gamma = 0.99
gae_lambda = 0.95
episodes = 500
value_coef = 0.5
entropy_coef = 0.002

best_reward = -float("inf")

with open("finetune_a2c_log.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["episode", "reward", "steps", "loss"])

    for episode in range(episodes):
        obs, info = env.reset(seed=1000 + episode)

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
        next_value = torch.tensor(0.0)
        gae = torch.tensor(0.0)

        for index in reversed(range(len(rewards))):
            current_value = values[index].detach()
            delta = rewards[index] + gamma * next_value - current_value
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            next_value = current_value

        advantages_tensor = torch.stack(advantages)
        values_tensor = torch.stack(values)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        returns_tensor = advantages_tensor + values_tensor.detach()
        normalized_advantages = (
            advantages_tensor - advantages_tensor.mean()
        ) / (advantages_tensor.std() + 1e-8)

        policy_loss = -(
            log_probs_tensor * normalized_advantages.detach()
        ).mean()

        value_loss = F.mse_loss(
            values_tensor,
            returns_tensor.detach()
        )

        entropy_bonus = entropies_tensor.mean()

        loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy_bonus
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_reward = float(sum(rewards))

        writer.writerow([
            episode + 1,
            total_reward,
            len(rewards),
            float(loss.item())
        ])

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(
                model.state_dict(),
                "a2c_finetuned.pt"
            )

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {len(rewards)} | "
                f"Loss: {loss.item():.2f} | "
                f"Best: {best_reward:.2f}"
            )

print("Fine-tuning finished.")