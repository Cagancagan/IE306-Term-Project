import copy
import random
from collections import deque

import gymnasium as gym
import drone_dispatch_env
import numpy as np
import torch
import torch.nn.functional as F

from ddpg_model import Actor, Critic


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = map(
            np.array,
            zip(*batch)
        )

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

env = gym.make("DroneControl-v0")

actor = Actor()
critic = Critic()

target_actor = copy.deepcopy(actor)
target_critic = copy.deepcopy(critic)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

buffer = ReplayBuffer()

gamma = 0.99
tau = 0.005
batch_size = 128
warmup_steps = 1000
total_steps = 20000
noise_std = 0.15

state, info = env.reset(seed=0)

episode_reward = 0.0
episode_count = 0
best_reward = -float("inf")

for step in range(1, total_steps + 1):
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        action = actor(state_tensor).squeeze(0).numpy()

    if step < warmup_steps:
        action = env.action_space.sample()
    else:
        noise = np.random.normal(0, noise_std, size=2)
        action = action + noise
        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high
        )

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    buffer.add(state, action, reward, next_state, float(done))

    state = next_state
    episode_reward += reward

    if len(buffer) >= batch_size:
        states, actions, rewards, next_states, dones = buffer.sample(
            batch_size
        )

        with torch.no_grad():
            next_actions = target_actor(next_states)
            target_q = target_critic(next_states, next_actions)
            target = rewards + gamma * (1 - dones) * target_q

        current_q = critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_loss = -critic(states, actor(states)).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        for target_param, param in zip(
            target_actor.parameters(),
            actor.parameters()
        ):
            target_param.data.copy_(
                tau * param.data
                + (1 - tau) * target_param.data
            )

        for target_param, param in zip(
            target_critic.parameters(),
            critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data
                + (1 - tau) * target_param.data
            )

    if done:
        episode_count += 1

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(actor.state_dict(), "ddpg_actor.pt")
            torch.save(critic.state_dict(), "ddpg_critic.pt")

        if episode_count % 10 == 0:
            print(
                f"Episode {episode_count} | "
                f"Reward: {episode_reward:.2f} | "
                f"Best: {best_reward:.2f} | "
                f"Step: {step}"
            )

        state, info = env.reset()
        episode_reward = 0.0

print("DDPG training finished.")