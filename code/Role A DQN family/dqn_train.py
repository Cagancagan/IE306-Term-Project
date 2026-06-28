import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import gymnasium as gym
import drone_dispatch_env
import yaml
import torch
import csv
import numpy as np
from dqn_agent import DQNAgent

def flatten_obs(obs_dict):
    drones = np.array(obs_dict["drones"]).flatten()
    orders = np.array(obs_dict["orders"]).flatten()
    time = np.array(obs_dict["time"]).flatten()
    grid = np.array(obs_dict["grid"]).flatten()
    return np.concatenate([drones, orders, time, grid])

with open("../../configs/dqn.yaml", "r") as f: config = yaml.safe_load(f)
my_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
env = gym.make(config["env_id"])
obs_sample, info_sample = env.reset(seed=my_seed)

agent = DQNAgent(config)
agent.memory.buffer.clear()
sample_state = flatten_obs(obs_sample)
config["obs_dim"] = len(sample_state)
config["action_dim"] = env.action_space.n
logs = []
best_reward = -float('inf')

print(f"DQN Eğitimi Başlıyor (Seed: {my_seed})...")
for episode in range(config["total_episodes"]):
    obs, info = env.reset()
    total_reward, done = 0, False
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = flatten_obs(obs)
        next_state = flatten_obs(next_obs)
        agent.memory.push(state, action, reward, next_state, done)
        agent.learn()

        obs = next_obs
        total_reward += reward

    logs.append([episode, total_reward, info.get("metrics", {}).get("cost_per_order", 0)])
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.q_net.state_dict(), f"../../weights/best_dqn_seed{my_seed}.pt")

    if episode % 10 == 0: print(f"Bölüm: {episode} | Toplam Ödül: {total_reward:.2f} | Rekor: {best_reward:.2f}")

with open(f"../../logs/dqn_seed{my_seed}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "cost_per_order"])
    writer.writerows(logs)
print(f"Eğitim tamamlandı! En iyi model kaydedildi.")