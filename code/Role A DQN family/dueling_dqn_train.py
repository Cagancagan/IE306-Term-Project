import sys
import gymnasium as gym
import drone_dispatch_env
import yaml
import torch
import csv
import numpy as np
from dueling_dqn_agent import DuelingDQNAgent

with open("../../configs/duel_dqn.yaml", "r") as f:
    config = yaml.safe_load(f)

my_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

env = gym.make(config["env_id"])
obs_sample, info_sample = env.reset(seed=my_seed)

state_key = [k for k in obs_sample.keys() if k != "action_mask"][0]
print(f"Çevreden gelen asıl veri anahtarı bulundu: '{state_key}'")

sample_state = np.array(obs_sample[state_key])
config["state_key"] = state_key
config["obs_dim"] = int(np.prod(sample_state.shape))
config["action_dim"] = env.action_space.n

agent = DuelingDQNAgent(config)
logs = []

print(f"Dueling DQN Eğitimi başlıyor... Toplam {config['total_episodes']} bölüm sürecek.")
for episode in range(config["total_episodes"]):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = np.array(obs[state_key]).flatten()
        next_state = np.array(next_obs[state_key]).flatten()

        agent.memory.push(state, action, reward, next_state, done)
        agent.learn()

        obs = next_obs
        total_reward += reward

    logs.append([episode, total_reward, info.get("metrics", {}).get("cost_per_order", 0)])
    if episode % 10 == 0:
        print(f"Bölüm: {episode} | Toplam Ödül: {total_reward:.2f}")

# Ağırlıkların ve logların kaydedilmesi
torch.save(agent.q_net.state_dict(), f"../../weights/dueling_dqn_seed{my_seed}.pt")

with open(f"../../logs/dueling_dqn_seed{my_seed}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "cost_per_order"])
    writer.writerows(logs)

print("Eğitim tamamlandı! Ağırlıklar '../../weights/dueling_dqn.pt' konumuna kaydedildi.")