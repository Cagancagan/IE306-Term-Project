import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gymnasium as gym
import drone_dispatch_env
import yaml
import torch
import numpy as np

from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from drone_dispatch_env import GreedyNearest, evaluate, Config

with open("../../configs/dqn.yaml", "r") as f:
    config = yaml.safe_load(f)

env = gym.make(config["env_id"])
obs_sample, info_sample = env.reset(seed=42)
state_key = [k for k in obs_sample.keys() if k != "action_mask"][0]
config["state_key"] = state_key
config["obs_dim"] = int(np.prod(np.array(obs_sample[state_key]).shape))
config["action_dim"] = env.action_space.n

def evaluate_agent(agent_name, agent, weight_path):
    print(f"\n--- {agent_name} Sınavı Başlıyor ---")
    try:
        agent.q_net.load_state_dict(torch.load(weight_path))
        print(f"BEYİN YÜKLENDİ: {weight_path}")
    except FileNotFoundError:
        print(f"HATA: {weight_path} bulunamadı! Atlanıyor.")
        return None

    agent.epsilon = 0.0
    agent.epsilon_min = 0.0
    obs, info = env.reset(seed=42)
    total_reward = 0
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f">>> {agent_name} Gerçek Skoru: {total_reward:.2f}")
    return total_reward

# Checkpoint isimlerine göre yolları güncelledik (seed=42 için)
agents_to_test = {
    "Klasik DQN": (DQNAgent(config), "../../weights/best_dqn_seed42.pt"),
    "Double DQN": (DoubleDQNAgent(config), "../../weights/best_double_dqn_seed42.pt"),
    "Dueling DQN": (DuelingDQNAgent(config), "../../weights/best_dueling_dqn_seed42.pt")
}

results = {}
for name, (agent, path) in agents_to_test.items():
    score = evaluate_agent(name, agent, path)
    if score is not None:
        results[name] = score

print("\n--- Greedy (Açgözlü) Baseline Sınavı Başlıyor ---")
cfg = Config()
for k, v in config.items():
    if hasattr(cfg, k):
        setattr(cfg, k, v)

greedy_policy = GreedyNearest(cfg)
greedy_results = evaluate(greedy_policy, cfg, seeds=[42])["mean"]
greedy_score = greedy_results["episode_return"]
results["GreedyNearest"] = greedy_score
print(f">>> Greedy Gerçek Skoru: {greedy_score:.2f}")

print("\n" + "="*35)
print("🏆 NİHAİ SINAV SONUÇLARI 🏆")
print("="*35)
for name, score in results.items():
    print(f"{name:<15} : {score:.2f}")
print("="*35)