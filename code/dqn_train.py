import gymnasium as gym
import drone_dispatch_env
import yaml
import torch
import csv
import numpy as np
from dqn_agent import DQNAgent

# 1. Ayarları Yükle
with open("../configs/dqn.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Simülatörü Başlat ve Veri Yapısını Çöz
env = gym.make(config["env_id"])
obs_sample, info_sample = env.reset(seed=42)

state_key = [k for k in obs_sample.keys() if k != "action_mask"][0]
print(f"Çevreden gelen asıl veri anahtarı bulundu: '{state_key}'")

# HATA ÇÖZÜMÜ: Veriyi tablo olmaktan çıkarıp düzleştiriyoruz (flatten)
sample_state = np.array(obs_sample[state_key])
config["state_key"] = state_key
config["obs_dim"] = int(np.prod(sample_state.shape))  # 8 ve 10'u çarpıp tam 80 boyutunu bulacak
config["action_dim"] = env.action_space.n

agent = DQNAgent(config)
logs = []

# 3. Eğitim Döngüsü
print("Eğitim başlıyor... Bu biraz zaman alabilir.")
for episode in range(config["total_episodes"]):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Verileri hafızaya atarken dümdüz (flatten) yapıp atıyoruz
        state = np.array(obs[state_key]).flatten()
        next_state = np.array(next_obs[state_key]).flatten()

        agent.memory.push(state, action, reward, next_state, done)
        agent.learn()

        obs = next_obs
        total_reward += reward

    logs.append([episode, total_reward, info.get("metrics", {}).get("cost_per_order", 0)])
    if episode % 10 == 0:
        print(f"Bölüm: {episode} | Toplam Ödül: {total_reward:.2f}")

# 4. Kayıt İşlemleri
torch.save(agent.q_net.state_dict(), "../weights/dqn.pt")

with open("../logs/dqn_seed42.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "cost_per_order"])
    writer.writerows(logs)

print("Eğitim tamamlandı! Ağırlıklar 'weights/' klasörüne kaydedildi.")