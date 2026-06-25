import gymnasium as gym
import drone_dispatch_env
import numpy as np
import torch

from b_utils import flatten_obs
from b_model import PolicyNetwork


def choose_best_action(model, obs):
    state = flatten_obs(obs)

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)

    with torch.no_grad():
        logits = model(state_tensor)
        masked_logits = logits.masked_fill(~mask_tensor, -1e9)
        action = torch.argmax(masked_logits, dim=1)

    return int(action.item())


model = PolicyNetwork()
model.load_state_dict(torch.load("reinforce_test.pt", map_location="cpu"))
model.eval()

rewards = []

for seed in [100, 101, 102]:
    env = gym.make("DroneDispatch-v0")
    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action = choose_best_action(model, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    rewards.append(total_reward)
    print(f"Seed {seed} | Reward: {total_reward:.2f} | Info: {info}")

print("Mean reward:", float(np.mean(rewards)))
print("Std reward:", float(np.std(rewards)))