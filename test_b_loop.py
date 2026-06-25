import gymnasium as gym
import drone_dispatch_env
import numpy as np

from b_utils import flatten_obs, choose_random_valid_action


env = gym.make("DroneDispatch-v0")
obs, info = env.reset(seed=0)

state = flatten_obs(obs)

print("State shape:", state.shape)
print("Expected state size: 741")
print("First valid action count:", int(obs["action_mask"].sum()))

rng = np.random.default_rng(0)

total_reward = 0.0
terminated = False
truncated = False
steps = 0

while not terminated and not truncated:
    action = choose_random_valid_action(obs, rng)
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1

print("Episode finished.")
print("Decision steps:", steps)
print("Total reward:", total_reward)