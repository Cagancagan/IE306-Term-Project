import gymnasium as gym
import drone_dispatch_env

env = gym.make("DroneDispatch-v0")
obs, info = env.reset(seed=0)

for key, value in obs.items():
    print(key, "->", getattr(value, "shape", None))

print("valid actions:", int(obs["action_mask"].sum()))
print("total actions:", len(obs["action_mask"]))
print("action space:", env.action_space)