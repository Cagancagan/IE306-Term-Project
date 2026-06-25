import gymnasium as gym
import drone_dispatch_env
import torch

from b_utils import flatten_obs
from b_model import PolicyNetwork


env = gym.make("DroneDispatch-v0")
obs, info = env.reset(seed=0)

state = flatten_obs(obs)

model = PolicyNetwork()

state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

logits = model(state_tensor)

mask = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
masked_logits = logits.masked_fill(~mask, -1e9)

print("State tensor shape:", state_tensor.shape)
print("Logits shape:", logits.shape)
print("Masked logits shape:", masked_logits.shape)
print("Valid action count:", int(mask.sum()))
print("Best valid action:", int(torch.argmax(masked_logits, dim=1).item()))