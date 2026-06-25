import gymnasium as gym
import drone_dispatch_env
import torch

from pairwise_features import build_pairwise_features
from pairwise_model import PairwisePolicy


env = gym.make("DroneDispatch-v0")
obs, info = env.reset(seed=0)

pair_features, charge_features = build_pairwise_features(obs)

model = PairwisePolicy()

pair_tensor = torch.tensor(pair_features, dtype=torch.float32)
charge_tensor = torch.tensor(charge_features, dtype=torch.float32)

scores = model(pair_tensor, charge_tensor)

mask = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
masked_scores = scores.masked_fill(~mask, -1e9)

print("Pair feature shape:", pair_tensor.shape)
print("Expected pair feature shape: torch.Size([1, 160, 11])")
print("Charge feature shape:", charge_tensor.shape)
print("Score shape:", scores.shape)
print("Valid actions:", int(mask.sum()))
print("Best valid action:", int(torch.argmax(masked_scores, dim=1).item()))