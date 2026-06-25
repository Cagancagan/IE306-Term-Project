import numpy as np
import torch
import torch.nn.functional as F
import sys

from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from drone_dispatch_env.baselines import MILPRolling

from pairwise_features import build_pairwise_features
from pairwise_model import PairwisePolicy


train_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

torch.manual_seed(train_seed)
np.random.seed(train_seed)

config = Config.from_yaml("configs/eval_standard.yaml")
env = DroneDispatchEnv(config)
expert = MILPRolling(config)

pair_data = []
charge_data = []
mask_data = []
action_data = []

for collected, seed in enumerate(
    range(train_seed * 75, (train_seed + 1) * 75),
    start=1
):
    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = expert.act(obs)

        pair_features, charge_features = build_pairwise_features(obs)

        pair_data.append(pair_features[0])
        charge_data.append(charge_features[0])
        mask_data.append(obs["action_mask"].copy())
        action_data.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
    if collected % 5 == 0:
        print(f"Collected seed {collected}/75")

pair_data = torch.tensor(np.array(pair_data), dtype=torch.float32)
charge_data = torch.tensor(np.array(charge_data), dtype=torch.float32)
mask_data = torch.tensor(np.array(mask_data), dtype=torch.bool)
action_data = torch.tensor(np.array(action_data), dtype=torch.long)

print("Samples:", len(action_data))

model = PairwisePolicy()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 256
epochs = 40

for epoch in range(epochs):
    indices = torch.randperm(len(action_data))

    total_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, len(action_data), batch_size):
        batch_indices = indices[start:start + batch_size]

        batch_pairs = pair_data[batch_indices]
        batch_charges = charge_data[batch_indices]
        batch_masks = mask_data[batch_indices]
        batch_actions = action_data[batch_indices]

        scores = model(batch_pairs, batch_charges)
        masked_scores = scores.masked_fill(~batch_masks, -1e9)

        loss = F.cross_entropy(masked_scores, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_indices)

        predictions = torch.argmax(masked_scores, dim=1)
        correct += int((predictions == batch_actions).sum())
        total += len(batch_indices)

    print(
        f"Epoch {epoch + 1} | "
        f"Loss: {total_loss / total:.4f} | "
        f"Accuracy: {correct / total:.4f}"
    )

torch.save(model.state_dict(), f"pairwise_milp_seed{train_seed}.pt")

print("Pairwise pretraining finished.")