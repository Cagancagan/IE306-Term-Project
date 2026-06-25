import numpy as np
import torch
import torch.nn.functional as F

from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from drone_dispatch_env.baselines import GreedyNearest

from b_utils import flatten_obs
from b_a2c_model import ActorCritic


torch.manual_seed(0)
np.random.seed(0)

config = Config.from_yaml("configs/eval_standard.yaml")
env = DroneDispatchEnv(config)
expert = GreedyNearest(config)

states = []
masks = []
actions = []

for seed in range(100):
    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = expert.act(obs)

        states.append(flatten_obs(obs))
        masks.append(obs["action_mask"].copy())
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

states = torch.tensor(np.array(states), dtype=torch.float32)
masks = torch.tensor(np.array(masks), dtype=torch.bool)
actions = torch.tensor(np.array(actions), dtype=torch.long)

print("Samples:", len(actions))

model = ActorCritic()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 256
epochs = 100

for epoch in range(epochs):
    indices = torch.randperm(len(actions))

    total_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, len(actions), batch_size):
        batch_indices = indices[start:start + batch_size]

        batch_states = states[batch_indices]
        batch_masks = masks[batch_indices]
        batch_actions = actions[batch_indices]

        logits, values = model(batch_states)
        masked_logits = logits.masked_fill(~batch_masks, -1e9)

        loss = F.cross_entropy(masked_logits, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_indices)
        predictions = torch.argmax(masked_logits, dim=1)
        correct += int((predictions == batch_actions).sum())
        total += len(batch_indices)

    print(
        f"Epoch {epoch + 1} | "
        f"Loss: {total_loss / total:.4f} | "
        f"Accuracy: {correct / total:.4f}"
    )

torch.save(model.state_dict(), "a2c_bc.pt")

print("Pretraining finished.")