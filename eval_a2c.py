import json
import numpy as np
import torch

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate

from b_utils import flatten_obs
from b_a2c_model import ActorCritic


class A2CAdapter:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def act(self, obs):
        state = flatten_obs(obs)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            obs["action_mask"],
            dtype=torch.bool
        ).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(state_tensor)
            masked_logits = logits.masked_fill(~mask_tensor, -1e9)
            action = torch.argmax(masked_logits, dim=1)

        return int(action.item())

    def action_probs(self, obs):
        state = flatten_obs(obs)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            obs["action_mask"],
            dtype=torch.bool
        ).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(state_tensor)
            masked_logits = logits.masked_fill(~mask_tensor, -1e9)
            probs = torch.softmax(masked_logits, dim=1)

        return probs.squeeze(0).numpy()


model = ActorCritic()
model.load_state_dict(torch.load("a2c_bc.pt", map_location="cpu"))

policy = A2CAdapter(model)

config = Config.from_yaml("configs/eval_standard.yaml")

results = evaluate(
    policy,
    config,
    seeds=[100, 101, 102]
)

print(json.dumps(results["mean"], indent=2))

print()

for index, result in enumerate(results["per_seed"]):
    print(f"Seed {index}:")
    print(json.dumps(result, indent=2))