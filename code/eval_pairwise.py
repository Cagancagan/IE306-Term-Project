import json
import torch

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate

from pairwise_features import build_pairwise_features
from pairwise_model import PairwisePolicy


class PairwiseAdapter:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def act(self, obs):
        pair_features, charge_features = build_pairwise_features(obs)

        pair_tensor = torch.tensor(pair_features, dtype=torch.float32)
        charge_tensor = torch.tensor(charge_features, dtype=torch.float32)
        mask_tensor = torch.tensor(
            obs["action_mask"],
            dtype=torch.bool
        ).unsqueeze(0)

        with torch.no_grad():
            scores = self.model(pair_tensor, charge_tensor)
            masked_scores = scores.masked_fill(~mask_tensor, -1e9)
            action = torch.argmax(masked_scores, dim=1)

        return int(action.item())

    def action_probs(self, obs):
        pair_features, charge_features = build_pairwise_features(obs)

        pair_tensor = torch.tensor(pair_features, dtype=torch.float32)
        charge_tensor = torch.tensor(charge_features, dtype=torch.float32)
        mask_tensor = torch.tensor(
            obs["action_mask"],
            dtype=torch.bool
        ).unsqueeze(0)

        with torch.no_grad():
            scores = self.model(pair_tensor, charge_tensor)
            masked_scores = scores.masked_fill(~mask_tensor, -1e9)
            probs = torch.softmax(masked_scores, dim=1)

        return probs.squeeze(0).numpy()


model = PairwisePolicy()
model.load_state_dict(torch.load("pairwise_milp_seed0.pt", map_location="cpu"))

policy = PairwiseAdapter(model)

config = Config.from_yaml("configs/eval_standard.yaml")

results = evaluate(
    policy,
    config,
    seeds=list(range(300, 320))
)

print(json.dumps(results["mean"], indent=2))

print()

for index, result in enumerate(results["per_seed"]):
    print(f"Seed {index}:")
    print(json.dumps(result, indent=2))