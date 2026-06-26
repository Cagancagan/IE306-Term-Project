import json
import numpy as np
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


config = Config.from_yaml("configs/eval_standard.yaml")
eval_seeds = list(range(300, 320))

model_files = [
    "pairwise_milp_seed0.pt",
    "pairwise_milp_seed1.pt",
    "pairwise_milp_seed2.pt"
]

all_results = []

for model_file in model_files:
    model = PairwisePolicy()
    model.load_state_dict(torch.load(model_file, map_location="cpu"))

    policy = PairwiseAdapter(model)

    results = evaluate(
        policy,
        config,
        seeds=eval_seeds
    )

    mean_result = results["mean"]
    all_results.append(mean_result)

    print()
    print(model_file)
    print(json.dumps(mean_result, indent=2))

metrics = [
    "cost_per_order",
    "success_rate",
    "ontime_rate",
    "energy_per_order",
    "depletion_events",
    "n_delivered",
    "n_dropped",
    "episode_return"
]

print()
print("MEAN ± STD ACROSS TRAINING SEEDS")

for metric in metrics:
    values = [result[metric] for result in all_results]
    print(
        f"{metric}: "
        f"{np.mean(values):.4f} ± {np.std(values):.4f}"
    )