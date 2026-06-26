import torch
import torch.nn as nn


class PairwisePolicy(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.charge_head = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.noop_score = nn.Parameter(torch.zeros(1))

    def forward(self, pair_features, charge_features):
        assignment_scores = self.scorer(pair_features).squeeze(-1)
        charge_scores = self.charge_head(charge_features).squeeze(-1)
        noop_score = self.noop_score.expand(pair_features.shape[0], 1)

        return torch.cat(
            [assignment_scores, charge_scores, noop_score],
            dim=1
        )