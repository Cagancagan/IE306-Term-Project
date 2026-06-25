import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=581, action_dim=169):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def choose_action(model, state, action_mask):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)

    logits = model(state_tensor)
    masked_logits = logits.masked_fill(~mask_tensor, -1e9)

    distribution = torch.distributions.Categorical(logits=masked_logits)

    action = distribution.sample()
    log_prob = distribution.log_prob(action)

    return int(action.item()), log_prob