import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim=741, action_dim=169):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        hidden = self.shared(x)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value


def choose_a2c_action(model, state, action_mask):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)

    logits, value = model(state_tensor)
    masked_logits = logits.masked_fill(~mask_tensor, -1e9)

    distribution = torch.distributions.Categorical(logits=masked_logits)

    action = distribution.sample()
    log_prob = distribution.log_prob(action)
    entropy = distribution.entropy()

    return int(action.item()), log_prob.squeeze(), value.squeeze(), entropy.squeeze()