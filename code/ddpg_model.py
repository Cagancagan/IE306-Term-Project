import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim=7, action_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        raw_action = self.net(state)

        speed = torch.sigmoid(raw_action[:, 0:1])
        heading_delta = torch.tanh(raw_action[:, 1:2])

        return torch.cat([speed, heading_delta], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim=7, action_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)