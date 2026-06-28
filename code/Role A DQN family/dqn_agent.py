import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        # Artık '128' yerine 'obs_dim' kullanıyoruz!
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),  # Giriş doğrudan obs_dim
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100000): self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done): self.buffer.append(
        (state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self): return len(self.buffer)


class DQNAgent:
    def __init__(self, config):
        self.obs_dim, self.action_dim, self.state_key = config.get("obs_dim", 20), config.get("action_dim",
                                                                                              10), config.get(
            "state_key", "observation")
        self.q_net, self.target_net = QNetwork(self.obs_dim, self.action_dim), QNetwork(self.obs_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.get("lr", 0.0005))
        self.memory = ReplayBuffer(config.get("buffer_size", 100000))
        self.batch_size, self.gamma = config.get("batch_size", 64), config.get("gamma", 0.99)
        self.epsilon, self.epsilon_min, self.epsilon_decay = 1.0, 0.05, 0.998

    def act(self, obs):
        # Yeni görme yeteneği: Verileri birleştiriyoruz
        drones = np.array(obs["drones"]).flatten()
        orders = np.array(obs["orders"]).flatten()
        time = np.array(obs["time"]).flatten()
        state = np.concatenate([drones, orders, time])

        action_mask = obs["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            return 0

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0).numpy()

        q_values[action_mask == 0] = -np.inf
        return int(np.argmax(q_values))

    def learn(self):
        if len(self.memory) < self.batch_size: return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = torch.FloatTensor(states), torch.LongTensor(actions).unsqueeze(
            1), torch.FloatTensor(rewards).unsqueeze(1), torch.FloatTensor(next_states), torch.FloatTensor(
            dones).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(0.005 * local_param.data + 0.995 * target_param.data)
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay