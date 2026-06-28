import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DuelingQNetwork, self).__init__()

        # Ortak Özellik Çıkarıcı Katman
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )

        # 1. Akış: Durumun genel değerini hesaplar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 2. Akış: Her bir hamlenin ekstra avantajını hesaplar A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Matematiksel birleştirme: Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent:
    def __init__(self, config):
        self.obs_dim = config.get("obs_dim", 20)
        self.action_dim = config.get("action_dim", 10)
        self.state_key = config.get("state_key", "observation")

        # Yeni Dueling ağımızı çağırıyoruz
        self.q_net = DuelingQNetwork(self.obs_dim, self.action_dim)
        self.target_net = DuelingQNetwork(self.obs_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.get("lr", 0.001))
        self.memory = ReplayBuffer(config.get("buffer_size", 10000))

        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998

    def act(self, obs):
        state = np.array(obs[self.state_key]).flatten()
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
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN öğrenme mantığını burada da koruyoruz (Dueling + Double DQN kombinasyonu)
            next_actions = self.q_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        tau = 0.005
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay