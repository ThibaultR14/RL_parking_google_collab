import random
from collections import deque
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done, duration):
        self.buffer.append((state, action, reward, next_state, done, duration))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, durations = zip(*batch)

        # 🔹 Conversion plus rapide pour PyTorch
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        durations = torch.tensor(durations, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones, durations

    def __len__(self):
        return len(self.buffer)