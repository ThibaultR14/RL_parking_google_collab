import torch
import torch.optim as optim
import torch.nn.functional as F
from .networks import DQNNetwork
from .replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, obs_dim, n_actions, config, device):
        self.device = device
        self.gamma = config.GAMMA

        self.policy_net = DQNNetwork(obs_dim, n_actions).to(device)
        self.target_net = DQNNetwork(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE, device)

    def select_action(self, state, epsilon):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.policy_net.net[-1].out_features, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.policy_net(
                    torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                )
                return q_values.argmax().item()

    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
