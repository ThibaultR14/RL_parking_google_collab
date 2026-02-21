import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


# ============================================
# =============== REPLAY BUFFER ==============
# ============================================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.device = device

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.obs_buf[idx]).to(self.device),
            torch.FloatTensor(self.act_buf[idx]).to(self.device),
            torch.FloatTensor(self.rew_buf[idx]).unsqueeze(1).to(self.device),
            torch.FloatTensor(self.next_obs_buf[idx]).to(self.device),
            torch.FloatTensor(self.done_buf[idx]).unsqueeze(1).to(self.device),
        )


# ============================================
# =================== ACTOR ==================
# ============================================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, act_dim)
        self.act_limit = act_limit

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x)) * self.act_limit


# ============================================
# ================== CRITIC ==================
# ============================================
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Q1
        self.q1_l1 = nn.Linear(obs_dim + act_dim, 256)
        self.q1_l2 = nn.Linear(256, 256)
        self.q1_l3 = nn.Linear(256, 1)

        # Q2
        self.q2_l1 = nn.Linear(obs_dim + act_dim, 256)
        self.q2_l2 = nn.Linear(256, 256)
        self.q2_l3 = nn.Linear(256, 1)

    def forward(self, obs, act):
        xu = torch.cat([obs, act], dim=1)

        # Q1
        q1 = F.relu(self.q1_l1(xu))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        # Q2
        q2 = F.relu(self.q2_l1(xu))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)

        return q1, q2

    def q1(self, obs, act):
        xu = torch.cat([obs, act], dim=1)
        q1 = F.relu(self.q1_l1(xu))
        q1 = F.relu(self.q1_l2(q1))
        return self.q1_l3(q1)


# ============================================
# ================= TD3 AGENT ================
# ============================================
class TD3Agent:
    def __init__(self, obs_dim, act_dim, act_limit, config, device):

        self.device = device
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.policy_noise = config.POLICY_NOISE
        self.noise_clip = config.NOISE_CLIP
        self.policy_freq = config.POLICY_FREQ
        self.batch_size = config.BATCH_SIZE
        self.act_limit = act_limit

        # Networks
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target = Actor(obs_dim, act_dim, act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim, act_dim, config.BUFFER_SIZE, device
        )

        self.total_it = 0

    # ====================================
    # ACTION SELECTION
    # ====================================
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy()[0]
        return action

    # ====================================
    # STORE TRANSITION
    # ====================================
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    # ====================================
    # TRAIN STEP
    # ====================================
    def train_step(self):

        if self.replay_buffer.size < self.batch_size:
            return

        self.total_it += 1

        state, action, reward, next_state, done = \
            self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.act_limit, self.act_limit)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + self.gamma * (1 - done) * torch.min(target_q1, target_q2)

        # Critic update
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + \
                      F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:

            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update
            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    # Compatible avec Trainer (même si inutile)
    def update_target(self):
        pass