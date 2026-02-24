import gymnasium as gym
import torch
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.dqn import DQNAgent
from agents.td3 import TD3Agent


class Trainer:
    def __init__(self, config, skills=None, skill_set_name=None, algo="dqn"):
        self.config = config
        self.skills = skills
        self.skill_set_name = skill_set_name
        self.algo = algo.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===== Environment =====
        self.env = gym.make(
            config.ENV_ID,
            max_episode_steps=config.MAX_STEPS_PER_EPISODE,
            render_mode="rgb_array"
        )
        obs_dim = self.env.observation_space["observation"].shape[0]

        # ===== Agent Selection =====
        if self.algo == "dqn":
            if skills is None:
                raise ValueError("DQN requires a skill set")
            n_actions = len(skills)
            self.agent = DQNAgent(obs_dim, n_actions, config, self.device)

        elif self.algo == "td3":
            act_dim = self.env.action_space.shape[0]
            act_limit = self.env.action_space.high[0]
            self.agent = TD3Agent(obs_dim, act_dim, act_limit, config, self.device)

        else:
            raise ValueError("Algo must be 'dqn' or 'td3'")

        # ===== Logging =====
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        base_dir = "/content/drive/MyDrive/RL_parking/runs"
        self.run_dir = os.path.join(
            base_dir,
            self.algo.upper(),
            skill_set_name if skill_set_name else "no_skills",
            timestamp
        )

        os.makedirs(self.run_dir, exist_ok=True)
        print("Saving runs to:", self.run_dir)

        self.writer = SummaryWriter(self.run_dir)

    # ================= TRAIN LOOP =================
    def train(self):

        epsilon = self.config.EPS_START if self.algo == "dqn" else None
        total_steps = 0
        frames = []

        for episode in range(1, self.config.MAX_EPISODES + 1):

            obs_dict, _ = self.env.reset()
            state = obs_dict["observation"]
            episode_reward = 0

            # ===== Skill variables (DQN only) =====
            current_skill = None
            skill_steps_remaining = 0
            skill_start_state = None
            skill_total_reward = 0
            skill_duration = 0
            skill_idx = None

            for step in range(self.config.MAX_STEPS_PER_EPISODE):

                if total_steps >= self.config.TOTAL_STEPS:
                    break

                total_steps += 1

                # ===== ACTION SELECTION =====
                if self.algo == "dqn":

                    # 🔥 Start new skill if needed
                    if skill_steps_remaining <= 0:

                        skill_idx = self.agent.select_action(state, epsilon)
                        current_skill = self.skills[skill_idx]
                        current_skill.reset()

                        skill_steps_remaining = current_skill.length

                        # Initialize SMDP rollout
                        skill_start_state = state
                        skill_total_reward = 0
                        skill_duration = 0

                    action = current_skill.step()
                    skill_steps_remaining -= 1

                elif self.algo == "td3":
                    action = self.agent.select_action(state)

                # ===== ENV STEP =====
                next_obs_dict, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = next_obs_dict["observation"]

                # ===== STORE TRANSITION =====
                if self.algo == "dqn":

                    skill_total_reward += reward
                    skill_duration += 1

                    # 🔥 Only store when skill ends (SMDP)
                    if skill_steps_remaining == 0 or done:
                        self.agent.store_transition(
                            skill_start_state,
                            skill_idx,
                            skill_total_reward,
                            next_state,
                            done,
                            skill_duration
                        )

                elif self.algo == "td3":
                    self.agent.store_transition(
                        state,
                        action,
                        reward,
                        next_state,
                        done
                    )

                # ===== TRAIN STEP =====
                if self.algo == "dqn":
                    self.agent.train_step(self.config.BATCH_SIZE)

                elif self.algo == "td3":
                    self.agent.train_step()

                state = next_state
                episode_reward += reward

                # ===== Capture video frames =====
                if total_steps % 10000 == 0:
                    frame = self.env.render()
                    frames.append(frame)

                if done:
                    break

            # ===== DQN updates =====
            if self.algo == "dqn":
                epsilon = max(self.config.EPS_END, epsilon * self.config.EPS_DECAY)

                if episode % self.config.TARGET_UPDATE == 0:
                    self.agent.update_target()

            # ===== Logging =====
            self.writer.add_scalar("Reward/episode", episode_reward, total_steps)

            if self.algo == "dqn":
                self.writer.add_scalar("Epsilon", epsilon, total_steps)

            print(
                f"[{self.algo.upper()}] Episode {episode:4d} | "
                f"Reward = {episode_reward:8.2f} | Total Steps = {total_steps:6d}"
            )

            # ===== Save video =====
            if frames:
                video_tensor = (
                    torch.tensor(np.array(frames))
                    .permute(0, 3, 1, 2)
                    .unsqueeze(0)
                )
                self.writer.add_video(
                    "Agent/video",
                    video_tensor,
                    global_step=total_steps,
                    fps=15
                )
                frames = []

            if total_steps >= self.config.TOTAL_STEPS:
                break

        self.writer.close()
        self.env.close()