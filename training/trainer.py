import highway_env
import gymnasium as gym
import torch
import os
import time
from torch.utils.tensorboard import SummaryWriter
from agents.dqn import DQNAgent
import numpy as np

class Trainer:
    def __init__(self, config, skills, skill_set_name):
        self.config = config
        self.skills = skills
        self.skill_set_name = skill_set_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Correction Gymnasium : render_mode="rgb_array"
        self.env = gym.make(
            config.ENV_ID,
            max_episode_steps=config.MAX_STEPS_PER_EPISODE,
            render_mode="rgb_array"
        )

        obs_dim = self.env.observation_space["observation"].shape[0]
        n_actions = len(skills)

        self.agent = DQNAgent(obs_dim, n_actions, config, self.device)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            "runs",
            "DQN",
            skill_set_name,
            timestamp
        )
        os.makedirs(self.run_dir, exist_ok=True)

        self.writer = SummaryWriter(self.run_dir)

    def train(self):
        epsilon = self.config.EPS_START
        total_steps = 0
        frames = []

        for episode in range(1, self.config.MAX_EPISODES + 1):
            obs_dict, _ = self.env.reset()
            state = obs_dict["observation"]
            episode_reward = 0

            current_skill = None
            skill_steps_remaining = 0

            for step in range(1, self.config.MAX_STEPS_PER_EPISODE + 1):
                if total_steps >= self.config.TOTAL_STEPS:
                    break
                total_steps += 1

                # Choisir un skill
                if skill_steps_remaining <= 0:
                    skill_idx = self.agent.select_action(state, epsilon)
                    current_skill = self.skills[skill_idx]
                    current_skill.reset()
                    skill_steps_remaining = current_skill.length

                action = current_skill.step()
                skill_steps_remaining -= 1

                next_obs_dict, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = next_obs_dict["observation"]

                # Ajout dans le replay buffer et apprentissage
                self.agent.replay_buffer.push(state, skill_idx, reward, next_state, done)
                self.agent.train_step(self.config.BATCH_SIZE)

                state = next_state
                episode_reward += reward

                # ======== Capture frames pour vidéo ========
                if total_steps % 10000 == 0:  # tous les 10k steps
                    frame = self.env.render()  # plus de mode="rgb_array"
                    frames.append(frame)

                if done:
                    break

            epsilon = max(self.config.EPS_END, epsilon * self.config.EPS_DECAY)

            if episode % self.config.TARGET_UPDATE == 0:
                self.agent.update_target()

            # Logs TensorBoard
            self.writer.add_scalar("Reward/episode", episode_reward, total_steps)  # steps totaux en x
            self.writer.add_scalar("Epsilon", epsilon, total_steps)

            print(
                f"Episode {episode:4d} | "
                f"Reward = {episode_reward:8.2f} | "
                f"Epsilon = {epsilon:.3f} | Total Steps = {total_steps:6d}"
            )

            # ======== Sauvegarde vidéo dans TensorBoard ========
            if frames:
                self.writer.add_video(
                    "Agent/video",
                    torch.tensor(np.array(frames)).permute(0, 3, 1, 2).unsqueeze(0),
                    global_step=total_steps,
                    fps=15
                )
                frames = []  # reset frames

            if total_steps >= self.config.TOTAL_STEPS:
                break

        self.writer.close()
        self.env.close()
