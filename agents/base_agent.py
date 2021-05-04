import logging
import time
import os
import sys
from pathlib import Path

import wandb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from models import critic_feed_forward_model_constructor, feed_forward_discrete_policy_constructor
from code_utils import ConfigManager, CheckpointsManager, TrainingConfig
from environments import get_env, Episode

logger = logging.getLogger()


class BaseActorCriticAgent(object):
    """
    Base class for actor critic algorithms:
     - Creates a FFNN model to use as policy (actor)
     - Creates a FFNN model to use as critic
     - Has training logic
    """

    def __init__(self, agent_path: Path, config: ConfigManager):
        """Creates an actor critic agent that uses FFNNs to represent both.

        Args:
            agent_path: The output folder for the model files
            config: The configurations for this agent
        """
        self.env = get_env(config.agent_config.env)()
        self.agent_path = agent_path
        self.config = config
        self.models_path = Path(agent_path, "models")

        critic_constructor = critic_feed_forward_model_constructor(self.env.state_space_n)
        self.critic = critic_constructor(self.config.critic_config)

        actor_constructor = feed_forward_discrete_policy_constructor(self.env.state_space_n, self.env.action_space_n)
        self.actor = actor_constructor(self.config.actor_config)

        self.ckpts_manager = CheckpointsManager(self.models_path, self.actor, self.critic)

    def generate_episode(self) -> Episode:

        self.env.reset_environment()
        done = False
        states = []
        rewards = []
        actions = []
        while not done:
            state = self.env.get_environment_state()
            tf_current_state = tf.constant(np.array([state]), dtype=tf.float32)
            action = self.actor.produce_actions(tf_current_state)[0][0]
            next_state, reward, done = self.env.environment_step(int(action))

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        episode = Episode(states, actions, rewards, "discrete", self.config.agent_config.discount)

        return episode

    def train_policy(self, training_config: TrainingConfig) -> float:
        """Train the agent to solve the current environment.

        Args:
            training_config: The training configurations
        Returns
            The final policy test mean reward
        """

        train_steps_rewards = []
        start_time = time.time()
        training_steps = 0
        progress_save = int(training_config.train_steps * 0.05)
        best_step = None
        best_checkpoints = None
        best_score = float("-inf")
        for i in tqdm(range(training_config.train_steps)):
            episode = self.generate_episode()
            train_steps_rewards.append(episode.total_reward)

            if training_config.show_every is not None:
                if i > 0 and not i % training_config.show_every:
                    logger.info(f"Training step N° {i} - "
                                f"Episodes reward: {episode.total_reward} - "
                                f"Batch time = {time.time() - start_time} sec")
                    start_time = time.time()

            states_batch = tf.constant(episode.states, dtype=np.float32)
            values_batch = self.critic(states_batch)

            discounted_rewards_batch = tf.constant(episode.discounted_rewards, dtype=np.float32)
            actions_batch = tf.constant(episode.actions, dtype=np.int32)
            advantage_batch = discounted_rewards_batch - tf.reshape(values_batch, -1)

            # noinspection PyArgumentList
            action_probabilities = self.actor.get_probabilities(states_batch)
            policy_outputs, actor_loss, log_probabilities = self.actor.train_step(
                states_batch, actions_batch, advantage_batch)

            values, critic_loss, _ = self.critic.train_step(states_batch, discounted_rewards_batch)

            if self.env.state_names is not None:
                for state_idx, state in enumerate(self.env.state_names):
                    state_attribute_hist = states_batch[:, state_idx]
                    wandb.log({f"{state}": wandb.Histogram(state_attribute_hist)}, step=training_steps)

            if self.env.actions is not None:
                for action_idx, action in enumerate(self.env.actions):
                    action_attribute_hist = action_probabilities[:, action_idx]
                    wandb.log({f"{action}": wandb.Histogram(action_attribute_hist)}, step=training_steps)

            wandb.log({'training_step': training_steps, 'actor_loss': actor_loss,
                       'critic_loss': critic_loss, 'episode_reward': episode.total_reward},
                      step=training_steps)
            try:
                wandb.log({"log_probabilities": wandb.Histogram(log_probabilities)}, step=training_steps)
            except ValueError:
                logger.info(f"Failed to save log probabilities: {log_probabilities}")
            wandb.log({"discounted_rewards": wandb.Histogram(discounted_rewards_batch)}, step=training_steps)
            try:
                wandb.log({"advantages": wandb.Histogram(advantage_batch)}, step=training_steps)
            except ValueError:
                logger.info(f"Failed to save advantages: {advantage_batch}")
            try:
                wandb.log({"values": wandb.Histogram(values_batch)}, step=training_steps)
            except ValueError:
                logger.info(f"Failed to save values: {values_batch}")

            training_steps += 1
            self.ckpts_manager.step_checkpoints()
            if not i % progress_save:
                self.ckpts_manager.save_progress_ckpts()
                logger.info(f"Progress checkpoints saved for step {training_steps}")
            if training_config.save_policy_every is not None:
                if not i % training_config.save_policy_every:
                    if episode.total_reward >= best_score:
                        best_score = episode.total_reward
                        best_step = i
                        best_checkpoints = self.ckpts_manager.save_ckpts()
                        logger.info(f"New best model - Reward = {episode.total_reward}")
                        logger.info(f"Checkpoint saved for step {training_steps}")

            if self.env.pass_test(train_steps_rewards[-20:]):
                logger.info("The agent trained successfully!!")
                best_step = i
                best_checkpoints = self.ckpts_manager.save_ckpts()
                logger.info(f"New best model - Reward = {episode.total_reward}")
                logger.info(f"Checkpoint saved for step {training_steps}")
                break

        # Load best checkpoint and save it
        logger.info(f"Best model in step {best_step} - {best_checkpoints[0]}")
        self.ckpts_manager.actor.restore(best_checkpoints[0])
        test_reward = self.test_agent(episodes=100)
        logger.info(f"Best model test: {100} episodes mean reward = {test_reward}")
        self.save_agent()
        self.plot_training_info(train_steps_rewards, self.agent_path)

        return test_reward

    def save_agent(self):
        """Save the policy neural network to files in the model path."""

        logger.info(f"Saving trained policy to {self.models_path}")
        start = time.time()
        self.actor.save(self.models_path)
        logger.info(f"Saving time {time.time() - start}")

    def load_model(self, model_dir: Path):
        """Loads a trained policy from files. If no save model is found,
        it loads the latest checkpoint available.

        Args:
            model_dir: Where the trained model is stored.
        """
        if Path(model_dir, "saved_model.pb").exists():
            self.actor = tf.keras.models.load_model(model_dir)
        else:
            self.ckpts_manager.actor.restore(self.ckpts_manager.actor_manager.latest_checkpoint)
            logger.info(f"Restored model from checkpoint {self.ckpts_manager.actor_manager.latest_checkpoint}")

    @staticmethod
    def plot_training_info(rewards: np.array, agent_folder: Path = None):
        """Plots the training reward moving average.

        Args:
            rewards: The rewards
            agent_folder: Where to save the generated plot
        """
        plt.figure(figsize=(5, 5))

        # Moving average plot
        plt.plot([i for i in range(len(rewards))], rewards)
        plt.ylabel(f"Reward")
        plt.xlabel("Training step #")
        plt.title("Rewards")

        if agent_folder is not None:
            plt.savefig(Path(agent_folder, "training_rewards.png"))

    def play_game(self, plot_game: bool = False, delay: float = None) -> Episode:
        """Plays a full episode using the current policy.

        Args:
            plot_game: If the environment should be plotted
            delay: Delay between environment steps (frames)

        Returns:
            The full played episode
        """
        self.env.reset_environment()
        done = False
        states = []
        rewards = []
        actions = []
        while not done:
            if plot_game:
                self.env.render_environment()
                if delay is not None:
                    time.sleep(delay)

            state = self.env.get_environment_state()
            tf_current_state = tf.constant(np.array([state]), dtype=tf.float32)
            action = self.actor.produce_actions(tf_current_state)[0][0]

            new_state, reward, done = self.env.environment_step(int(action))

            states.append(state)
            rewards.append(reward)
            actions.append(action)

        episode = Episode(states, actions, rewards, "discrete", self.config.agent_config.discount)

        self.env.reset_environment()

        return episode

    def test_agent(self, episodes: int = 100):
        total_rewards = []
        for i in range(episodes):
            episode = self.play_game(plot_game=False, delay=None)
            total_rewards.append(episode.total_reward)

        mean_reward = np.mean(total_rewards)
        return mean_reward
