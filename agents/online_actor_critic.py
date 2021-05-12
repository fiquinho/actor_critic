import logging
from pathlib import Path

import wandb
import tensorflow as tf
import numpy as np

from .base_agent import BaseActorCriticAgent, ConfigManager, Episode


logger = logging.getLogger()


class OnlineActorCriticAgent(BaseActorCriticAgent):
    """
    Implementing online actor critic algorithm from
    http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf
    """

    def __init__(self, agent_path: Path, config: ConfigManager):

        super().__init__(agent_path, config)

        self.env.reset_environment()
        self.last_complete_episode = None
        self.last_complete_episode_reward = 0.
        self.finished_episode = False
        self.current_episode = self.reset_current_episode()

    @staticmethod
    def reset_current_episode() -> dict:
        new_episode = {"states": [],
                       "actions": [],
                       "rewards": []}
        return new_episode

    def train_step(self, step_n: int) -> int:
        """ Make a single training step for this method.

        Args:
            step_n: The current training step number

        Returns:
            The total reward from the last completed episode
        """

        if self.finished_episode:
            self.env.reset_environment()
            self.finished_episode = False

        # Take an action in the environment and save to current episode history
        state = self.env.get_normalized_state()
        tf_state = tf.constant(np.array([state]), dtype=tf.float32)
        self.current_episode["states"].append(state)

        action = self.actor.produce_actions(tf_state)[0][0]
        tf_action = tf.constant(np.array([action]), dtype=tf.int32)
        self.current_episode["actions"].append(action.numpy())

        _, reward, done = self.env.environment_step(int(action))
        norm_next_state = self.env.get_normalized_state()
        tf_next_state = tf.constant(np.array([norm_next_state]), dtype=tf.float32)
        self.current_episode["rewards"].append(reward)

        # Calculate critic target and train critic
        if done:
            next_state_value = tf.constant(np.array([[0.]]), dtype=tf.float32)
        else:
            next_state_value = self.critic(tf_next_state)

        critic_target = reward + self.config.agent_config.discount * next_state_value.numpy()[0][0]
        tf_critic_target = tf.constant(np.array([critic_target]), dtype=tf.float32)
        old_critic_values, critic_loss, _ = self.critic.train_step(tf_state, tf_critic_target)

        # Evaluate advantages
        new_state_value = self.critic(tf_state)
        if done:
            new_next_state_value = tf.constant(np.array([0.]), dtype=tf.float32)
        else:
            new_next_state_value = self.critic(tf_next_state)
        advantage = reward + self.config.agent_config.discount * tf.reshape(new_next_state_value, -1) - \
            tf.reshape(new_state_value, -1)
        advantage_batch = tf.constant(advantage, dtype=np.float32)

        # Train actor
        action_probabilities = self.actor.get_probabilities(tf_state)
        policy_outputs, actor_loss, log_probabilities = self.actor.train_step(
            tf_state, tf_action, advantage_batch)

        # Save metrics to WandB
        if self.env.state_names is not None:
            for state_idx, state in enumerate(self.env.state_names):
                state_attribute_hist = tf_state[:, state_idx]
                wandb.log({f"{state}": wandb.Histogram(state_attribute_hist)}, step=step_n)

        if self.env.actions is not None:
            for action_idx, action in enumerate(self.env.actions):
                action_attribute_hist = action_probabilities[:, action_idx]
                wandb.log({f"{action}": wandb.Histogram(action_attribute_hist)}, step=step_n)

        wandb.log({'training_step': step_n, 'actor_loss': actor_loss,
                   'critic_loss': critic_loss, 'episode_reward': self.last_complete_episode_reward},
                  step=step_n)
        try:
            wandb.log({"log_probabilities": wandb.Histogram(log_probabilities)}, step=step_n)
        except ValueError:
            logger.info(f"Failed to save log probabilities: {log_probabilities}")
        try:
            wandb.log({"advantages": wandb.Histogram(advantage_batch)}, step=step_n)
        except ValueError:
            logger.info(f"Failed to save advantages: {advantage_batch}")
        try:
            wandb.log({"values": wandb.Histogram(new_state_value)}, step=step_n)
        except ValueError:
            logger.info(f"Failed to save values: {new_state_value}")

        if done:
            episode = Episode(self.current_episode["states"], self.current_episode["actions"],
                              self.current_episode["rewards"], "discrete", self.config.agent_config.discount)
            self.last_complete_episode = episode
            self.finished_episode = True
            self.last_complete_episode_reward = self.last_complete_episode.total_reward
            self.current_episode = self.reset_current_episode()

        return self.last_complete_episode_reward, self.finished_episode
