import logging
from pathlib import Path

import wandb
import tensorflow as tf
import numpy as np

from .base_agent import BaseActorCriticAgent, ConfigManager


logger = logging.getLogger()


class BatchActorCriticAgent(BaseActorCriticAgent):
    """
    Implementing bach actor critic algorithm from
    http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf
    """

    def __init__(self, agent_path: Path, config: ConfigManager):

        super().__init__(agent_path, config)

    def train_step(self, step_n: int) -> int:
        """ Make a single training step for this method.

        Args:
            step_n: The current training step number

        Returns:
            The total reward from the last completed episode
        """

        episode = self.generate_episode()

        # Fit critic with the returns of the episode
        states_batch = tf.constant(episode.states, dtype=np.float32)
        discounted_rewards_batch = tf.constant(episode.discounted_rewards, dtype=np.float32)
        old_critic_values, critic_loss, _ = self.critic.train_step(states_batch, discounted_rewards_batch)

        # Approximate the advantages using the critic
        values = self.critic(states_batch)
        next_state_values = values.numpy()[1:, :]  # First state doesn't has a previous state
        next_state_values = np.append(next_state_values, [[0.]], axis=0)  # Terminal state value = 0
        next_state_values = tf.constant(next_state_values, dtype=np.float32)
        rewards_batch = tf.constant(episode.rewards, dtype=np.float32)
        advantage_batch = rewards_batch + self.config.agent_config.discount * tf.reshape(next_state_values, -1) - \
            tf.reshape(values, -1)

        action_probabilities = self.actor.get_probabilities(states_batch)
        actions_batch = tf.constant(episode.actions, dtype=np.int32)
        policy_outputs, actor_loss, log_probabilities = self.actor.train_step(
            states_batch, actions_batch, advantage_batch)

        # Save metrics to WandB
        if self.env.state_names is not None:
            for state_idx, state in enumerate(self.env.state_names):
                state_attribute_hist = states_batch[:, state_idx]
                wandb.log({f"{state}": wandb.Histogram(state_attribute_hist)}, step=step_n)

        if self.env.actions is not None:
            for action_idx, action in enumerate(self.env.actions):
                action_attribute_hist = action_probabilities[:, action_idx]
                wandb.log({f"{action}": wandb.Histogram(action_attribute_hist)}, step=step_n)

        wandb.log({'training_step': step_n, 'actor_loss': actor_loss,
                   'critic_loss': critic_loss, 'episode_reward': episode.total_reward},
                  step=step_n)
        wandb.log({"discounted_rewards": wandb.Histogram(discounted_rewards_batch)}, step=step_n)

        try:
            wandb.log({"log_probabilities": wandb.Histogram(log_probabilities)}, step=step_n)
        except ValueError:
            logger.info(f"Failed to save log probabilities: {log_probabilities}")
        try:
            wandb.log({"advantages": wandb.Histogram(advantage_batch)}, step=step_n)
        except ValueError:
            logger.info(f"Failed to save advantages: {advantage_batch}")
        try:
            wandb.log({"values": wandb.Histogram(values)}, step=step_n)
        except ValueError:
            logger.info(f"Failed to save values: {values}")

        return episode.total_reward
