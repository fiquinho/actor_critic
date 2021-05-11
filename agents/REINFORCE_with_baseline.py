import logging
from pathlib import Path

import wandb
import numpy as np
import tensorflow as tf

from .base_agent import BaseActorCriticAgent, ConfigManager


logger = logging.getLogger()


class REINFORCEwBaselineAgent(BaseActorCriticAgent):
    """
    Implementing REINFORCE algorithm with baseline. Similar to:
    https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
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

        states_batch = tf.constant(episode.states, dtype=np.float32)
        values_batch = self.critic(states_batch)
        discounted_rewards_batch = tf.constant(episode.discounted_rewards, dtype=np.float32)
        advantage_batch = discounted_rewards_batch - tf.reshape(values_batch, -1)
        advantage_batch = tf.constant(advantage_batch, dtype=np.float32)

        action_probabilities = self.actor.get_probabilities(states_batch)
        actions_batch = tf.constant(episode.actions, dtype=np.int32)

        # Fit actor and critic
        policy_outputs, actor_loss, log_probabilities = self.actor.train_step(
            states_batch, actions_batch, advantage_batch)
        values, critic_loss, _ = self.critic.train_step(states_batch, discounted_rewards_batch)

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
