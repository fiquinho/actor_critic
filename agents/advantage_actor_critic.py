import logging
from pathlib import Path
from typing import List

import wandb
import tensorflow as tf
import numpy as np

from .base_agent import BaseActorCriticAgent, ConfigManager, get_env, Environment
from tensorflow.keras import Model


logger = logging.getLogger()


class CollectedExperience(object):
    """The collected experience from a single environment,
    for a determined amount of steps."""

    def __init__(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.last_state = None

        self.rewards_to_go = None

    def sanity_check(self):
        """Assert the correct lengths of the attributes"""
        assert len(self.states) == len(self.actions) == len(self.rewards) == len(self.dones)

    def calculate_rewards_to_go(self, critic: Model, discount: float):
        rewards_to_go = []
        current_episode_rtg = []
        tf_last_state = tf.constant(np.array([self.last_state]), dtype=tf.float32)
        last_state_value = critic(tf_last_state)
        for i in range(len(self.dones) - 1, -1, -1):

            if self.dones[i]:
                # noinspection PyListCreation
                current_episode_rtg = []
                current_episode_rtg.append(self.rewards[i])
                rewards_to_go.insert(0, self.rewards[i])
            else:
                if i + 1 == len(self.rewards):
                    predicted_rtg = self.rewards[i] + discount * last_state_value
                    current_episode_rtg.append(predicted_rtg.numpy()[0][0])
                    rewards_to_go.insert(0, predicted_rtg.numpy()[0][0])

                else:
                    predicted_rtg = self.rewards[i] + current_episode_rtg[-1]
                    current_episode_rtg.append(predicted_rtg)
                    rewards_to_go.insert(0, predicted_rtg)

        assert len(self.rewards) == len(rewards_to_go)
        self.rewards_to_go = rewards_to_go


class MultipleEnvironments(object):
    """Manage multiple environments easily."""
    
    def __init__(self, env: Environment, env_num: int):

        self.envs = [env() for _ in range(env_num)]

    def restart_environments(self):
        for env in self.envs:
            env.reset_environment()

    def get_normalized_states(self) -> List[np.array]:
        current_states = []
        for env_num, env in enumerate(self.envs):
            state = env.get_normalized_state()
            current_states.append(state)

        return current_states

    @staticmethod
    def calculate_rewards_to_go(experience: List[CollectedExperience],
                                critic: Model, discount: float):
        for e in experience:
            e.calculate_rewards_to_go(critic, discount)

    @staticmethod
    def concatenated_states(experience: List[CollectedExperience]):
        states = []
        for e in experience:
            states += e.states
        return np.array(states)

    @staticmethod
    def concatenated_actions(experience: List[CollectedExperience]):
        actions = []
        for e in experience:
            actions += e.actions
        return np.array(actions)

    @staticmethod
    def concatenated_rewards_to_go(experience: List[CollectedExperience]):
        rtg = []
        for e in experience:
            rtg += e.rewards_to_go
        return np.array(rtg)

    def collect_experience(self, actor: Model, batch_env_steps: int) -> List[CollectedExperience]:

        collected_experience = [CollectedExperience() for _ in range(len(self.envs))]

        # Collect a batch of experience from each environment
        for env_step in range(batch_env_steps):

            # Sample actions from current policy
            current_states = self.get_normalized_states()
            tf_states = tf.constant(np.array(current_states), dtype=tf.float32)
            actions = actor.produce_actions(tf_states).numpy()

            # Step in each environment and collect results
            for env_num, env in enumerate(self.envs):
                _, reward, done = env.environment_step(int(actions[env_num]))
                collected_experience[env_num].rewards.append(reward)
                collected_experience[env_num].dones.append(done)
                collected_experience[env_num].states.append(current_states[env_num])
                collected_experience[env_num].actions.append(actions[env_num][0])
                if done:
                    env.reset_environment()

        # Collect final state of each episode
        final_states = self.get_normalized_states()
        for i, ce in enumerate(collected_experience):
            ce.last_state = final_states[i]
            # Sanity check
            ce.sanity_check()

        return collected_experience


class A2C(BaseActorCriticAgent):
    """
    Implementing advantage actor critic (A2C) algorithm.
    I found several different implementation details for this
    algorithm, so this is my own attempt to implement it.
    I'll be using the "online actor critic" from
    http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf
    but with parallel environments (kind of) and a batched training.
    Other Used theory resources are listed in README.md.
    """

    def __init__(self, agent_path: Path, config: ConfigManager):

        super().__init__(agent_path, config)

        self.periodic_test = True
        self.environments = MultipleEnvironments(get_env(config.agent_config.env), config.agent_config.num_processes)
        self.environments.restart_environments()

    def pass_test(self, **kwargs) -> bool:
        """Standard test of agent

        Args:
            **kwargs:

        Returns: If the agent solved the environment

        """
        train_episodes_rewards = kwargs["train_episodes_rewards"]
        return self.env.pass_test(train_episodes_rewards[-5:])

    def train_step(self, step_n: int) -> int:
        """ Make a single training step for this method.

        Args:
            step_n: The current training step number

        Returns:
            The total reward from the last completed episode
        """

        # Collect a batch of experience from each environment
        collected_experience = self.environments.collect_experience(
            self.actor, self.config.agent_config.batch_env_steps)
        self.environments.calculate_rewards_to_go(
            collected_experience, self.critic, self.config.agent_config.discount)

        # Train critic
        collected_states = self.environments.concatenated_states(collected_experience)
        tf_states = tf.constant(np.array(collected_states), dtype=tf.float32)
        collected_rtg = self.environments.concatenated_rewards_to_go(collected_experience)
        tf_rewards_to_go = tf.constant(np.array(collected_rtg), dtype=tf.float32)
        old_critic_values, critic_loss, _ = self.critic.train_step(tf_states, tf_rewards_to_go)

        # Calculate advantages
        self.environments.calculate_rewards_to_go(
            collected_experience, self.critic, self.config.agent_config.discount)
        collected_rtg = self.environments.concatenated_rewards_to_go(collected_experience)
        tf_rewards_to_go = tf.constant(np.array(collected_rtg), dtype=tf.float32)
        values = self.critic(tf_states)
        advantages = tf_rewards_to_go - tf.reshape(values, -1)
        advantage_batch = tf.constant(advantages, dtype=np.float32)

        # Train actor
        action_probabilities = self.actor.get_probabilities(tf_states)
        collected_actions = self.environments.concatenated_actions(collected_experience)
        tf_actions = tf.constant(np.array(collected_actions), dtype=tf.int32)
        policy_outputs, actor_loss, log_probabilities = self.actor.train_step(
            tf_states, tf_actions, advantage_batch)

        # Save metrics to WandB
        if self.env.state_names is not None:
            for state_idx, state in enumerate(self.env.state_names):
                state_attribute_hist = tf_states[:, state_idx]
                wandb.log({f"{state}": wandb.Histogram(state_attribute_hist)}, step=step_n)

        if self.env.actions is not None:
            for action_idx, action in enumerate(self.env.actions):
                action_attribute_hist = action_probabilities[:, action_idx]
                wandb.log({f"{action}": wandb.Histogram(action_attribute_hist)}, step=step_n)

        wandb.log({'training_step': step_n, 'actor_loss': actor_loss,
                   'critic_loss': critic_loss},
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
            wandb.log({"values": wandb.Histogram(values)}, step=step_n)
        except ValueError:
            logger.info(f"Failed to save values: {values}")

        return 0., False
