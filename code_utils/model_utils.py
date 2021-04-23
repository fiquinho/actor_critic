from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model


class CheckpointsManager(object):

    def __init__(self, models_path: Path, actor: Model, critic: Model):
        self.actor = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=actor.optimizer, net=actor)
        self.critic = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=critic.optimizer, net=critic)

        self.progress_actor = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=actor.optimizer, net=actor)
        self.progress_critic = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=critic.optimizer, net=critic)

        self.actor_manager = tf.train.CheckpointManager(
            self.actor, str(models_path) + "\\actor_ckpts", max_to_keep=3)
        self.critic_manager = tf.train.CheckpointManager(
            self.critic, str(models_path) + "\\critic_ckpts",  max_to_keep=3)

        self.progress_actor_manager = tf.train.CheckpointManager(
            self.progress_actor, str(models_path) + "\\progress_actor_ckpts", max_to_keep=None)
        self.progress_critic_manager = tf.train.CheckpointManager(
            self.progress_critic, str(models_path) + "\\progress_critic_ckpts", max_to_keep=None)

    def step_checkpoints(self):
        self.actor.step.assign_add(1)
        self.critic.step.assign_add(1)
        self.progress_actor.step.assign_add(1)
        self.progress_critic.step.assign_add(1)

    def save_progress_ckpts(self):
        self.progress_actor_manager.save()
        self.progress_critic_manager.save()

    def save_ckpts(self):
        actor_file = self.actor_manager.save()
        critic_file = self.critic_manager.save()

        return actor_file, critic_file
