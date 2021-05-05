"""
Neural Network models used to represent stochastic policies.
The models are implemented as sub-classes of tf.keras.Models.
The training loop is custom as well as the training step.
Using @tf.function decorators for performance optimization.
"""

import logging
from typing import Tuple, List

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from dataclasses import dataclass


logger = logging.getLogger()


@dataclass
class ActorConfig(object):
    layer_sizes: List[int]
    learning_rate: float
    hidden_activation: str = "relu"


def feed_forward_discrete_policy_constructor(input_dim, output_dim):
    """Creates a tf.keras.Model subclass for a Feed Forward Neural Network
    that represents a categorical stochastic policy.
    Args:
        input_dim: The length of the state vector
        output_dim: The number of possible actions

    Returns:
        A class to instantiate the model object.
    """
    class FeedForwardPolicyGradientModel(Model):
        """Feed Forward Neural Network that represents a categorical stochastic policy.
        The input and output sizes are already defined.
        """

        def __init__(self, actor_config: ActorConfig):
            """
            Creates a new FFNN model to represent a policy. Implements all needed
            methods from tf.keras.Model.
            Args:
                actor_config: The model configurations
            """

            super(FeedForwardPolicyGradientModel, self).__init__()
            self.output_size = output_dim
            self.input_size = input_dim
            self.model_config = actor_config

            self.hidden_layers = []
            for i in self.model_config.layer_sizes:
                self.hidden_layers.append(Dense(i, activation=self.model_config.hidden_activation,
                                                name=f"hidden_{len(self.hidden_layers)}"))

            self.output_logits = Dense(output_dim, activation=None, name="output_logits")

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.model_config.learning_rate)

        def get_config(self):
            """Used by tf.keras to load a saved model."""
            return {"layer_sizes": self.model_config.layer_sizes,
                    "learning_rate": self.model_config.learning_rate,
                    "hidden_activation": self.model_config.hidden_activation}

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32), ))
        def call(self, inputs: tf.Tensor):
            """See base Class."""

            logger.info("[Retrace] call")
            x = inputs
            for layer in self.hidden_layers:
                x = layer(x)
            logits = self.output_logits(x)
            return logits

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None], dtype=tf.int32),
                                      tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def train_step(self, sates: tf.Tensor, actions: tf.Tensor,
                       weights: tf.Tensor) -> (Tuple[tf.Tensor], tf.Tensor, tf.Tensor):
            """See base Class."""

            logger.info("[Retrace] train_step")
            with tf.GradientTape() as tape:
                logits = self(sates)
                action_masks = tf.one_hot(actions, self.output_size)
                log_probabilities = tf.reduce_sum(action_masks * self._get_log_probabilities(logits), axis=-1)
                loss = -tf.reduce_sum(weights * log_probabilities)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            return (logits, ), loss, log_probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
        def get_probabilities(self, states: tf.Tensor) -> tf.Tensor:
            """Gets the actual probabilities of each action for each set of logits.

            Args:
                states: The states from where to get the action probabilities

            Returns:
                The probabilities (for each state they add up to 1)
            """

            logger.info("[Retrace] get_probabilities")
            logits = self(states)
            probabilities = tf.nn.softmax(logits)
            return probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, output_dim], dtype=tf.float32)])
        def _get_log_probabilities(self, logits: tf.Tensor) -> tf.Tensor:
            """Gets the logarithmic probabilities of each action for each set of logits.

            Args:
                logits: The output of this model: self(states)

            Returns:
                The logarithmic probabilities
            """

            logger.info("[Retrace] get_log_probabilities")
            log_probabilities = tf.nn.log_softmax(logits)
            return log_probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
        def produce_actions(self, states: tf.Tensor) -> tf.Tensor:
            """Get a sample from the action probability distribution produced
            by the model, for each passed state.

            Args:
                states: The list of states representations

            Returns:
                The sampled action for each state
            """

            logger.info("[Retrace] produce_actions")
            logits = self(states)
            actions = tf.random.categorical(logits, 1)
            return actions

    return FeedForwardPolicyGradientModel


def test():
    tf.config.run_functions_eagerly(True)
    tf.random.set_seed(0)

    sample_config = {
        "layer_sizes": [3, 3],
        "learning_rate": 0.001,
        "hidden_activation": "relu",
    }

    model_config = ActorConfig(**sample_config)
    actor_constructor = feed_forward_discrete_policy_constructor(3, 2)
    model = actor_constructor(model_config)

    states = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rewards = np.array([[0.], [0.], [1.]])
    actions = np.array([1, 0, 1])

    (logits,), loss, log_probabilities = model.train_step(states, actions, rewards)
    print(f"logits = {logits}")
    print(f"loss = {loss}")
    print(f"log_probabilities= {log_probabilities}")

    produced_actions = model.produce_actions(states)
    print(f"Produced actions = {produced_actions}")
    print(f"Produced actions probabilities = {model.get_probabilities(states)}")


if __name__ == '__main__':
    test()
