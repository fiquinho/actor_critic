import logging
import sys
import os
from typing import List
from pathlib import Path

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from code_utils import CriticConfig


logger = logging.getLogger()


class Critic(Model):
    """Feed Forward Neural Network for value function approximation."""

    def __init__(self, critic_config: CriticConfig):
        """Creates a new FFNN for value function approximation. Implements all needed
        methods from tf.keras.Model.

        Args:
            critic_config: The model configurations
        """

        self.model_config = critic_config
        super(Critic, self).__init__()

        self.hidden_layers = []
        for i in self.model_config.layer_sizes:
            self.hidden_layers.append(Dense(i, activation=self.model_config.hidden_activation,
                                            name=f"hidden_{len(self.hidden_layers)}"))

        self.value = Dense(1, activation=self.model_config.output_activation, name="value")

        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.model_config.learning_rate)

    def get_config(self):
        """Used by tf.keras to load a saved model."""
        return {"layer_sizes": self.model_config.layer_sizes,
                "learning_rate": self.model_config.learning_rate,
                "hidden_activation": self.model_config.hidden_activation,
                "output_activation": self.model_config.output_activation}

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """See base Class."""

        logger.info("[Retrace] call")
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        value = self.value(x)

        return value

    @tf.function
    def train_step(self, states: tf.Tensor, discounted_rewards: tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """See base Class."""

        logger.info("[Retrace] train_step")
        with tf.GradientTape() as tape:
            values = self(states)
            loss = self.loss_object(discounted_rewards, values)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return values, loss, gradients


def test():
    tf.config.run_functions_eagerly(True)
    tf.random.set_seed(0)

    sample_config = {
        "layer_sizes": [3, 3],
        "learning_rate": 0.001,
        "hidden_activation": "relu",
        "output_activation": "linear"
    }

    model_config = CriticConfig(**sample_config)
    model = Critic(model_config)

    state = np.array([[1.], [2.], [3.]])
    discounted_rewards = np.array([[0.5], [1.], [1.]])

    values, loss, gradients = model.train_step(state, discounted_rewards)
    print(f"Values = {values}")
    print(f"loss = {loss}")
    print(f"gradients train= {gradients}")


if __name__ == '__main__':
    test()
