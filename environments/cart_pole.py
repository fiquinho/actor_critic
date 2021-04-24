from typing import List

import gym
import numpy as np

from .environments import Episode, Environment


class CartPoleEnvironment(Environment):

    def __init__(self):
        env = gym.make("CartPole-v0")
        action_space = 2
        state_space = env.observation_space.shape[0]
        actions = ["left", "right"]
        state_names = ["cart_position", "cart_velocity",
                       "pole_angle", "pole_angular_velocity"]
        assert state_space == len(state_names)

        Environment.__init__(self, env, action_space, state_space, actions, state_names)

    def reset_environment(self):
        self.env.reset()

    def get_environment_state(self) -> np.array:
        return np.array(self.env.state)

    def environment_step(self, action: int) -> (np.array, int, bool):
        """Do a move in the environment.

        Args:
            action: The action to take

        Returns:
            The next state, the reward obtained by doing the action, and if the environment is terminated
        """
        next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done

    def render_environment(self):
        self.env.render()

    @staticmethod
    def pass_test(rewards: List[float]):
        if np.mean(rewards) >= 195.:
            return True
        else:
            return False

    def close(self):
        self.env.close()

    @staticmethod
    def win_condition(episode: Episode):
        if episode.total_reward >= 195.:
            return True
        else:
            return False


def main():
    env = CartPoleEnvironment()

    for i in range(2):
        print(f"Episode: {i}")
        done = False
        env.reset_environment()
        while not done:
            action = np.random.randint(0, 2)
            print(action)
            next_state, reward, done = env.environment_step(action)
            env.render_environment()
    env.close()
    pass


if __name__ == '__main__':
    main()
