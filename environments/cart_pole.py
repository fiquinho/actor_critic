import gym
import numpy as np


class CartPoleEnvironment(object):

    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.action_space = 2
        self.state_space = self.env.observation_space.shape[0]
        self.actions = ["left", "right"]
        self.state_names = ["cart_position", "cart_velocity",
                            "pole_angle", "pole_angular_velocity"]
        assert self.state_space == len(self.state_names)

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

    def close(self):
        self.env.close()


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
