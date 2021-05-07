import numpy as np
from typing import List


# Small epsilon value for stabilizing division operations
EPS = np.finfo(np.float32).eps.item()


class Episode(object):
    """A single episode of an environment."""

    def __init__(self, states: list, actions: list, rewards: list, action_space: str,
                 discount: float):
        """Creates a new episode.

        Args:
            states: A list with all the states that occurred in the episode
            rewards: A list with all the rewards obtained in the episode
            actions: A list with all the actions taken in the episode
            action_space: The type of action space. One of ["continuous", "discrete"]
            discount: The discount factor for discounted rewards
        """
        assert len(states) == len(rewards) == len(actions)
        self.states = np.array(states, dtype=np.float32)
        self.rewards = np.array(rewards, dtype=np.float32)

        if action_space == "discrete":
            self.actions = np.array(actions, dtype=np.int32)
        elif action_space == "continuous":
            self.actions = np.array(actions, dtype=np.float32)
        else:
            raise ValueError(f"Found unsupported action space type = {action_space}")

        self.total_reward = np.sum(self.rewards)

        self.discount = discount
        self.discounted_rewards = self._get_discounted_rewards(rewards)

    def _get_discounted_rewards(self, reward_list) -> np.array:

        discounted_reward_list = []
        reward_list.reverse()
        for i in range(len(reward_list)):
            if i == 0:
                discounted_reward = reward_list[i]
                discounted_reward_list.append(discounted_reward)
            else:
                discounted_reward = self.discount * discounted_reward_list[i - 1] + reward_list[i]
                discounted_reward_list.append(discounted_reward)

        discounted_reward_list.reverse()
        discounted_reward_list = np.array(discounted_reward_list, dtype=np.float32)

        return discounted_reward_list

    def __len__(self) -> int:
        """
        Returns:
            The length of the episode.
        """
        return len(self.states)


class Environment(object):
    """Base class to create environments that can be used to train an
    actor critic agent. All methods need to be implemented.
    """
    def __init__(self, env, action_space_n: int, state_space_n: int,
                 actions: List[str], state_names: List[str]=None):
        """Create a new Environment object.

        Args:
            env: An object with the environment implementation
            action_space_n: The number of possible actions
            state_space_n: The length of the state vector representation
            actions: A list with the actions names
            state_names: A list with the state attributes names
        """
        self.env = env
        self.action_space_n = action_space_n
        self.state_space_n = state_space_n
        self.actions = actions
        self.state_names = state_names

    def reset_environment(self):
        """Reset the environment to start a new episode."""
        raise NotImplementedError

    def get_normalized_state(self) -> np.array:
        """Get the current state of the environment with each
        state attribute normalized in [0, 1], ready to be fed to a NN.

        Returns:
            The current normalized state (np.array)
        """
        raise NotImplementedError

    def get_state(self) -> np.array:
        """Get the current state of the environment.

        Returns:
            The current state (np.array)
        """
        raise NotImplementedError

    def environment_step(self, action: int) -> (np.array, float, bool):
        """Make a move in the environment with given action.

        Args:
            action: The action index or action value

        Returns:
            next_environment_state (np.array), reward (float), terminated_environment (bool)
        """
        raise NotImplementedError

    def render_environment(self):
        """Render the environment."""
        raise NotImplementedError

    @staticmethod
    def win_condition(episode: Episode):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    @staticmethod
    def pass_test(rewards: List[float]):
        raise NotImplementedError
