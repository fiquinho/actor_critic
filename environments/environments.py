import numpy as np


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
            discounted_reward = self.discount * sum(discounted_reward_list) + reward_list[i]
            discounted_reward_list.append(discounted_reward)

        discounted_reward_list.reverse()
        return np.array(discounted_reward_list, dtype=np.float32)

    def __len__(self) -> int:
        """
        Returns:
            The length of the episode.
        """
        return len(self.states)
