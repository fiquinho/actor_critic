import os
import sys
from typing import List
from pathlib import Path

import pygame
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent))


from environments import Episode, Environment


class FlappyBirdEnvironment(Environment):

    def __init__(self):
        env = FlappyBird()
        self.p = PLE(env, add_noop_action=True)
        self.p.init()
        self.win_score = 10.
        action_space = len(self.p.getActionSet())
        state_space = len(self.p.getGameState())
        actions = ["up", "nothing"]
        state_names = list(self.p.getGameState().keys())

        Environment.__init__(self, env, action_space, state_space, actions, state_names)

    def reset_environment(self):
        self.p.reset_game()

    def get_state(self) -> np.array:
        state = list(self.p.getGameState().values())
        state = np.array(state)
        return state

    def get_normalized_state(self) -> np.array:
        """Get the current state of the environment with each
        state attribute normalized in [0, 1], ready to be fed to a NN.

        Returns:
            The current normalized state (np.array)
        """

        state = self.get_state()

        states_mins = np.array([0., -10., 0., 0., 0., 0., 0., 0.])
        states_maxs = np.array([512., 10., 288 * 2., 512., 512., 288 * 2., 512., 512.])
        state = (state - states_mins) / (states_maxs - states_mins)
        return state

    def environment_step(self, action: int) -> (np.array, int, bool):
        """Do a move in the environment.

        Args:
            action: The action to take

        Returns:
            The next state, the reward obtained by doing the action, and if the environment is terminated
        """
        p_action = self.p.getActionSet()[action]
        reward = self.p.act(p_action)
        done = self.p.game_over()
        if self.p.score() >= self.win_score:
            done = True
        next_state = self.get_state()
        return next_state, reward, done

    def render_environment(self):
        self.p.display_screen = True
        self.p.force_fps = False

    def pass_test(self, rewards: List[float]):
        if np.mean(rewards) >= self.win_score:
            return True
        else:
            return False

    def close(self):
        pygame.quit()

    def win_condition(self, episode: Episode):
        if episode.total_reward >= self.win_score:
            return True
        else:
            return False


def main():
    env = FlappyBirdEnvironment()

    for i in range(1):
        print(f"Episode: {i}")
        done = False
        env.reset_environment()
        print(f"Start state = {env.get_state()}")

        while not done:
            # action = np.random.randint(0, 2)
            action = 1
            print(f"Action = {action}")
            next_state, reward, done = env.environment_step(action)
            env.render_environment()
            print(f"New state = {next_state}")
            print(f"Reward = {reward}")

    env.close()
    pass


if __name__ == '__main__':
    main()
