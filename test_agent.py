import argparse
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent))

from code_utils import ConfigManager
from agents import BaseActorCriticAgent


def main():
    parser = argparse.ArgumentParser(description="Test a trained agent on it's environment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('REQUIRED named arguments')
    required_named.add_argument("--experiment_dir", type=str, required=True,
                                help="The path to a trained agent directory.")
    parser.add_argument("--episodes", type=int, default=200,
                        help="The number of episodes to play during testing.")
    parser.add_argument("--render_games", action="store_true", default=False,
                        help="Activate to render the agent playing each episode.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    config_file = Path(experiment_dir, "configurations.json")
    config = ConfigManager.from_json_file(config_file)

    # Load agent
    agent = BaseActorCriticAgent(agent_path=experiment_dir, config=config)

    agent.load_model(Path(experiment_dir, "models"))

    results = []
    rewards = []
    for i in range(args.episodes):
        episode = agent.play_game(plot_game=args.render_games, delay=0.001)
        win = agent.env.win_condition(episode)
        results.append(win)
        rewards.append(episode.total_reward)

        print(f"Episode = {i} - Total Reward = {episode.total_reward} - Victory = {win} - "
              f"Episode length = {len(episode)}")

    if results[0] is not None:
        print(f"Agent performance = {sum(results) * 100 / len(results)} % of Wins")
        print(f"Mean reward = {sum(rewards) / len(rewards)}")
    agent.env.close()


if __name__ == '__main__':
    main()
