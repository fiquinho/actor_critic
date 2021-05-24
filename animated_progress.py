import sys
import os
import argparse
from pathlib import Path


SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent))

from code_utils import ConfigManager
from agents import BaseActorCriticAgent


def main():
    parser = argparse.ArgumentParser(description="See an agent saved progress during it's training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('REQUIRED named arguments')
    required_named.add_argument("--experiment_dir", type=str, required=True,
                                help="The path to a trained agent directory.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    config_file = Path(experiment_dir, "configurations.json")
    config = ConfigManager.from_json_file(config_file)

    # Load agent
    agent = BaseActorCriticAgent(agent_path=experiment_dir, config=config)

    for checkpoint in agent.ckpts_manager.progress_actor_manager.checkpoints:
        agent.ckpts_manager.progress_actor.restore(checkpoint)
        print(f"Restored model from checkpoint {Path(checkpoint).stem}")

        episode = agent.play_game(plot_game=True, delay=None)
        win = agent.env.win_condition(episode)

        print(f"Policy = {Path(checkpoint).stem} - Total Reward = {episode.total_reward} - Victory = {win} - "
              f"Episode length = {len(episode)}")

    agent.env.close()


if __name__ == '__main__':
    main()
