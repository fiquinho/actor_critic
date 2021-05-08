import sys
import os
from pathlib import Path

import tensorflow as tf


SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent))

from code_utils import ConfigManager
from agents import BaseActorCriticAgent


tf.config.run_functions_eagerly(True)


experiment_name = "flappybird_REINFORCE_default"

experiment_dir = Path("experiments", experiment_name)
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
