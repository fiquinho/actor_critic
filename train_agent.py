import argparse
import logging
import os
import sys
import shutil
import json
import time
from pathlib import Path

import wandb
import tensorflow as tf
from git import Repo

from code_utils import prepare_file_logger, prepare_stream_logger, ConfigManager
from agents import get_agent

logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


EXPERIMENTS_DIR = Path("experiments")
SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
CONFIGS_DIR = Path(SCRIPT_DIR.parent, "configurations")


def main():
    parser = argparse.ArgumentParser(description="Train an Actor-Critic agent that plays a specific environment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('REQUIRED named arguments')
    required_named.add_argument("--config_file", type=str, required=True,
                                help="Configuration file for the experiment.")
    parser.add_argument("--output_dir", type=str, default=EXPERIMENTS_DIR,
                        help="Where to save the experiment files")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Activate to run Tensorflow in eager mode.")
    parser.add_argument("--replace", action="store_true", default=False,
                        help="Activate to replace old experiment with the same name in the output folder.")
    args = parser.parse_args()

    # On debug mode all functions are executed normally (eager mode)
    if args.debug:
        tf.config.run_functions_eagerly(True)

    # Get git version
    repo = Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # Use provided configurations file
    config_file = Path(args.config_file)
    config = ConfigManager.from_json_file(config_file)

    # Create experiment folder and handle old results
    output_dir = Path(args.output_dir)
    agent_folder = Path(output_dir, config.agent_config.name)
    deleted_old = False
    if agent_folder.exists():
        if args.replace:
            shutil.rmtree(agent_folder)
            deleted_old = True
        else:
            raise FileExistsError(f"The experiment {agent_folder} already exists."
                                  f"Change output folder, experiment name or use -replace "
                                  f"to overwrite.")
    agent_folder.mkdir(parents=True)

    # Save experiments configurations and start experiment log
    prepare_file_logger(logger, logging.INFO, Path(agent_folder, "experiment.log"))
    logger.info(f"Running experiment {config.agent_config.name}")
    if deleted_old:
        logger.info(f"Deleted old experiment in {agent_folder}")
    config.log_configurations(logger)
    experiment_config_file = Path(agent_folder, "configurations.json")
    logger.info(f"Saving experiment configurations to {experiment_config_file}")
    config.to_json_file(experiment_config_file)

    wandbrun = wandb.init(project=f"AC-{config.agent_config.env}",
                          name=config.agent_config.name,
                          group=config.agent_config.agent_type,
                          notes=config.agent_config.desc,
                          config=config.as_single_dict(),
                          reinit=True,
                          dir=f"experiments/{config.agent_config.name}")

    # Create agent
    agent = get_agent(config.agent_config.agent_type)(agent_path=agent_folder, config=config)

    start_time = time.time()
    test_reward = agent.train_policy(training_config=config.training_config)
    train_time = time.time() - start_time

    experiment_info = {"mean_test_reward": float(test_reward),
                       "name": config.agent_config.name,
                       "description": config.agent_config.desc,
                       "git_hash": sha,
                       "train_time": train_time}
    with open(Path(agent_folder, "experiment_information.json"), "w") as outfile:
        json.dump(experiment_info, outfile, indent=4)

    wandbrun.finish()


if __name__ == '__main__':
    main()
