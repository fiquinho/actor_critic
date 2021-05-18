from subprocess import call
from pathlib import Path


configs_dir = Path("C://", "Users", "Fico", "rl_experiments",
                   "policy_gradient", "Pendulum-v0", "experiments_configurations", "21-10-20 - 2")

experiments = [
    {
        "name": "00.03",
        "env": "Pendulum-v0",
        "agent": "naive",
        "config_file": str(Path(configs_dir, "naive_default.json")),
        "desc": "Normalized rewards.",
    },
    {
        "name": "00.03",
        "env": "Pendulum-v0",
        "agent": "reward_to_go",
        "config_file": str(Path(configs_dir, "reward_to_go_default.json")),
        "desc": "Normalized rewards.",
    },
    {
        "name": "00.03",
        "env": "Pendulum-v0",
        "agent": "REINFORCE",
        "config_file": str(Path(configs_dir, "REINFORCE_default.json")),
        "desc": "Normalized rewards.",
    }

]

for config in experiments:
    call_list = ["python", "train_agent.py",
                 "--name", config["name"],
                 "--config_file", config["config_file"],
                 "--env", config["env"],
                 "--agent", config["agent"],
                 "--desc", config["desc"],
                 "--replace"]

    call(call_list)
