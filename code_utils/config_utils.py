import os
import sys
import json
import logging
from pathlib import Path

from dataclasses import dataclass

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from models import ActorConfig, CriticConfig


@dataclass
class TrainingConfig(object):
    train_steps: int
    save_policy_every: int
    show_every: int


@dataclass
class BaseAgentConfig(object):
    name: str
    desc: str
    env: str
    discount: float


class ConfigManager(object):

    def __init__(self, config_dict: dict):
        self.agent_config = BaseAgentConfig(**config_dict["agent_config"])
        self.critic_config = CriticConfig(**config_dict["critic_config"])
        self.actor_config = ActorConfig(**config_dict["actor_config"])
        self.training_config = TrainingConfig(**config_dict["training_config"])

    def log_configurations(self, logger: logging.Logger):

        logger.info("Used configurations:")
        for key, value in self.__dict__.items():
            logger.info(f"\t{key}: {value}")

    def to_json_file(self, output_file: Path):
        json_data = {}
        for key, value in self.__dict__.items():
            json_data[key] = value.__dict__

        with open(output_file, "w", encoding="utf8") as f:
            json.dump(json_data, f, indent=4)

    @classmethod
    def from_json_file(cls, config_file: Path):
        config_dict = cls.read_json_config(config_file)
        return cls(config_dict)

    @staticmethod
    def read_json_config(config_file: Path):

        with open(config_file, "r", encoding="utf8") as cfile:
            config_dict = json.load(cfile)
        return config_dict

    def as_single_dict(self) -> dict:
        data = {}
        for key, value in self.__dict__.items():
            data.update(value.__dict__)

        return data
