from .base_agent import BaseActorCriticAgent
from .batch_actor_critic import BatchActorCriticAgent


def get_agent(name: str) -> BaseActorCriticAgent:

    if name == "batch_AC":
        return BatchActorCriticAgent
    else:
        raise ValueError(f"Agent type {name} was not found.")
