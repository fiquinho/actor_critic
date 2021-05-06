from .base_agent import BaseActorCriticAgent
from .batch_actor_critic import BatchActorCriticAgent
from .REINFORCE_with_baseline import REINFORCEwBaselineAgent


def get_agent(name: str) -> BaseActorCriticAgent:

    if name == "batch_AC":
        return BatchActorCriticAgent
    if name == "REINFORCE":
        return REINFORCEwBaselineAgent
    else:
        raise ValueError(f"Agent type {name} was not found.")
