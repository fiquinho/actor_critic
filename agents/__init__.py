from .base_agent import BaseActorCriticAgent
from .batch_actor_critic import BatchActorCriticAgent
from .REINFORCE_with_baseline import REINFORCEwBaselineAgent
from .online_actor_critic import OnlineActorCriticAgent
from .advantage_actor_critic import A2C


def get_agent(name: str) -> BaseActorCriticAgent:

    if name == "batch_AC":
        return BatchActorCriticAgent
    if name == "REINFORCE":
        return REINFORCEwBaselineAgent
    if name == "online_AC":
        return OnlineActorCriticAgent
    if name == "A2C":
        return A2C
    else:
        raise ValueError(f"Agent type {name} was not found.")
