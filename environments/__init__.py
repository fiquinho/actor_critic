from .cart_pole import CartPoleEnvironment
from .environments import Episode, Environment


def get_env(name: str) -> Environment:

    if name == "cart_pole":
        return CartPoleEnvironment
    else:
        raise ValueError(f"Environment {name} was not found.")
