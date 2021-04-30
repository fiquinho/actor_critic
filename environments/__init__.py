from .cart_pole import CartPoleEnvironment
from .environments import Episode, Environment
from .pixelcopter import PixelcopterEnvironment


def get_env(name: str) -> Environment:

    if name == "cart_pole":
        return CartPoleEnvironment
    elif name == "pixelcopter":
        return PixelcopterEnvironment
    else:
        raise ValueError(f"Environment {name} was not found.")
