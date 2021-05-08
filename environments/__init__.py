from .cart_pole import CartPoleEnvironment
from .environments import Episode, Environment
from .pixelcopter import PixelcopterEnvironment
from .flappy_bird import FlappyBirdEnvironment


def get_env(name: str) -> Environment:

    if name == "cart_pole":
        return CartPoleEnvironment
    elif name == "pixelcopter":
        return PixelcopterEnvironment
    elif name == "flappybird":
        return FlappyBirdEnvironment
    else:
        raise ValueError(f"Environment {name} was not found.")
