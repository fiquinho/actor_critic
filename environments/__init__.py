from .cart_pole import CartPoleEnvironment


def get_env(name: str):

    if name == "cart_pole":
        return CartPoleEnvironment
    else:
        raise ValueError(f"Environment {name} was not found.")
