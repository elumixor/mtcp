from typing import Literal


def get(path: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__api_method__ = "GET"
        wrapper.__api_path__ = path

        return wrapper

    return decorator


def post(path: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__api_method__ = "POST"
        wrapper.__api_path__ = path

        return wrapper

    return decorator


def get_handlers(instance, method: Literal["GET", "POST"]):
    return [
        getattr(instance, attr)
        for attr in dir(instance)
        if callable(getattr(instance, attr))
        and hasattr(getattr(instance, attr), "__api_method__")
        and getattr(getattr(instance, attr), "__api_method__") == method
    ]
