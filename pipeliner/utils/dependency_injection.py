from typing import TypeVar, Type
from functools import wraps

__services_dict = {}


T = TypeVar('T')


def inject(service: Type[T]) -> T:
    if service not in __services_dict:
        raise Exception(f"Service \"{service.__name__}\" not provided")

    return __services_dict[service]


def provide(service, instance):
    if service in __services_dict:
        raise Exception(f"Service \"{service.__name__}\" already provided")

    __services_dict[service] = instance


def auto_provided(service):
    @wraps(service)
    def wrapper(*args, **kwargs):
        instance = service(*args, **kwargs)
        provide(wrapper, instance)
        return instance
    return wrapper
